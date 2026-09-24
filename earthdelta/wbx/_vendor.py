"""Import the vendored official WeatherBench-X / WeatherBench 2 code, fail-closed.

Same pattern as ``stormer_bridge._build_stormer_model``'s use of
``reference/stormer``: the official repos are real git clones under
``reference/`` and are made importable by ``sys.path`` insertion, never by a
pip install. Before anything is imported this module verifies that each clone's
checked-out HEAD is the commit pinned in ``reference/_manifest.json`` (and, by
default, that the tracked files are unmodified). Any mismatch raises
``VendorPinError``; there is no "warn and continue" path.

Environment isolation
---------------------
WeatherBench-X's metric modules ``import jax`` at module top level, and the
frozen ``.pydeps`` directory (the environment every pinned GPU job ran with)
has no jax. CPU jax is therefore installed into a separate ``.pydeps_wbx``
directory (jax 0.6.2 / jaxlib 0.6.2 / ml_dtypes 0.5.1 -- the newest jax that
supports Python 3.10 and numpy 1.26). ``.pydeps_wbx`` is APPENDED to
``sys.path``, never prepended: it can only supply modules that are otherwise
missing and can never shadow a package the pinned code imports. Its top-level
names are also checked to be disjoint from ``.pydeps``.

Nothing in the pinned Fs/bank code imports this module.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference"
MANIFEST_PATH = REFERENCE_ROOT / "_manifest.json"
PYDEPS_DIR = REPO_ROOT / ".pydeps"
PYDEPS_WBX_DIR = REPO_ROOT / ".pydeps_wbx"

#: Vendored repo directory name -> top-level Python package inside it.
VENDORED_PACKAGES: Dict[str, str] = {
    "weatherbenchX": "weatherbenchX",
    "weatherbench2": "weatherbench2",
}

#: Modules that pull apache_beam (directly or transitively) and must never be
#: imported by the EarthDelta evaluation path.
FORBIDDEN_MODULES: Tuple[str, ...] = (
    "apache_beam",
    "weatherbench2.evaluation",
    "weatherbench2.metrics",
    "weatherbenchX.beam_pipeline",
    "weatherbenchX.beam_utils",
)

#: The official modules EarthDelta is allowed to use (for provenance digests).
USED_VENDORED_FILES: Dict[str, Tuple[str, ...]] = {
    "weatherbenchX": (
        "weatherbenchX/__init__.py",
        "weatherbenchX/aggregation.py",
        "weatherbenchX/binning.py",
        "weatherbenchX/weighting.py",
        "weatherbenchX/xarray_tree.py",
        "weatherbenchX/interpolations.py",
        "weatherbenchX/metrics/__init__.py",
        "weatherbenchX/metrics/base.py",
        "weatherbenchX/metrics/deterministic.py",
        "weatherbenchX/metrics/spatial.py",
        "weatherbenchX/metrics/wrappers.py",
        "weatherbenchX/data_loaders/__init__.py",
        "weatherbenchX/data_loaders/base.py",
        "weatherbenchX/data_loaders/xarray_loaders.py",
        "public_benchmark/run_benchmark_evaluation.py",
        "public_benchmark/public_configs.py",
    ),
    "weatherbench2": (
        "weatherbench2/__init__.py",
        "weatherbench2/regridding.py",
        "weatherbench2/schema.py",
    ),
}

#: Minimum numpy required by jax 0.6.2 (the frozen .pydeps ships 1.26.4; the
#: bare system numpy 1.24.4 is too old).
_MIN_NUMPY = (1, 26)


class VendorPinError(RuntimeError):
    """A vendored official repo is missing, unpinned, modified or mis-imported."""


@dataclass(frozen=True)
class VendoredRepo:
    name: str
    path: Path
    pinned_head: str
    actual_head: str
    remote: str
    license_hint: str
    clean: Optional[bool]
    used_files_sha256: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "path": str(self.path),
            "pinned_head": self.pinned_head,
            "actual_head": self.actual_head,
            "remote": self.remote,
            "license_hint": self.license_hint,
            "worktree_clean": self.clean,
            "used_files_sha256": self.used_files_sha256,
        }


# ----------------------------------------------------------------------------
# Pin verification
# ----------------------------------------------------------------------------

def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        raise VendorPinError(f"reference manifest not found: {path}")
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise VendorPinError(f"reference manifest {path} is not a JSON list")
    return {e["dir"]: e for e in entries if isinstance(e, dict) and "dir" in e}


def read_git_head(repo_dir: Path) -> str:
    """Resolve HEAD to a full commit SHA by reading ``.git`` files directly.

    Does not shell out to git (git 2.34 refuses these AFS-owned clones as
    "dubious ownership" unless given ``--git-dir``), so it works anywhere.
    """
    git_dir = repo_dir / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        raise VendorPinError(f"{repo_dir} is not a git clone (no .git/HEAD)")
    head = head_file.read_text().strip()
    if not head.startswith("ref:"):
        sha = head
    else:
        ref = head.split(":", 1)[1].strip()
        loose = git_dir / ref
        sha = ""
        if loose.is_file():
            sha = loose.read_text().strip()
        else:
            packed = git_dir / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 2 and parts[1] == ref:
                        sha = parts[0]
                        break
        if not sha:
            raise VendorPinError(f"cannot resolve {ref} in {git_dir}")
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise VendorPinError(f"malformed HEAD sha {sha!r} in {repo_dir}")
    return sha


#: Tracked source subtrees whose Python files must equal the pinned commit.
_CLEAN_CHECK_TREES: Dict[str, Tuple[str, ...]] = {
    "weatherbenchX": ("weatherbenchX", "public_benchmark"),
    "weatherbench2": ("weatherbench2",),
}


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def modified_tracked_sources(repo_dir: Path, subtrees: Iterable[str], timeout: float = 120.0) -> List[str]:
    """Tracked ``*.py`` files under ``subtrees`` whose content differs from HEAD.

    Compares git blob hashes of the working files with ``git ls-tree -r HEAD``
    (fast on AFS, unlike a full ``git status``). Missing files count as
    modified. Any failure to run git raises; it is never treated as clean.
    """
    proc = subprocess.run(
        ["git", f"--git-dir={repo_dir / '.git'}", "ls-tree", "-r", "HEAD", "--", *subtrees],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise VendorPinError(
            f"git ls-tree failed for {repo_dir} (rc={proc.returncode}): {proc.stderr.strip()[:300]}")
    modified: List[str] = []
    n = 0
    for line in proc.stdout.splitlines():
        meta, _, rel = line.partition("\t")
        parts = meta.split()
        if len(parts) != 3 or parts[1] != "blob" or not rel.endswith(".py"):
            continue
        n += 1
        path = repo_dir / rel
        if not path.is_file() or _git_blob_sha1(path.read_bytes()) != parts[2]:
            modified.append(rel)
    if n == 0:
        raise VendorPinError(f"no tracked Python sources found under {list(subtrees)} in {repo_dir}")
    return modified


def worktree_is_clean(repo_dir: Path) -> bool:
    """True iff every tracked Python source of the used subtrees matches HEAD."""
    subtrees = _CLEAN_CHECK_TREES.get(repo_dir.name, (".",))
    return not modified_tracked_sources(repo_dir, subtrees)


def used_files_digest(repo_dir: Path, rel_paths: Iterable[str]) -> str:
    h = hashlib.sha256()
    for rel in rel_paths:
        p = repo_dir / rel
        if not p.is_file():
            raise VendorPinError(f"expected vendored file missing: {p}")
        h.update(rel.encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


_VERIFIED: Dict[str, VendoredRepo] = {}


def verify_vendored_repo(
    name: str,
    *,
    manifest: Optional[Dict[str, Dict[str, str]]] = None,
    reference_root: Path = REFERENCE_ROOT,
    check_clean: bool = True,
) -> VendoredRepo:
    """Verify one vendored repo against the manifest pin. Raises on mismatch."""
    manifest = manifest if manifest is not None else load_manifest()
    if name not in manifest:
        raise VendorPinError(f"{name} is not listed in reference/_manifest.json")
    entry = manifest[name]
    pinned = str(entry.get("head", "")).strip().lower()
    if len(pinned) < 7:
        raise VendorPinError(f"manifest pin for {name} is too short: {pinned!r}")
    repo_dir = reference_root / name
    actual = read_git_head(repo_dir)
    if not actual.startswith(pinned):
        raise VendorPinError(
            f"{name}: checked-out HEAD {actual} does not match the pinned "
            f"commit {pinned} in reference/_manifest.json"
        )
    clean: Optional[bool] = None
    if check_clean:
        clean = worktree_is_clean(repo_dir)
        if not clean:
            raise VendorPinError(
                f"{name}: tracked files are modified relative to the pinned "
                f"commit {pinned}; refusing to evaluate with patched official code"
            )
    return VendoredRepo(
        name=name,
        path=repo_dir,
        pinned_head=pinned,
        actual_head=actual,
        remote=str(entry.get("remote", "")),
        license_hint=str(entry.get("license_hint", "")),
        clean=clean,
        used_files_sha256=used_files_digest(repo_dir, USED_VENDORED_FILES.get(name, ())),
    )


# ----------------------------------------------------------------------------
# sys.path handling
# ----------------------------------------------------------------------------

def _top_level_names(directory: Path) -> set:
    names = set()
    if not directory.is_dir():
        return names
    for child in directory.iterdir():
        n = child.name
        if n.endswith((".dist-info", ".egg-info")) or n in ("bin", "__pycache__"):
            continue
        names.add(n.split(".")[0])
    return names


def add_wbx_deps_path() -> Optional[Path]:
    """Append ``.pydeps_wbx`` (CPU jax) to ``sys.path`` if it exists.

    Appending (not prepending) means it can only supply modules that are not
    importable otherwise. Also refuses to proceed if it overlaps ``.pydeps``.
    Returns the path added (or already present), or None if absent.
    """
    if not PYDEPS_WBX_DIR.is_dir():
        return None
    overlap = _top_level_names(PYDEPS_WBX_DIR) & _top_level_names(PYDEPS_DIR)
    if overlap:
        raise VendorPinError(
            f".pydeps_wbx shadows frozen .pydeps packages {sorted(overlap)}; "
            "it must contain only packages absent from .pydeps (jax/jaxlib/ml_dtypes)"
        )
    entry = str(PYDEPS_WBX_DIR)
    if entry not in sys.path:
        sys.path.append(entry)
    return PYDEPS_WBX_DIR


def _insert_front(path: Path) -> None:
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _check_module_origin(module_name: str, expected_root: Path) -> None:
    mod = sys.modules.get(module_name)
    if mod is None:
        return
    origin = Path(getattr(mod, "__file__", "") or "").resolve()
    if expected_root.resolve() not in origin.parents:
        raise VendorPinError(
            f"module {module_name!r} is already imported from {origin}, not from "
            f"the pinned clone under {expected_root}"
        )


def assert_no_beam() -> None:
    """Raise if any beam-dependent official module has been imported."""
    loaded = [m for m in sys.modules if any(
        m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)]
    if loaded:
        raise VendorPinError(
            f"forbidden beam-dependent modules imported: {sorted(loaded)}"
        )


def ensure_vendored(*, check_clean: bool = True, need_jax: bool = True) -> Dict[str, VendoredRepo]:
    """Verify pins, then make the official packages importable. Idempotent.

    Args:
        check_clean: also require an unmodified worktree (``git status``).
        need_jax: also make CPU jax importable (append ``.pydeps_wbx``) and
            verify it imports from there with a new-enough numpy.

    Returns:
        name -> VendoredRepo provenance record.
    """
    manifest = load_manifest()
    records: Dict[str, VendoredRepo] = {}
    for name in VENDORED_PACKAGES:
        cache_key = f"{name}:{check_clean}"
        if cache_key not in _VERIFIED:
            _VERIFIED[cache_key] = verify_vendored_repo(
                name, manifest=manifest, check_clean=check_clean)
        records[name] = _VERIFIED[cache_key]
    for name, package in VENDORED_PACKAGES.items():
        repo_dir = records[name].path
        _insert_front(repo_dir)
        _check_module_origin(package, repo_dir)
    if need_jax:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        add_wbx_deps_path()
        import numpy as np
        if tuple(int(x) for x in np.__version__.split(".")[:2]) < _MIN_NUMPY:
            raise VendorPinError(
                f"numpy {np.__version__} is too old for jax 0.6.2 (needs >=1.26); "
                "run with PYTHONPATH=.pydeps (numpy 1.26.4)"
            )
        try:
            jax = importlib.import_module("jax")
        except ImportError as exc:
            raise VendorPinError(
                "jax is not importable; install CPU jax into .pydeps_wbx "
                f"(see earthdelta/wbx/_vendor.py): {exc}"
            ) from exc
        jax_origin = Path(jax.__file__).resolve()
        if PYDEPS_WBX_DIR.is_dir() and PYDEPS_WBX_DIR.resolve() not in jax_origin.parents:
            raise VendorPinError(
                f"jax imported from {jax_origin}, expected the isolated {PYDEPS_WBX_DIR}"
            )
    assert_no_beam()
    return records


def import_official(module_name: str, *, check_clean: bool = True):
    """Import an allowed official module after ``ensure_vendored``."""
    if any(module_name == f or module_name.startswith(f + ".") for f in FORBIDDEN_MODULES):
        raise VendorPinError(f"{module_name} is beam-dependent and forbidden")
    need_jax = module_name.startswith(("weatherbenchX.metrics", "weatherbench2.regridding"))
    ensure_vendored(check_clean=check_clean, need_jax=need_jax)
    mod = importlib.import_module(module_name)
    top = module_name.split(".")[0]
    _check_module_origin(top, REFERENCE_ROOT / top)
    assert_no_beam()
    return mod


def provenance(check_clean: bool = True) -> Dict[str, object]:
    """JSON-able provenance of the official code and jax used."""
    records = ensure_vendored(check_clean=check_clean, need_jax=True)
    import jax
    import numpy as np
    import xarray as xr
    return {
        "vendored": {k: v.to_dict() for k, v in records.items()},
        "jax_version": jax.__version__,
        "jax_origin": str(Path(jax.__file__).resolve().parent),
        "numpy_version": np.__version__,
        "xarray_version": xr.__version__,
        "pydeps_wbx": str(PYDEPS_WBX_DIR),
        "pydeps_wbx_position": "appended (cannot shadow)",
    }


__all__: List[str] = [
    "VendorPinError", "VendoredRepo", "ensure_vendored", "import_official",
    "verify_vendored_repo", "read_git_head", "assert_no_beam", "provenance",
    "add_wbx_deps_path", "FORBIDDEN_MODULES", "PYDEPS_WBX_DIR",
]
