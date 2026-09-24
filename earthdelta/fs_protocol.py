"""FP-03 pre-registration binding for real Fs runs.

Everything a real (non-synthetic) Fs process must establish BEFORE it loads a
backbone or reads a sample, and the provenance block every output carries:

  * the protocol file is byte-identical to the SHA-256 on the command line and
    declares the configuration id being run; the command line may not differ
    from the declared configuration in any declared field;
  * the S0 certificate is the declared file, is a committed PASS, and was
    issued for the checkpoint, normalization identity, torch version and the
    exact bridge/contract/lowrank source files this process has imported;
  * the S0-certified official backend (xformers attention through the pinned
    upstream Stormer class) is importable BEFORE construction and is what was
    actually built AFTER construction -- the bridge's silent SDPA fallback is
    refused rather than recorded after the fact.

Nothing here edits or monkeypatches the certified modules; it only reads them.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

PROTOCOL_SCHEMA = "ed-fs-protocol/1"
OFFICIAL_MODEL_MODULE = "stormer.models.hub.stormer"
OFFICIAL_MODEL_QUALNAME = "Stormer"
#: Modules whose file hashes the S0 certificate records and this process must
#: have loaded byte-identically.
S0_BOUND_MODULES = (
    "earthdelta.contracts",
    "earthdelta.bridge.stormer_bridge",
    "earthdelta.bridge.stormer_arch",
    "earthdelta.lowrank",
)
TF32_ENV_OVERRIDES = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")


class ProtocolViolation(ValueError):
    """A pre-registration, certificate or backend precondition failed."""

    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.detail: Dict[str, Any] = dict(detail or {})


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# Protocol file and declared configurations
# =============================================================================

def load_protocol(path: Path | str, expected_sha256: str) -> Dict[str, Any]:
    """Load the protocol only if its bytes hash to the pre-registered SHA-256."""
    actual = file_sha256(path)
    if actual != str(expected_sha256):
        raise ProtocolViolation(
            "PROTOCOL_SHA256_MISMATCH",
            f"{path} hashes to {actual}, not the declared {expected_sha256}.",
            {"path": str(path), "actual": actual, "expected": str(expected_sha256)},
        )
    protocol = json.loads(Path(path).read_text())
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ProtocolViolation(
            "PROTOCOL_SCHEMA_UNKNOWN",
            f"protocol schema {protocol.get('schema_version')!r} != {PROTOCOL_SCHEMA!r}.",
            {},
        )
    return protocol


def declared_config(protocol: Mapping[str, Any], config_id: str) -> Dict[str, Any]:
    configs = protocol.get("configs") or {}
    if config_id not in configs:
        raise ProtocolViolation(
            "CONFIG_ID_UNDECLARED",
            f"config id {config_id!r} is not declared by the protocol.",
            {"declared": sorted(configs)},
        )
    return dict(configs[config_id])


def _normalize(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return value
    return value


def check_cli_against_config(declared: Mapping[str, Any], actual: Mapping[str, Any]) -> List[str]:
    """Every field the configuration declares must equal the command line.

    Keys whose declared value is a mapping with ``path``/``sha256`` compare
    both the path and the file's actual bytes. Returns the mismatches.
    """
    mismatches: List[str] = []
    for key, want in declared.items():
        if key.startswith("_") or key in ("job", "role", "description", "expected"):
            continue
        if key not in actual:
            mismatches.append(f"{key}: declared {want!r} but the CLI has no such value")
            continue
        got = actual[key]
        if isinstance(want, Mapping) and "one_of" in want:
            # e.g. the formal replicas: byte-identical argv except the device.
            if _normalize(got) not in [_normalize(v) for v in want["one_of"]]:
                mismatches.append(f"{key}: {got!r} not in declared {want['one_of']!r}")
            continue
        if isinstance(want, Mapping) and "sha256" in want:
            if got is None or str(Path(str(got)).resolve()) != str(Path(str(want["path"])).resolve()):
                mismatches.append(f"{key}: path {got!r} != declared {want.get('path')!r}")
            elif file_sha256(got) != want["sha256"]:
                mismatches.append(f"{key}: sha256 of {got} != declared {want['sha256']}")
            continue
        if isinstance(want, float) or isinstance(got, float):
            try:
                if float(want) != float(got):
                    mismatches.append(f"{key}: {got!r} != declared {want!r}")
            except (TypeError, ValueError):
                mismatches.append(f"{key}: {got!r} != declared {want!r}")
            continue
        if _normalize(got) != _normalize(want):
            mismatches.append(f"{key}: {got!r} != declared {want!r}")
    return mismatches


# =============================================================================
# S0 certificate consumption
# =============================================================================

def loaded_module_hashes(names: Iterable[str] = S0_BOUND_MODULES) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for name in names:
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None) if module is not None else None
        out[name] = {"path": str(path), "sha256": file_sha256(path) if path else "NOT_LOADED"}
    return out


def consume_s0_certificate(
    path: Path | str,
    *,
    expected_sha256: str,
    gate_config: Any,
    torch_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Refuse to proceed unless the S0 certificate binds THIS process.

    ``gate_config`` is the `GateIdentityConfig` the process will load its
    backbone from. Checks: file SHA-256, committed PASS with every criterion
    True, checkpoint SHA, normalization identity, config digest, torch version,
    and the SHA-256 of every S0-bound module actually imported here.
    """
    actual_sha = file_sha256(path)
    failures: List[str] = []
    if actual_sha != str(expected_sha256):
        failures.append(f"certificate sha256 {actual_sha} != declared {expected_sha256}")
    cert = json.loads(Path(path).read_text())
    if cert.get("s0_gate_pass") is not True or cert.get("verdict_committed") is not True:
        failures.append("s0_gate_pass and verdict_committed are not both true")
    if cert.get("status") != "ok":
        failures.append(f"status {cert.get('status')!r} != 'ok'")
    criteria = cert.get("gate_criteria") or {}
    if not criteria or not all(v is True for v in criteria.values()):
        failures.append(f"not every gate criterion is True: {criteria}")
    if cert.get("computed_checkpoint_sha256") != getattr(gate_config, "expected_checkpoint_sha256", None):
        failures.append("checkpoint sha256 differs from the gate config")
    if cert.get("normalization_identity") != getattr(gate_config, "expected_normalization_identity", None):
        failures.append("normalization identity differs from the gate config")
    if cert.get("config_digest") != getattr(gate_config, "config_digest", None):
        failures.append("gate config digest differs from the certificate")
    running = torch_version or torch.__version__
    if cert.get("torch_version") != running:
        failures.append(f"certificate torch {cert.get('torch_version')!r} != running {running!r}")
    recorded = (cert.get("source_identity") or {}).get("modules") or {}
    loaded = loaded_module_hashes()
    for name, info in loaded.items():
        want = (recorded.get(name) or {}).get("sha256")
        if want is None or want != info["sha256"]:
            failures.append(f"module {name} sha256 {info['sha256']} != certified {want}")
    report = {
        "check": "s0_certificate_consumed",
        "path": str(path),
        "sha256": actual_sha,
        "run_id": cert.get("run_id"),
        "torch_version": cert.get("torch_version"),
        "normalization_identity": cert.get("normalization_identity"),
        "checkpoint_sha256": cert.get("computed_checkpoint_sha256"),
        "loaded_module_hashes": loaded,
        "passed": not failures,
        "failures": failures,
    }
    if failures:
        raise ProtocolViolation("S0_CERTIFICATE_NOT_BINDING", "; ".join(failures), report)
    return report


# =============================================================================
# Official backend: precondition before construction, postcondition after
# =============================================================================

def official_backend_precondition(
    *, expected_xformers: str, expected_torch: Optional[str] = None
) -> Dict[str, Any]:
    """xformers of the certified version must import BEFORE the model is built."""
    failures: List[str] = []
    xformers_version = None
    try:
        import xformers  # noqa: F401
        import xformers.ops  # noqa: F401

        xformers_version = getattr(xformers, "__version__", None)
    except Exception as exc:  # noqa: BLE001 - any import failure is a refusal
        failures.append(f"xformers is not importable: {type(exc).__name__}: {exc}")
    if xformers_version is not None and xformers_version != expected_xformers:
        failures.append(f"xformers {xformers_version} != certified {expected_xformers}")
    if expected_torch is not None and torch.__version__ != expected_torch:
        failures.append(f"torch {torch.__version__} != certified {expected_torch}")
    report = {"check": "official_backend_precondition", "xformers_version": xformers_version,
              "torch_version": torch.__version__, "passed": not failures, "failures": failures}
    if failures:
        raise ProtocolViolation("OFFICIAL_BACKEND_UNAVAILABLE", "; ".join(failures), report)
    return report


def model_class_identity(model: Any) -> Dict[str, str]:
    cls = type(model)
    return {"module": cls.__module__, "qualname": cls.__qualname__}


def official_backend_postcondition(model: Any) -> Dict[str, Any]:
    """The constructed backbone must BE the pinned upstream class."""
    identity = model_class_identity(model)
    attention = None
    try:
        attention = type(model.blocks[0].attn)
    except (AttributeError, IndexError, TypeError):
        pass
    attention_identity = (
        {"module": attention.__module__, "qualname": attention.__qualname__} if attention else None
    )
    passed = (identity["module"] == OFFICIAL_MODEL_MODULE
              and identity["qualname"] == OFFICIAL_MODEL_QUALNAME
              and attention_identity is not None
              and attention_identity["module"] == OFFICIAL_MODEL_MODULE)
    report = {"check": "official_backend_postcondition", "model_class": identity,
              "attention_class": attention_identity, "official_backend": passed,
              "passed": passed}
    if not passed:
        raise ProtocolViolation(
            "OFFICIAL_BACKEND_NOT_BUILT",
            f"backbone class {identity} is not the pinned upstream "
            f"{OFFICIAL_MODEL_MODULE}.{OFFICIAL_MODEL_QUALNAME}; the bridge fell back.",
            report,
        )
    return report


# =============================================================================
# Provenance block written with every output
# =============================================================================

def _version_of(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:  # noqa: BLE001 - absence is recorded, not raised
        return None
    return getattr(module, "__version__", None)


def imported_source_hashes(roots: Iterable[Path]) -> Dict[str, str]:
    """SHA-256 of every imported module file living under one of ``roots``."""
    resolved = [Path(r).resolve() for r in roots]
    out: Dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        p = Path(path).resolve()
        if any(str(p).startswith(str(root) + os.sep) for root in resolved) and p.is_file():
            out[str(p)] = file_sha256(p)
    return out


def collect_provenance(
    *,
    argv: List[str],
    device: Any,
    source_roots: Iterable[Path],
    model: Any = None,
    input_files: Optional[Mapping[str, Optional[Path | str]]] = None,
    started_utc: Optional[str] = None,
) -> Dict[str, Any]:
    cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    try:
        device_name = torch.cuda.get_device_name(torch.device(device)) if cuda else str(device)
    except Exception:  # noqa: BLE001
        device_name = str(device)
    files = {}
    for key, path in (input_files or {}).items():
        files[key] = ({"path": str(path), "sha256": file_sha256(path)}
                      if path is not None and Path(str(path)).is_file() else None)
    return {
        "schema_version": "ed-fs-provenance/1",
        "argv": list(argv),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "versions": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torchvision": _version_of("torchvision"),
            "xformers": _version_of("xformers"),
            "lightning": _version_of("lightning"),
            "timm": _version_of("timm"),
        },
        "device": str(device),
        "device_name": device_name,
        "tf32": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "env": {k: os.environ.get(k) for k in TF32_ENV_OVERRIDES},
        },
        "determinism": {
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "model_class": model_class_identity(model) if model is not None else None,
        "official_backend": (
            model_class_identity(model)["module"] == OFFICIAL_MODEL_MODULE if model is not None else None
        ),
        "imported_source_sha256": imported_source_hashes(source_roots),
        "input_files": files,
        "started_utc": started_utc,
        "recorded_utc": utc_now(),
    }


def tf32_is_off(provenance: Mapping[str, Any]) -> bool:
    """TF32 is off iff the EFFECTIVE flags, read after the process configured
    torch, are off.

    ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`` is recorded but not decisive: in the
    certified torch 2.3.1 it only initializes ``float32_matmul_precision``
    (ATen/Context.h member initializer); the explicit
    ``torch.backends.cuda.matmul.allow_tf32 = False`` every Fs process issues at
    import resets it to HIGHEST, and cuBLAS consults only that member
    (``Context::allowTF32CuBLAS``), which is exactly what the recorded getter
    returns. When the precision itself was recorded it must be "highest".
    """
    tf32 = provenance.get("tf32") or {}
    precision = tf32.get("float32_matmul_precision")
    return (tf32.get("cuda_matmul_allow_tf32") is False and tf32.get("cudnn_allow_tf32") is False
            and (precision is None or precision == "highest"))
