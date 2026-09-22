#!/usr/bin/env python3
"""S0 Gate verification script for EarthDelta Stormer bridge.

Runs S0-level gate checks to verify bridge correctness before downstream use.
All gate criteria are fail-closed: initialized False, remain False on exceptions.

Gate Criteria:
- identity_config_bound: A frozen --config with a complete expected identity was supplied
- source_identity_match: Imported module hashes recorded; reference was produced
  under the same frozen config and the same pinned upstream source commit
- variable_coordinate_identity_match: Variable order, grid shape and coordinate
  VALUES match the frozen identity (not just shapes)
- ckpt_sha256_bound: Checkpoint SHA-256 hash matches expected value (not just length check)
- strict_load_zero_diff: Checkpoint loads with zero missing/unexpected keys
- version_identity_match: The bridge's computed ArtifactVersion equals the version
  the FROZEN config requires (never compared against itself)
- manifest_identity_match: The reference manifest's checkpoint/normalization identity
  is enforced against both the frozen config AND the loaded runtime objects
- upstream_parity: Bridge output matches official xformers Stormer (<=1e-5)
- input_norm_binding: bridge.normalize(x_raw) reproduces the saved input_norm.pt
- multistep_parity: Every registered (interval, steps) rollout matches the reference
  NUMERICALLY (not merely present, shaped and finite)
- multistep_reference_present: All required multistep reference files are present
- zero_edit_equals_official: Bridge internal consistency (<=1e-6)
- normalization_parity: Normalization identity matches the frozen expected identity
- no_state_leak: Identical inputs produce bit-identical outputs
- outputs_finite: All outputs contain no NaN/Inf
- raw_input_binding: Raw input hash matches expected (if reference exists)

Usage:
    python scripts/s0_gate.py --config PATH [--reference-dir PATH] [--output-dir PATH]

The upstream_parity check requires running export_upstream_reference.py first
in a GPU+xformers environment, against THE SAME --config. Without the reference
artifacts, this check fails closed with a clear message.

Reference directory selection is identity-based, never sort-order based. Pass
--reference-dir to name one explicitly; auto-selection only succeeds when
exactly one candidate fully matches the expected identity.

Environment variables:
    EARTHDELTA_REPO_ROOT: Override SOURCE root detection (code snapshot)
    S0_OUTPUT_DIR: Override the base output directory for results

Exit codes:
    0: All gate criteria passed
    1: One or more gate criteria failed
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import numbers
import os
import shutil
import socket
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

print("S0 Gate script starting...", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)


# =============================================================================
# Path Configuration (no hardcoded absolute paths)
# =============================================================================

def get_repo_root() -> Path:
    """Get the SOURCE root (importable code snapshot).

    This is deliberately distinct from the ASSET root (checkpoints, NPZ, NPY),
    which is named explicitly by the frozen --config. The two may live in
    completely different directories.
    """
    if "EARTHDELTA_REPO_ROOT" in os.environ:
        return Path(os.environ["EARTHDELTA_REPO_ROOT"]).resolve()
    # Default: two levels up from this script
    return Path(__file__).resolve().parent.parent


def get_paths(asset_root: Path, patch_size: int = 4) -> Dict[str, Path]:
    """Get conventional ASSET paths relative to an asset root.

    These are fallbacks only. When a frozen config is supplied, `resolve_paths`
    takes the checkpoint / normalization / input / reference locations straight
    from the config instead.
    """
    return {
        "input_dir": asset_root / "scripts" / "s0_gate_inputs",
        "checkpoint_ps2": asset_root / "checkpoints" / "stormer_1.40625_patch_size_2.ckpt",
        "checkpoint_ps4": asset_root / "checkpoints" / "stormer_1.40625_patch_size_4.ckpt",
        "checkpoint": asset_root / "checkpoints" / f"stormer_1.40625_patch_size_{patch_size}.ckpt",
        "norm_dir": asset_root / "reference" / "stormer" / "normalization_constants",
        "upstream_reference_base": asset_root / "artifacts",
        "default_output_dir": asset_root / "artifacts" / "s0_gate",
    }


SOURCE_ROOT = get_repo_root()
REPO_ROOT = SOURCE_ROOT  # backwards-compatible alias

# Add the source snapshot to Python path
sys.path.insert(0, str(SOURCE_ROOT))
print(f"Source root (code snapshot): {SOURCE_ROOT}", flush=True)

# =============================================================================
# Imports (dependencies must be pre-installed, no auto-pip-install)
# =============================================================================

import numpy as np
import torch
import torch.nn as nn

print(f"NumPy version: {np.__version__}", flush=True)
print(f"PyTorch version: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

try:
    from earthdelta.bridge import (
        Stormer,
        NormalizationContract,
        WeatherStepBridge,
        controlled_rollout,
        load_stormer_checkpoint_detailed,
        CheckpointLoadResult,
        DEFAULT_VARIABLES,
        POLICY_LEGACY,
        POLICY_OFFICIAL_ZERO_DIFF_MEAN,
        _compute_file_sha256,
        check_version_match,
    )
    from earthdelta.contracts import (
        EditPlan,
        reference_plan,
        GateIdentityConfig,
        ArtifactVersion,
        NormalizationIdentityFormatError,
        compare_normalization_identity,
        compute_coordinate_digest,
        compute_normalization_asset_sha256,
    )
    from earthdelta.lowrank import ExpertLoRA
    print("EarthDelta imports successful", flush=True)
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}", flush=True)
    print("Dependencies must be pre-installed. Do not auto-install.", flush=True)
    sys.exit(1)

# Z500 channel index in canonical 69-variable order
Z500_IDX = 11  # geopotential_500 in the canonical 69-variable order
# The index is resolved by NAME at use time so a different (config-supplied)
# variable list cannot silently sample the wrong channel.
Z500_NAME = "geopotential_500"

# Modules whose content hash identifies the source snapshot that actually ran.
SOURCE_IDENTITY_MODULES = (
    "earthdelta.contracts",
    "earthdelta.bridge.stormer_bridge",
    "earthdelta.bridge.stormer_arch",
    "earthdelta.lowrank",
)


def record_source_identity(
    module_names: Sequence[str] = SOURCE_IDENTITY_MODULES,
) -> Dict[str, Any]:
    """Record which module files were actually imported, plus their content hash.

    The source snapshot is identified by what Python really loaded, not by an
    assumption that the code and the assets share a directory.
    """
    modules: Dict[str, Any] = {}
    for name in module_names:
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None) if module is not None else None
        if path and os.path.exists(path):
            modules[name] = {
                "path": os.path.realpath(path),
                "sha256": _compute_file_sha256(path),
            }
        else:
            modules[name] = {"path": path, "sha256": None}
    gate_script = os.path.realpath(__file__)
    modules["scripts.s0_gate"] = {
        "path": gate_script,
        "sha256": _compute_file_sha256(gate_script),
    }
    return {"source_root": str(SOURCE_ROOT), "modules": modules}

# Historical reference RMSE values from Stormer paper (informational only)
# These are approximate and NOT used for gate pass/fail decisions
HISTORICAL_REFERENCE = {
    "z500_rmse_6h": {"value": 30, "unit": "m^2/s^2", "note": "rough estimate from paper"},
    "z500_rmse_24h": {"value": 80, "unit": "m^2/s^2", "note": "rough estimate from paper"},
}


# =============================================================================
# Data loading utilities
# =============================================================================

def load_npy_inputs(
    input_dir: Path,
    intervals: Tuple[int, ...] = (6, 24),
    input_file: str = 'jan2020_full.npy',
) -> Dict[str, np.ndarray]:
    """Load pre-extracted .npy inputs."""
    inputs = {
        'data': np.load(input_dir / input_file),  # (T, V, H, W)
        'lat': np.load(input_dir / 'lat.npy'),
        'lon': np.load(input_dir / 'lon.npy'),
        'inp_mean': np.load(input_dir / 'inp_mean.npy'),
        'inp_std': np.load(input_dir / 'inp_std.npy'),
    }
    for interval in intervals:
        inputs[f'diff_mean_{interval}'] = np.load(input_dir / f'diff_mean_{interval}.npy')
        inputs[f'diff_std_{interval}'] = np.load(input_dir / f'diff_std_{interval}.npy')
    return inputs


def create_normalization_from_npy(
    inputs: Dict[str, np.ndarray],
    policy: str = POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    variables: Optional[List[str]] = None,
    intervals: Tuple[int, ...] = (6, 24),
) -> NormalizationContract:
    """Create NormalizationContract from pre-extracted numpy arrays.

    Args:
        inputs: Dictionary of numpy arrays from load_npy_inputs
        policy: Normalization policy. Default is POLICY_OFFICIAL_ZERO_DIFF_MEAN
                to match official inference.py semantics.
        variables: Variable list/order (default: DEFAULT_VARIABLES).
        intervals: Diff intervals to build transforms for.
    """
    diff_mean: Dict[int, torch.Tensor] = {}
    diff_std: Dict[int, torch.Tensor] = {}
    for interval in intervals:
        raw_mean = inputs[f'diff_mean_{interval}']
        # Under official policy, force diff_mean to zero
        if policy == POLICY_OFFICIAL_ZERO_DIFF_MEAN:
            raw_mean = np.zeros_like(raw_mean)
        diff_mean[int(interval)] = torch.from_numpy(raw_mean).float()
        diff_std[int(interval)] = torch.from_numpy(inputs[f'diff_std_{interval}']).float()

    return NormalizationContract(
        inp_mean=torch.from_numpy(inputs['inp_mean']).float(),
        inp_std=torch.from_numpy(inputs['inp_std']).float(),
        diff_mean=diff_mean,
        diff_std=diff_std,
        variables=list(variables) if variables is not None else DEFAULT_VARIABLES,
        policy=policy,
    )


def compute_raw_input_hash(x_raw: np.ndarray) -> str:
    """Compute SHA-256 hash of raw input for identity binding."""
    return hashlib.sha256(x_raw.tobytes()).hexdigest()[:16]


def resolve_paths(
    config: Optional[GateIdentityConfig],
    asset_root: Optional[Path] = None,
    patch_size: int = 4,
) -> Dict[str, Path]:
    """Resolve asset locations, preferring the frozen config over conventions."""
    if config is None:
        return get_paths(asset_root or SOURCE_ROOT, patch_size)

    root = Path(config.asset_root) if config.asset_root else (asset_root or SOURCE_ROOT)
    defaults = get_paths(root, config.patch_size)
    paths = dict(defaults)
    paths["asset_root"] = root
    paths["checkpoint"] = Path(config.checkpoint_path)
    paths["norm_dir"] = Path(config.normalization_dir)
    if config.input_dir:
        paths["input_dir"] = Path(config.input_dir)
    if config.reference_base_dir:
        paths["upstream_reference_base"] = Path(config.reference_base_dir)
        paths["default_output_dir"] = Path(config.reference_base_dir) / "s0_gate"
    return paths


# =============================================================================
# Gate criterion: ckpt_sha256_bound (FULL comparison, not just length)
# =============================================================================

def verify_ckpt_sha256(
    ckpt_path: Path,
    load_result: Optional[CheckpointLoadResult],
    expected_sha256: Optional[str] = None,
    computed_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify checkpoint SHA-256 matches expected value.

    This criterion now performs FULL SHA-256 comparison, not just length check.
    If expected_sha256 is None, we just compute and record the hash (first run).
    `computed_sha256` lets the caller reuse a hash it already computed from the
    same file instead of re-reading a multi-gigabyte checkpoint.
    """
    result = {
        "criterion": "ckpt_sha256_bound",
        "passed": False,  # fail-closed
        "checkpoint": str(ckpt_path),
        "computed_sha256": None,
        "expected_sha256": expected_sha256,
    }

    try:
        if not ckpt_path.exists():
            result["error"] = f"Checkpoint not found: {ckpt_path}"
            return result

        if computed_sha256 is not None:
            pass
        elif load_result is not None:
            computed_sha256 = load_result.checkpoint_sha256
        else:
            computed_sha256 = _compute_file_sha256(str(ckpt_path))

        result["computed_sha256"] = computed_sha256

        # FULL validation: compare against expected, not just check length
        if expected_sha256 is not None:
            result["match"] = computed_sha256.lower() == expected_sha256.lower()
            result["passed"] = result["match"]
            if not result["passed"]:
                result["error"] = (
                    f"SHA-256 mismatch: expected {expected_sha256}, "
                    f"got {computed_sha256}"
                )
        else:
            # No expected value provided - fail closed (B02 fix)
            # Computed hash is recorded for enrollment/discovery, but cannot certify identity
            result["passed"] = False
            result["reason"] = "IDENTITY_NOT_BOUND: no expected SHA-256 provided"
            result["note"] = "Computed hash recorded for enrollment; pass --expected-sha256 to certify"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# Gate criterion: strict_load_zero_diff
# =============================================================================

def verify_strict_load_zero_diff(load_result: CheckpointLoadResult) -> Dict[str, Any]:
    """Verify strict checkpoint load had zero missing/unexpected keys."""
    result = {
        "criterion": "strict_load_zero_diff",
        "passed": False,  # fail-closed
        "missing_keys_count": len(load_result.missing_keys),
        "unexpected_keys_count": len(load_result.unexpected_keys),
    }

    try:
        result["passed"] = load_result.strict_load_zero_diff

        if load_result.missing_keys:
            result["missing_keys"] = load_result.missing_keys[:10]  # First 10
        if load_result.unexpected_keys:
            result["unexpected_keys"] = load_result.unexpected_keys[:10]

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# Gate criterion: upstream_parity
# =============================================================================

class ReferenceSelectionError(RuntimeError):
    """Raised when a reference directory cannot be selected unambiguously.

    `missing` distinguishes "there is nothing here at all" from "there is
    something here but it does not match / is ambiguous". Only the former is
    reported as an absent reference; the latter must surface its real reason.
    """

    def __init__(self, message: str, missing: bool = False):
        super().__init__(message)
        self.missing = missing


def _load_reference_manifest(ref_dir: Path) -> Optional[Dict[str, Any]]:
    """Load a candidate's manifest.json, or None when unusable."""
    manifest_path = ref_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as handle:
            return json.load(handle)
    except Exception:
        return None


def reference_identity_mismatches(
    manifest: Dict[str, Any], config: GateIdentityConfig
) -> List[str]:
    """Return the identity fields on which a reference manifest disagrees.

    A candidate reference directory only qualifies when every one of these is
    satisfied. An empty list means a full identity match.
    """
    reasons: List[str] = []

    sha = (manifest.get("checkpoint") or {}).get("sha256")
    if not config.expected_checkpoint_sha256:
        reasons.append("config_expected_checkpoint_sha256_unset")
    elif not sha:
        reasons.append("manifest_checkpoint_sha256_missing")
    elif sha.lower() != config.expected_checkpoint_sha256.lower():
        reasons.append("checkpoint_sha256")

    normalization = manifest.get("normalization") or {}
    if normalization.get("policy") != config.normalization_policy:
        reasons.append("normalization_policy")

    actual_identity = normalization.get("identity_digest")
    if config.expected_normalization_identity is None:
        reasons.append("config_expected_normalization_identity_unset")
    elif actual_identity is None:
        # Pre-schema reference: it only carries a bare digest. Treat the bare
        # value as a legacy-schema identity so the comparison fails closed with
        # an explicit format error instead of silently upgrading it.
        legacy = normalization.get("digest")
        if legacy is None:
            reasons.append("normalization_identity_missing")
        else:
            try:
                if not compare_normalization_identity(
                    config.expected_normalization_identity, legacy
                ):
                    reasons.append("normalization_identity")
            except NormalizationIdentityFormatError as exc:
                reasons.append(f"normalization_identity_format_version_mismatch: {exc}")
    else:
        try:
            if not compare_normalization_identity(
                config.expected_normalization_identity, actual_identity
            ):
                reasons.append("normalization_identity")
        except NormalizationIdentityFormatError as exc:
            reasons.append(f"normalization_identity_format_version_mismatch: {exc}")

    pinned = (manifest.get("official_source") or {}).get("pinned_commit")
    if pinned != config.official_commit:
        reasons.append("source_identity")

    manifest_config_digest = (manifest.get("gate_config") or {}).get("digest")
    if manifest_config_digest is None:
        reasons.append("gate_config_digest_missing")
    elif manifest_config_digest != config.config_digest:
        reasons.append("gate_config_digest")

    variables_digest = manifest.get("variables_digest")
    if variables_digest is None:
        reasons.append("variables_digest_missing")
    elif variables_digest != config.variables_digest:
        reasons.append("variable_identity")

    grid_shape = manifest.get("grid_shape")
    if grid_shape is None or tuple(int(v) for v in grid_shape) != tuple(config.grid_shape):
        reasons.append("grid_shape")

    if config.expected_coordinate_digest is not None:
        if manifest.get("coordinate_digest") != config.expected_coordinate_digest:
            reasons.append("coordinate_identity")

    return reasons


def select_reference_dir(
    base_dir: Path,
    patch_size: int,
    config: Optional[GateIdentityConfig] = None,
    explicit: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Select the upstream reference directory by IDENTITY, never by sort order.

    Selection rules:
      * ``explicit`` (from --reference-dir) always wins and must exist.
      * With a frozen config, auto-selection succeeds only when EXACTLY ONE
        candidate fully matches the expected identity (checkpoint SHA, norm
        policy + identity digest, source identity, frozen-config digest,
        variable identity, grid/coordinate identity).
      * Two candidates that merely share the same ps{N} tag are ambiguous and
        must be resolved with --reference-dir; they are never broken by
        alphabetical order.
      * There is NO legacy/fuzzy fallback directory.

    Raises:
        ReferenceSelectionError: When no candidate matches, or more than one does.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_dir():
            raise ReferenceSelectionError(
                f"--reference-dir does not exist or is not a directory: {path}"
            )
        return path, {"selection": "explicit", "reference_dir": str(path)}

    pattern = f"upstream_reference_ps{patch_size}_*"
    candidates = sorted(p for p in base_dir.glob(pattern) if p.is_dir()) if base_dir.exists() else []
    info: Dict[str, Any] = {
        "selection": "auto",
        "candidate_count": len(candidates),
        "candidates": [str(c) for c in candidates],
    }

    if not candidates:
        raise ReferenceSelectionError(
            f"Upstream reference not found in {base_dir} (pattern {pattern}). "
            "Run export_upstream_reference.py against the same --config first.",
            missing=True,
        )

    if config is None:
        # Without a frozen identity we cannot verify a match. Accept only the
        # unambiguous single-candidate case; never break a tie by sort order.
        if len(candidates) == 1:
            info["selection"] = "auto_single_candidate_no_config"
            return candidates[0], info
        raise ReferenceSelectionError(
            f"Ambiguous reference selection: {len(candidates)} candidate directories "
            f"in {base_dir} and no --config to disambiguate them. "
            "Pass --reference-dir to name the directory explicitly."
        )

    matching: List[Path] = []
    rejected: Dict[str, List[str]] = {}
    for candidate in candidates:
        manifest = _load_reference_manifest(candidate)
        if manifest is None:
            rejected[str(candidate)] = ["manifest_missing_or_unreadable"]
            continue
        reasons = reference_identity_mismatches(manifest, config)
        if reasons:
            rejected[str(candidate)] = reasons
        else:
            matching.append(candidate)

    info["rejected"] = rejected
    info["matching"] = [str(m) for m in matching]

    if len(matching) == 1:
        info["selection"] = "auto_identity_match"
        info["reference_dir"] = str(matching[0])
        return matching[0], info

    if not matching:
        raise ReferenceSelectionError(
            f"No reference directory in {base_dir} fully matches the expected identity. "
            f"Rejections: {json.dumps(rejected)}. "
            "Pass --reference-dir to name the directory explicitly."
        )

    raise ReferenceSelectionError(
        f"Ambiguous reference selection: {len(matching)} directories claim the same "
        f"identity ({[str(m) for m in matching]}). Refusing to resolve by sort order; "
        "pass --reference-dir to name the directory explicitly."
    )


def _candidate_reference_dirs(base_dir: Path, patch_size: int) -> List[Path]:
    """All directories matching the ps{N} naming convention, unfiltered."""
    if not base_dir.exists():
        return []
    return sorted(p for p in base_dir.glob(f"upstream_reference_ps{patch_size}_*") if p.is_dir())


def diagnostic_reference_dir(
    base_dir: Path,
    patch_size: int,
    config: Optional[GateIdentityConfig] = None,
    explicit: Optional[Path] = None,
) -> Tuple[Optional[Path], Optional[ReferenceSelectionError]]:
    """Pick a directory to DIAGNOSE, even when it fails the identity match.

    Identity-based selection deliberately refuses a non-matching directory, but
    then the gate can only say "nothing matched". For diagnosis we fall back to
    the single candidate (when there is exactly one) so the failing criterion
    can name the field that actually differs instead of a generic selection
    error. This never grants a pass: the caller still compares every field and
    the selection error is reported alongside.
    """
    try:
        return select_reference_dir(base_dir, patch_size, config, explicit)[0], None
    except ReferenceSelectionError as exc:
        candidates = _candidate_reference_dirs(base_dir, patch_size)
        if len(candidates) == 1:
            return candidates[0], exc
        return None, exc


def find_upstream_reference_dir(
    base_dir: Path,
    patch_size: int,
    config: Optional[GateIdentityConfig] = None,
    explicit: Optional[Path] = None,
) -> Optional[Path]:
    """Identity-based reference lookup; None only when nothing is there.

    Ambiguity is propagated as a ReferenceSelectionError so the calling gate
    criterion reports the real reason instead of silently picking one.
    """
    try:
        return select_reference_dir(base_dir, patch_size, config, explicit)[0]
    except ReferenceSelectionError as exc:
        if exc.missing:
            return None
        raise


def verify_upstream_parity(
    bridge: WeatherStepBridge,
    upstream_base_dir: Path,
    patch_size: int,
    x_raw: np.ndarray,
    device: torch.device,
    tolerance: float = 1e-5,
    config: Optional[GateIdentityConfig] = None,
    reference_dir: Optional[Path] = None,
    input_norm_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Verify bridge output matches official xformers Stormer output.

    This is the core A01 audit fix: comparing bridge output against the
    REAL official code path, not against itself.

    The numeric chain is anchored to the raw input: `x_raw` is pushed through
    THIS bridge's own normalizer and the result must reproduce the reference's
    saved `input_norm.pt`. Without that step the parity number would only prove
    "the bridge agrees with upstream on some tensor read off disk", which says
    nothing about whether that tensor came from this raw input.

    The upstream reference must be exported by export_upstream_reference.py
    running in a GPU+xformers environment. If the reference is missing,
    this check fails closed.
    """
    interval = int(config.interval_hours) if config is not None else 6
    result = {
        "criterion": "upstream_parity",
        "passed": False,  # fail-closed
        "tolerance": tolerance,
        "input_norm_tolerance": input_norm_tolerance,
        "upstream_available": False,
        "input_norm_binding": False,
    }

    try:
        # Find the upstream reference directory (identity-based selection)
        upstream_dir = find_upstream_reference_dir(
            upstream_base_dir, patch_size, config, reference_dir
        )
        if upstream_dir is None:
            result["error"] = (
                f"Upstream reference not found in {upstream_base_dir}. "
                "Run export_upstream_reference.py in a GPU+xformers environment first."
            )
            return result

        result["upstream_dir"] = str(upstream_dir)

        manifest_path = upstream_dir / "manifest.json"
        if not manifest_path.exists():
            result["error"] = (
                f"Manifest not found at {manifest_path}. "
                "Run export_upstream_reference.py in a GPU+xformers environment first."
            )
            return result

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        result["upstream_manifest"] = {
            "timestamp_utc": manifest.get("timestamp_utc"),
            "xformers_version": manifest.get("xformers_version"),
            "checkpoint_sha256": manifest.get("checkpoint", {}).get("sha256"),
            "normalization_policy": manifest.get("normalization", {}).get("policy"),
            "normalization_digest": manifest.get("normalization", {}).get("digest"),
            "normalization_identity": manifest.get("normalization", {}).get("identity_digest"),
        }
        result["upstream_available"] = True

        if config is not None:
            required_files = [
                f"official_output_{i}h_{s}step.pt" for i, s in config.registered_rollouts
            ]
        else:
            required_files = ["official_output_6h_1step.pt", "official_output_6h_4step.pt"]
        missing_files = [f for f in required_files if not (upstream_dir / f).exists()]

        if missing_files:
            result["error"] = f"Missing required reference files: {missing_files}"
            result["passed"] = False
            return result

        input_norm_path = upstream_dir / "input_norm.pt"
        if not input_norm_path.exists():
            result["error"] = f"Missing saved normalized input: {input_norm_path}"
            return result

        # Load upstream outputs
        one_step_name = f"official_output_{interval}h_1step.pt"
        upstream_output_1step = torch.load(upstream_dir / one_step_name, map_location="cpu")
        upstream_input_norm = torch.load(input_norm_path, map_location="cpu")

        # Verify shapes and dtypes
        result["upstream_output_shape"] = list(upstream_output_1step.shape)
        result["upstream_output_dtype"] = str(upstream_output_1step.dtype)

        if config is not None:
            expected_shape = (1, len(config.variables), int(config.grid_shape[0]), int(config.grid_shape[1]))
        else:
            expected_shape = (1, 69, 128, 256)
        if tuple(upstream_output_1step.shape) != expected_shape:
            result["error"] = (
                f"Unexpected upstream output shape: {tuple(upstream_output_1step.shape)}, "
                f"expected {expected_shape}"
            )
            return result

        # Move to device
        upstream_output = upstream_output_1step.to(device)
        input_norm = upstream_input_norm.to(device)

        # --- Raw-input binding: this normalized input must be derived from x_raw
        # by THIS bridge's own normalizer, not merely be some file on disk. ---
        x_raw_t = torch.from_numpy(np.asarray(x_raw)).float().unsqueeze(0).to(device)
        if tuple(x_raw_t.shape) != tuple(input_norm.shape):
            result["error"] = (
                f"Raw input shape {tuple(x_raw_t.shape)} does not match saved "
                f"normalized input shape {tuple(input_norm.shape)}"
            )
            return result
        with torch.no_grad():
            bridge_input_norm = bridge.normalize(x_raw_t)
        norm_diff = float((bridge_input_norm - input_norm).abs().max().item())
        result["input_norm_max_abs_diff"] = norm_diff
        result["input_norm_binding"] = norm_diff <= input_norm_tolerance
        if not result["input_norm_binding"]:
            result["error"] = (
                f"Saved input_norm.pt was NOT produced from this raw input: "
                f"bridge.normalize(x_raw) differs by {norm_diff:.3e} "
                f"(tolerance {input_norm_tolerance:.0e})"
            )
            return result

        # Run bridge on the bound normalized input
        with torch.no_grad():
            bridge_output = bridge.forward_validation(
                bridge_input_norm, bridge.variables, interval=interval, steps=1
            )

        # Verify bridge output shape matches
        if bridge_output.shape != upstream_output.shape:
            result["error"] = (
                f"Shape mismatch: bridge {bridge_output.shape} vs "
                f"upstream {upstream_output.shape}"
            )
            return result

        # Compare
        diff = (bridge_output - upstream_output).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())

        result["max_abs_diff"] = max_diff
        result["mean_abs_diff"] = mean_diff
        result["passed"] = max_diff <= tolerance and result["input_norm_binding"]

        if not result["passed"]:
            result["note"] = (
                f"Bridge output differs from official xformers output by {max_diff:.2e} "
                f"(tolerance: {tolerance:.0e})"
            )

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


# =============================================================================
# Gate criterion: input_norm_binding (raw input -> saved normalized input)
# =============================================================================

def verify_input_norm_binding(
    bridge: WeatherStepBridge,
    x_raw: np.ndarray,
    upstream_base_dir: Path,
    patch_size: int,
    device: torch.device,
    tolerance: float = 1e-6,
    config: Optional[GateIdentityConfig] = None,
    reference_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Require the saved input_norm.pt to be derivable from x_raw by the bridge.

    The raw-content hash check (`verify_raw_input_binding`) proves the raw array
    is the enrolled one. This check proves the saved *normalized* tensor was
    actually derived from that raw array under this bridge's normalization, so
    an unrelated normalized tensor cannot be swapped in underneath a correct
    raw hash.
    """
    result = {
        "criterion": "input_norm_binding",
        "passed": False,  # fail-closed
        "tolerance": tolerance,
    }

    try:
        upstream_dir = find_upstream_reference_dir(
            upstream_base_dir, patch_size, config, reference_dir
        )
        if upstream_dir is None:
            result["reason"] = "IDENTITY_NOT_BOUND: no reference directory to bind against"
            result["error"] = f"Upstream reference not found in {upstream_base_dir}"
            return result

        result["upstream_dir"] = str(upstream_dir)
        input_norm_path = upstream_dir / "input_norm.pt"
        if not input_norm_path.exists():
            result["error"] = f"Missing saved normalized input: {input_norm_path}"
            return result

        saved_norm = torch.load(input_norm_path, map_location="cpu").to(device)
        x_raw_t = torch.from_numpy(np.asarray(x_raw)).float().unsqueeze(0).to(device)

        result["raw_shape"] = list(x_raw_t.shape)
        result["saved_norm_shape"] = list(saved_norm.shape)
        if tuple(x_raw_t.shape) != tuple(saved_norm.shape):
            result["error"] = (
                f"Raw input shape {tuple(x_raw_t.shape)} does not match saved "
                f"normalized input shape {tuple(saved_norm.shape)}"
            )
            return result

        with torch.no_grad():
            derived = bridge.normalize(x_raw_t)

        max_diff = float((derived - saved_norm).abs().max().item())
        result["max_abs_diff"] = max_diff
        result["exact_equality"] = bool(torch.equal(derived, saved_norm))
        result["passed"] = max_diff <= tolerance
        if not result["passed"]:
            result["error"] = (
                f"input_norm.pt is not bridge.normalize(x_raw): max abs diff "
                f"{max_diff:.3e} exceeds tolerance {tolerance:.0e}. The saved "
                "normalized input does not correspond to this raw input."
            )

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


# =============================================================================
# Gate criterion: raw_input_binding
# =============================================================================

def verify_raw_input_binding(
    x_raw: np.ndarray,
    upstream_base_dir: Path,
    patch_size: int,
    config: Optional[GateIdentityConfig] = None,
    reference_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify raw input hash matches expected value from upstream reference.

    This is the raw-content half of the input binding. The normalized-input
    half lives in `verify_input_norm_binding`; a raw hash alone does not prove
    the saved normalized tensor came from this raw input.
    """
    result = {
        "criterion": "raw_input_binding",
        "passed": False,  # fail-closed
    }

    try:
        computed_hash = compute_raw_input_hash(x_raw)
        result["computed_hash"] = computed_hash

        # A frozen expected hash certifies the input without needing a reference.
        if config is not None and config.expected_raw_input_hash is not None:
            result["config_expected_hash"] = config.expected_raw_input_hash
            result["config_match"] = computed_hash == config.expected_raw_input_hash
            if not result["config_match"]:
                result["error"] = (
                    f"Raw input hash mismatch vs frozen config: expected "
                    f"{config.expected_raw_input_hash}, got {computed_hash}"
                )
                return result

        # Find upstream reference
        upstream_dir = find_upstream_reference_dir(
            upstream_base_dir, patch_size, config, reference_dir
        )
        if upstream_dir is None:
            # No reference directory - fail closed (B02 fix)
            result["passed"] = False
            result["reason"] = "IDENTITY_NOT_BOUND: no reference raw input hash available to bind against"
            result["note"] = "Computed hash recorded for enrollment; create upstream reference first"
            return result

        # Check for raw input hash file
        hash_file = upstream_dir / "raw_input_hash.txt"
        if not hash_file.exists():
            # Check manifest for hash
            manifest_path = upstream_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                expected_hash = manifest.get("raw_input_hash")
                if expected_hash:
                    result["expected_hash"] = expected_hash
                    result["match"] = computed_hash == expected_hash
                    result["passed"] = result["match"]
                    if not result["passed"]:
                        result["error"] = f"Raw input hash mismatch: expected {expected_hash}, got {computed_hash}"
                    return result

            # No expected hash in manifest either - fail closed (B02 fix)
            result["passed"] = False
            result["reason"] = "IDENTITY_NOT_BOUND: no reference raw input hash available to bind against"
            result["note"] = "Computed hash recorded for enrollment; add raw_input_hash to manifest"
            return result

        with open(hash_file) as f:
            expected_hash = f.read().strip()

        result["expected_hash"] = expected_hash
        result["match"] = computed_hash == expected_hash
        result["passed"] = result["match"]

        if not result["passed"]:
            result["error"] = f"Raw input hash mismatch: expected {expected_hash}, got {computed_hash}"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# Gate criterion: multistep_reference_present
# =============================================================================

def verify_multistep_reference(
    upstream_base_dir: Path,
    patch_size: int,
    required_steps: Tuple[int, ...] = (1, 4),
    expected_shape: Tuple[int, ...] = (1, 69, 128, 256),
    interval_hours: int = 6,
    config: Optional[GateIdentityConfig] = None,
    reference_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify all required multistep reference files are present and loadable.

    STRUCTURAL check only (presence / shape / finiteness). A finite, correctly
    shaped but numerically wrong reference passes here by design; the numeric
    comparison lives in `verify_multistep_parity` and is a separate criterion.
    """
    if config is not None:
        required_steps = tuple(config.rollout_steps)
        interval_hours = int(config.interval_hours)
        expected_shape = (
            1, len(config.variables),
            int(config.grid_shape[0]), int(config.grid_shape[1]),
        )

    result = {
        "criterion": "multistep_reference_present",
        "passed": False,  # fail-closed
        "required_steps": list(required_steps),
        "expected_shape": list(expected_shape),
    }

    try:
        # An empty registration must never pass vacuously.
        if not required_steps:
            result["error"] = (
                "Empty multistep registration: no (interval, steps) pairs are "
                "registered, so nothing would be verified. Register at least one."
            )
            return result

        upstream_dir = find_upstream_reference_dir(
            upstream_base_dir, patch_size, config, reference_dir
        )
        if upstream_dir is None:
            result["error"] = f"Upstream reference not found in {upstream_base_dir}"
            return result

        result["upstream_dir"] = str(upstream_dir)

        present_files = {}
        missing_files = []
        shape_mismatch = []

        for steps in required_steps:
            fname = f"official_output_{interval_hours}h_{steps}step.pt"
            fpath = upstream_dir / fname

            if not fpath.exists():
                missing_files.append(fname)
                present_files[fname] = False
            else:
                present_files[fname] = True
                # Load and verify shape
                try:
                    tensor = torch.load(fpath, map_location="cpu")
                    if tuple(tensor.shape) != tuple(expected_shape):
                        shape_mismatch.append(
                            f"{fname}: shape {tuple(tensor.shape)} != {tuple(expected_shape)}"
                        )
                    if not torch.isfinite(tensor).all():
                        shape_mismatch.append(f"{fname}: contains non-finite values")
                except Exception as e:
                    shape_mismatch.append(f"{fname}: load error: {e}")

        result["present_files"] = present_files
        result["missing_files"] = missing_files
        result["shape_errors"] = shape_mismatch

        result["passed"] = len(missing_files) == 0 and len(shape_mismatch) == 0

        if not result["passed"]:
            errors = []
            if missing_files:
                errors.append(f"Missing: {missing_files}")
            if shape_mismatch:
                errors.append(f"Shape errors: {shape_mismatch}")
            result["error"] = "; ".join(errors)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# Gate criterion: multistep_parity (NUMERIC, per registered rollout)
# =============================================================================

def verify_multistep_parity(
    bridge: WeatherStepBridge,
    x_raw: np.ndarray,
    upstream_base_dir: Path,
    patch_size: int,
    device: torch.device,
    registered_rollouts: Sequence[Tuple[int, int]] = ((6, 1), (6, 4)),
    tolerance: float = 1e-5,
    config: Optional[GateIdentityConfig] = None,
    reference_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare bridge rollouts against the reference NUMERICALLY, per rollout.

    Every registered (interval, steps) pair is run through the bridge from the
    bound normalized input and compared value-by-value with the corresponding
    exported reference tensor. Existence, shape and finiteness are necessary
    but nowhere near sufficient: a finite, correctly shaped, numerically wrong
    4-step tensor fails here.

    An empty registration fails: nothing verified is not the same as verified.
    """
    if config is not None:
        registered_rollouts = config.registered_rollouts
        tolerance = config.rollout_tolerance

    result = {
        "criterion": "multistep_parity",
        "passed": False,  # fail-closed
        "tolerance": tolerance,
        "registered_rollouts": [list(r) for r in registered_rollouts],
        "comparisons": {},
    }

    try:
        if not registered_rollouts:
            result["error"] = (
                "Empty rollout registration: no (interval, steps) pair is registered, "
                "so no numeric comparison would run. This must fail, not pass vacuously."
            )
            return result

        upstream_dir = find_upstream_reference_dir(
            upstream_base_dir, patch_size, config, reference_dir
        )
        if upstream_dir is None:
            result["error"] = f"Upstream reference not found in {upstream_base_dir}"
            return result
        result["upstream_dir"] = str(upstream_dir)

        input_norm_path = upstream_dir / "input_norm.pt"
        if not input_norm_path.exists():
            result["error"] = f"Missing saved normalized input: {input_norm_path}"
            return result
        saved_norm = torch.load(input_norm_path, map_location="cpu").to(device)

        # Start from the bridge's OWN normalization of the raw input so that the
        # rollout chain is anchored to the enrolled raw input, not to a tensor
        # of unknown provenance read off disk.
        x_raw_t = torch.from_numpy(np.asarray(x_raw)).float().unsqueeze(0).to(device)
        if tuple(x_raw_t.shape) != tuple(saved_norm.shape):
            result["error"] = (
                f"Raw input shape {tuple(x_raw_t.shape)} does not match saved "
                f"normalized input shape {tuple(saved_norm.shape)}"
            )
            return result
        with torch.no_grad():
            x_norm = bridge.normalize(x_raw_t)

        all_passed = True
        for interval, steps in registered_rollouts:
            key = f"{interval}h_{steps}step"
            entry: Dict[str, Any] = {"passed": False}
            result["comparisons"][key] = entry

            fpath = upstream_dir / f"official_output_{key}.pt"
            if not fpath.exists():
                entry["error"] = f"Missing reference file: {fpath.name}"
                all_passed = False
                continue

            reference = torch.load(fpath, map_location="cpu").to(device)
            entry["reference_shape"] = list(reference.shape)

            with torch.no_grad():
                actual = bridge.forward_validation(
                    x_norm, bridge.variables, interval=int(interval), steps=int(steps)
                )
            entry["actual_shape"] = list(actual.shape)

            if actual.shape != reference.shape:
                entry["error"] = (
                    f"Shape mismatch: bridge {tuple(actual.shape)} vs "
                    f"reference {tuple(reference.shape)}"
                )
                all_passed = False
                continue

            if not bool(torch.isfinite(reference).all()):
                entry["error"] = "Reference tensor contains non-finite values"
                all_passed = False
                continue
            if not bool(torch.isfinite(actual).all()):
                entry["error"] = "Bridge rollout produced non-finite values"
                all_passed = False
                continue

            diff = (actual - reference).abs()
            entry["max_abs_diff"] = float(diff.max().item())
            entry["mean_abs_diff"] = float(diff.mean().item())
            entry["passed"] = entry["max_abs_diff"] <= tolerance
            if not entry["passed"]:
                entry["error"] = (
                    f"Numeric parity failure for {key}: max abs diff "
                    f"{entry['max_abs_diff']:.3e} exceeds tolerance {tolerance:.0e}"
                )
                all_passed = False

        result["passed"] = all_passed
        if not all_passed:
            failures = {
                k: v.get("error") for k, v in result["comparisons"].items()
                if not v.get("passed")
            }
            result["error"] = f"Multistep numeric parity failed: {failures}"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


# =============================================================================
# Gate criterion: zero_edit_equals_official (internal consistency)
# =============================================================================

def verify_zero_edit_internal_consistency(
    bridge: WeatherStepBridge,
    x_raw: np.ndarray,
    interval: int,
    steps: int,
    device: torch.device,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Verify controlled_rollout with zero coefficients equals forward_validation.

    NOTE: This is an INTERNAL CONSISTENCY check (bridge vs bridge).
    It verifies that the controlled_rollout path with zero edits produces
    the same output as forward_validation. This is a useful check, but it
    is NOT a substitute for upstream_parity (which compares against the
    real official xformers code).
    """
    result = {
        "criterion": "zero_edit_equals_official",
        "note": "Internal consistency check (bridge vs bridge), NOT upstream parity",
        "interval": interval,
        "steps": steps,
        "tolerance": tolerance,
        "passed": False,  # fail-closed
    }

    try:
        x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
        x_norm = bridge.normalization.normalize(x_raw_t)

        plan = reference_plan(num_experts=8)

        with torch.no_grad():
            ref_out = bridge.forward_validation(
                x_norm, bridge.variables, interval=interval, steps=steps
            )
            ctrl_out = controlled_rollout(
                bridge, x_norm, bridge.variables, interval=interval, steps=steps,
                plan=plan, expert_loras={},
            )

        diff = (ref_out - ctrl_out).abs()
        result["max_abs_diff"] = float(diff.max().item())
        result["mean_abs_diff"] = float(diff.mean().item())
        result["passed"] = result["max_abs_diff"] <= tolerance

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# Gate criterion: normalization_parity
# =============================================================================

def verify_normalization_parity(
    norm_from_npy: NormalizationContract,
    norm_dir: Path,
    expected_digest: Optional[str] = None,
    config: Optional[GateIdentityConfig] = None,
) -> Dict[str, Any]:
    """Verify the normalization identity against the frozen expected identity.

    `expected_digest` must be the SCHEMA-QUALIFIED identity string
    (``ed-norm-identity/1:<hex>``). A bare, unversioned digest is treated as a
    legacy-format value and fails closed with a format-version error rather
    than being reinterpreted under the current schema.
    """
    variables = list(config.variables) if config is not None else DEFAULT_VARIABLES
    intervals = tuple(config.normalization_intervals) if config is not None else (6, 24)
    if expected_digest is None and config is not None:
        expected_digest = config.expected_normalization_identity

    result = {
        "criterion": "normalization_parity",
        "passed": False,  # fail-closed
        "computed_digest": norm_from_npy.digest,
        "computed_identity": norm_from_npy.identity_digest,
        "identity_schema": norm_from_npy.identity_schema_version,
        "policy": norm_from_npy.policy,
        "policy_content_consistent": norm_from_npy.policy_content_consistent,
        "n_variables": len(norm_from_npy.variables),
        "intervals_present": sorted(norm_from_npy.diff_std.keys()),
    }

    try:
        # The declared policy must reflect the actual contents, not just label them.
        if not norm_from_npy.policy_content_consistent:
            result["error"] = (
                f"Normalization contract declares policy '{norm_from_npy.policy}' but "
                "its stored diff_mean is nonzero. The policy label must match the "
                "real contents."
            )
            return result

        if config is not None and norm_from_npy.policy != config.normalization_policy:
            result["expected_policy"] = config.normalization_policy
            result["error"] = (
                f"Normalization policy mismatch: config expects "
                f"'{config.normalization_policy}', runtime contract declares "
                f"'{norm_from_npy.policy}'"
            )
            return result

        if config is not None and list(norm_from_npy.variables) != variables:
            result["error"] = (
                "Normalization contract variable list/order does not match the frozen config"
            )
            return result

        if not norm_dir.exists():
            result["error"] = f"Normalization directory not found: {norm_dir}"
            return result

        # Load from NPZ with same policy
        norm_from_npz = NormalizationContract.from_npz_dir(
            str(norm_dir), variables=variables, intervals=intervals,
            policy=norm_from_npy.policy,
        )
        result["digest_from_npz"] = norm_from_npz.digest
        result["identity_from_npz"] = norm_from_npz.identity_digest
        result["digests_match_npz"] = norm_from_npy.digest == norm_from_npz.digest

        # The raw NPZ content hash is a DIFFERENT question from the identity
        # digest; record it separately and never compare the two to each other.
        npz_content = compute_normalization_asset_sha256(norm_dir)
        result["npz_content_sha256"] = npz_content
        if config is not None and config.expected_normalization_npz_sha256 is not None:
            result["expected_npz_content_sha256"] = config.expected_normalization_npz_sha256
            result["npz_content_match"] = npz_content == config.expected_normalization_npz_sha256
            if not result["npz_content_match"]:
                result["error"] = (
                    f"Normalization NPZ content hash mismatch: expected "
                    f"{config.expected_normalization_npz_sha256}, got {npz_content}"
                )
                return result

        # Check against expected identity if provided
        if expected_digest is not None:
            result["expected_digest"] = expected_digest
            try:
                matched = compare_normalization_identity(
                    expected_digest, norm_from_npy.identity_digest
                )
            except NormalizationIdentityFormatError as exc:
                result["digests_match_expected"] = False
                result["format_version_mismatch"] = True
                result["error"] = str(exc)
                return result
            result["digests_match_expected"] = matched
            result["passed"] = matched and result["digests_match_npz"]
            if not result["passed"]:
                result["error"] = (
                    f"Digest mismatch: computed={norm_from_npy.identity_digest}, "
                    f"expected={expected_digest}, from_npz={norm_from_npz.identity_digest}"
                )
        else:
            # No expected digest provided - fail closed (B02 fix)
            # Self-consistency with NPZ is recorded but cannot certify identity
            result["passed"] = False
            result["reason"] = "IDENTITY_NOT_BOUND: no expected normalization digest provided"
            result["note"] = "Computed digest recorded for enrollment; pass --expected-norm-digest to certify"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# Gate criterion: version_identity_match (real expected-vs-actual comparison)
# =============================================================================

def verify_version_identity(
    bridge: WeatherStepBridge,
    config: Optional[GateIdentityConfig],
    load_result: Optional[CheckpointLoadResult] = None,
    computed_checkpoint_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare the bridge's computed ArtifactVersion against the FROZEN config.

    The expected version is reconstructed entirely from the enrolled config
    (checkpoint SHA, backbone geometry, grid, patch size, normalization
    identity). Nothing on the expected side is read off the object under test,
    so this can never degenerate into comparing a value with itself.
    """
    result = {
        "criterion": "version_identity_match",
        "passed": False,  # fail-closed
    }

    try:
        actual_version = bridge.version
        result["actual_version"] = asdict(actual_version)

        if config is None:
            result["reason"] = "IDENTITY_NOT_BOUND: no frozen --config supplied"
            result["error"] = "Cannot verify version identity without an expected identity"
            return result

        expected_norm_digest = config.expected_normalization_digest
        if expected_norm_digest is None:
            result["reason"] = "IDENTITY_NOT_BOUND: config has no expected_normalization_identity"
            result["error"] = "Cannot build an expected version without the expected normalization identity"
            return result

        expected_version = config.expected_artifact_version(expected_norm_digest)
        result["expected_version"] = asdict(expected_version)

        # Bind the runtime-loaded checkpoint to the hash computed from the file.
        if load_result is not None and computed_checkpoint_sha256 is not None:
            result["runtime_checkpoint_sha256"] = load_result.checkpoint_sha256
            if load_result.checkpoint_sha256.lower() != computed_checkpoint_sha256.lower():
                result["error"] = (
                    "Loaded checkpoint hash does not match the hash computed from the "
                    f"asset file: runtime {load_result.checkpoint_sha256} vs "
                    f"file {computed_checkpoint_sha256}"
                )
                return result

        try:
            check_version_match(expected_version, actual_version)
        except ValueError as exc:
            result["error"] = f"Version identity mismatch: {exc}"
            result["differing_fields"] = [
                name for name, value in asdict(expected_version).items()
                if value != asdict(actual_version)[name]
            ]
            return result

        result["passed"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


# =============================================================================
# Gate criterion: manifest_identity_match (manifest vs config AND vs runtime)
# =============================================================================

def verify_manifest_identity(
    upstream_base_dir: Path,
    patch_size: int,
    config: Optional[GateIdentityConfig],
    norm: Optional[NormalizationContract],
    computed_checkpoint_sha256: Optional[str],
    reference_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Enforce the reference manifest against the frozen config AND the runtime.

    The manifest's checkpoint/normalization identity used to be read and
    printed but never enforced. Here it must agree with (a) the frozen config
    and (b) the objects this run actually loaded: the checkpoint hash computed
    from the asset on disk and the live NormalizationContract.
    """
    result = {
        "criterion": "manifest_identity_match",
        "passed": False,  # fail-closed
    }

    try:
        if config is None:
            result["reason"] = "IDENTITY_NOT_BOUND: no frozen --config supplied"
            result["error"] = "Cannot enforce a manifest without an expected identity"
            return result

        upstream_dir, selection_error = diagnostic_reference_dir(
            upstream_base_dir, patch_size, config, reference_dir
        )
        if selection_error is not None:
            result["selection_error"] = str(selection_error)
        if upstream_dir is None:
            result["error"] = f"ReferenceSelectionError: {selection_error}"
            return result
        result["upstream_dir"] = str(upstream_dir)

        manifest = _load_reference_manifest(upstream_dir)
        if manifest is None:
            result["error"] = f"Manifest missing or unreadable in {upstream_dir}"
            return result

        # (a) manifest vs frozen config
        config_mismatches = reference_identity_mismatches(manifest, config)
        result["config_mismatches"] = config_mismatches

        # (b) manifest vs the objects actually loaded by this run
        runtime_mismatches: List[str] = []
        manifest_sha = (manifest.get("checkpoint") or {}).get("sha256")
        result["manifest_checkpoint_sha256"] = manifest_sha
        result["runtime_checkpoint_sha256"] = computed_checkpoint_sha256
        if computed_checkpoint_sha256 is None:
            runtime_mismatches.append("runtime_checkpoint_sha256_unavailable")
        elif not manifest_sha:
            runtime_mismatches.append("manifest_checkpoint_sha256_missing")
        elif manifest_sha.lower() != computed_checkpoint_sha256.lower():
            runtime_mismatches.append(
                f"checkpoint_sha256: manifest {manifest_sha} vs runtime {computed_checkpoint_sha256}"
            )

        manifest_norm = manifest.get("normalization") or {}
        if norm is None:
            runtime_mismatches.append("runtime_normalization_unavailable")
        else:
            result["manifest_normalization_policy"] = manifest_norm.get("policy")
            result["runtime_normalization_policy"] = norm.policy
            if manifest_norm.get("policy") != norm.policy:
                runtime_mismatches.append(
                    f"normalization_policy: manifest {manifest_norm.get('policy')} vs "
                    f"runtime {norm.policy}"
                )
            manifest_identity = manifest_norm.get("identity_digest") or manifest_norm.get("digest")
            result["manifest_normalization_identity"] = manifest_identity
            result["runtime_normalization_identity"] = norm.identity_digest
            if manifest_identity is None:
                runtime_mismatches.append("manifest_normalization_identity_missing")
            else:
                try:
                    if not compare_normalization_identity(manifest_identity, norm.identity_digest):
                        runtime_mismatches.append(
                            f"normalization_identity: manifest {manifest_identity} vs "
                            f"runtime {norm.identity_digest}"
                        )
                except NormalizationIdentityFormatError as exc:
                    runtime_mismatches.append(
                        f"normalization_identity_format_version_mismatch: {exc}"
                    )

        result["runtime_mismatches"] = runtime_mismatches
        result["passed"] = not config_mismatches and not runtime_mismatches
        if not result["passed"]:
            result["error"] = (
                "Reference manifest identity not enforced clean: "
                f"config mismatches={config_mismatches}, runtime mismatches={runtime_mismatches}"
            )

    except ReferenceSelectionError as exc:
        result["error"] = f"ReferenceSelectionError: {exc}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


# =============================================================================
# Gate criterion: source_identity_match
# =============================================================================

def verify_source_identity(
    upstream_base_dir: Path,
    patch_size: int,
    config: Optional[GateIdentityConfig],
    reference_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Record the imported source snapshot and enforce the pinned upstream source.

    Records which module files were actually imported plus their content hash
    (source root), and requires the reference to have been produced under the
    same pinned upstream commit AND the same frozen config identity.
    """
    result = {
        "criterion": "source_identity_match",
        "passed": False,  # fail-closed
        "source_identity": record_source_identity(),
    }

    try:
        if config is None:
            result["reason"] = "IDENTITY_NOT_BOUND: no frozen --config supplied"
            result["error"] = "Cannot verify source identity without an expected identity"
            return result

        result["expected_official_commit"] = config.official_commit
        result["expected_config_digest"] = config.config_digest

        upstream_dir, selection_error = diagnostic_reference_dir(
            upstream_base_dir, patch_size, config, reference_dir
        )
        if selection_error is not None:
            result["selection_error"] = str(selection_error)
        if upstream_dir is None:
            result["error"] = f"ReferenceSelectionError: {selection_error}"
            return result
        result["upstream_dir"] = str(upstream_dir)

        manifest = _load_reference_manifest(upstream_dir)
        if manifest is None:
            result["error"] = f"Manifest missing or unreadable in {upstream_dir}"
            return result

        pinned = (manifest.get("official_source") or {}).get("pinned_commit")
        result["manifest_official_commit"] = pinned
        manifest_config_digest = (manifest.get("gate_config") or {}).get("digest")
        result["manifest_config_digest"] = manifest_config_digest
        result["reference_source_identity"] = manifest.get("source_identity")

        problems: List[str] = []
        if pinned != config.official_commit:
            problems.append(
                f"pinned upstream commit mismatch: config {config.official_commit} vs "
                f"reference {pinned}"
            )
        if manifest_config_digest is None:
            problems.append("reference manifest carries no frozen gate_config digest")
        elif manifest_config_digest != config.config_digest:
            problems.append(
                f"frozen config digest mismatch: gate {config.config_digest} vs "
                f"reference {manifest_config_digest}"
            )

        result["problems"] = problems
        result["passed"] = not problems
        if problems:
            result["error"] = "Source identity mismatch: " + "; ".join(problems)

    except ReferenceSelectionError as exc:
        result["error"] = f"ReferenceSelectionError: {exc}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


# =============================================================================
# Gate criterion: variable_coordinate_identity_match
# =============================================================================

def verify_variable_coordinate_identity(
    inputs: Dict[str, np.ndarray],
    config: Optional[GateIdentityConfig],
    upstream_base_dir: Optional[Path] = None,
    patch_size: Optional[int] = None,
    reference_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify variable ORDER and coordinate VALUES, not just shapes.

    Two grids of identical shape but different lat/lon values are different
    identities; so are two runs whose variable lists contain the same names in
    a different order. Every comparison here is expected-vs-actual: the frozen
    config supplies the expected coordinate digest and variable order, while
    the actual values come from the input assets on disk and (when available)
    from the reference manifest.
    """
    result = {
        "criterion": "variable_coordinate_identity_match",
        "passed": False,  # fail-closed
    }

    try:
        lat = np.asarray(inputs["lat"])
        lon = np.asarray(inputs["lon"])
        data = np.asarray(inputs["data"])
        coordinate_digest = compute_coordinate_digest(lat, lon)
        result["asset_coordinate_digest"] = coordinate_digest
        result["grid_shape"] = [int(lat.shape[0]), int(lon.shape[0])]
        result["data_shape"] = [int(s) for s in data.shape]

        if config is None:
            result["reason"] = "IDENTITY_NOT_BOUND: no frozen --config supplied"
            result["error"] = "Cannot verify variable/coordinate identity without an expected identity"
            return result

        result["expected_variables_digest"] = config.variables_digest

        problems: List[str] = []

        # Variable ORDER is checked against what the reference actually recorded,
        # never against the config's own copy of itself.
        if upstream_base_dir is not None:
            upstream_dir, selection_error = diagnostic_reference_dir(
                upstream_base_dir,
                patch_size if patch_size is not None else config.patch_size,
                config, reference_dir,
            )
            if selection_error is not None:
                result["selection_error"] = str(selection_error)
            manifest = _load_reference_manifest(upstream_dir) if upstream_dir else None
            if manifest is None:
                problems.append(
                    f"reference unavailable for variable identity: {selection_error}"
                )
            else:
                reference_variables = manifest.get("variables")
                result["reference_variables_digest"] = manifest.get("variables_digest")
                if reference_variables is None:
                    problems.append("reference manifest records no variable list")
                elif list(reference_variables) != list(config.variables):
                    first = next(
                        (i for i, (a, b) in enumerate(zip(reference_variables, config.variables))
                         if a != b),
                        min(len(reference_variables), len(config.variables)),
                    )
                    problems.append(
                        "variable order differs from the reference at index "
                        f"{first}: reference has "
                        f"{reference_variables[first] if first < len(reference_variables) else '<end>'}, "
                        f"config has "
                        f"{config.variables[first] if first < len(config.variables) else '<end>'}"
                    )
                reference_digest = manifest.get("variables_digest")
                if reference_digest is not None and reference_digest != config.variables_digest:
                    problems.append(
                        f"variables digest mismatch: reference {reference_digest} vs "
                        f"config {config.variables_digest}"
                    )
                reference_coordinate = manifest.get("coordinate_digest")
                result["reference_coordinate_digest"] = reference_coordinate
                if reference_coordinate is not None and reference_coordinate != coordinate_digest:
                    problems.append(
                        f"coordinate digest mismatch vs reference: reference "
                        f"{reference_coordinate} vs asset {coordinate_digest}"
                    )

        expected_grid = tuple(int(v) for v in config.grid_shape)
        actual_grid = (int(lat.shape[0]), int(lon.shape[0]))
        if actual_grid != expected_grid:
            problems.append(f"grid shape {actual_grid} != expected {expected_grid}")
        if data.ndim != 4:
            problems.append(f"input data must be (T, V, H, W), got shape {data.shape}")
        else:
            if int(data.shape[1]) != len(config.variables):
                problems.append(
                    f"input has {int(data.shape[1])} channels but config declares "
                    f"{len(config.variables)} variables"
                )
            if (int(data.shape[2]), int(data.shape[3])) != expected_grid:
                problems.append(
                    f"input grid {(int(data.shape[2]), int(data.shape[3]))} != expected {expected_grid}"
                )

        if config.expected_coordinate_digest is None:
            problems.append(
                "IDENTITY_NOT_BOUND: config has no expected_coordinate_digest"
            )
        elif config.expected_coordinate_digest != coordinate_digest:
            problems.append(
                f"coordinate values differ from the enrolled identity: expected "
                f"{config.expected_coordinate_digest}, computed {coordinate_digest}"
            )

        result["problems"] = problems
        result["passed"] = not problems
        if problems:
            result["error"] = "Variable/coordinate identity mismatch: " + "; ".join(problems)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


# =============================================================================
# Gate criterion: no_state_leak
# =============================================================================

def verify_no_state_leak(
    bridge: WeatherStepBridge,
    x_raw: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """Verify no state leak - identical inputs produce bit-identical outputs."""
    result = {
        "criterion": "no_state_leak",
        "passed": False,  # fail-closed
    }

    try:
        x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
        x_norm = bridge.normalization.normalize(x_raw_t)
        plan = reference_plan(num_experts=8)

        # Scope variable that should NOT leak
        _ = torch.randn_like(x_norm)

        with torch.no_grad():
            out_a = controlled_rollout(
                bridge, x_norm, bridge.variables, interval=6, steps=2,
                plan=plan, expert_loras={},
            ).clone()

        # Change scope variable
        _ = torch.randn_like(x_norm) * 100

        with torch.no_grad():
            out_b = controlled_rollout(
                bridge, x_norm, bridge.variables, interval=6, steps=2,
                plan=plan, expert_loras={},
            )

        result["bit_identical"] = bool(torch.equal(out_a, out_b))
        result["passed"] = result["bit_identical"]

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# Gate criterion: outputs_finite
# =============================================================================

def verify_outputs_finite(
    bridge: WeatherStepBridge,
    x_raw: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """Verify all gate-relevant outputs contain no NaN/Inf."""
    result = {
        "criterion": "outputs_finite",
        "passed": False,  # fail-closed
        "checks": {},
    }

    try:
        x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
        x_norm = bridge.normalization.normalize(x_raw_t)

        outputs_to_check = {}

        # forward_validation
        with torch.no_grad():
            out_fv = bridge.forward_validation(x_norm, bridge.variables, interval=6, steps=1)
            outputs_to_check["forward_validation_6h_1step"] = out_fv

            # controlled_rollout
            plan = reference_plan(num_experts=8)
            out_cr = controlled_rollout(
                bridge, x_norm, bridge.variables, interval=6, steps=1,
                plan=plan, expert_loras={},
            )
            outputs_to_check["controlled_rollout_6h_1step"] = out_cr

        all_finite = True
        for name, tensor in outputs_to_check.items():
            is_finite = bool(torch.isfinite(tensor).all())
            result["checks"][name] = is_finite
            if not is_finite:
                all_finite = False

        result["passed"] = all_finite

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# RMSE sanity check (informational only)
# =============================================================================

def compute_rmse_sanity(
    bridge: WeatherStepBridge,
    data: np.ndarray,
    lat: np.ndarray,
    device: torch.device,
    variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute RMSE sanity check metrics (informational, not a gate criterion)."""
    result = {
        "note": "Historical reference only - NOT used for gate pass/fail",
        "historical_reference": HISTORICAL_REFERENCE,
    }

    names = list(variables) if variables is not None else list(bridge.variables)
    if Z500_NAME in names:
        z500_idx = names.index(Z500_NAME)
    else:
        result["skipped"] = f"{Z500_NAME} not present in variable list"
        return result

    try:
        lat_rad = np.deg2rad(lat)
        lat_weights = np.cos(lat_rad)
        lat_weights = lat_weights / lat_weights.mean()
        lat_weights_t = torch.from_numpy(lat_weights).float().to(device).view(1, 1, -1, 1)

        for label, interval, steps in [("6h", 6, 1), ("24h", 6, 4)]:
            n_samples = min(5, data.shape[0] - steps)
            rmse_vals = []

            for i in range(n_samples):
                x_raw = data[i]
                y_true = data[i + steps]

                x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
                y_true_t = torch.from_numpy(y_true).float().unsqueeze(0).to(device)

                x_norm = bridge.normalization.normalize(x_raw_t)

                with torch.no_grad():
                    y_pred_norm = bridge.forward_validation(
                        x_norm, bridge.variables, interval=interval, steps=steps
                    )
                    y_pred = bridge.normalization.denormalize(y_pred_norm)

                z500_pred = y_pred[:, z500_idx, :, :]
                z500_true = y_true_t[:, z500_idx, :, :]

                sq_err = (z500_pred - z500_true) ** 2
                rmse = torch.sqrt((sq_err * lat_weights_t.squeeze(1)).mean()).item()
                rmse_vals.append(rmse)

            result[label] = {
                "rmse_z500_weighted": float(np.mean(rmse_vals)) if rmse_vals else None,
                "rmse_z500_std": float(np.std(rmse_vals)) if len(rmse_vals) > 1 else None,
                "n_samples": n_samples,
            }

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# Verdict commitment and cleanup classification (B13)
# =============================================================================

#: Every criterion that MUST have been really evaluated before a PASS verdict
#: may be committed. Seeding `gate_criteria` with False already makes a
#: never-evaluated criterion fail closed; this tuple additionally makes a
#: never-evaluated criterion impossible to hide behind a True default should
#: the seeding or the recording ever drift apart.
REQUIRED_GATE_CRITERIA: Tuple[str, ...] = (
    "identity_config_bound",
    "source_identity_match",
    "variable_coordinate_identity_match",
    "ckpt_sha256_bound",
    "normalization_parity",
    "manifest_identity_match",
    "raw_input_binding",
    "strict_load_zero_diff",
    "version_identity_match",
    "input_norm_binding",
    "upstream_parity",
    "multistep_reference_present",
    "multistep_parity",
    "zero_edit_equals_official",
    "no_state_leak",
    "outputs_finite",
)


class GateVerdictNotCommittable(RuntimeError):
    """A provisional PASS verdict failed its pre-commit checks.

    Raised from inside `run_s0_gate`'s main try block so that the outer
    exception handler revokes the verdict along every other late failure.
    """


def _assert_gate_verdict_committable(result: Dict[str, Any]) -> None:
    """Refuse to commit a PASS that is not backed by the full evidence set.

    A PASS is only committable once (a) every required criterion was actually
    evaluated and recorded -- not merely left at its fail-closed default -- and
    (b) the identity evidence that makes the run attributable was recorded.
    Anything short of that raises, and `run_s0_gate`'s outer handler then
    revokes the provisional verdict unconditionally.

    Raises:
        GateVerdictNotCommittable: If required evidence is missing.
    """
    criteria = result.get("gate_criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise GateVerdictNotCommittable(
            "Refusing to commit PASS: no gate criteria were recorded."
        )

    details = result.get("criteria_details")
    if not isinstance(details, dict):
        raise GateVerdictNotCommittable(
            "Refusing to commit PASS: criteria_details is missing."
        )

    never_evaluated = [name for name in REQUIRED_GATE_CRITERIA if name not in details]
    if never_evaluated:
        raise GateVerdictNotCommittable(
            "Refusing to commit PASS: required criteria were never evaluated: "
            f"{never_evaluated}"
        )

    absent = [name for name in REQUIRED_GATE_CRITERIA if name not in criteria]
    if absent:
        raise GateVerdictNotCommittable(
            "Refusing to commit PASS: required criteria absent from the verdict "
            f"set: {absent}"
        )

    not_passing = sorted(name for name, passed in criteria.items() if not passed)
    if not_passing:
        raise GateVerdictNotCommittable(
            f"Refusing to commit PASS: criteria are not all passing: {not_passing}"
        )

    # Identity evidence: a PASS that cannot be attributed to a frozen config and
    # a recorded source snapshot is not a PASS we are willing to publish.
    if not result.get("config_digest"):
        raise GateVerdictNotCommittable(
            "Refusing to commit PASS: no frozen config digest was recorded."
        )
    source_identity = result.get("source_identity")
    if not isinstance(source_identity, dict) or not source_identity.get("modules"):
        raise GateVerdictNotCommittable(
            "Refusing to commit PASS: no source identity snapshot was recorded."
        )


def _empty_cuda_cache() -> None:
    """Release cached CUDA blocks. Non-essential; may fail on a sick device."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_classified_cleanup(
    stage: str, action, warnings: List[Dict[str, Any]]
) -> None:
    """Run one non-essential cleanup step, recording failures on their own axis.

    Cleanup runs after the verdict has already been decided, so a cleanup
    failure must never crash past a legitimately computed result and must never
    be conflated with the gate's own pass/fail or with `result["error"]`. It is
    recorded as a classified warning instead.
    """
    try:
        action()
    except BaseException as exc:  # noqa: BLE001 - cleanup must not escape
        warnings.append({
            "stage": stage,
            "classification": "non_essential_cleanup",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "affects_gate_verdict": False,
        })
        print(
            f"WARNING: non-essential cleanup step {stage!r} failed and was "
            f"recorded as a cleanup warning: {type(exc).__name__}: {exc}",
            flush=True,
        )


# =============================================================================
# Main gate execution
# =============================================================================

def run_s0_gate(
    patch_size: int = 4,
    check_only: Optional[str] = None,
    output_dir: Optional[Path] = None,
    expected_checkpoint_sha256: Optional[str] = None,
    expected_normalization_digest: Optional[str] = None,
    config: Optional[GateIdentityConfig] = None,
    config_path: Optional[Path] = None,
    reference_dir: Optional[Path] = None,
    asset_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all S0 gate verifications.

    All gate criteria are fail-closed: they start False and stay False
    if any exception occurs during evaluation.

    Ordering is deliberate. Everything that can be decided from identity alone
    (config binding, source identity, variable/coordinate identity, checkpoint
    hash, normalization identity, manifest enforcement, raw-input hash) runs
    BEFORE the expensive checkpoint load and forward passes. When any of those
    fails the expensive phase is skipped and its criteria stay False, so a
    CPU-only fixture can exercise the real identity logic cheaply and a failure
    is attributed to the field that actually broke.

    Args:
        patch_size: Patch size (2 or 4); overridden by `config` when supplied.
        check_only: Run only a specific criterion (not implemented yet).
        output_dir: Output directory for results (informational here; main()
            owns the run-scoped publication).
        expected_checkpoint_sha256: Expected checkpoint SHA-256 (CLI override).
        expected_normalization_digest: Expected schema-qualified normalization
            identity (CLI override).
        config: Frozen GateIdentityConfig (the enrolled expected identity).
        config_path: Path the config was loaded from, for provenance.
        reference_dir: Explicit reference directory (--reference-dir).
        asset_root: Fallback asset root when the config does not name one.
    """
    if config is not None:
        patch_size = config.patch_size
        if expected_checkpoint_sha256 is None:
            expected_checkpoint_sha256 = config.expected_checkpoint_sha256
        if expected_normalization_digest is None:
            expected_normalization_digest = config.expected_normalization_identity

    PATHS = resolve_paths(config, asset_root, patch_size)

    result = {
        "status": "failed",
        "run_id": (
            's0-gate-'
            f'{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dt%H%M%S%fz")}'
        ),
        "hostname": socket.gethostname(),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "repo_root": str(SOURCE_ROOT),
        "asset_root": str(PATHS.get("asset_root", SOURCE_ROOT)),
        "config_path": str(config_path) if config_path else None,
        "config_digest": config.config_digest if config is not None else None,
        "reference_dir_arg": str(reference_dir) if reference_dir else None,
        "patch_size": patch_size,
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "source_identity": record_source_identity(),
        "gate_criteria": {
            "identity_config_bound": False,
            "source_identity_match": False,
            "variable_coordinate_identity_match": False,
            "ckpt_sha256_bound": False,
            "normalization_parity": False,
            "manifest_identity_match": False,
            "raw_input_binding": False,
            "strict_load_zero_diff": False,
            "version_identity_match": False,
            "input_norm_binding": False,
            "upstream_parity": False,
            "multistep_reference_present": False,
            "multistep_parity": False,
            "zero_edit_equals_official": False,
            "no_state_leak": False,
            "outputs_finite": False,
        },
        "criteria_details": {},
        "s0_gate_pass": False,
        # True only once a PASS verdict has cleared its pre-commit checks. A
        # legitimate FAIL leaves this False: there was no PASS to commit.
        "verdict_committed": False,
        # Non-essential cleanup failures live on their OWN axis: they are never
        # folded into s0_gate_pass, status or error, and never crash past an
        # already-computed verdict (B13).
        "cleanup_warnings": [],
    }

    def record(name: str, detail: Dict[str, Any]) -> bool:
        result["criteria_details"][name] = detail
        passed = bool(detail.get("passed"))
        result["gate_criteria"][name] = passed
        print(f"  - {name}: {'PASS' if passed else 'FAIL'}", flush=True)
        return passed

    model = None
    bridge = None
    try:
        # Determine device
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            result["cuda_device"] = torch.cuda.get_device_name(0)
        else:
            device = torch.device('cpu')
            result["cuda_device"] = "CPU"
        print(f"Using device: {device}", flush=True)
        print(f"Patch size: {patch_size}", flush=True)

        # ------------------------------------------------------------------
        # Phase 1: identity binding (cheap, runs before any model load)
        # ------------------------------------------------------------------
        print("Running identity binding criteria...", flush=True)

        config_detail: Dict[str, Any] = {
            "criterion": "identity_config_bound",
            "passed": False,
            "config_path": str(config_path) if config_path else None,
        }
        if config is None:
            config_detail["reason"] = "IDENTITY_NOT_BOUND: no frozen --config supplied"
            config_detail["error"] = (
                "S0 requires a frozen identity config shared with the exporter. "
                "Pass --config."
            )
        else:
            missing = [
                name for name, value in (
                    ("expected_checkpoint_sha256", config.expected_checkpoint_sha256),
                    ("expected_normalization_identity", config.expected_normalization_identity),
                    ("expected_coordinate_digest", config.expected_coordinate_digest),
                    ("expected_raw_input_hash", config.expected_raw_input_hash),
                ) if not value
            ]
            config_detail["config_digest"] = config.config_digest
            config_detail["identity_tag"] = config.identity_tag
            config_detail["registered_rollouts"] = [list(r) for r in config.registered_rollouts]
            config_detail["missing_expected_fields"] = missing
            if missing:
                config_detail["reason"] = "IDENTITY_NOT_BOUND"
                config_detail["error"] = (
                    f"Frozen config is missing expected identity fields: {missing}"
                )
            elif not config.registered_rollouts:
                config_detail["error"] = "Empty rollout registration"
            else:
                config_detail["passed"] = True
        record("identity_config_bound", config_detail)

        # Load inputs (asset root, from the config when present)
        print("Loading inputs...", flush=True)
        input_intervals = (
            tuple(config.normalization_intervals) if config is not None else (6, 24)
        )
        input_file = config.input_file if config is not None else 'jan2020_full.npy'
        inputs = load_npy_inputs(PATHS["input_dir"], input_intervals, input_file)
        x_raw_0 = inputs['data'][0]
        result["input_dir"] = str(PATHS["input_dir"])

        record(
            "variable_coordinate_identity_match",
            verify_variable_coordinate_identity(
                inputs, config, PATHS["upstream_reference_base"], patch_size, reference_dir
            ),
        )

        record(
            "source_identity_match",
            verify_source_identity(
                PATHS["upstream_reference_base"], patch_size, config, reference_dir
            ),
        )

        # Create normalization with the configured policy (official: zero diff_mean)
        policy = config.normalization_policy if config is not None else POLICY_OFFICIAL_ZERO_DIFF_MEAN
        norm = create_normalization_from_npy(
            inputs,
            policy=policy,
            variables=list(config.variables) if config is not None else None,
            intervals=input_intervals,
        )
        result["normalization_policy"] = norm.policy
        result["normalization_digest"] = norm.digest
        result["normalization_identity"] = norm.identity_digest

        record(
            "normalization_parity",
            verify_normalization_parity(
                norm, PATHS["norm_dir"], expected_normalization_digest, config
            ),
        )

        # Checkpoint hash comes from the file, before any model construction.
        computed_ckpt_sha256: Optional[str] = None
        if PATHS["checkpoint"].exists():
            computed_ckpt_sha256 = _compute_file_sha256(str(PATHS["checkpoint"]))
        result["checkpoint_path"] = str(PATHS["checkpoint"])
        result["computed_checkpoint_sha256"] = computed_ckpt_sha256

        record(
            "ckpt_sha256_bound",
            verify_ckpt_sha256(
                PATHS["checkpoint"], None, expected_checkpoint_sha256,
                computed_sha256=computed_ckpt_sha256,
            ),
        )

        record(
            "manifest_identity_match",
            verify_manifest_identity(
                PATHS["upstream_reference_base"], patch_size, config, norm,
                computed_ckpt_sha256, reference_dir,
            ),
        )

        record(
            "raw_input_binding",
            verify_raw_input_binding(
                x_raw_0, PATHS["upstream_reference_base"], patch_size, config, reference_dir
            ),
        )

        record(
            "multistep_reference_present",
            verify_multistep_reference(
                PATHS["upstream_reference_base"], patch_size,
                required_steps=tuple(config.rollout_steps) if config else (1, 4),
                config=config, reference_dir=reference_dir,
            ),
        )

        identity_phase = (
            "identity_config_bound", "source_identity_match",
            "variable_coordinate_identity_match", "ckpt_sha256_bound",
            "normalization_parity", "manifest_identity_match",
            "raw_input_binding", "multistep_reference_present",
        )
        identity_failures = [n for n in identity_phase if not result["gate_criteria"][n]]
        result["identity_phase_failures"] = identity_failures

        if identity_failures:
            # Fail closed and skip the expensive phase: the remaining criteria
            # stay False and say why, rather than crashing somewhere unrelated.
            skip_reason = (
                "SKIPPED: identity binding failed before model load "
                f"({identity_failures})"
            )
            for name in (
                "strict_load_zero_diff", "version_identity_match",
                "input_norm_binding", "upstream_parity", "multistep_parity",
                "zero_edit_equals_official", "no_state_leak", "outputs_finite",
            ):
                result["criteria_details"][name] = {
                    "criterion": name, "passed": False, "reason": skip_reason,
                }
                result["gate_criteria"][name] = False
            result["s0_gate_pass"] = False
            result["status"] = "gate_failed"
            result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return result

        # ------------------------------------------------------------------
        # Phase 2: expensive checks (model load + forward passes)
        # ------------------------------------------------------------------
        print(f"Loading checkpoint (detailed) for ps{patch_size}...", flush=True)
        load_kwargs: Dict[str, Any] = {}
        if config is not None:
            load_kwargs = {
                "variables": list(config.variables),
                "in_img_size": tuple(int(v) for v in config.grid_shape),
                "hidden_size": config.hidden_size,
                "depth": config.depth,
                "num_heads": config.num_heads,
                "mlp_ratio": config.mlp_ratio,
            }
        load_result = load_stormer_checkpoint_detailed(
            str(PATHS["checkpoint"]), patch_size=patch_size, **load_kwargs
        )
        model = load_result.model.to(device)
        bridge = WeatherStepBridge(model, norm, load_result.version)

        print("Running runtime criteria...", flush=True)

        record("strict_load_zero_diff", verify_strict_load_zero_diff(load_result))

        # Real expected-vs-actual version comparison (never self-vs-self).
        version_detail = verify_version_identity(
            bridge, config, load_result, computed_ckpt_sha256
        )
        record("version_identity_match", version_detail)
        result["version_match"] = version_detail["passed"]
        if not version_detail["passed"]:
            result["version_match_error"] = version_detail.get("error")

        record(
            "input_norm_binding",
            verify_input_norm_binding(
                bridge, x_raw_0, PATHS["upstream_reference_base"], patch_size, device,
                tolerance=config.input_norm_tolerance if config else 1e-6,
                config=config, reference_dir=reference_dir,
            ),
        )

        record(
            "upstream_parity",
            verify_upstream_parity(
                bridge, PATHS["upstream_reference_base"], patch_size, x_raw_0, device,
                tolerance=config.parity_tolerance if config else 1e-5,
                config=config, reference_dir=reference_dir,
                input_norm_tolerance=config.input_norm_tolerance if config else 1e-6,
            ),
        )

        record(
            "multistep_parity",
            verify_multistep_parity(
                bridge, x_raw_0, PATHS["upstream_reference_base"], patch_size, device,
                registered_rollouts=config.registered_rollouts if config else ((6, 1), (6, 4)),
                tolerance=config.rollout_tolerance if config else 1e-5,
                config=config, reference_dir=reference_dir,
            ),
        )

        interval_hours = int(config.interval_hours) if config is not None else 6
        rollout_steps = tuple(config.rollout_steps) if config is not None else (1, 4)
        ze_details: Dict[str, Any] = {}
        ze_all_pass = True
        for steps in rollout_steps:
            key = f"{interval_hours}h_{steps}step"
            detail = verify_zero_edit_internal_consistency(
                bridge, x_raw_0, interval=interval_hours, steps=int(steps), device=device
            )
            ze_details[key] = detail
            ze_all_pass = ze_all_pass and detail["passed"]
        ze_details["all_passed"] = ze_all_pass
        record(
            "zero_edit_equals_official",
            {"criterion": "zero_edit_equals_official", "passed": ze_all_pass, **ze_details},
        )

        record("no_state_leak", verify_no_state_leak(bridge, x_raw_0, device))
        record("outputs_finite", verify_outputs_finite(bridge, x_raw_0, device))

        # RMSE sanity check (informational)
        print("Computing RMSE sanity check (informational)...", flush=True)
        result["rmse_sanity"] = compute_rmse_sanity(
            bridge, inputs['data'], inputs['lat'], device,
            variables=list(config.variables) if config is not None else None,
        )

        # ------------------------------------------------------------------
        # Overall pass/fail. The verdict assigned here is PROVISIONAL: it is
        # only committed once the pre-commit checks below confirm that every
        # required computation really ran and the identity evidence is on the
        # record. Anything that raises from here on lands in the outer handler,
        # which revokes the verdict unconditionally (B13).
        # ------------------------------------------------------------------
        result["s0_gate_pass"] = all(result["gate_criteria"].values())
        result["status"] = "ok" if result["s0_gate_pass"] else "gate_failed"

        if result["s0_gate_pass"]:
            _assert_gate_verdict_committable(result)
            result["verdict_committed"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        # A late exception revokes the verdict unconditionally: whatever was
        # provisionally assigned above is reset here, fail-closed.
        result["s0_gate_pass"] = False
        result["status"] = "exception"
        result["verdict_committed"] = False
    finally:
        # Cleanup is non-essential and runs AFTER the verdict is decided. Each
        # step is individually contained so a failing cleanup can neither
        # destroy the computed result nor escape this function; failures are
        # classified onto result["cleanup_warnings"] instead (B13).
        cleanup_warnings: List[Dict[str, Any]] = []

        def _drop_model_references() -> None:
            nonlocal model, bridge
            model = None
            bridge = None

        _run_classified_cleanup(
            "release_model_references", _drop_model_references, cleanup_warnings
        )
        _run_classified_cleanup(
            "torch_cuda_empty_cache", _empty_cuda_cache, cleanup_warnings
        )

        try:
            result["cleanup_warnings"] = cleanup_warnings
        except BaseException:  # pragma: no cover - result is a plain dict
            pass

    result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return result


def format_finite(value: Any, spec: str, missing: str = "N/A") -> str:
    """Apply a numeric format spec ONLY to a present, finite, real number (B14).

    Every other case -- absent, None, non-numeric, NaN, +/-Inf, or a value the
    spec simply cannot render -- degrades to a labelled placeholder instead of
    raising. A report is evidence about a run; it must never be the thing that
    destroys that evidence, and a missing or non-finite number must be visible
    as missing rather than printed as a plausible-looking `nan`.

    Args:
        value: Candidate value to format.
        spec: Format spec (e.g. ".2e", ".1f") applied only to finite reals.
        missing: Placeholder prefix used when the value cannot be formatted.

    Returns:
        The formatted number, or `missing` annotated with why it was skipped.
    """
    if value is None:
        return f"{missing} (missing)"
    # bool is a subclass of int; formatting it as a float would be misleading.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return f"{missing} (non-numeric: {type(value).__name__})"
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return f"{missing} (non-numeric: {type(value).__name__})"
    if math.isnan(numeric):
        return f"{missing} (nan)"
    if math.isinf(numeric):
        return f"{missing} ({'+' if numeric > 0 else '-'}inf)"
    try:
        return format(numeric, spec)
    except (TypeError, ValueError):  # pragma: no cover - spec is a literal
        return f"{missing} (unformattable)"


def generate_report(result: Dict[str, Any]) -> str:
    """Generate markdown report from S0 gate results.

    Every numeric field is rendered through `format_finite`, so a missing or
    non-finite value produces a visible placeholder rather than an exception or
    a misleading `nan`.
    """
    lines = [
        '# S0 Gate Report',
        '',
        f'**Run ID:** `{result.get("run_id", "unknown")}`',
        f'**Executed:** {result.get("started_utc", "unknown")}',
        f'**Hostname:** {result.get("hostname", "unknown")}',
        f'**Repo Root:** `{result.get("repo_root", "unknown")}`',
        f'**Patch Size:** {result.get("patch_size", "unknown")}',
        f'**Status:** {"PASS" if result.get("s0_gate_pass") else "FAIL"}',
        '',
        '## Environment',
        '',
        f'- PyTorch: {result.get("torch_version", "unknown")}',
        f'- CUDA available: {result.get("cuda_available", False)}',
        f'- Device: {result.get("cuda_device", "unknown")}',
        f'- Normalization policy: {result.get("normalization_policy", "unknown")}',
        f'- Normalization digest: `{result.get("normalization_digest", "unknown")}`',
        '',
        '## Gate Criteria Results',
        '',
        '| Criterion | Status | Notes |',
        '|-----------|--------|-------|',
    ]

    for criterion, passed in result.get('gate_criteria', {}).items():
        status = 'PASS' if passed else 'FAIL'
        details = result.get('criteria_details', {}).get(criterion, {})
        if isinstance(details, dict) and 'error' in details:
            error_msg = str(details['error'])
            notes = f"Error: {error_msg[:50]}..." if len(error_msg) > 50 else f"Error: {error_msg}"
        elif criterion == 'upstream_parity' and isinstance(details, dict) and not details.get('upstream_available'):
            notes = "Reference not found - run export_upstream_reference.py first"
        else:
            notes = ""
        lines.append(f'| {criterion} | {status} | {notes} |')

    lines.extend([
        '',
        '## Details',
        '',
    ])

    # Checkpoint SHA-256
    sha_details = result.get('criteria_details', {}).get('ckpt_sha256_bound', {})
    if isinstance(sha_details, dict) and sha_details.get('computed_sha256'):
        lines.extend([
            '### Checkpoint Identity',
            f'- Computed SHA-256: `{sha_details["computed_sha256"]}`',
        ])
        if sha_details.get('expected_sha256'):
            lines.append(f'- Expected SHA-256: `{sha_details["expected_sha256"]}`')
            lines.append(f'- Match: {sha_details.get("match", "N/A")}')
        lines.append('')

    # Upstream parity
    up_details = result.get('criteria_details', {}).get('upstream_parity', {})
    lines.extend([
        '### Upstream Parity (vs Official xformers Stormer)',
        f'- Available: {up_details.get("upstream_available", False) if isinstance(up_details, dict) else "N/A"}',
    ])
    if isinstance(up_details, dict) and up_details.get('upstream_available'):
        # The lines are emitted unconditionally and `format_finite` decides how
        # each value renders: a missing or non-finite number must be VISIBLE as
        # missing, not silently dropped from the report (B14).
        lines.append(
            f'- Max absolute diff: {format_finite(up_details.get("max_abs_diff"), ".2e")}'
        )
        lines.append(
            f'- Tolerance: {format_finite(up_details.get("tolerance"), ".0e")}'
        )
    if isinstance(up_details, dict) and up_details.get('error'):
        lines.append(f'- Error: {up_details["error"]}')
    lines.append('')

    # Internal consistency
    ze_details = result.get('criteria_details', {}).get('zero_edit_equals_official', {})
    lines.extend([
        '### Internal Consistency (controlled_rollout vs forward_validation)',
        '',
        'Note: This is bridge-internal consistency, NOT upstream parity.',
        '',
    ])
    if isinstance(ze_details, dict):
        for key in ['6h_1step', '6h_4step']:
            if key in ze_details:
                d = ze_details[key]
                if isinstance(d, dict):
                    lines.append(
                        f'- {key}: '
                        f'max_diff={format_finite(d.get("max_abs_diff"), ".2e")}, '
                        f'passed={d.get("passed")}'
                    )
    lines.append('')

    # RMSE sanity (informational)
    rmse = result.get('rmse_sanity', {})
    lines.extend([
        '## RMSE Sanity Check (Informational Only)',
        '',
        '**Note:** These values are for reference only and do NOT affect gate pass/fail.',
        '',
        '| Lead Time | RMSE Z500 | Historical Reference |',
        '|-----------|-----------|---------------------|',
    ])
    for key in ['6h', '24h']:
        if isinstance(rmse, dict) and key in rmse:
            val = rmse[key].get('rmse_z500_weighted') if isinstance(rmse[key], dict) else None
            ref = HISTORICAL_REFERENCE.get(f'z500_rmse_{key}', {}).get('value', 'N/A')
            lines.append(f'| {key} | {format_finite(val, ".1f")} | ~{ref} |')

    # Non-essential cleanup failures are reported on their own axis, never
    # folded into the gate verdict (B13).
    cleanup_warnings = result.get('cleanup_warnings') or []
    if isinstance(cleanup_warnings, list) and cleanup_warnings:
        lines.extend([
            '',
            '## Cleanup Warnings (Do NOT Affect Gate Pass/Fail)',
            '',
        ])
        for warning in cleanup_warnings:
            if isinstance(warning, dict):
                lines.append(
                    f'- `{warning.get("stage", "unknown")}` '
                    f'[{warning.get("classification", "unclassified")}]: '
                    f'{warning.get("error", "unknown error")}'
                )
            else:
                lines.append(f'- {warning}')

    lines.extend([
        '',
        f'*Report generated at {result.get("finished_utc", "unknown")}*',
    ])

    return '\n'.join(lines)


def generate_minimal_report(result: Dict[str, Any], report_error: str) -> str:
    """Fallback report used when the full report could not be generated (B14).

    Report generation is presentation, not evidence. If it fails, the run's
    computed JSON result must still be published; this stands in for the
    markdown so publication can proceed and records why the full report is
    missing.
    """
    lines = [
        '# S0 Gate Report (DEGRADED)',
        '',
        'The full markdown report could not be generated. The JSON result '
        'published alongside this file is the authoritative record of the run '
        'and was NOT affected by this failure.',
        '',
        f'**Report generation error:** `{report_error}`',
        '',
        f'**Run ID:** `{result.get("run_id", "unknown")}`',
        f'**Status:** {"PASS" if result.get("s0_gate_pass") else "FAIL"}',
        f'**Gate status field:** {result.get("status", "unknown")}',
        '',
        '## Gate Criteria Results',
        '',
    ]
    criteria = result.get('gate_criteria')
    if isinstance(criteria, dict) and criteria:
        for criterion, passed in criteria.items():
            lines.append(f'- {criterion}: {"PASS" if passed else "FAIL"}')
    else:
        lines.append('- (no criteria recorded)')
    return '\n'.join(lines)


def publish_gate_outputs(
    base_dir: Path, result: Dict[str, Any], report: str
) -> Path:
    """Write the run's outputs into a unique run-scoped directory, atomically.

    Every run gets its own directory (never a shared, reused path), everything
    is written into a staging directory first, the JSON is re-read and checked,
    and only then is the directory published under its final name. The PASS
    marker is written last, via a temp file plus an atomic rename, so a
    half-written or crashed run can never be consumed as a stale PASS.
    """
    run_id = str(result.get("run_id") or "s0-gate-unknown")
    base_dir.mkdir(parents=True, exist_ok=True)
    final_dir = base_dir / run_id
    if final_dir.exists():
        raise RuntimeError(
            f"Run-scoped output directory already exists and will not be reused: {final_dir}"
        )

    staging = base_dir / f".staging-{run_id}-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        json_path = staging / 's0_gate_result.json'
        with open(json_path, 'w') as handle:
            json.dump(result, handle, indent=2, default=str)
        report_path = staging / 'S0_GATE_REPORT.md'
        with open(report_path, 'w') as handle:
            handle.write(report)

        # Verify the staged artifacts before publishing them.
        with open(json_path) as handle:
            reread = json.load(handle)
        if reread.get("s0_gate_pass") != result.get("s0_gate_pass"):
            raise RuntimeError("Staged gate result does not round-trip; refusing to publish")
        if not report_path.read_text():
            raise RuntimeError("Staged gate report is empty; refusing to publish")

        os.replace(staging, final_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # PASS marker is published LAST and atomically.
    if result.get("s0_gate_pass"):
        marker_tmp = final_dir / ".PASS.tmp"
        with open(marker_tmp, 'w') as handle:
            handle.write(f"{run_id}\n{result.get('finished_utc', '')}\n")
        os.replace(marker_tmp, final_dir / "PASS")

    return final_dir


def main():
    parser = argparse.ArgumentParser(description="S0 Gate Verification")
    parser.add_argument(
        "--config", type=Path, default=None,
        help=(
            "Frozen GateIdentityConfig JSON holding the expected identity. "
            "The SAME file must be passed to scripts/export_upstream_reference.py."
        )
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=None,
        help=(
            "Explicit upstream reference directory. Required whenever more than "
            "one candidate directory matches the expected identity; auto-selection "
            "refuses to break such a tie."
        )
    )
    parser.add_argument(
        "--checkpoint", choices=["ps2", "ps4"], default="ps4",
        help="Checkpoint to use when --config does not specify one (default: ps4)"
    )
    parser.add_argument(
        "--check", type=str, default=None,
        help="Run only a specific criterion (not implemented yet)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="BASE output directory; each run publishes into <base>/<run_id>"
    )
    parser.add_argument(
        "--expected-sha256", type=str, default=None,
        help="Expected checkpoint SHA-256 for comparison (overrides --config)"
    )
    parser.add_argument(
        "--expected-norm-digest", type=str, default=None,
        help=(
            "Expected SCHEMA-QUALIFIED normalization identity "
            "(ed-norm-identity/1:<hex>). A bare legacy digest fails closed."
        )
    )
    parser.add_argument(
        "--asset-root", type=Path, default=None,
        help="Asset root fallback when --config does not name one"
    )
    parser.add_argument(
        "--help-criteria", action="store_true",
        help="Show gate criteria descriptions"
    )
    args = parser.parse_args()

    if args.help_criteria:
        print("S0 Gate Criteria:")
        print("  identity_config_bound  - Frozen --config with a complete expected identity")
        print("  source_identity_match  - Pinned upstream commit + frozen config digest match")
        print("  variable_coordinate_identity_match - Variable order and coordinate VALUES match")
        print("  ckpt_sha256_bound      - Checkpoint SHA-256 matches expected value")
        print("  normalization_parity   - Normalization identity matches expected (schema-checked)")
        print("  manifest_identity_match- Reference manifest enforced vs config AND runtime")
        print("  raw_input_binding      - Raw input hash matches expected")
        print("  multistep_reference_present - All registered reference files present")
        print("  strict_load_zero_diff  - Checkpoint loads with zero missing/unexpected keys")
        print("  version_identity_match - Bridge version equals the version the config requires")
        print("  input_norm_binding     - bridge.normalize(x_raw) reproduces input_norm.pt")
        print("  upstream_parity        - Bridge output matches official xformers Stormer (<=1e-5)")
        print("  multistep_parity       - Every registered rollout matches the reference numerically")
        print("  zero_edit_equals_official - Bridge internal consistency (<=1e-6)")
        print("  no_state_leak          - Identical inputs produce bit-identical outputs")
        print("  outputs_finite         - All outputs contain no NaN/Inf")
        return 0

    config: Optional[GateIdentityConfig] = None
    if args.config is not None:
        try:
            config = GateIdentityConfig.load_json(args.config)
        except Exception as exc:
            print(f'ERROR: could not load --config {args.config}: {exc}', flush=True)
            return 1

    patch_size = config.patch_size if config is not None else (
        2 if args.checkpoint == "ps2" else 4
    )
    PATHS = resolve_paths(config, args.asset_root, patch_size)

    base_output_dir = args.output_dir or Path(os.environ.get(
        'S0_OUTPUT_DIR', str(PATHS["default_output_dir"])
    ))

    print('=' * 60, flush=True)
    print('S0 Gate Verification - EarthDelta Stormer Bridge', flush=True)
    print('=' * 60, flush=True)
    print(f'Checkpoint: ps{patch_size}', flush=True)
    if config is not None:
        print(f'Config: {args.config} (digest {config.config_digest})', flush=True)
    else:
        print('Config: NONE - identity-bound criteria will fail closed', flush=True)
    if args.reference_dir is not None:
        print(f'Reference dir (explicit): {args.reference_dir}', flush=True)

    # Run all gate verifications
    result = run_s0_gate(
        patch_size=patch_size,
        check_only=args.check,
        output_dir=base_output_dir,
        expected_checkpoint_sha256=args.expected_sha256,
        expected_normalization_digest=args.expected_norm_digest,
        config=config,
        config_path=args.config,
        reference_dir=args.reference_dir,
        asset_root=args.asset_root,
    )

    # Write outputs into a unique run-scoped directory, published atomically.
    #
    # Report generation is presentation and is therefore isolated from
    # publication: `run_s0_gate` has already computed the run's evidence, and a
    # formatting failure in the markdown must never prevent that evidence from
    # being persisted (B14). The failure is recorded on its own field of the
    # result and a degraded report stands in so publication can proceed.
    try:
        report = generate_report(result)
    except Exception as exc:
        report_error = f'{type(exc).__name__}: {exc}'
        result["report_generation_error"] = report_error
        result["report_generation_traceback"] = traceback.format_exc()
        print(
            f'WARNING: markdown report generation failed ({report_error}); '
            'publishing the computed JSON result with a degraded report.',
            flush=True,
        )
        report = generate_minimal_report(result, report_error)

    try:
        run_dir = publish_gate_outputs(base_output_dir, result, report)
    except Exception as exc:
        print(f'ERROR: could not publish gate outputs: {exc}', flush=True)
        return 1

    result["output_dir"] = str(run_dir)
    print(f'\nRun-scoped output directory: {run_dir}', flush=True)
    print(f'JSON result written to: {run_dir / "s0_gate_result.json"}', flush=True)
    print(f'Markdown report written to: {run_dir / "S0_GATE_REPORT.md"}', flush=True)
    if result.get("s0_gate_pass"):
        print(f'PASS marker published: {run_dir / "PASS"}', flush=True)

    # Print summary
    print('\n' + '=' * 60, flush=True)
    print('SUMMARY', flush=True)
    print('=' * 60, flush=True)
    print(f'Status: {result.get("status")}', flush=True)
    print(f'S0 Gate Pass: {result.get("s0_gate_pass")}', flush=True)
    print('Gate Criteria:', flush=True)
    for criterion, passed in result.get('gate_criteria', {}).items():
        status = 'PASS' if passed else 'FAIL'
        print(f'  - {criterion}: {status}', flush=True)

    if result.get('error'):
        print(f'\nError: {result.get("error")}', flush=True)

    # Classified non-essential cleanup failures are reported on their own line,
    # separate from the verdict and from result["error"] (B13).
    cleanup_warnings = result.get('cleanup_warnings') or []
    if cleanup_warnings:
        print(
            f'\nCleanup warnings (do NOT affect gate pass/fail): '
            f'{len(cleanup_warnings)}',
            flush=True,
        )
        for warning in cleanup_warnings:
            if isinstance(warning, dict):
                print(
                    f'  - {warning.get("stage", "unknown")} '
                    f'[{warning.get("classification", "unclassified")}]: '
                    f'{warning.get("error", "unknown error")}',
                    flush=True,
                )
            else:
                print(f'  - {warning}', flush=True)

    if result.get('report_generation_error'):
        print(
            f'\nReport generation error (result JSON was still published): '
            f'{result.get("report_generation_error")}',
            flush=True,
        )

    # Exit code: 0 if all pass, 1 otherwise
    return 0 if result.get('s0_gate_pass') else 1


if __name__ == '__main__':
    sys.exit(main())
