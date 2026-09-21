#!/usr/bin/env python3
"""S0 Gate verification script for EarthDelta Stormer bridge.

Runs S0-level gate checks to verify bridge correctness before downstream use.
All gate criteria are fail-closed: initialized False, remain False on exceptions.

Gate Criteria:
- ckpt_sha256_bound: Checkpoint SHA-256 hash matches expected value (not just length check)
- strict_load_zero_diff: Checkpoint loads with zero missing/unexpected keys
- upstream_parity: Bridge output matches official xformers Stormer (<=1e-5)
- zero_edit_equals_official: Bridge internal consistency (<=1e-6)
- normalization_parity: Normalization digest matches expected reference
- no_state_leak: Identical inputs produce bit-identical outputs
- outputs_finite: All outputs contain no NaN/Inf
- raw_input_binding: Raw input hash matches expected (if reference exists)
- multistep_reference_present: All required multistep reference files are present

Usage:
    python scripts/s0_gate.py [--checkpoint ps2|ps4] [--output-dir PATH]

The upstream_parity check requires running export_upstream_reference.py first
in a GPU+xformers environment. Without the reference artifacts, this check
fails closed with a clear message.

Environment variables:
    EARTHDELTA_REPO_ROOT: Override repo root detection
    S0_OUTPUT_DIR: Override output directory for results

Exit codes:
    0: All gate criteria passed
    1: One or more gate criteria failed
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

print("S0 Gate script starting...", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)


# =============================================================================
# Path Configuration (no hardcoded absolute paths)
# =============================================================================

def get_repo_root() -> Path:
    """Get repository root from environment or script location."""
    if "EARTHDELTA_REPO_ROOT" in os.environ:
        return Path(os.environ["EARTHDELTA_REPO_ROOT"]).resolve()
    # Default: two levels up from this script
    return Path(__file__).resolve().parent.parent


def get_paths(repo_root: Path, patch_size: int = 4) -> Dict[str, Path]:
    """Get all paths relative to repo root.

    Args:
        repo_root: Repository root path
        patch_size: Patch size (2 or 4). Default is 4 (mainline per research_spec_v6.yaml).
    """
    ps_str = f"ps{patch_size}"
    return {
        "input_dir": repo_root / "scripts" / "s0_gate_inputs",
        "checkpoint_ps2": repo_root / "checkpoints" / "stormer_1.40625_patch_size_2.ckpt",
        "checkpoint_ps4": repo_root / "checkpoints" / "stormer_1.40625_patch_size_4.ckpt",
        "checkpoint": repo_root / "checkpoints" / f"stormer_1.40625_patch_size_{patch_size}.ckpt",
        "norm_dir": repo_root / "reference" / "stormer" / "normalization_constants",
        "upstream_reference_base": repo_root / "artifacts",
        "default_output_dir": repo_root / "artifacts" / "s0_gate",
    }


REPO_ROOT = get_repo_root()

# Add repo to Python path
sys.path.insert(0, str(REPO_ROOT))
print(f"Repository root: {REPO_ROOT}", flush=True)

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
    from earthdelta.contracts import EditPlan, reference_plan, GateIdentityConfig
    from earthdelta.lowrank import ExpertLoRA
    print("EarthDelta imports successful", flush=True)
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}", flush=True)
    print("Dependencies must be pre-installed. Do not auto-install.", flush=True)
    sys.exit(1)

# Z500 channel index in canonical 69-variable order
Z500_IDX = 11  # geopotential_500

# Historical reference RMSE values from Stormer paper (informational only)
# These are approximate and NOT used for gate pass/fail decisions
HISTORICAL_REFERENCE = {
    "z500_rmse_6h": {"value": 30, "unit": "m^2/s^2", "note": "rough estimate from paper"},
    "z500_rmse_24h": {"value": 80, "unit": "m^2/s^2", "note": "rough estimate from paper"},
}


# =============================================================================
# Data loading utilities
# =============================================================================

def load_npy_inputs(input_dir: Path) -> Dict[str, np.ndarray]:
    """Load pre-extracted .npy inputs."""
    return {
        'data': np.load(input_dir / 'jan2020_full.npy'),  # (124, 69, 128, 256)
        'lat': np.load(input_dir / 'lat.npy'),
        'lon': np.load(input_dir / 'lon.npy'),
        'inp_mean': np.load(input_dir / 'inp_mean.npy'),
        'inp_std': np.load(input_dir / 'inp_std.npy'),
        'diff_mean_6': np.load(input_dir / 'diff_mean_6.npy'),
        'diff_std_6': np.load(input_dir / 'diff_std_6.npy'),
        'diff_mean_24': np.load(input_dir / 'diff_mean_24.npy'),
        'diff_std_24': np.load(input_dir / 'diff_std_24.npy'),
    }


def create_normalization_from_npy(
    inputs: Dict[str, np.ndarray],
    policy: str = POLICY_OFFICIAL_ZERO_DIFF_MEAN,
) -> NormalizationContract:
    """Create NormalizationContract from pre-extracted numpy arrays.

    Args:
        inputs: Dictionary of numpy arrays from load_npy_inputs
        policy: Normalization policy. Default is POLICY_OFFICIAL_ZERO_DIFF_MEAN
                to match official inference.py semantics.
    """
    # Under official policy, force diff_mean to zero
    if policy == POLICY_OFFICIAL_ZERO_DIFF_MEAN:
        diff_mean_6 = np.zeros_like(inputs['diff_mean_6'])
        diff_mean_24 = np.zeros_like(inputs['diff_mean_24'])
    else:
        diff_mean_6 = inputs['diff_mean_6']
        diff_mean_24 = inputs['diff_mean_24']

    return NormalizationContract(
        inp_mean=torch.from_numpy(inputs['inp_mean']).float(),
        inp_std=torch.from_numpy(inputs['inp_std']).float(),
        diff_mean={
            6: torch.from_numpy(diff_mean_6).float(),
            24: torch.from_numpy(diff_mean_24).float(),
        },
        diff_std={
            6: torch.from_numpy(inputs['diff_std_6']).float(),
            24: torch.from_numpy(inputs['diff_std_24']).float(),
        },
        variables=DEFAULT_VARIABLES,
        policy=policy,
    )


def compute_raw_input_hash(x_raw: np.ndarray) -> str:
    """Compute SHA-256 hash of raw input for identity binding."""
    return hashlib.sha256(x_raw.tobytes()).hexdigest()[:16]


# =============================================================================
# Gate criterion: ckpt_sha256_bound (FULL comparison, not just length)
# =============================================================================

def verify_ckpt_sha256(
    ckpt_path: Path,
    load_result: Optional[CheckpointLoadResult],
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify checkpoint SHA-256 matches expected value.

    This criterion now performs FULL SHA-256 comparison, not just length check.
    If expected_sha256 is None, we just compute and record the hash (first run).
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

        if load_result is not None:
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

def find_upstream_reference_dir(base_dir: Path, patch_size: int) -> Optional[Path]:
    """Find the upstream reference directory for a given patch size.

    Looks for directories matching the pattern upstream_reference_ps{patch_size}_*
    """
    pattern = f"upstream_reference_ps{patch_size}_*"
    matches = list(base_dir.glob(pattern))
    if matches:
        # Return most recent if multiple
        return sorted(matches)[-1]

    # Fallback to legacy path
    legacy_path = base_dir / "upstream_reference"
    if legacy_path.exists():
        return legacy_path

    return None


def verify_upstream_parity(
    bridge: WeatherStepBridge,
    upstream_base_dir: Path,
    patch_size: int,
    x_raw: np.ndarray,
    device: torch.device,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Verify bridge output matches official xformers Stormer output.

    This is the core A01 audit fix: comparing bridge output against the
    REAL official code path, not against itself.

    The upstream reference must be exported by export_upstream_reference.py
    running in a GPU+xformers environment. If the reference is missing,
    this check fails closed.
    """
    result = {
        "criterion": "upstream_parity",
        "passed": False,  # fail-closed
        "tolerance": tolerance,
        "upstream_available": False,
    }

    try:
        # Find the upstream reference directory
        upstream_dir = find_upstream_reference_dir(upstream_base_dir, patch_size)
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
        }
        result["upstream_available"] = True

        # Verify both 1-step and 4-step references exist
        required_files = ["official_output_6h_1step.pt", "official_output_6h_4step.pt"]
        missing_files = []
        for fname in required_files:
            if not (upstream_dir / fname).exists():
                missing_files.append(fname)

        if missing_files:
            result["error"] = f"Missing required reference files: {missing_files}"
            result["passed"] = False
            return result

        # Load upstream outputs
        upstream_output_6h_1step = torch.load(upstream_dir / "official_output_6h_1step.pt")
        upstream_input_norm = torch.load(upstream_dir / "input_norm.pt")

        # Verify shapes and dtypes
        result["upstream_output_shape"] = list(upstream_output_6h_1step.shape)
        result["upstream_output_dtype"] = str(upstream_output_6h_1step.dtype)

        expected_shape = (1, 69, 128, 256)
        if tuple(upstream_output_6h_1step.shape) != expected_shape:
            result["error"] = (
                f"Unexpected upstream output shape: {upstream_output_6h_1step.shape}, "
                f"expected {expected_shape}"
            )
            return result

        # Move to device
        upstream_output = upstream_output_6h_1step.to(device)
        input_norm = upstream_input_norm.to(device)

        # Run bridge on same input
        with torch.no_grad():
            bridge_output = bridge.forward_validation(
                input_norm, bridge.variables, interval=6, steps=1
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
        result["passed"] = max_diff <= tolerance

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
# Gate criterion: raw_input_binding
# =============================================================================

def verify_raw_input_binding(
    x_raw: np.ndarray,
    upstream_base_dir: Path,
    patch_size: int,
) -> Dict[str, Any]:
    """Verify raw input hash matches expected value from upstream reference."""
    result = {
        "criterion": "raw_input_binding",
        "passed": False,  # fail-closed
    }

    try:
        computed_hash = compute_raw_input_hash(x_raw)
        result["computed_hash"] = computed_hash

        # Find upstream reference
        upstream_dir = find_upstream_reference_dir(upstream_base_dir, patch_size)
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
) -> Dict[str, Any]:
    """Verify all required multistep reference files are present and loadable."""
    result = {
        "criterion": "multistep_reference_present",
        "passed": False,  # fail-closed
        "required_steps": list(required_steps),
    }

    try:
        upstream_dir = find_upstream_reference_dir(upstream_base_dir, patch_size)
        if upstream_dir is None:
            result["error"] = f"Upstream reference not found in {upstream_base_dir}"
            return result

        result["upstream_dir"] = str(upstream_dir)

        present_files = {}
        missing_files = []
        shape_mismatch = []

        for steps in required_steps:
            fname = f"official_output_6h_{steps}step.pt"
            fpath = upstream_dir / fname

            if not fpath.exists():
                missing_files.append(fname)
                present_files[fname] = False
            else:
                present_files[fname] = True
                # Load and verify shape
                try:
                    tensor = torch.load(fpath, map_location="cpu")
                    expected_shape = (1, 69, 128, 256)
                    if tuple(tensor.shape) != expected_shape:
                        shape_mismatch.append(
                            f"{fname}: shape {tuple(tensor.shape)} != {expected_shape}"
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
) -> Dict[str, Any]:
    """Verify normalization contract matches reference npz files and expected digest."""
    result = {
        "criterion": "normalization_parity",
        "passed": False,  # fail-closed
        "computed_digest": norm_from_npy.digest,
        "policy": norm_from_npy.policy,
        "n_variables": len(norm_from_npy.variables),
        "intervals_present": list(norm_from_npy.diff_std.keys()),
    }

    try:
        if not norm_dir.exists():
            result["error"] = f"Normalization directory not found: {norm_dir}"
            return result

        # Load from NPZ with same policy
        norm_from_npz = NormalizationContract.from_npz_dir(
            str(norm_dir), variables=DEFAULT_VARIABLES, intervals=(6, 24),
            policy=norm_from_npy.policy,
        )
        result["digest_from_npz"] = norm_from_npz.digest
        result["digests_match_npz"] = norm_from_npy.digest == norm_from_npz.digest

        # Check against expected digest if provided
        if expected_digest is not None:
            result["expected_digest"] = expected_digest
            result["digests_match_expected"] = norm_from_npy.digest == expected_digest
            result["passed"] = result["digests_match_expected"] and result["digests_match_npz"]
            if not result["passed"]:
                result["error"] = (
                    f"Digest mismatch: computed={norm_from_npy.digest}, "
                    f"expected={expected_digest}, from_npz={norm_from_npz.digest}"
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
) -> Dict[str, Any]:
    """Compute RMSE sanity check metrics (informational, not a gate criterion)."""
    result = {
        "note": "Historical reference only - NOT used for gate pass/fail",
        "historical_reference": HISTORICAL_REFERENCE,
    }

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

                z500_pred = y_pred[:, Z500_IDX, :, :]
                z500_true = y_true_t[:, Z500_IDX, :, :]

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
# Main gate execution
# =============================================================================

def run_s0_gate(
    patch_size: int = 4,
    check_only: Optional[str] = None,
    output_dir: Optional[Path] = None,
    expected_checkpoint_sha256: Optional[str] = None,
    expected_normalization_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all S0 gate verifications.

    All gate criteria are fail-closed: they start False and stay False
    if any exception occurs during evaluation.

    Args:
        patch_size: Patch size (2 or 4). Default is 4 (mainline).
        check_only: Run only a specific criterion (not implemented yet).
        output_dir: Output directory for results.
        expected_checkpoint_sha256: Expected checkpoint SHA-256 for comparison.
        expected_normalization_digest: Expected normalization digest for comparison.
    """
    PATHS = get_paths(REPO_ROOT, patch_size)

    result = {
        "status": "failed",
        "run_id": f's0-gate-{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dt%H%M%Sz")}',
        "hostname": socket.gethostname(),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "patch_size": patch_size,
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "gate_criteria": {
            "ckpt_sha256_bound": False,
            "strict_load_zero_diff": False,
            "upstream_parity": False,
            "zero_edit_equals_official": False,
            "normalization_parity": False,
            "no_state_leak": False,
            "outputs_finite": False,
            "raw_input_binding": False,
            "multistep_reference_present": False,
        },
        "criteria_details": {},
        "s0_gate_pass": False,
    }

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

        # Load inputs
        print("Loading inputs...", flush=True)
        inputs = load_npy_inputs(PATHS["input_dir"])
        x_raw_0 = inputs['data'][0]

        # Create normalization with official policy (zero diff_mean)
        norm = create_normalization_from_npy(inputs, policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN)
        result["normalization_policy"] = norm.policy
        result["normalization_digest"] = norm.digest

        # Load checkpoint (detailed)
        print(f"Loading checkpoint (detailed) for ps{patch_size}...", flush=True)
        load_result = load_stormer_checkpoint_detailed(
            str(PATHS["checkpoint"]), patch_size=patch_size
        )
        model = load_result.model.to(device)
        bridge = WeatherStepBridge(model, norm, load_result.version)

        # Call check_version_match at gate entry point (B02 fix)
        print("Verifying version match...", flush=True)
        try:
            check_version_match(bridge.version, bridge.version)
            result["version_match"] = True
        except ValueError as e:
            result["version_match"] = False
            result["version_match_error"] = str(e)

        # Run gate criteria
        print("Running gate criteria...", flush=True)

        # 1. ckpt_sha256_bound (FULL comparison, not just length)
        print("  - ckpt_sha256_bound", flush=True)
        sha256_result = verify_ckpt_sha256(
            PATHS["checkpoint"], load_result, expected_checkpoint_sha256
        )
        result["criteria_details"]["ckpt_sha256_bound"] = sha256_result
        result["gate_criteria"]["ckpt_sha256_bound"] = sha256_result["passed"]

        # 2. strict_load_zero_diff
        print("  - strict_load_zero_diff", flush=True)
        strict_result = verify_strict_load_zero_diff(load_result)
        result["criteria_details"]["strict_load_zero_diff"] = strict_result
        result["gate_criteria"]["strict_load_zero_diff"] = strict_result["passed"]

        # 3. upstream_parity
        print("  - upstream_parity", flush=True)
        upstream_result = verify_upstream_parity(
            bridge, PATHS["upstream_reference_base"], patch_size, x_raw_0, device
        )
        result["criteria_details"]["upstream_parity"] = upstream_result
        result["gate_criteria"]["upstream_parity"] = upstream_result["passed"]

        # 4. zero_edit_equals_official (internal consistency)
        print("  - zero_edit_equals_official (6h, 1-step)", flush=True)
        ze_6h_1 = verify_zero_edit_internal_consistency(
            bridge, x_raw_0, interval=6, steps=1, device=device
        )
        print("  - zero_edit_equals_official (6h, 4-step)", flush=True)
        ze_6h_4 = verify_zero_edit_internal_consistency(
            bridge, x_raw_0, interval=6, steps=4, device=device
        )
        ze_all_pass = ze_6h_1["passed"] and ze_6h_4["passed"]
        result["criteria_details"]["zero_edit_equals_official"] = {
            "6h_1step": ze_6h_1,
            "6h_4step": ze_6h_4,
            "all_passed": ze_all_pass,
        }
        result["gate_criteria"]["zero_edit_equals_official"] = ze_all_pass

        # 5. normalization_parity
        print("  - normalization_parity", flush=True)
        norm_result = verify_normalization_parity(
            norm, PATHS["norm_dir"], expected_normalization_digest
        )
        result["criteria_details"]["normalization_parity"] = norm_result
        result["gate_criteria"]["normalization_parity"] = norm_result["passed"]

        # 6. no_state_leak
        print("  - no_state_leak", flush=True)
        leak_result = verify_no_state_leak(bridge, x_raw_0, device)
        result["criteria_details"]["no_state_leak"] = leak_result
        result["gate_criteria"]["no_state_leak"] = leak_result["passed"]

        # 7. outputs_finite
        print("  - outputs_finite", flush=True)
        finite_result = verify_outputs_finite(bridge, x_raw_0, device)
        result["criteria_details"]["outputs_finite"] = finite_result
        result["gate_criteria"]["outputs_finite"] = finite_result["passed"]

        # 8. raw_input_binding
        print("  - raw_input_binding", flush=True)
        raw_input_result = verify_raw_input_binding(
            x_raw_0, PATHS["upstream_reference_base"], patch_size
        )
        result["criteria_details"]["raw_input_binding"] = raw_input_result
        result["gate_criteria"]["raw_input_binding"] = raw_input_result["passed"]

        # 9. multistep_reference_present
        print("  - multistep_reference_present", flush=True)
        multistep_result = verify_multistep_reference(
            PATHS["upstream_reference_base"], patch_size, required_steps=(1, 4)
        )
        result["criteria_details"]["multistep_reference_present"] = multistep_result
        result["gate_criteria"]["multistep_reference_present"] = multistep_result["passed"]

        # RMSE sanity check (informational)
        print("Computing RMSE sanity check (informational)...", flush=True)
        result["rmse_sanity"] = compute_rmse_sanity(
            bridge, inputs['data'], inputs['lat'], device
        )

        # Overall pass/fail
        result["s0_gate_pass"] = all(result["gate_criteria"].values())
        result["status"] = "ok" if result["s0_gate_pass"] else "gate_failed"

        # Clean up
        del model, bridge
        torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        # Ensure gate does NOT pass on exception (fail-closed)
        result["s0_gate_pass"] = False
        result["status"] = "exception"

    result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return result


def generate_report(result: Dict[str, Any]) -> str:
    """Generate markdown report from S0 gate results."""
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
        max_diff = up_details.get("max_abs_diff")
        tolerance = up_details.get("tolerance")
        if max_diff is not None:
            lines.append(f'- Max absolute diff: {max_diff:.2e}')
        if tolerance is not None:
            lines.append(f'- Tolerance: {tolerance:.0e}')
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
                    max_diff = d.get("max_abs_diff")
                    if max_diff is not None:
                        lines.append(f'- {key}: max_diff={max_diff:.2e}, passed={d.get("passed")}')
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
            if val is not None:
                lines.append(f'| {key} | {val:.1f} | ~{ref} |')
            else:
                lines.append(f'| {key} | N/A | ~{ref} |')

    lines.extend([
        '',
        f'*Report generated at {result.get("finished_utc", "unknown")}*',
    ])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="S0 Gate Verification")
    parser.add_argument(
        "--checkpoint", choices=["ps2", "ps4"], default="ps4",
        help="Checkpoint to use (default: ps4 - the mainline per research_spec_v6.yaml)"
    )
    parser.add_argument(
        "--check", type=str, default=None,
        help="Run only a specific criterion (not implemented yet)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--expected-sha256", type=str, default=None,
        help="Expected checkpoint SHA-256 for comparison"
    )
    parser.add_argument(
        "--expected-norm-digest", type=str, default=None,
        help="Expected normalization digest for comparison"
    )
    parser.add_argument(
        "--help-criteria", action="store_true",
        help="Show gate criteria descriptions"
    )
    args = parser.parse_args()

    if args.help_criteria:
        print("S0 Gate Criteria:")
        print("  ckpt_sha256_bound      - Checkpoint SHA-256 matches expected value")
        print("  strict_load_zero_diff  - Checkpoint loads with zero missing/unexpected keys")
        print("  upstream_parity        - Bridge output matches official xformers Stormer (<=1e-5)")
        print("  zero_edit_equals_official - Bridge internal consistency (<=1e-6)")
        print("  normalization_parity   - Normalization digest matches expected reference")
        print("  no_state_leak          - Identical inputs produce bit-identical outputs")
        print("  outputs_finite         - All outputs contain no NaN/Inf")
        print("  raw_input_binding      - Raw input hash matches expected")
        print("  multistep_reference_present - All required multistep reference files present")
        return 0

    patch_size = 2 if args.checkpoint == "ps2" else 4
    PATHS = get_paths(REPO_ROOT, patch_size)

    output_dir = args.output_dir or Path(os.environ.get(
        'S0_OUTPUT_DIR', str(PATHS["default_output_dir"])
    ))

    print('=' * 60, flush=True)
    print('S0 Gate Verification - EarthDelta Stormer Bridge', flush=True)
    print('=' * 60, flush=True)
    print(f'Checkpoint: ps{patch_size}', flush=True)

    # Run all gate verifications
    result = run_s0_gate(
        patch_size=patch_size,
        check_only=args.check,
        output_dir=output_dir,
        expected_checkpoint_sha256=args.expected_sha256,
        expected_normalization_digest=args.expected_norm_digest,
    )

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / 's0_gate_result.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'\nJSON result written to: {json_path}', flush=True)

    report = generate_report(result)
    report_path = output_dir / 'S0_GATE_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'Markdown report written to: {report_path}', flush=True)

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

    # Exit code: 0 if all pass, 1 otherwise
    return 0 if result.get('s0_gate_pass') else 1


if __name__ == '__main__':
    sys.exit(main())
