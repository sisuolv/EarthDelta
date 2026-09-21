#!/usr/bin/env python3
"""S0 Gate verification script for EarthDelta Stormer bridge.

Runs S0-level gate checks to verify bridge correctness before downstream use.
All gate criteria are fail-closed: initialized False, remain False on exceptions.

Gate Criteria:
- ckpt_sha256_bound: Checkpoint SHA-256 hash computed and recorded
- strict_load_zero_diff: Checkpoint loads with zero missing/unexpected keys
- upstream_parity: Bridge output matches official xformers Stormer (<=1e-5)
- zero_edit_equals_official: Bridge internal consistency (<=1e-6)
- normalization_parity: Normalization digest matches reference
- no_state_leak: Identical inputs produce bit-identical outputs
- outputs_finite: All outputs contain no NaN/Inf

Usage:
    python scripts/s0_gate.py [--check CRITERION] [--output-dir PATH]

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


def get_paths(repo_root: Path) -> Dict[str, Path]:
    """Get all paths relative to repo root."""
    return {
        "input_dir": repo_root / "scripts" / "s0_gate_inputs",
        "checkpoint_ps2": repo_root / "checkpoints" / "stormer_1.40625_patch_size_2.ckpt",
        "checkpoint_ps4": repo_root / "checkpoints" / "stormer_1.40625_patch_size_4.ckpt",
        "norm_dir": repo_root / "reference" / "stormer" / "normalization_constants",
        "upstream_reference_dir": repo_root / "artifacts" / "upstream_reference",
        "default_output_dir": repo_root / "artifacts" / "s0_gate",
    }


REPO_ROOT = get_repo_root()
PATHS = get_paths(REPO_ROOT)

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
        _compute_file_sha256,
    )
    from earthdelta.contracts import EditPlan, reference_plan
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


def create_normalization_from_npy(inputs: Dict[str, np.ndarray]) -> NormalizationContract:
    """Create NormalizationContract from pre-extracted numpy arrays."""
    return NormalizationContract(
        inp_mean=torch.from_numpy(inputs['inp_mean']).float(),
        inp_std=torch.from_numpy(inputs['inp_std']).float(),
        diff_mean={
            6: torch.from_numpy(inputs['diff_mean_6']).float(),
            24: torch.from_numpy(inputs['diff_mean_24']).float(),
        },
        diff_std={
            6: torch.from_numpy(inputs['diff_std_6']).float(),
            24: torch.from_numpy(inputs['diff_std_24']).float(),
        },
        variables=DEFAULT_VARIABLES,
    )


# =============================================================================
# Gate criterion: ckpt_sha256_bound
# =============================================================================

def verify_ckpt_sha256(
    ckpt_path: Path,
    load_result: Optional[CheckpointLoadResult],
) -> Dict[str, Any]:
    """Verify checkpoint SHA-256 is computed and bound.

    This criterion passes if we can compute and record the SHA-256 hash
    of the checkpoint file.
    """
    result = {
        "criterion": "ckpt_sha256_bound",
        "passed": False,  # fail-closed
        "checkpoint": str(ckpt_path),
        "sha256": None,
    }

    try:
        if not ckpt_path.exists():
            result["error"] = f"Checkpoint not found: {ckpt_path}"
            return result

        if load_result is not None:
            result["sha256"] = load_result.checkpoint_sha256
        else:
            result["sha256"] = _compute_file_sha256(str(ckpt_path))

        result["passed"] = result["sha256"] is not None and len(result["sha256"]) == 64

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

def verify_upstream_parity(
    bridge: WeatherStepBridge,
    upstream_dir: Path,
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
        manifest_path = upstream_dir / "manifest.json"
        if not manifest_path.exists():
            result["error"] = (
                f"Upstream reference not found at {upstream_dir}. "
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
        }
        result["upstream_available"] = True

        # Load upstream outputs
        upstream_output_6h_1step = torch.load(upstream_dir / "official_output_6h_1step.pt")
        upstream_input_norm = torch.load(upstream_dir / "input_norm.pt")

        # Move to device
        upstream_output = upstream_output_6h_1step.to(device)
        input_norm = upstream_input_norm.to(device)

        # Run bridge on same input
        with torch.no_grad():
            bridge_output = bridge.forward_validation(
                input_norm, bridge.variables, interval=6, steps=1
            )

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
) -> Dict[str, Any]:
    """Verify normalization contract matches reference npz files."""
    result = {
        "criterion": "normalization_parity",
        "passed": False,  # fail-closed
        "digest": norm_from_npy.digest,
        "n_variables": len(norm_from_npy.variables),
        "intervals_present": list(norm_from_npy.diff_std.keys()),
    }

    try:
        if not norm_dir.exists():
            result["error"] = f"Normalization directory not found: {norm_dir}"
            return result

        norm_from_npz = NormalizationContract.from_npz_dir(
            str(norm_dir), variables=DEFAULT_VARIABLES, intervals=(6, 24)
        )
        result["digest_from_npz"] = norm_from_npz.digest
        result["digests_match"] = norm_from_npy.digest == norm_from_npz.digest
        result["passed"] = result["digests_match"]

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
                "rmse_z500_weighted": float(np.mean(rmse_vals)),
                "rmse_z500_std": float(np.std(rmse_vals)),
                "n_samples": n_samples,
            }

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# Main gate execution
# =============================================================================

def run_s0_gate(
    check_only: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all S0 gate verifications.

    All gate criteria are fail-closed: they start False and stay False
    if any exception occurs during evaluation.
    """
    result = {
        "status": "failed",
        "run_id": f's0-gate-{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dt%H%M%Sz")}',
        "hostname": socket.gethostname(),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
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

        # Load inputs
        print("Loading inputs...", flush=True)
        inputs = load_npy_inputs(PATHS["input_dir"])
        x_raw_0 = inputs['data'][0]

        # Create normalization
        norm = create_normalization_from_npy(inputs)

        # Load checkpoint (detailed)
        print("Loading checkpoint (detailed)...", flush=True)
        load_result = load_stormer_checkpoint_detailed(
            str(PATHS["checkpoint_ps2"]), patch_size=2
        )
        model = load_result.model.to(device)
        bridge = WeatherStepBridge(model, norm, load_result.version)

        # Run gate criteria
        print("Running gate criteria...", flush=True)

        # 1. ckpt_sha256_bound
        print("  - ckpt_sha256_bound", flush=True)
        sha256_result = verify_ckpt_sha256(PATHS["checkpoint_ps2"], load_result)
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
            bridge, PATHS["upstream_reference_dir"], x_raw_0, device
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
        norm_result = verify_normalization_parity(norm, PATHS["norm_dir"])
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
        f'**Status:** {"PASS" if result.get("s0_gate_pass") else "FAIL"}',
        '',
        '## Environment',
        '',
        f'- PyTorch: {result.get("torch_version", "unknown")}',
        f'- CUDA available: {result.get("cuda_available", False)}',
        f'- Device: {result.get("cuda_device", "unknown")}',
        '',
        '## Gate Criteria Results',
        '',
        '| Criterion | Status | Notes |',
        '|-----------|--------|-------|',
    ]

    for criterion, passed in result.get('gate_criteria', {}).items():
        status = 'PASS' if passed else 'FAIL'
        details = result.get('criteria_details', {}).get(criterion, {})
        if 'error' in details:
            notes = f"Error: {details['error'][:50]}..."
        elif criterion == 'upstream_parity' and not details.get('upstream_available'):
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
    if sha_details.get('sha256'):
        lines.extend([
            '### Checkpoint Identity',
            f'- SHA-256: `{sha_details["sha256"]}`',
            '',
        ])

    # Upstream parity
    up_details = result.get('criteria_details', {}).get('upstream_parity', {})
    lines.extend([
        '### Upstream Parity (vs Official xformers Stormer)',
        f'- Available: {up_details.get("upstream_available", False)}',
    ])
    if up_details.get('upstream_available'):
        lines.extend([
            f'- Max absolute diff: {up_details.get("max_abs_diff", "N/A"):.2e}',
            f'- Tolerance: {up_details.get("tolerance", "N/A"):.0e}',
        ])
    if up_details.get('error'):
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
    for key in ['6h_1step', '6h_4step']:
        if key in ze_details:
            d = ze_details[key]
            lines.append(f'- {key}: max_diff={d.get("max_abs_diff", "N/A"):.2e}, passed={d.get("passed")}')
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
        if key in rmse:
            val = rmse[key].get('rmse_z500_weighted', float('nan'))
            ref = HISTORICAL_REFERENCE.get(f'z500_rmse_{key}', {}).get('value', 'N/A')
            lines.append(f'| {key} | {val:.1f} | ~{ref} |')

    lines.extend([
        '',
        f'*Report generated at {result.get("finished_utc", "unknown")}*',
    ])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="S0 Gate Verification")
    parser.add_argument(
        "--check", type=str, default=None,
        help="Run only a specific criterion (not implemented yet)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--help-criteria", action="store_true",
        help="Show gate criteria descriptions"
    )
    args = parser.parse_args()

    if args.help_criteria:
        print("S0 Gate Criteria:")
        print("  ckpt_sha256_bound      - Checkpoint SHA-256 hash computed and recorded")
        print("  strict_load_zero_diff  - Checkpoint loads with zero missing/unexpected keys")
        print("  upstream_parity        - Bridge output matches official xformers Stormer (<=1e-5)")
        print("  zero_edit_equals_official - Bridge internal consistency (<=1e-6)")
        print("  normalization_parity   - Normalization digest matches reference")
        print("  no_state_leak          - Identical inputs produce bit-identical outputs")
        print("  outputs_finite         - All outputs contain no NaN/Inf")
        return 0

    output_dir = args.output_dir or Path(os.environ.get(
        'S0_OUTPUT_DIR', str(PATHS["default_output_dir"])
    ))

    print('=' * 60, flush=True)
    print('S0 Gate Verification - EarthDelta Stormer Bridge', flush=True)
    print('=' * 60, flush=True)

    # Run all gate verifications
    result = run_s0_gate(check_only=args.check, output_dir=output_dir)

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
