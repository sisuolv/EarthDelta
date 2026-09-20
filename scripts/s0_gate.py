#!/usr/bin/env python3
"""S0 Gate verification script for EarthDelta Stormer bridge.

Runs on ACP GPU cluster to verify:
1. Strict checkpoint loading (both ps2 and ps4)
2. Zero-edit equivalence (controlled_rollout with zero coeffs == forward_validation)
3. Normalization parity (digest matches expected)
4. No state leak (verified via existing test patterns)
5. RMSE sanity check for 6h and 24h rollouts on real ERA5 data

Writes results to JSON and generates S0_GATE_REPORT.md.

Usage (in ACP container):
    python /mnt/afs/260010168/EarthDelta/scripts/s0_gate.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

print("S0 Gate script starting...", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)

# Check and install missing dependencies
def ensure_dependencies():
    """Ensure required dependencies are installed."""
    # timm is needed for model architecture
    required = ['timm']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"Found {pkg}", flush=True)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing packages: {missing}", flush=True)
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            '--quiet', '--no-cache-dir', *missing
        ])
        print("Dependencies installed.", flush=True)

ensure_dependencies()

import numpy as np
import torch
import torch.nn as nn

print(f"NumPy version: {np.__version__}", flush=True)
print(f"PyTorch version: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

# Add EarthDelta to path
sys.path.insert(0, '/mnt/afs/260010168/EarthDelta')
print("EarthDelta path added.", flush=True)

try:
    from earthdelta.bridge import (
        Stormer,
        NormalizationContract,
        WeatherStepBridge,
        controlled_rollout,
        load_stormer_checkpoint,
        DEFAULT_VARIABLES,
    )
    print("earthdelta.bridge imported OK", flush=True)
except Exception as e:
    print(f"ERROR importing earthdelta.bridge: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    from earthdelta.contracts import EditPlan, reference_plan
    print("earthdelta.contracts imported OK", flush=True)
except Exception as e:
    print(f"ERROR importing earthdelta.contracts: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    from earthdelta.lowrank import ExpertLoRA
    print("earthdelta.lowrank imported OK", flush=True)
except Exception as e:
    print(f"ERROR importing earthdelta.lowrank: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("All imports successful.", flush=True)

# =============================================================================
# Map climate_learn namespace to our own implementation
# =============================================================================
# The Stormer checkpoint was saved with references to climate_learn.models.hub.stormer.Stormer.
# We map that class to our own Stormer implementation (which uses SDPA instead of xformers).
# This allows torch.load to unpickle the checkpoint correctly.

import types

def create_mock_package(name):
    """Create a mock package module."""
    mod = types.ModuleType(name)
    mod.__path__ = [f'/mock/{name.replace(".", "/")}']
    mod.__package__ = name
    return mod

# Create mock climate_learn hierarchy with our Stormer class
climate_learn = create_mock_package('climate_learn')
climate_learn.models = create_mock_package('climate_learn.models')
climate_learn.models.hub = create_mock_package('climate_learn.models.hub')
climate_learn.models.hub.stormer = create_mock_package('climate_learn.models.hub.stormer')
climate_learn.models.iterative_module = create_mock_package('climate_learn.models.iterative_module')
climate_learn.utils = create_mock_package('climate_learn.utils')
climate_learn.utils.data_utils = create_mock_package('climate_learn.utils.data_utils')
climate_learn.utils.lr_scheduler = create_mock_package('climate_learn.utils.lr_scheduler')
climate_learn.utils.metrics = create_mock_package('climate_learn.utils.metrics')

# Map the Stormer class to our implementation
climate_learn.models.hub.stormer.Stormer = Stormer

# Import our MemEffAttention for completeness
from earthdelta.bridge.stormer_arch import MemEffAttention, Block, FinalLayer
climate_learn.models.hub.stormer.MemEffAttention = MemEffAttention
climate_learn.models.hub.stormer.Block = Block
climate_learn.models.hub.stormer.FinalLayer = FinalLayer

# Import data_utils constants
from earthdelta.bridge.stormer_bridge import CONSTANTS, DEFAULT_VARIABLES as DV
climate_learn.utils.data_utils.CONSTANTS = CONSTANTS

# Create a mock lr_scheduler
class MockLinearWarmupCosineAnnealingLR:
    pass
climate_learn.utils.lr_scheduler.LinearWarmupCosineAnnealingLR = MockLinearWarmupCosineAnnealingLR

# Register all modules
sys.modules['climate_learn'] = climate_learn
sys.modules['climate_learn.models'] = climate_learn.models
sys.modules['climate_learn.models.hub'] = climate_learn.models.hub
sys.modules['climate_learn.models.hub.stormer'] = climate_learn.models.hub.stormer
sys.modules['climate_learn.models.iterative_module'] = climate_learn.models.iterative_module
sys.modules['climate_learn.utils'] = climate_learn.utils
sys.modules['climate_learn.utils.data_utils'] = climate_learn.utils.data_utils
sys.modules['climate_learn.utils.lr_scheduler'] = climate_learn.utils.lr_scheduler
sys.modules['climate_learn.utils.metrics'] = climate_learn.utils.metrics

print("climate_learn namespace mapped to earthdelta implementation.", flush=True)


# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = Path('/mnt/afs/260010168/EarthDelta/scripts/s0_gate_inputs')
CHECKPOINT_PS2 = Path('/mnt/afs/260010168/EarthDelta/checkpoints/stormer_1.40625_patch_size_2.ckpt')
CHECKPOINT_PS4 = Path('/mnt/afs/260010168/EarthDelta/checkpoints/stormer_1.40625_patch_size_4.ckpt')
NORM_DIR = Path('/mnt/afs/260010168/EarthDelta/reference/stormer/normalization_constants')
REPORT_DIR = Path('/mnt/afs/260010168/EarthDelta/plans/plans_v1_0919/v6_draft')

# Expected file sizes for checkpoints (verified)
EXPECTED_PS2_SIZE = 5625590679
EXPECTED_PS4_SIZE = 5570407547

# Z500 channel index in canonical 69-variable order
Z500_IDX = 11  # geopotential_500

# Stormer paper approximate RMSE values for reference (from paper Table 1/2)
# These are rough order-of-magnitude references, not exact targets
STORMER_PAPER_RMSE_Z500_6H = 30  # m^2/s^2 (rough estimate)
STORMER_PAPER_RMSE_Z500_24H = 80  # m^2/s^2 (rough estimate)


# =============================================================================
# Data loading utilities (no zarr/xarray - uses pre-extracted .npy)
# =============================================================================

def load_npy_inputs() -> Dict[str, np.ndarray]:
    """Load pre-extracted .npy inputs."""
    return {
        'data': np.load(INPUT_DIR / 'jan2020_full.npy'),  # (124, 69, 128, 256)
        'lat': np.load(INPUT_DIR / 'lat.npy'),
        'lon': np.load(INPUT_DIR / 'lon.npy'),
        'inp_mean': np.load(INPUT_DIR / 'inp_mean.npy'),
        'inp_std': np.load(INPUT_DIR / 'inp_std.npy'),
        'diff_mean_6': np.load(INPUT_DIR / 'diff_mean_6.npy'),
        'diff_std_6': np.load(INPUT_DIR / 'diff_std_6.npy'),
        'diff_mean_24': np.load(INPUT_DIR / 'diff_mean_24.npy'),
        'diff_std_24': np.load(INPUT_DIR / 'diff_std_24.npy'),
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
# Gate verification functions
# =============================================================================

def verify_checkpoint_load(
    ckpt_path: Path,
    patch_size: int,
    expected_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Load checkpoint with strict=True and verify structure.

    Returns dict with load status, missing/unexpected keys, etc.
    """
    result = {
        'checkpoint': str(ckpt_path),
        'patch_size': patch_size,
        'exists': ckpt_path.exists(),
        'size_bytes': 0,
        'size_matches': False,
        'load_success': False,
        'missing_keys': [],
        'unexpected_keys': [],
        'error': None,
    }

    if not ckpt_path.exists():
        result['error'] = 'Checkpoint file not found'
        return result

    result['size_bytes'] = ckpt_path.stat().st_size
    result['size_matches'] = result['size_bytes'] == expected_size

    try:
        model, version = load_stormer_checkpoint(
            str(ckpt_path),
            patch_size=patch_size,
        )
        model = model.to(device)
        result['load_success'] = True
        result['artifact_version'] = asdict(version)

        # Verify model is frozen
        frozen = all(not p.requires_grad for p in model.parameters())
        result['model_frozen'] = frozen

        # Quick forward pass sanity
        x = torch.randn(1, 69, 128, 256, device=device, dtype=torch.float32)
        interval = torch.tensor([0.6], device=device, dtype=torch.float32)
        with torch.no_grad():
            out = model(x, DEFAULT_VARIABLES, interval)
        result['forward_shape'] = list(out.shape)
        result['forward_finite'] = bool(torch.isfinite(out).all())

        # Clean up
        del model, out
        torch.cuda.empty_cache()

    except RuntimeError as e:
        result['error'] = str(e)
        # Try to extract missing/unexpected keys from error message
        if 'Missing key(s)' in str(e):
            result['error_type'] = 'missing_keys'
        elif 'Unexpected key(s)' in str(e):
            result['error_type'] = 'unexpected_keys'
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'

    return result


def verify_zero_edit_equivalence(
    bridge: WeatherStepBridge,
    x_raw: np.ndarray,
    interval: int,
    steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Verify controlled_rollout with zero coefficients equals forward_validation.

    This is the core S0 gate: zero-edit must equal official reference path.
    """
    result = {
        'interval': interval,
        'steps': steps,
        'equivalent': False,
        'max_abs_diff': float('inf'),
        'mean_abs_diff': float('inf'),
        'rtol_1e6': False,
        'error': None,
    }

    try:
        # Normalize input
        x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)  # (1, 69, 128, 256)
        x_norm = bridge.normalization.normalize(x_raw_t)

        # Reference plan (all zeros)
        plan = reference_plan(num_experts=8)

        # Run both paths
        with torch.no_grad():
            # Forward validation (official reference equivalent)
            ref_out = bridge.forward_validation(
                x_norm, bridge.variables, interval=interval, steps=steps
            )

            # Controlled rollout with zero coefficients
            ctrl_out = controlled_rollout(
                bridge, x_norm, bridge.variables, interval=interval, steps=steps,
                plan=plan, expert_loras={},
            )

        # Compare
        diff = (ref_out - ctrl_out).abs()
        result['max_abs_diff'] = float(diff.max().item())
        result['mean_abs_diff'] = float(diff.mean().item())
        result['rtol_1e6'] = bool(torch.allclose(ref_out, ctrl_out, atol=1e-6, rtol=1e-6))
        result['equivalent'] = result['rtol_1e6']

    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
        result['traceback'] = traceback.format_exc()

    return result


def verify_no_state_leak(
    bridge: WeatherStepBridge,
    x_raw: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """Verify no state leak - identical inputs produce bit-identical outputs."""
    result = {
        'bit_identical': False,
        'error': None,
    }

    try:
        x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
        x_norm = bridge.normalization.normalize(x_raw_t)
        plan = reference_plan(num_experts=8)

        # Scope variable that should NOT leak
        future_truth_a = torch.randn_like(x_norm)
        _ = future_truth_a

        with torch.no_grad():
            out_a = controlled_rollout(
                bridge, x_norm, bridge.variables, interval=6, steps=2,
                plan=plan, expert_loras={},
            ).clone()

        # Change scope variable
        future_truth_b = torch.randn_like(x_norm) * 100
        _ = future_truth_b

        with torch.no_grad():
            out_b = controlled_rollout(
                bridge, x_norm, bridge.variables, interval=6, steps=2,
                plan=plan, expert_loras={},
            )

        result['bit_identical'] = bool(torch.equal(out_a, out_b))

    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'

    return result


def compute_rmse_z500(
    bridge: WeatherStepBridge,
    data: np.ndarray,
    lat: np.ndarray,
    interval: int,
    steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Compute latitude-weighted RMSE for Z500.

    Args:
        bridge: WeatherStepBridge with loaded model
        data: (T, 69, 128, 256) ERA5 data
        lat: (128,) latitude array
        interval: Forecast interval in hours
        steps: Number of autoregressive steps
        device: CUDA device

    Returns:
        Dict with RMSE values and metadata
    """
    result = {
        'interval': interval,
        'steps': steps,
        'lead_time_hours': interval * steps,
        'rmse_z500': float('nan'),
        'rmse_z500_weighted': float('nan'),
        'n_samples': 0,
        'error': None,
    }

    try:
        # Compute latitude weights (cosine weighting)
        lat_rad = np.deg2rad(lat)
        lat_weights = np.cos(lat_rad)
        lat_weights = lat_weights / lat_weights.mean()  # Normalize
        lat_weights_t = torch.from_numpy(lat_weights).float().to(device).view(1, 1, -1, 1)

        # Use multiple starting points for robust estimate
        n_samples = min(10, data.shape[0] - steps)
        rmse_vals = []
        rmse_weighted_vals = []

        for i in range(n_samples):
            x_raw = data[i]  # (69, 128, 256)
            y_true = data[i + steps]  # Target

            x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
            y_true_t = torch.from_numpy(y_true).float().unsqueeze(0).to(device)

            # Normalize input
            x_norm = bridge.normalization.normalize(x_raw_t)

            # Run rollout
            with torch.no_grad():
                y_pred_norm = bridge.forward_validation(
                    x_norm, bridge.variables, interval=interval, steps=steps
                )
                y_pred = bridge.normalization.denormalize(y_pred_norm)

            # Extract Z500 channel
            z500_pred = y_pred[:, Z500_IDX, :, :]  # (1, 128, 256)
            z500_true = y_true_t[:, Z500_IDX, :, :]

            # Compute RMSE
            sq_err = (z500_pred - z500_true) ** 2
            rmse = torch.sqrt(sq_err.mean()).item()
            rmse_weighted = torch.sqrt((sq_err * lat_weights_t.squeeze(1)).mean()).item()

            rmse_vals.append(rmse)
            rmse_weighted_vals.append(rmse_weighted)

        result['rmse_z500'] = float(np.mean(rmse_vals))
        result['rmse_z500_std'] = float(np.std(rmse_vals))
        result['rmse_z500_weighted'] = float(np.mean(rmse_weighted_vals))
        result['rmse_z500_weighted_std'] = float(np.std(rmse_weighted_vals))
        result['n_samples'] = n_samples

    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
        result['traceback'] = traceback.format_exc()

    return result


def verify_normalization_parity(
    norm_from_npy: NormalizationContract,
) -> Dict[str, Any]:
    """Verify normalization contract matches reference."""
    result = {
        'digest': norm_from_npy.digest,
        'n_variables': len(norm_from_npy.variables),
        'intervals_present': list(norm_from_npy.diff_std.keys()),
        'inp_mean_finite': bool(torch.isfinite(norm_from_npy.inp_mean).all()),
        'inp_std_positive': bool((norm_from_npy.inp_std > 0).all()),
        'parity_ok': True,
    }

    # Also load directly from npz to verify match
    try:
        norm_from_npz = NormalizationContract.from_npz_dir(
            str(NORM_DIR), variables=DEFAULT_VARIABLES, intervals=(6, 24)
        )
        result['digest_from_npz'] = norm_from_npz.digest
        result['digests_match'] = norm_from_npy.digest == norm_from_npz.digest
        result['parity_ok'] = result['digests_match']
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
        result['parity_ok'] = False

    return result


# =============================================================================
# Main execution
# =============================================================================

def run_s0_gate() -> Dict[str, Any]:
    """Run all S0 gate verifications."""
    result = {
        'status': 'failed',
        'run_id': f's0-gate-{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dt%H%M%Sz")}',
        'hostname': socket.gethostname(),
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'executor_model': 'claude-opus-4-5-20251101',  # Model that wrote/ran this
        'torch_version': None,
        'cuda_version': None,
        'checkpoints': {},
        'zero_edit_equivalence': {},
        'normalization_parity': {},
        'no_state_leak': {},
        'rmse_sanity': {},
        's0_gate_pass': False,
        'gate_criteria': {
            'zero_edit_equals_official': False,
            'normalization_parity': False,
            'no_state_leak': False,
        },
    }

    try:
        # Environment info
        result['torch_version'] = str(torch.__version__)
        result['cuda_version'] = torch.version.cuda
        result['cuda_available'] = torch.cuda.is_available()
        result['cuda_device_count'] = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if not torch.cuda.is_available():
            result['error'] = 'CUDA not available'
            return result

        device = torch.device('cuda:0')
        print(f"Using device: {device}", flush=True)
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

        # Load inputs
        print("Loading pre-extracted inputs...", flush=True)
        inputs = load_npy_inputs()
        print(f"Data shape: {inputs['data'].shape}", flush=True)

        # Create normalization contract
        norm = create_normalization_from_npy(inputs)

        # Verify normalization parity
        print("Verifying normalization parity...", flush=True)
        result['normalization_parity'] = verify_normalization_parity(norm)
        result['gate_criteria']['normalization_parity'] = result['normalization_parity']['parity_ok']

        # Test with ps2 checkpoint (primary for S0)
        print("Loading ps2 checkpoint with strict=True...", flush=True)
        ckpt_result = verify_checkpoint_load(CHECKPOINT_PS2, 2, EXPECTED_PS2_SIZE, device)
        result['checkpoints']['ps2'] = ckpt_result

        if not ckpt_result['load_success']:
            result['error'] = f'ps2 checkpoint load failed: {ckpt_result.get("error")}'
            return result

        print("Loading ps4 checkpoint with strict=True...", flush=True)
        ckpt_result_ps4 = verify_checkpoint_load(CHECKPOINT_PS4, 4, EXPECTED_PS4_SIZE, device)
        result['checkpoints']['ps4'] = ckpt_result_ps4

        if not ckpt_result_ps4['load_success']:
            result['error'] = f'ps4 checkpoint load failed: {ckpt_result_ps4.get("error")}'
            return result

        # Load ps2 model for remaining tests
        print("Creating WeatherStepBridge with ps2 checkpoint...", flush=True)
        model, version = load_stormer_checkpoint(str(CHECKPOINT_PS2), patch_size=2)
        model = model.to(device)
        bridge = WeatherStepBridge(model, norm, version)

        # Zero-edit equivalence tests
        print("Testing zero-edit equivalence (6h, 1-step)...", flush=True)
        x_raw_0 = inputs['data'][0]  # First timestep
        result['zero_edit_equivalence']['6h_1step'] = verify_zero_edit_equivalence(
            bridge, x_raw_0, interval=6, steps=1, device=device
        )

        print("Testing zero-edit equivalence (6h, 4-step = 24h)...", flush=True)
        result['zero_edit_equivalence']['6h_4step'] = verify_zero_edit_equivalence(
            bridge, x_raw_0, interval=6, steps=4, device=device
        )

        # Check if all zero-edit tests pass
        ze_pass = all(
            v.get('equivalent', False)
            for v in result['zero_edit_equivalence'].values()
        )
        result['gate_criteria']['zero_edit_equals_official'] = ze_pass

        # No state leak test
        print("Testing no state leak...", flush=True)
        result['no_state_leak'] = verify_no_state_leak(bridge, x_raw_0, device)
        result['gate_criteria']['no_state_leak'] = result['no_state_leak'].get('bit_identical', False)

        # RMSE sanity check
        print("Computing RMSE sanity check (6h)...", flush=True)
        result['rmse_sanity']['6h'] = compute_rmse_z500(
            bridge, inputs['data'], inputs['lat'], interval=6, steps=1, device=device
        )
        result['rmse_sanity']['6h']['paper_reference'] = STORMER_PAPER_RMSE_Z500_6H

        print("Computing RMSE sanity check (24h)...", flush=True)
        result['rmse_sanity']['24h'] = compute_rmse_z500(
            bridge, inputs['data'], inputs['lat'], interval=6, steps=4, device=device
        )
        result['rmse_sanity']['24h']['paper_reference'] = STORMER_PAPER_RMSE_Z500_24H

        # Overall pass/fail
        result['s0_gate_pass'] = all(result['gate_criteria'].values())
        result['status'] = 'ok' if result['s0_gate_pass'] else 'gate_failed'

        # Clean up
        del model, bridge
        torch.cuda.empty_cache()

    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
        result['traceback'] = traceback.format_exc()

    result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return result


def generate_report(result: Dict[str, Any]) -> str:
    """Generate markdown report from S0 gate results."""
    lines = [
        '# S0 Gate Report',
        '',
        f'**Run ID:** `{result.get("run_id", "unknown")}`',
        f'**Executed:** {result.get("started_utc", "unknown")}',
        f'**Hostname:** {result.get("hostname", "unknown")}',
        f'**Executor Model:** {result.get("executor_model", "unknown")}',
        f'**Status:** {"PASS" if result.get("s0_gate_pass") else "FAIL"}',
        '',
        '## Environment',
        '',
        f'- PyTorch: {result.get("torch_version", "unknown")}',
        f'- CUDA: {result.get("cuda_version", "unknown")}',
        f'- GPU count: {result.get("cuda_device_count", 0)}',
        '',
        '## Gate Criteria Results',
        '',
        '| Criterion | Status |',
        '|-----------|--------|',
    ]

    for criterion, passed in result.get('gate_criteria', {}).items():
        status = 'PASS' if passed else 'FAIL'
        lines.append(f'| {criterion} | {status} |')

    lines.extend([
        '',
        '## Checkpoint Loading',
        '',
    ])

    for name, ckpt in result.get('checkpoints', {}).items():
        status = 'SUCCESS' if ckpt.get('load_success') else 'FAILED'
        lines.append(f'### {name.upper()} Checkpoint')
        lines.append(f'- Path: `{ckpt.get("checkpoint")}`')
        lines.append(f'- Size: {ckpt.get("size_bytes", 0):,} bytes')
        lines.append(f'- Size matches expected: {ckpt.get("size_matches")}')
        lines.append(f'- Load status: **{status}**')
        if ckpt.get('error'):
            lines.append(f'- Error: {ckpt.get("error")}')
        lines.append('')

    lines.extend([
        '## Zero-Edit Equivalence',
        '',
    ])

    for test_name, ze_result in result.get('zero_edit_equivalence', {}).items():
        status = 'PASS' if ze_result.get('equivalent') else 'FAIL'
        lines.append(f'### {test_name}')
        lines.append(f'- Interval: {ze_result.get("interval")}h, Steps: {ze_result.get("steps")}')
        lines.append(f'- Max absolute diff: {ze_result.get("max_abs_diff", "N/A"):.2e}')
        lines.append(f'- Mean absolute diff: {ze_result.get("mean_abs_diff", "N/A"):.2e}')
        lines.append(f'- Within 1e-6 tolerance: {ze_result.get("rtol_1e6")}')
        lines.append(f'- Status: **{status}**')
        lines.append('')

    lines.extend([
        '## Normalization Parity',
        '',
    ])

    norm = result.get('normalization_parity', {})
    lines.append(f'- Digest: `{norm.get("digest", "unknown")}`')
    lines.append(f'- Variables: {norm.get("n_variables", 0)}')
    lines.append(f'- Intervals: {norm.get("intervals_present", [])}')
    lines.append(f'- Parity OK: **{norm.get("parity_ok", False)}**')
    lines.append('')

    lines.extend([
        '## No State Leak',
        '',
    ])

    leak = result.get('no_state_leak', {})
    status = 'PASS' if leak.get('bit_identical') else 'FAIL'
    lines.append(f'- Bit-identical outputs: {leak.get("bit_identical")}')
    lines.append(f'- Status: **{status}**')
    lines.append('')

    lines.extend([
        '## RMSE Sanity Check (Z500)',
        '',
        '| Lead Time | RMSE (m^2/s^2) | Weighted RMSE | Paper Reference | Order of Magnitude |',
        '|-----------|----------------|---------------|-----------------|-------------------|',
    ])

    for lt, rmse in result.get('rmse_sanity', {}).items():
        rmse_val = rmse.get('rmse_z500', float('nan'))
        rmse_w = rmse.get('rmse_z500_weighted', float('nan'))
        ref = rmse.get('paper_reference', 'N/A')
        # Check if within 2x of reference
        if isinstance(rmse_val, (int, float)) and isinstance(ref, (int, float)) and not np.isnan(rmse_val):
            ratio = rmse_val / ref if ref > 0 else float('inf')
            magnitude_ok = 'OK' if 0.2 < ratio < 5 else 'CHECK'
        else:
            magnitude_ok = 'N/A'
        lines.append(f'| {lt} | {rmse_val:.1f} | {rmse_w:.1f} | ~{ref} | {magnitude_ok} |')

    lines.extend([
        '',
        '## Notes',
        '',
        '- RMSE values are computed on January 2020 ERA5 pilot data (124 timesteps)',
        '- Paper reference values are approximate; exact match not expected',
        '- Order of magnitude check verifies we are within 0.2x-5x of paper values',
        '- Zero-edit equivalence verifies controlled_rollout with zero coefficients',
        '  exactly matches forward_validation (the official inference path)',
        '',
    ])

    if result.get('error'):
        lines.extend([
            '## Errors',
            '',
            f'```',
            result.get('error', ''),
            '```',
            '',
        ])

    lines.append(f'*Report generated at {result.get("finished_utc", "unknown")}*')

    return '\n'.join(lines)


def main():
    print('=' * 60, flush=True)
    print('S0 Gate Verification - EarthDelta Stormer Bridge', flush=True)
    print('=' * 60, flush=True)

    # Run all gate verifications
    result = run_s0_gate()

    # Determine output directory
    output_dir = Path(os.environ.get('S0_OUTPUT_DIR', '/mnt/afs/260010168/EarthDelta/plans/plans_v1_0919/v6_draft'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON result
    json_path = output_dir / 's0_gate_result.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'\nJSON result written to: {json_path}', flush=True)

    # Generate and write markdown report
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

    # Also print full JSON for logs
    print('\n' + '=' * 60, flush=True)
    print('FULL RESULT JSON', flush=True)
    print('=' * 60, flush=True)
    print(json.dumps(result, indent=2, default=str), flush=True)

    return 0 if result.get('s0_gate_pass') else 1


if __name__ == '__main__':
    sys.exit(main())
