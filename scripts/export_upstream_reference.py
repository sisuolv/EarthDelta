#!/usr/bin/env python3
"""Export upstream reference outputs for S0 gate parity verification.

This script MUST run inside a GPU job container with xformers installed.
It builds the real official Stormer model using the actual code in
reference/stormer/ (which uses xformers.ops.memory_efficient_attention,
NOT the SDPA version in earthdelta.bridge).

The script:
1. Verifies xformers is available (exits BLOCKED otherwise)
2. Loads the official Stormer model from reference/stormer/
3. Loads weights from the same checkpoint used by the bridge
4. Runs inference on the pinned input tensor (jan2020_full.npy)
5. Exports outputs + environment manifest to artifacts/upstream_reference/

This output is then used by s0_gate.py's upstream_parity check to verify
the bridge produces equivalent output.

Usage (in ACP GPU container):
    python scripts/export_upstream_reference.py [--checkpoint ps2|ps4] [--output-dir PATH]

Exit codes:
    0 - Success, reference output exported
    1 - Error during execution
    2 - BLOCKED: xformers not available (expected on CPU containers)
    3 - BLOCKED: CUDA not available
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
from pathlib import Path
from typing import Any, Dict

# =============================================================================
# Environment check - must happen before any torch/xformers imports
# =============================================================================

def check_xformers_available() -> bool:
    """Check if xformers is installed and importable."""
    try:
        import xformers.ops  # noqa: F401
        return True
    except ImportError:
        return False


def check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def exit_blocked(reason: str, code: int = 2) -> None:
    """Exit with BLOCKED status."""
    print(f"BLOCKED: {reason}", file=sys.stderr)
    result = {
        "status": "blocked",
        "reason": reason,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    print(json.dumps(result, indent=2))
    sys.exit(code)


# Pre-import environment checks
if not check_xformers_available():
    exit_blocked(
        "xformers not available. This script requires xformers to execute "
        "the official Stormer code path. Install xformers and re-run, or "
        "run this script in a GPU container with xformers pre-installed.",
        code=2
    )

if not check_cuda_available():
    exit_blocked(
        "CUDA not available. This script requires a GPU to run the official "
        "Stormer model with xformers memory-efficient attention.",
        code=3
    )


# =============================================================================
# Now safe to import torch and reference code
# =============================================================================

import numpy as np
import torch

# Add reference stormer to path (before earthdelta to avoid conflicts)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
REFERENCE_STORMER = REPO_ROOT / "reference" / "stormer"
sys.path.insert(0, str(REFERENCE_STORMER))

# Import official Stormer (uses xformers)
from stormer.models.hub.stormer import Stormer as OfficialStormer  # noqa: E402

# Import earthdelta for normalization (but NOT the bridge Stormer)
sys.path.insert(0, str(REPO_ROOT))
from earthdelta.bridge import (  # noqa: E402
    NormalizationContract,
    DEFAULT_VARIABLES,
    _compute_file_sha256,
)


# =============================================================================
# Configuration
# =============================================================================

def get_repo_root() -> Path:
    """Get repository root from script location."""
    return Path(__file__).resolve().parent.parent


def get_default_paths(repo_root: Path) -> Dict[str, Path]:
    """Get default paths relative to repo root."""
    return {
        "input_dir": repo_root / "scripts" / "s0_gate_inputs",
        "checkpoint_ps2": repo_root / "checkpoints" / "stormer_1.40625_patch_size_2.ckpt",
        "checkpoint_ps4": repo_root / "checkpoints" / "stormer_1.40625_patch_size_4.ckpt",
        "norm_dir": repo_root / "reference" / "stormer" / "normalization_constants",
        "output_dir": repo_root / "artifacts" / "upstream_reference",
    }


# =============================================================================
# Core functions
# =============================================================================

def load_official_stormer(
    checkpoint_path: Path,
    patch_size: int,
    variables: list,
    in_img_size: tuple = (128, 256),
    device: torch.device = None,
) -> OfficialStormer:
    """Load the official Stormer model using reference code.

    This loads the real xformers-based Stormer, not the SDPA bridge version.

    Args:
        checkpoint_path: Path to checkpoint file
        patch_size: Patch size (2 or 4)
        variables: List of variable names
        in_img_size: Input image size (H, W)
        device: Device to load model to

    Returns:
        Loaded and frozen OfficialStormer model
    """
    if device is None:
        device = torch.device("cuda:0")

    # Build official model
    model = OfficialStormer(
        in_img_size=list(in_img_size),
        variables=variables,
        patch_size=patch_size,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    )

    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Strip 'net.' prefix (same as bridge)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            new_state_dict[k[4:]] = v
        else:
            new_state_dict[k] = v

    # Load with strict=True
    model.load_state_dict(new_state_dict, strict=True)

    # Freeze and move to device
    model.requires_grad_(False)
    model.eval()
    model = model.to(device)

    return model


def run_official_inference(
    model: OfficialStormer,
    x_norm: torch.Tensor,
    variables: list,
    normalization: NormalizationContract,
    interval: int,
    steps: int,
) -> torch.Tensor:
    """Run official autoregressive inference.

    Matches the rollout logic in GlobalForecastIterativeModule.forward_validation.

    Args:
        model: Official Stormer model
        x_norm: Normalized input of shape (B, V, H, W)
        variables: Variable names
        normalization: Normalization contract
        interval: Forecast interval (hours)
        steps: Number of autoregressive steps

    Returns:
        Normalized output at final step
    """
    device = x_norm.device
    patch_size = model.patch_size

    # Scale interval (matching iterative_module.py)
    interval_tensor = torch.tensor([interval], device=device, dtype=x_norm.dtype) / 10.0
    interval_tensor = interval_tensor.repeat(x_norm.shape[0])

    x = x_norm
    for _ in range(steps):
        # Pad if needed
        h = x.shape[-2]
        if h % patch_size != 0:
            pad_size = patch_size - h % patch_size
            padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), 'constant', 0)
        else:
            padded_x = x
            pad_size = 0

        # Forward pass
        with torch.no_grad():
            output = model(padded_x, variables, interval_tensor)

        # Remove padding
        pred_diff = output[:, :, pad_size:] if pad_size > 0 else output

        # Zero out constant channels
        pred_diff = normalization.replace_constant(pred_diff, variables)

        # Denormalize diff, add to denormalized input, renormalize
        pred_diff = normalization.denormalize_diff(pred_diff, interval)
        pred = normalization.denormalize(x) + pred_diff
        x = normalization.normalize(pred)

    return x


def create_environment_manifest(checkpoint_path: Path, patch_size: int) -> Dict[str, Any]:
    """Create manifest documenting the execution environment."""
    import xformers

    manifest = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "xformers_version": xformers.__version__,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _compute_file_sha256(str(checkpoint_path)),
            "patch_size": patch_size,
        },
        "model_config": {
            "in_img_size": [128, 256],
            "hidden_size": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
        },
    }
    return manifest


def export_reference(
    checkpoint_path: Path,
    patch_size: int,
    input_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Export official reference outputs.

    Args:
        checkpoint_path: Path to checkpoint
        patch_size: Patch size (2 or 4)
        input_dir: Directory with pinned inputs
        output_dir: Directory to write outputs

    Returns:
        Result dict with status and metadata
    """
    result = {
        "status": "failed",
        "checkpoint": str(checkpoint_path),
        "patch_size": patch_size,
    }

    try:
        device = torch.device("cuda:0")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

        # Load pinned input
        print("Loading pinned input from s0_gate_inputs...")
        jan2020_data = np.load(input_dir / "jan2020_full.npy")  # (124, 69, 128, 256)
        x_raw = jan2020_data[0]  # First timestep
        x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)

        # Load normalization
        print("Loading normalization constants...")
        inputs = {
            'inp_mean': np.load(input_dir / 'inp_mean.npy'),
            'inp_std': np.load(input_dir / 'inp_std.npy'),
            'diff_mean_6': np.load(input_dir / 'diff_mean_6.npy'),
            'diff_std_6': np.load(input_dir / 'diff_std_6.npy'),
            'diff_mean_24': np.load(input_dir / 'diff_mean_24.npy'),
            'diff_std_24': np.load(input_dir / 'diff_std_24.npy'),
        }
        normalization = NormalizationContract(
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

        # Normalize input
        x_norm = normalization.normalize(x_raw_t)

        # Load official model
        print(f"Loading official Stormer (patch_size={patch_size}) with xformers...")
        model = load_official_stormer(
            checkpoint_path, patch_size, DEFAULT_VARIABLES, device=device
        )
        print("Model loaded successfully")

        # Run inference for multiple configurations
        outputs = {}
        for interval, steps in [(6, 1), (6, 4)]:
            key = f"{interval}h_{steps}step"
            print(f"Running inference: {key}...")
            with torch.no_grad():
                out = run_official_inference(
                    model, x_norm, DEFAULT_VARIABLES, normalization, interval, steps
                )
            outputs[key] = out.cpu()
            print(f"  Output shape: {out.shape}, finite: {torch.isfinite(out).all()}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save outputs
        for key, tensor in outputs.items():
            out_path = output_dir / f"official_output_{key}.pt"
            torch.save(tensor, out_path)
            print(f"Saved: {out_path}")

        # Save the normalized input (for exact reproducibility)
        torch.save(x_norm.cpu(), output_dir / "input_norm.pt")

        # Save manifest
        manifest = create_environment_manifest(checkpoint_path, patch_size)
        manifest["outputs"] = {k: list(v.shape) for k, v in outputs.items()}
        manifest["outputs_finite"] = {k: bool(torch.isfinite(v).all()) for k, v in outputs.items()}
        manifest["normalization_digest"] = normalization.digest

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest: {manifest_path}")

        result["status"] = "ok"
        result["output_dir"] = str(output_dir)
        result["manifest"] = manifest

        # Clean up
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Export upstream reference outputs for S0 gate verification"
    )
    parser.add_argument(
        "--checkpoint", choices=["ps2", "ps4"], default="ps2",
        help="Checkpoint to use (default: ps2)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: artifacts/upstream_reference)"
    )
    parser.add_argument(
        "--repo-root", type=Path, default=None,
        help="Repository root (default: detected from script location)"
    )
    args = parser.parse_args()

    repo_root = args.repo_root or get_repo_root()
    paths = get_default_paths(repo_root)

    checkpoint_path = paths["checkpoint_ps2"] if args.checkpoint == "ps2" else paths["checkpoint_ps4"]
    patch_size = 2 if args.checkpoint == "ps2" else 4
    output_dir = args.output_dir or paths["output_dir"]

    print("=" * 60)
    print("Export Upstream Reference - Official Stormer with xformers")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Patch size: {patch_size}")
    print(f"Output dir: {output_dir}")
    print()

    result = export_reference(
        checkpoint_path=checkpoint_path,
        patch_size=patch_size,
        input_dir=paths["input_dir"],
        output_dir=output_dir,
    )

    print()
    print("=" * 60)
    print(f"Status: {result['status']}")
    print("=" * 60)

    if result["status"] != "ok":
        print(json.dumps(result, indent=2, default=str))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
