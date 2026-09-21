#!/usr/bin/env python3
"""Export upstream reference outputs for S0 gate parity verification.

This script MUST run inside a GPU job container with xformers installed.
It imports and uses the ACTUAL official GlobalForecastIterativeModule.forward_validation
from reference/stormer/ (which uses xformers.ops.memory_efficient_attention),
NOT the SDPA version in earthdelta.bridge.

CRITICAL: This script does NOT reuse the earthdelta NormalizationContract to build
the reference outputs. That contract is exactly what's under audit. Instead, it
constructs transforms independently from the official NPZ files using the official
inference.py semantics (zero diff_mean).

The script:
1. Verifies xformers is available (exits BLOCKED otherwise)
2. Loads the official Stormer model from reference/stormer/
3. Sets up transforms INDEPENDENTLY using zero diff_mean (matching official inference.py)
4. Runs the official forward_validation method
5. Exports outputs + environment manifest to a namespaced directory

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
from typing import Any, Dict, Optional

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
        "run this script in a GPU container with xformers pre-installed. "
        "IMPORTANT: This script will NOT fall back to SDPA - that would defeat "
        "the purpose of generating an independent reference.",
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
from torchvision.transforms import transforms

# Add reference stormer to path (before earthdelta to avoid conflicts)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
REFERENCE_STORMER = REPO_ROOT / "reference" / "stormer"
sys.path.insert(0, str(REFERENCE_STORMER))

# Import official Stormer modules (uses xformers)
from stormer.models.hub.stormer import Stormer as OfficialStormer  # noqa: E402
from stormer.models.iterative_module import GlobalForecastIterativeModule  # noqa: E402

# Import ONLY the file hash utility from earthdelta - NOT NormalizationContract
sys.path.insert(0, str(REPO_ROOT))
from earthdelta.bridge.stormer_bridge import _compute_file_sha256  # noqa: E402
from earthdelta.contracts import GateIdentityConfig, OFFICIAL_STORMER_COMMIT  # noqa: E402

# Official 69-variable list (from reference/stormer/configs/finetune_multi_step.yaml)
# We hardcode this here to avoid importing from earthdelta.bridge.DEFAULT_VARIABLES
# which would couple the reference path to the audited code.
OFFICIAL_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000",
]


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
        "output_base": repo_root / "artifacts",
    }


# =============================================================================
# Official Transform Construction (INDEPENDENT from earthdelta)
# =============================================================================

def build_official_transforms(
    norm_dir: Path,
    variables: list,
    intervals: tuple = (6, 12, 24),
) -> tuple:
    """Build transforms using OFFICIAL inference.py semantics.

    CRITICAL: This function constructs transforms independently, NOT using
    the earthdelta NormalizationContract. It matches the official inference.py
    which uses:
    - inp_transform: Normalize(mean, std)
    - diff_transform: Normalize(ZEROS, std)  <-- Zero diff_mean!

    See reference/stormer/inference.py lines 116-118:
        out_transforms[l] = transforms.Normalize(
            np.zeros_like(normalize_diff_std), normalize_diff_std
        )

    Returns:
        (inp_transform, diff_transforms_dict)
    """
    # Load input normalization
    normalize_mean = dict(np.load(norm_dir / "normalize_mean.npz"))
    normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
    normalize_std = dict(np.load(norm_dir / "normalize_std.npz"))
    normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)

    inp_transform = transforms.Normalize(normalize_mean, normalize_std)

    # Load diff normalization - USING ZEROS for mean (official semantic)
    diff_transforms = {}
    for interval in intervals:
        diff_std_path = norm_dir / f"normalize_diff_std_{interval}.npz"
        if diff_std_path.exists():
            normalize_diff_std = dict(np.load(diff_std_path))
            normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
            # OFFICIAL SEMANTIC: Zero mean, not the NPZ diff_mean values
            diff_transforms[interval] = transforms.Normalize(
                np.zeros_like(normalize_diff_std), normalize_diff_std
            )

    return inp_transform, diff_transforms


def compute_normalization_digest(
    norm_dir: Path,
    variables: list,
    intervals: tuple = (6, 24),
) -> str:
    """Compute a digest for the normalization constants.

    This is computed independently for the manifest, using zero diff_mean
    policy to match the official semantics.
    """
    parts = []
    parts.append(b"policy=official_zero_diff_mean")
    parts.append((",".join(variables)).encode())

    # Input normalization
    mean_dict = dict(np.load(norm_dir / "normalize_mean.npz"))
    std_dict = dict(np.load(norm_dir / "normalize_std.npz"))
    inp_mean = np.concatenate([mean_dict[v] for v in variables], axis=0)
    inp_std = np.concatenate([std_dict[v] for v in variables], axis=0)

    parts.append(f"inp_mean_shape={inp_mean.shape}".encode())
    parts.append(inp_mean.tobytes())
    parts.append(f"inp_std_shape={inp_std.shape}".encode())
    parts.append(inp_std.tobytes())

    # Diff normalization (zeros for mean)
    for interval in sorted(intervals):
        diff_std_path = norm_dir / f"normalize_diff_std_{interval}.npz"
        if diff_std_path.exists():
            ds = dict(np.load(diff_std_path))
            diff_std = np.concatenate([ds[v] for v in variables], axis=0)
            # Zero mean
            diff_mean = np.zeros_like(diff_std)
            parts.append(f"diff_mean_{interval}_shape={diff_mean.shape}".encode())
            parts.append(diff_mean.tobytes())
            parts.append(f"diff_std_{interval}_shape={diff_std.shape}".encode())
            parts.append(diff_std.tobytes())

    return hashlib.sha256(b"".join(parts)).hexdigest()[:16]


# =============================================================================
# Core functions
# =============================================================================

def load_official_module(
    checkpoint_path: Path,
    patch_size: int,
    variables: list,
    inp_transform,
    diff_transforms: dict,
    in_img_size: tuple = (128, 256),
    device: torch.device = None,
) -> GlobalForecastIterativeModule:
    """Load the official GlobalForecastIterativeModule with transforms set.

    This uses the ACTUAL official module, not a reimplementation.
    """
    if device is None:
        device = torch.device("cuda:0")

    # Build official Stormer network
    net = OfficialStormer(
        in_img_size=list(in_img_size),
        variables=variables,
        patch_size=patch_size,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    )

    # Wrap in GlobalForecastIterativeModule
    module = GlobalForecastIterativeModule(net)

    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    module.load_state_dict(state_dict)

    # Set transforms (this is how official inference.py does it)
    module.set_transforms(inp_transform, diff_transforms)

    # Freeze and move to device
    module.requires_grad_(False)
    module.eval()
    module = module.to(device)

    return module


def create_environment_manifest(
    checkpoint_path: Path,
    patch_size: int,
    norm_dir: Path,
    variables: list,
) -> Dict[str, Any]:
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
        "normalization": {
            "dir": str(norm_dir),
            "policy": "official_zero_diff_mean",
            "digest": compute_normalization_digest(norm_dir, variables),
            "note": "Uses zero diff_mean matching official inference.py semantics",
        },
        "official_source": {
            "pinned_commit": OFFICIAL_STORMER_COMMIT,
            "note": "This reference was generated using the official GlobalForecastIterativeModule.forward_validation, NOT the earthdelta bridge implementation.",
        },
        "variables_count": len(variables),
    }
    return manifest


def export_reference(
    checkpoint_path: Path,
    patch_size: int,
    input_dir: Path,
    output_dir: Path,
    norm_dir: Path,
) -> Dict[str, Any]:
    """Export official reference outputs.

    CRITICAL: This uses the ACTUAL official forward_validation method,
    NOT a reimplementation. The transforms are constructed independently
    using official semantics (zero diff_mean).
    """
    result = {
        "status": "failed",
        "checkpoint": str(checkpoint_path),
        "patch_size": patch_size,
    }

    try:
        device = torch.device("cuda:0")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

        # Build transforms INDEPENDENTLY using official semantics
        print("Building official transforms (zero diff_mean)...")
        inp_transform, diff_transforms = build_official_transforms(
            norm_dir, OFFICIAL_VARIABLES
        )

        # Load pinned input
        print("Loading pinned input from s0_gate_inputs...")
        jan2020_data = np.load(input_dir / "jan2020_full.npy")  # (124, 69, 128, 256)
        x_raw = jan2020_data[0]  # First timestep
        x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)

        # Compute raw input hash for identity binding
        raw_input_hash = hashlib.sha256(x_raw.tobytes()).hexdigest()[:16]
        print(f"Raw input hash: {raw_input_hash}")

        # Normalize input using official transform
        x_norm = inp_transform(x_raw_t)

        # Load official module
        print(f"Loading official GlobalForecastIterativeModule (patch_size={patch_size}) with xformers...")
        module = load_official_module(
            checkpoint_path, patch_size, OFFICIAL_VARIABLES,
            inp_transform, diff_transforms, device=device
        )
        print("Module loaded successfully")

        # Run inference using the ACTUAL official forward_validation method
        outputs = {}
        for interval, steps in [(6, 1), (6, 4)]:
            key = f"{interval}h_{steps}step"
            print(f"Running official forward_validation: {key}...")
            with torch.no_grad():
                # This is the ACTUAL official method, not a reimplementation
                out = module.forward_validation(
                    x_norm, OFFICIAL_VARIABLES, interval, steps
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

        # Save raw input hash
        with open(output_dir / "raw_input_hash.txt", "w") as f:
            f.write(raw_input_hash)

        # Save manifest
        manifest = create_environment_manifest(
            checkpoint_path, patch_size, norm_dir, OFFICIAL_VARIABLES
        )
        manifest["outputs"] = {k: list(v.shape) for k, v in outputs.items()}
        manifest["outputs_finite"] = {k: bool(torch.isfinite(v).all()) for k, v in outputs.items()}
        manifest["raw_input_hash"] = raw_input_hash
        manifest["input_path"] = str(input_dir / "jan2020_full.npy")

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest: {manifest_path}")

        result["status"] = "ok"
        result["output_dir"] = str(output_dir)
        result["manifest"] = manifest
        result["raw_input_hash"] = raw_input_hash

        # Clean up
        del module
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
        "--checkpoint", choices=["ps2", "ps4"], default="ps4",
        help="Checkpoint to use (default: ps4 - the mainline per research_spec_v6.yaml)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: artifacts/upstream_reference_<identity>)"
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

    # Compute checkpoint SHA for identity
    ckpt_sha256 = _compute_file_sha256(str(checkpoint_path)) if checkpoint_path.exists() else "unknown"

    # Build namespaced output directory
    identity_tag = f"ps{patch_size}_{ckpt_sha256[:8]}_zd"
    output_dir = args.output_dir or (paths["output_base"] / f"upstream_reference_{identity_tag}")

    print("=" * 60)
    print("Export Upstream Reference - Official Stormer with xformers")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Patch size: {patch_size}")
    print(f"Checkpoint SHA-256: {ckpt_sha256}")
    print(f"Output dir: {output_dir}")
    print(f"Note: Using official forward_validation with zero diff_mean")
    print()

    result = export_reference(
        checkpoint_path=checkpoint_path,
        patch_size=patch_size,
        input_dir=paths["input_dir"],
        output_dir=output_dir,
        norm_dir=paths["norm_dir"],
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
