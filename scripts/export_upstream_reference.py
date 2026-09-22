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
    1 - Error during execution (including expected-identity mismatch)
    2 - BLOCKED: xformers not available (expected on CPU containers)
    3 - BLOCKED: CUDA not available

Note on module structure: the xformers/CUDA guards and the official Stormer
imports are deliberately performed inside main()/load_official_module() rather
than at import time. That keeps the identity/digest helpers in this module
importable (and therefore testable) on a CPU box, while the CLI still refuses
to run and still returns exit code 2/3 without xformers or CUDA. The script
still NEVER falls back to SDPA.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import socket
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Source root (importable code snapshot) vs asset root (checkpoint/NPZ/NPY).
# These are separate concepts: SOURCE_ROOT is where this code lives, the asset
# root comes from the frozen --config and may be anywhere on disk.
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SOURCE_ROOT  # backwards-compatible alias

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# Import ONLY identity/hash helpers from earthdelta - NOT NormalizationContract
# and NOT any of the audited numeric transforms.
from earthdelta.bridge.stormer_bridge import (  # noqa: E402
    _compute_file_sha256,
    _ensure_climate_learn_pickle_compat,
)
from earthdelta.contracts import (  # noqa: E402
    NORM_IDENTITY_SCHEMA_VERSION,
    GateIdentityConfig,
    OFFICIAL_STORMER_COMMIT,
    compare_normalization_identity,
    compute_coordinate_digest,
    compute_normalization_asset_sha256,
    compute_normalization_identity_digest,
    compute_variables_digest,
    format_normalization_identity,
)


# =============================================================================
# Environment checks
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
        import torch as _torch
        return _torch.cuda.is_available()
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


def enforce_environment() -> None:
    """Refuse to run outside a GPU + xformers environment."""
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
    """Get the SOURCE root (importable code snapshot) from script location."""
    return SOURCE_ROOT


def get_default_paths(asset_root: Path) -> Dict[str, Path]:
    """Get default ASSET paths relative to an asset root.

    The asset root is where checkpoints / NPZ / NPY live. It defaults to the
    source root only as a convenience; `--config` may point it elsewhere.
    """
    return {
        "input_dir": asset_root / "scripts" / "s0_gate_inputs",
        "checkpoint_ps2": asset_root / "checkpoints" / "stormer_1.40625_patch_size_2.ckpt",
        "checkpoint_ps4": asset_root / "checkpoints" / "stormer_1.40625_patch_size_4.ckpt",
        "norm_dir": asset_root / "reference" / "stormer" / "normalization_constants",
        "output_base": asset_root / "artifacts",
    }


def record_source_identity(module_names: Sequence[str]) -> Dict[str, Any]:
    """Record which module files were actually imported, plus their content hash.

    The source snapshot is identified by what Python really loaded, not by an
    assumption that source and assets share a directory.
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
    return {"source_root": str(SOURCE_ROOT), "modules": modules}


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
    # Imported lazily so this module stays importable on CPU-only boxes.
    from torchvision.transforms import transforms

    inp_mean, inp_std, diff_mean, diff_std = load_official_constants(
        norm_dir, variables, intervals
    )

    inp_transform = transforms.Normalize(inp_mean, inp_std)

    # Load diff normalization - USING ZEROS for mean (official semantic)
    diff_transforms = {}
    for interval in sorted(diff_std):
        diff_transforms[interval] = transforms.Normalize(
            diff_mean[interval], diff_std[interval]
        )

    return inp_transform, diff_transforms


def load_official_constants(
    norm_dir: Path,
    variables: Sequence[str],
    intervals: Sequence[int] = (6, 12, 24),
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Load the EFFECTIVE normalization constants using official semantics.

    Independent of earthdelta.NormalizationContract: this reads the NPZ files
    directly and applies the official inference.py rule that the diff mean is
    zeros. The returned arrays are both what the transforms are built from and
    what the identity digest is computed over, so the digest can never drift
    away from the numbers actually used.
    """
    norm_dir = Path(norm_dir)
    mean_dict = dict(np.load(norm_dir / "normalize_mean.npz"))
    std_dict = dict(np.load(norm_dir / "normalize_std.npz"))
    inp_mean = np.concatenate([mean_dict[v] for v in variables], axis=0)
    inp_std = np.concatenate([std_dict[v] for v in variables], axis=0)

    diff_mean: Dict[int, np.ndarray] = {}
    diff_std: Dict[int, np.ndarray] = {}
    for interval in sorted(int(i) for i in intervals):
        diff_std_path = norm_dir / f"normalize_diff_std_{interval}.npz"
        if not diff_std_path.exists():
            continue
        ds = dict(np.load(diff_std_path))
        std = np.concatenate([ds[v] for v in variables], axis=0)
        diff_std[interval] = std
        # OFFICIAL SEMANTIC: zero mean, not the NPZ diff_mean values.
        diff_mean[interval] = np.zeros_like(std)

    return inp_mean, inp_std, diff_mean, diff_std


def compute_normalization_digest(
    norm_dir: Path,
    variables: list,
    intervals: tuple = (6, 24),
    policy: str = "official_zero_diff_mean",
) -> str:
    """Bare identity digest of the EFFECTIVE normalization constants.

    The constants are loaded independently here (the exporter never calls the
    audited NormalizationContract), but the *serialization* used to derive the
    digest is the single shared one in earthdelta.contracts. That is what makes
    this value comparable with the bridge's NormalizationContract.digest.

    This is NOT a hash of the NPZ files on disk - see
    `compute_normalization_asset_sha256` for that, and keep the two apart.
    """
    inp_mean, inp_std, diff_mean, diff_std = load_official_constants(
        norm_dir, variables, intervals
    )
    return compute_normalization_identity_digest(
        policy=policy,
        variables=list(variables),
        inp_mean=inp_mean,
        inp_std=inp_std,
        diff_mean=diff_mean,
        diff_std=diff_std,
    )


def compute_normalization_identity(
    norm_dir: Path,
    variables: list,
    intervals: tuple = (6, 24),
    policy: str = "official_zero_diff_mean",
) -> str:
    """Schema-qualified identity string for the effective constants."""
    return format_normalization_identity(
        compute_normalization_digest(norm_dir, variables, intervals, policy),
        NORM_IDENTITY_SCHEMA_VERSION,
    )


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
    hidden_size: int = 1024,
    depth: int = 24,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
):
    """Load the official GlobalForecastIterativeModule with transforms set.

    This uses the ACTUAL official module, not a reimplementation. The official
    imports happen here (not at module import time) so that the identity
    helpers above stay importable without xformers.
    """
    if device is None:
        device = torch.device("cuda:0")

    reference_stormer = SOURCE_ROOT / "reference" / "stormer"
    if str(reference_stormer) not in sys.path:
        sys.path.insert(0, str(reference_stormer))

    # Import official Stormer modules (uses xformers)
    from stormer.models.hub.stormer import Stormer as OfficialStormer
    from stormer.models.iterative_module import GlobalForecastIterativeModule

    # Build official Stormer network
    net = OfficialStormer(
        in_img_size=list(in_img_size),
        variables=variables,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )

    # Wrap in GlobalForecastIterativeModule
    module = GlobalForecastIterativeModule(net)

    # Resolve only inert training-metadata classes; the official network and
    # Lightning module above remain the actual upstream implementations.
    _ensure_climate_learn_pickle_compat()
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


def build_reference_manifest(
    config: GateIdentityConfig,
    checkpoint_path: Path,
    norm_dir: Path,
    variables: list,
    checkpoint_sha256: str,
    coordinate_digest: Optional[str],
) -> Dict[str, Any]:
    """Create manifest documenting the identity AND the execution environment.

    Every field the gate enforces is written here explicitly so the gate never
    has to infer identity from a directory name.
    """
    try:
        import xformers
        xformers_version = xformers.__version__
    except ImportError:
        xformers_version = None

    identity = compute_normalization_identity(
        norm_dir, variables, config.normalization_intervals, config.normalization_policy
    )
    schema, bare_digest = identity.split(":", 1)

    manifest = {
        "manifest_schema": NORM_IDENTITY_SCHEMA_VERSION,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "xformers_version": xformers_version,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gate_config": {
            "digest": config.config_digest,
            "identity_tag": config.identity_tag,
            "frozen": config.to_manifest_dict(),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "patch_size": config.patch_size,
        },
        "model_config": {
            "in_img_size": list(config.grid_shape),
            "hidden_size": config.hidden_size,
            "depth": config.depth,
            "num_heads": config.num_heads,
            "mlp_ratio": config.mlp_ratio,
        },
        "normalization": {
            "dir": str(norm_dir),
            "policy": config.normalization_policy,
            # Identity of the EFFECTIVE constants (shared serialization).
            "identity_digest": identity,
            "identity_schema": schema,
            "digest": bare_digest,
            # Raw NPZ file bytes - a DIFFERENT question, kept under its own key.
            "npz_content_sha256": compute_normalization_asset_sha256(norm_dir),
            "intervals": list(config.normalization_intervals),
            "note": "Uses zero diff_mean matching official inference.py semantics",
        },
        "official_source": {
            "pinned_commit": config.official_commit,
            "note": "This reference was generated using the official GlobalForecastIterativeModule.forward_validation, NOT the earthdelta bridge implementation.",
        },
        "source_identity": record_source_identity([
            "earthdelta.contracts",
            "earthdelta.bridge.stormer_bridge",
            "__main__",
        ]),
        "variables": list(variables),
        "variables_count": len(variables),
        "variables_digest": compute_variables_digest(list(variables)),
        "grid_shape": list(config.grid_shape),
        "coordinate_digest": coordinate_digest,
        "registered_rollouts": [list(r) for r in config.registered_rollouts],
    }
    return manifest


# Backwards-compatible alias used by older call sites.
def create_environment_manifest(
    checkpoint_path: Path,
    patch_size: int,
    norm_dir: Path,
    variables: list,
) -> Dict[str, Any]:
    """Deprecated: kept so older callers keep working. Prefer build_reference_manifest."""
    config = GateIdentityConfig(
        checkpoint_path=str(checkpoint_path),
        expected_checkpoint_sha256=_compute_file_sha256(str(checkpoint_path)),
        patch_size=patch_size,
        normalization_policy="official_zero_diff_mean",
        normalization_dir=str(norm_dir),
        variables=tuple(variables),
        grid_shape=(128, 256),
    )
    return build_reference_manifest(
        config, checkpoint_path, norm_dir, variables,
        config.expected_checkpoint_sha256, None,
    )


def enforce_enrolled_identity(
    config: GateIdentityConfig,
    checkpoint_sha256: str,
    normalization_identity: str,
    npz_content_sha256: str,
    variables: Sequence[str],
    coordinate_digest: Optional[str],
    raw_input_hash: str,
) -> None:
    """Fail the export when observations disagree with the enrolled identity.

    The config is FROZEN before the export runs. The exporter's job is to prove
    it produced the enrolled identity, never to back-fill the config with
    whatever it happened to observe.

    Raises:
        ValueError: On any mismatch, naming the specific field.
    """
    problems: List[str] = []

    if not config.expected_checkpoint_sha256:
        problems.append("config.expected_checkpoint_sha256 is empty (identity not enrolled)")
    elif checkpoint_sha256.lower() != config.expected_checkpoint_sha256.lower():
        problems.append(
            f"checkpoint sha256 mismatch: config expects {config.expected_checkpoint_sha256}, "
            f"asset is {checkpoint_sha256}"
        )

    if config.expected_normalization_identity is None:
        problems.append("config.expected_normalization_identity is unset (identity not enrolled)")
    else:
        try:
            if not compare_normalization_identity(
                config.expected_normalization_identity, normalization_identity
            ):
                problems.append(
                    f"normalization identity mismatch: config expects "
                    f"{config.expected_normalization_identity}, computed {normalization_identity}"
                )
        except ValueError as exc:  # NormalizationIdentityFormatError
            problems.append(str(exc))

    if (config.expected_normalization_npz_sha256 is not None
            and config.expected_normalization_npz_sha256 != npz_content_sha256):
        problems.append(
            f"normalization NPZ content hash mismatch: config expects "
            f"{config.expected_normalization_npz_sha256}, asset is {npz_content_sha256}"
        )

    if tuple(variables) != tuple(config.variables):
        problems.append("variable list/order does not match the enrolled config")

    if (config.expected_coordinate_digest is not None
            and coordinate_digest is not None
            and config.expected_coordinate_digest != coordinate_digest):
        problems.append(
            f"coordinate identity mismatch: config expects "
            f"{config.expected_coordinate_digest}, asset is {coordinate_digest}"
        )

    if (config.expected_raw_input_hash is not None
            and config.expected_raw_input_hash != raw_input_hash):
        problems.append(
            f"raw input hash mismatch: config expects {config.expected_raw_input_hash}, "
            f"asset is {raw_input_hash}"
        )

    if problems:
        raise ValueError("Enrolled identity not satisfied: " + "; ".join(problems))


def publish_directory_atomically(staging: Path, final: Path) -> None:
    """Publish a fully-written staging directory to its final name atomically.

    A crashed or half-written export leaves only the staging directory behind,
    which no consumer looks at, so it can never be mistaken for a complete
    reference.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)


def export_reference(
    checkpoint_path: Path,
    patch_size: int,
    input_dir: Path,
    output_dir: Path,
    norm_dir: Path,
    config: Optional[GateIdentityConfig] = None,
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

    staging_dir: Optional[Path] = None
    try:
        if config is None:
            raise ValueError(
                "--config is required: the exporter must run against a frozen, "
                "pre-enrolled identity so the gate can verify the reference "
                "against the same expected values."
            )

        variables = list(config.variables)
        device = torch.device("cuda:0")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

        # Build transforms INDEPENDENTLY using official semantics
        print("Building official transforms (zero diff_mean)...")
        inp_transform, diff_transforms = build_official_transforms(
            norm_dir, variables, tuple(config.normalization_intervals)
        )

        # Load pinned input
        input_path = input_dir / config.input_file
        print(f"Loading pinned input from {input_path}...")
        jan2020_data = np.load(input_path)  # (T, V, H, W)
        x_raw = jan2020_data[0]  # First timestep
        x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)

        # Compute raw input hash for identity binding
        raw_input_hash = hashlib.sha256(x_raw.tobytes()).hexdigest()[:16]
        print(f"Raw input hash: {raw_input_hash}")

        coordinate_digest = None
        lat_path, lon_path = input_dir / "lat.npy", input_dir / "lon.npy"
        if lat_path.exists() and lon_path.exists():
            coordinate_digest = compute_coordinate_digest(np.load(lat_path), np.load(lon_path))

        checkpoint_sha256 = _compute_file_sha256(str(checkpoint_path))
        normalization_identity = compute_normalization_identity(
            norm_dir, variables, config.normalization_intervals, config.normalization_policy
        )
        npz_content_sha256 = compute_normalization_asset_sha256(norm_dir)

        # Enrolled identity is checked BEFORE the expensive model load.
        enforce_enrolled_identity(
            config, checkpoint_sha256, normalization_identity, npz_content_sha256,
            variables, coordinate_digest, raw_input_hash,
        )

        # Normalize input using official transform
        x_norm = inp_transform(x_raw_t)

        # Load official module
        print(f"Loading official GlobalForecastIterativeModule (patch_size={patch_size}) with xformers...")
        module = load_official_module(
            checkpoint_path, patch_size, variables,
            inp_transform, diff_transforms, device=device,
            in_img_size=tuple(config.grid_shape),
            hidden_size=config.hidden_size, depth=config.depth,
            num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
        )
        print("Module loaded successfully")

        # Run inference for every REGISTERED (interval, steps) pair.
        outputs = {}
        for interval, steps in config.registered_rollouts:
            key = f"{interval}h_{steps}step"
            print(f"Running official forward_validation: {key}...")
            with torch.no_grad():
                # This is the ACTUAL official method, not a reimplementation
                out = module.forward_validation(x_norm, variables, interval, steps)
            outputs[key] = out.cpu()
            print(f"  Output shape: {out.shape}, finite: {torch.isfinite(out).all()}")

        # Write everything into a staging directory, publish atomically last.
        staging_dir = output_dir.parent / f".staging-{output_dir.name}-{os.getpid()}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

        for key, tensor in outputs.items():
            out_path = staging_dir / f"official_output_{key}.pt"
            torch.save(tensor, out_path)
            print(f"Staged: {out_path}")

        # Save the normalized input (for exact reproducibility)
        torch.save(x_norm.cpu(), staging_dir / "input_norm.pt")

        # Save raw input hash
        with open(staging_dir / "raw_input_hash.txt", "w") as f:
            f.write(raw_input_hash)

        # Save manifest
        manifest = build_reference_manifest(
            config, checkpoint_path, norm_dir, variables,
            checkpoint_sha256, coordinate_digest,
        )
        manifest["outputs"] = {k: list(v.shape) for k, v in outputs.items()}
        manifest["outputs_finite"] = {k: bool(torch.isfinite(v).all()) for k, v in outputs.items()}
        manifest["raw_input_hash"] = raw_input_hash
        manifest["input_path"] = str(input_path)

        manifest_path = staging_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        publish_directory_atomically(staging_dir, output_dir)
        staging_dir = None
        print(f"Published reference: {output_dir}")

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
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    return result


def resolve_asset_paths(
    config: GateIdentityConfig, asset_root: Optional[Path] = None
) -> Dict[str, Path]:
    """Resolve asset locations from the frozen config.

    Assets are located by the config, not by assuming they sit next to the
    source tree. `asset_root` only supplies fallbacks for unset config fields.
    """
    root = Path(config.asset_root) if config.asset_root else (asset_root or SOURCE_ROOT)
    defaults = get_default_paths(root)
    return {
        "asset_root": root,
        "checkpoint": Path(config.checkpoint_path),
        "norm_dir": Path(config.normalization_dir),
        "input_dir": Path(config.input_dir) if config.input_dir else defaults["input_dir"],
        "output_base": (
            Path(config.reference_base_dir) if config.reference_base_dir
            else defaults["output_base"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export upstream reference outputs for S0 gate verification"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help=(
            "Frozen GateIdentityConfig JSON. REQUIRED. The same file must be "
            "passed to scripts/s0_gate.py so both sides share one expected identity."
        )
    )
    parser.add_argument(
        "--checkpoint", choices=["ps2", "ps4"], default=None,
        help="Legacy selector; ignored when --config supplies the checkpoint path"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <reference base>/upstream_reference_<identity>)"
    )
    parser.add_argument(
        "--repo-root", type=Path, default=None,
        help="Asset root fallback (default: detected from script location)"
    )
    args = parser.parse_args()

    # Environment guards run here (not at import) so this module stays
    # importable on CPU while the CLI still reports BLOCKED with code 2/3.
    enforce_environment()

    if args.config is None:
        print(
            "ERROR: --config is required. Enroll the expected identity in a "
            "GateIdentityConfig JSON before exporting; the exporter must not "
            "invent the expected values it is supposed to be checked against.",
            file=sys.stderr,
        )
        return 1

    try:
        config = GateIdentityConfig.load_json(args.config)
    except Exception as exc:
        print(f"ERROR: could not load --config {args.config}: {exc}", file=sys.stderr)
        return 1

    paths = resolve_asset_paths(config, args.repo_root)
    checkpoint_path = paths["checkpoint"]
    patch_size = config.patch_size

    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or (
        paths["output_base"] / config.reference_output_dir_name
    )

    print("=" * 60)
    print("Export Upstream Reference - Official Stormer with xformers")
    print("=" * 60)
    print(f"Config: {args.config} (digest {config.config_digest})")
    print(f"Source root: {SOURCE_ROOT}")
    print(f"Asset root: {paths['asset_root']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Patch size: {patch_size}")
    print(f"Expected checkpoint SHA-256: {config.expected_checkpoint_sha256}")
    print(f"Expected normalization identity: {config.expected_normalization_identity}")
    print(f"Registered rollouts: {list(config.registered_rollouts)}")
    print(f"Output dir: {output_dir}")
    print(f"Note: Using official forward_validation with zero diff_mean")
    print()

    result = export_reference(
        checkpoint_path=checkpoint_path,
        patch_size=patch_size,
        input_dir=paths["input_dir"],
        output_dir=output_dir,
        norm_dir=paths["norm_dir"],
        config=config,
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
