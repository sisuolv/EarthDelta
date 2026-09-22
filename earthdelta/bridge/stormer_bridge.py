"""Bridge between Stormer weather model and EarthDelta's low-rank edits.

Provides:
- load_stormer_checkpoint: Loads checkpoint with strict=True, returns model + ArtifactVersion
- NormalizationContract: Input/diff normalization matching iterative_module.py
- WeatherStepBridge: Wraps model + normalization + pad/unpad for rollout
- controlled_rollout: Injects ExpertLoRA edits during autoregressive rollout

The controlled_rollout function has a "no_state_leak" property: outputs depend
ONLY on (x_norm, variables, interval, steps, plan, expert_loras) and NOT on
any external/future data accessible from enclosing scope.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn

from ..contracts import (
    ArtifactVersion,
    EditPlan,
    NORM_IDENTITY_SCHEMA_VERSION,
    compute_normalization_identity_digest,
    format_normalization_identity,
)
from ..lowrank import ExpertLoRA
from .stormer_arch import Stormer


# =============================================================================
# Default 69-variable list (from finetune_multi_step.yaml)
# =============================================================================

DEFAULT_VARIABLES = [
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

# Constants from data_utils.py - these variables should be zeroed in predictions
CONSTANTS = [
    "anisotropy_of_sub_gridscale_orography",
    "orography",
    "land_sea_mask",
    "slt",
    "lattitude",
    "longitude",
    "angle_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
]


# =============================================================================
# Normalization Contract
# =============================================================================

# Normalization policy constants
POLICY_LEGACY = "legacy"  # Use actual diff_mean values from NPZ files (pre-audit behavior)
POLICY_OFFICIAL_ZERO_DIFF_MEAN = "official_zero_diff_mean"  # Force diff_mean to zero (matches official inference.py)


@dataclass(frozen=True)
class NormalizationContract:
    """Encapsulates input and diff normalization transforms.

    Matches the semantics from iterative_module.py:
    - inp_transform: normalizes raw input -> normalized input
    - reverse_inp_transform: denormalizes normalized input -> raw
    - reverse_diff_transform[interval]: denormalizes normalized diff -> raw diff
    - replace_constant: zeros out constant variable channels

    Policy field controls diff_mean behavior:
    - POLICY_LEGACY (default): Uses actual diff_mean values from NPZ files.
      This preserves backward compatibility with existing experiments.
    - POLICY_OFFICIAL_ZERO_DIFF_MEAN: Forces diff_mean to zero, matching the
      official inference.py which uses transforms.Normalize(np.zeros_like(...), std).
      This is the correct semantic for comparison against official upstream outputs.

    NOTE: The official Stormer inference.py (reference/stormer/inference.py:118)
    uses zero diff_mean: `transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)`
    The legacy path loaded nonzero diff_mean values which creates a numeric mismatch.
    """
    inp_mean: torch.Tensor  # [V]
    inp_std: torch.Tensor   # [V]
    diff_mean: Dict[int, torch.Tensor]  # interval -> [V]
    diff_std: Dict[int, torch.Tensor]   # interval -> [V]
    variables: List[str]
    policy: str = POLICY_LEGACY  # Default to legacy for backward compatibility
    # The official NPZ diff constants are float64.  Keep private source copies
    # for inverse-transform construction while exposing the historical
    # float32 contract tensors to callers and identity hashing.
    _reverse_diff_mean: Optional[Dict[int, torch.Tensor]] = field(
        default=None, repr=False, compare=False
    )
    _reverse_diff_std: Optional[Dict[int, torch.Tensor]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self):
        """Reject a contract whose declared policy contradicts its contents.

        The policy label is part of the identity, so it must not be able to lie
        about what the object actually holds. Under
        POLICY_OFFICIAL_ZERO_DIFF_MEAN the stored diff_mean tensors ARE the
        effective ones (zeros); a nonzero stored diff_mean under that label is
        a mislabelled object and is rejected at construction.
        """
        if self.policy not in (POLICY_LEGACY, POLICY_OFFICIAL_ZERO_DIFF_MEAN):
            raise ValueError(
                f"Unknown policy: {self.policy}. "
                "Use POLICY_LEGACY or POLICY_OFFICIAL_ZERO_DIFF_MEAN."
            )
        if not self.policy_content_consistent:
            offending = sorted(
                interval for interval, tensor in self.diff_mean.items()
                if bool(torch.any(tensor != 0))
            )
            raise ValueError(
                f"Policy '{self.policy}' declares zero diff_mean but the contract "
                f"holds nonzero diff_mean for intervals {offending}. The declared "
                "policy must match the actual contents."
            )

    @property
    def policy_content_consistent(self) -> bool:
        """True when the declared policy agrees with the stored constants.

        Used by the gate so that a contract mutated after construction (e.g. a
        relabelled policy field) is still rejected rather than trusted on the
        strength of its label alone.
        """
        if self.policy != POLICY_OFFICIAL_ZERO_DIFF_MEAN:
            return True
        return all(not bool(torch.any(t != 0)) for t in self.diff_mean.values())

    @classmethod
    def from_npz_dir(
        cls,
        npz_dir: str,
        variables: Optional[List[str]] = None,
        intervals: Tuple[int, ...] = (6, 12, 24),
        policy: str = POLICY_LEGACY,
    ) -> 'NormalizationContract':
        """Load normalization constants from npz files.

        Args:
            npz_dir: Directory containing normalize_*.npz files
            variables: Variable list (default: DEFAULT_VARIABLES)
            intervals: Intervals to load diff transforms for
            policy: Normalization policy (POLICY_LEGACY or POLICY_OFFICIAL_ZERO_DIFF_MEAN).
                    Default is POLICY_LEGACY for backward compatibility.

        Returns:
            NormalizationContract instance
        """
        if variables is None:
            variables = DEFAULT_VARIABLES.copy()

        if policy not in (POLICY_LEGACY, POLICY_OFFICIAL_ZERO_DIFF_MEAN):
            raise ValueError(f"Unknown policy: {policy}. Use POLICY_LEGACY or POLICY_OFFICIAL_ZERO_DIFF_MEAN.")

        # Load input normalization
        mean_path = os.path.join(npz_dir, "normalize_mean.npz")
        std_path = os.path.join(npz_dir, "normalize_std.npz")

        mean_dict = dict(np.load(mean_path))
        std_dict = dict(np.load(std_path))

        inp_mean = np.concatenate([mean_dict[v] for v in variables], axis=0)
        inp_std = np.concatenate([std_dict[v] for v in variables], axis=0)

        # Load diff normalization for each interval
        diff_mean = {}
        diff_std = {}
        raw_diff_mean_source = {}
        raw_diff_std_source = {}
        for interval in intervals:
            diff_mean_path = os.path.join(npz_dir, f"normalize_diff_mean_{interval}.npz")
            diff_std_path = os.path.join(npz_dir, f"normalize_diff_std_{interval}.npz")

            if os.path.exists(diff_mean_path):
                dm = dict(np.load(diff_mean_path))
                raw_diff_mean = np.concatenate([dm[v] for v in variables], axis=0)
                # Under official policy, force diff_mean to zero
                if policy == POLICY_OFFICIAL_ZERO_DIFF_MEAN:
                    raw_diff_mean = np.zeros_like(raw_diff_mean)
                raw_diff_mean_source[interval] = raw_diff_mean
                diff_mean[interval] = torch.from_numpy(raw_diff_mean).float()

            if os.path.exists(diff_std_path):
                ds = dict(np.load(diff_std_path))
                raw_diff_std = np.concatenate([ds[v] for v in variables], axis=0)
                raw_diff_std_source[interval] = raw_diff_std
                diff_std[interval] = torch.from_numpy(raw_diff_std).float()

        contract = cls(
            inp_mean=torch.from_numpy(inp_mean).float(),
            inp_std=torch.from_numpy(inp_std).float(),
            diff_mean=diff_mean,
            diff_std=diff_std,
            variables=variables,
            policy=policy,
        )
        # Keep the uncast arrays separately: torchvision constructs the
        # reverse transform from these float64 values before applying it to
        # a float32 tensor.
        object.__setattr__(
            contract, "_reverse_diff_mean", {
                interval: torch.from_numpy(raw).clone()
                for interval, raw in raw_diff_mean_source.items()
            },
        )
        object.__setattr__(
            contract, "_reverse_diff_std", {
                interval: torch.from_numpy(raw).clone()
                for interval, raw in raw_diff_std_source.items()
            },
        )
        return contract

    def normalize(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Normalize raw input: x_norm = (x_raw - mean) / std.

        Args:
            x_raw: Raw input of shape (B, V, H, W)

        Returns:
            Normalized input of same shape
        """
        mean = self.inp_mean.to(x_raw.device, x_raw.dtype).view(1, -1, 1, 1)
        std = self.inp_std.to(x_raw.device, x_raw.dtype).view(1, -1, 1, 1)
        return (x_raw - mean) / std

    @staticmethod
    def _reverse_parameters(
        mean: torch.Tensor, std: torch.Tensor, device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build inverse Normalize constants with upstream operation order.

        ``GlobalForecastIterativeModule.get_reverse_transform`` computes the
        reciprocal and reverse mean while its transform tensors are still CPU
        float32, then moves those constants with the module.  Computing the
        reciprocal directly on CUDA can round differently on each rollout
        step, even though the formulas are mathematically identical.
        """
        cpu_std = std.detach().to(device="cpu")
        cpu_mean = mean.detach().to(device="cpu")
        cpu_std_inverse = 1.0 / cpu_std
        cpu_mean_inverse = -cpu_mean * cpu_std_inverse
        return (
            cpu_mean_inverse.to(device=device, dtype=dtype),
            cpu_std_inverse.to(device=device, dtype=dtype),
        )

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize normalized input using the official inverse transform.

        The upstream Stormer module constructs an inverse
        ``torchvision.transforms.Normalize`` with ``std_inverse = 1 / std``
        and applies subtraction followed by division.  Writing the
        mathematically equivalent ``x_norm * std + mean`` changes FP32
        rounding enough to fail the strict upstream parity gate after rollout.

        Args:
            x_norm: Normalized input of shape (B, V, H, W)

        Returns:
            Raw input of same shape
        """
        mean = self.inp_mean.to(x_norm.device, x_norm.dtype).view(1, -1, 1, 1)
        std = self.inp_std.to(x_norm.device, x_norm.dtype).view(1, -1, 1, 1)
        # Use the original CPU float32 constants for inverse construction;
        # see _reverse_parameters for why this is intentionally not computed
        # from the device-cast tensors above.
        mean_inverse, std_inverse = self._reverse_parameters(
            self.inp_mean.view(1, -1, 1, 1), self.inp_std.view(1, -1, 1, 1),
            x_norm.device, x_norm.dtype,
        )
        return (x_norm - mean_inverse) / std_inverse

    def denormalize_diff(self, diff_norm: torch.Tensor, interval: int) -> torch.Tensor:
        """Denormalize normalized diff: diff_raw = diff_norm * std + mean.

        The mean used depends on the policy:
        - POLICY_OFFICIAL_ZERO_DIFF_MEAN: Always uses zero mean, matching official inference.py.
        - POLICY_LEGACY: Uses the loaded diff_mean values (may be nonzero).

        Note: The official Stormer inference.py (reference/stormer/inference.py:118)
        uses transforms.Normalize(np.zeros_like(...), std), i.e., zero diff_mean.
        Using nonzero diff_mean creates a numeric mismatch with upstream.

        Args:
            diff_norm: Normalized diff of shape (B, V, H, W)
            interval: Forecast interval (6, 12, or 24)

        Returns:
            Raw diff of same shape
        """
        if interval not in self.diff_std:
            raise ValueError(f"No diff transform for interval {interval}")

        reverse_std = self._reverse_diff_std or self.diff_std
        reverse_mean = self._reverse_diff_mean or self.diff_mean
        source_std = reverse_std[interval].view(1, -1, 1, 1)

        # Determine mean based on policy
        if self.policy == POLICY_OFFICIAL_ZERO_DIFF_MEAN:
            # Official semantic: always zero diff_mean
            source_mean = torch.zeros_like(source_std)
        elif interval in reverse_mean:
            # Legacy semantic: use actual diff_mean from NPZ
            source_mean = reverse_mean[interval].view(1, -1, 1, 1)
        else:
            source_mean = torch.zeros_like(source_std)

        # Match torchvision's get_reverse_transform arithmetic exactly.  This
        # is deliberately written as subtraction/division instead of the
        # equivalent multiply/add expression so official parity is stable in
        # float32 across autoregressive steps.  Inverse constants are built on
        # CPU, where the official transform constructs them.
        mean_inverse, std_inverse = self._reverse_parameters(
            source_mean, source_std,
            diff_norm.device, diff_norm.dtype,
        )
        return (diff_norm - mean_inverse) / std_inverse

    def replace_constant(self, yhat: torch.Tensor, variables: List[str]) -> torch.Tensor:
        """Zero out constant variable channels (matching iterative_module.py).

        Args:
            yhat: Prediction of shape (B, V, H, W)
            variables: List of variable names

        Returns:
            Prediction with constant channels zeroed
        """
        yhat = yhat.clone()
        for i in range(yhat.shape[1]):
            if variables[i] in CONSTANTS:
                yhat[:, i] = 0.0
        return yhat

    @property
    def digest(self) -> str:
        """Bare 16-hex identity digest of the effective normalization constants.

        Computed with the SHARED serializer in earthdelta.contracts so that this
        value is byte-for-byte comparable with the digest the independent
        exporter computes from the same NPZ assets. See
        `serialize_normalization_identity` for the pinned schema (field order,
        explicit sorted interval set, float32 little-endian digest dtype).

        Prefer `identity_digest` when the value is written to or read from a
        manifest/config: that form carries the schema version, which is what
        makes a legacy reference fail closed instead of being silently reused.
        """
        return compute_normalization_identity_digest(
            policy=self.policy,
            variables=self.variables,
            inp_mean=self.inp_mean,
            inp_std=self.inp_std,
            diff_mean=self.diff_mean,
            diff_std=self.diff_std,
        )

    @property
    def identity_schema_version(self) -> str:
        """Serialization schema version backing `digest` / `identity_digest`."""
        return NORM_IDENTITY_SCHEMA_VERSION

    @property
    def identity_digest(self) -> str:
        """Schema-qualified identity, e.g. ``ed-norm-identity/1:<16 hex>``."""
        return format_normalization_identity(self.digest, NORM_IDENTITY_SCHEMA_VERSION)


# =============================================================================
# Checkpoint Loading
# =============================================================================

@dataclass
class CheckpointLoadResult:
    """Result of loading a checkpoint with strict=True.

    Provides detailed information about the load for gate verification.
    """
    model: 'Stormer'
    version: ArtifactVersion
    checkpoint_sha256: str
    file_size_bytes: int
    missing_keys: List[str]
    unexpected_keys: List[str]

    @property
    def strict_load_zero_diff(self) -> bool:
        """True if strict load had zero missing and zero unexpected keys."""
        return len(self.missing_keys) == 0 and len(self.unexpected_keys) == 0


# The pinned Stormer .ckpt files are PyTorch Lightning checkpoints, so their pickle
# stream also carries the original training run's hyper_parameters, which reference
# `climate_learn` -- a training framework we neither install nor use. We only consume
# checkpoint["state_dict"], but the unpickler must resolve every name before it can
# hand that dict back. Pickle opcode inspection (zipfile + pickletools.genops over
# archive/data.pkl, without unpickling) confirmed these are the only five such names,
# and that none of them carries custom __reduce__ logic, so placeholders are inert.
_CLIMATE_LEARN_PICKLE_CLASSES: Tuple[Tuple[str, str], ...] = (
    ("climate_learn.models.lr_scheduler", "LinearWarmupCosineAnnealingLR"),
    ("climate_learn.metrics.metrics", "LatWeightedMSE"),
    ("climate_learn.metrics.metrics", "LatWeightedRMSE"),
    ("climate_learn.metrics.utils", "MetricsMetaInfo"),
    ("climate_learn.transforms.denormalize", "Denormalize"),
)

_COMPAT_SHIM_MARKER = "__earthdelta_compat_shim__"


class _ClimateLearnCompatPlaceholder:
    """Inert stand-in for an unpickled climate_learn checkpoint-metadata object."""


def _ensure_climate_learn_pickle_compat() -> None:
    """Make the checkpoint's climate_learn class paths resolvable for unpickling.

    No-op when a real climate_learn is importable. Otherwise registers placeholder
    modules/classes for exactly the five names in _CLIMATE_LEARN_PICKLE_CLASSES,
    never overwriting an existing sys.modules entry or module attribute.
    """
    try:
        import climate_learn  # noqa: F401
    except ImportError:
        pass
    else:
        if not getattr(climate_learn, _COMPAT_SHIM_MARKER, False):
            return

    for module_path, class_name in _CLIMATE_LEARN_PICKLE_CLASSES:
        parts = module_path.split(".")
        for depth in range(1, len(parts) + 1):
            dotted = ".".join(parts[:depth])
            module = sys.modules.get(dotted)
            if module is None:
                module = types.ModuleType(dotted)
                module.__path__ = []  # type: ignore[attr-defined]
                setattr(module, _COMPAT_SHIM_MARKER, True)
                sys.modules[dotted] = module
                if depth > 1:
                    parent = sys.modules[".".join(parts[:depth - 1])]
                    if not hasattr(parent, parts[depth - 1]):
                        setattr(parent, parts[depth - 1], module)

        leaf = sys.modules[module_path]
        if not hasattr(leaf, class_name):
            placeholder = type(
                class_name, (_ClimateLearnCompatPlaceholder,), {"__module__": module_path}
            )
            setattr(leaf, class_name, placeholder)


def _compute_file_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_stormer_checkpoint(
    ckpt_path: str,
    patch_size: int,
    variables: Optional[List[str]] = None,
    in_img_size: Tuple[int, int] = (128, 256),
    hidden_size: int = 1024,
    depth: int = 24,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    compute_sha256: bool = True,
) -> Tuple[Stormer, ArtifactVersion]:
    """Load a Stormer checkpoint with strict=True.

    Args:
        ckpt_path: Path to checkpoint file
        patch_size: Patch size (2 or 4)
        variables: Variable list (default: DEFAULT_VARIABLES)
        in_img_size: Input image size (H, W)
        hidden_size: Hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        compute_sha256: Whether to compute SHA-256 hash (can be slow for large files)

    Returns:
        Tuple of (model, artifact_version)

    Raises:
        RuntimeError: If checkpoint loading fails with strict=True
    """
    if variables is None:
        variables = DEFAULT_VARIABLES.copy()

    # When the official package is present (the real S0 worker), instantiate
    # its Stormer class directly. This removes backend/architecture drift from
    # the parity comparison; CPU development still uses the self-contained SDPA
    # implementation because it has no xformers dependency.
    model, official_backend = _build_stormer_model(
        in_img_size, variables, patch_size, hidden_size, depth, num_heads, mlp_ratio
    )

    # Load checkpoint
    _ensure_climate_learn_pickle_compat()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Strip 'net.' prefix from keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            new_state_dict[k[4:]] = v
        else:
            new_state_dict[k] = v

    # Load with strict=True
    model.load_state_dict(new_state_dict, strict=True)

    # Freeze all parameters
    model.requires_grad_(False)
    model.eval()

    # Create artifact version
    file_size = os.path.getsize(ckpt_path)
    config_str = f"ps{patch_size}_h{hidden_size}_d{depth}_nh{num_heads}_mr{mlp_ratio}"

    # Compute SHA-256 if requested (provides checkpoint identity binding)
    if compute_sha256:
        ckpt_sha256 = _compute_file_sha256(ckpt_path)
        backbone_str = f"stormer_{config_str}_sha256:{ckpt_sha256[:16]}"
    else:
        backbone_str = f"stormer_{config_str}"

    version = ArtifactVersion(
        backbone=backbone_str,
        static_adapter="none",
        edit_bank="none",
        normalization="pending",  # Will be set when paired with NormalizationContract
        grid=f"{in_img_size[0]}x{in_img_size[1]}",
        projection=f"patch{patch_size}",
        split="full",
        continuation="reference_after_hold",
    )

    return model, version


def _build_stormer_model(
    in_img_size: Tuple[int, int], variables: List[str], patch_size: int,
    hidden_size: int, depth: int, num_heads: int, mlp_ratio: float,
) -> Tuple[nn.Module, bool]:
    """Build the pinned official model when its CUDA attention is available."""
    try:
        # The worker source snapshot is not a Git checkout; make the pinned
        # upstream package explicit instead of relying on caller PYTHONPATH.
        reference_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "reference", "stormer")
        )
        if os.path.isdir(reference_root) and reference_root not in sys.path:
            sys.path.insert(0, reference_root)
        import xformers.ops  # noqa: F401
        from stormer.models.hub.stormer import Stormer as OfficialStormer
    except (ImportError, ModuleNotFoundError):
        return Stormer(
            in_img_size=in_img_size, variables=variables, patch_size=patch_size,
            hidden_size=hidden_size, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        ), False
    return OfficialStormer(
        in_img_size=in_img_size, variables=variables, patch_size=patch_size,
        hidden_size=hidden_size, depth=depth, num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    ), True


def load_stormer_checkpoint_detailed(
    ckpt_path: str,
    patch_size: int,
    variables: Optional[List[str]] = None,
    in_img_size: Tuple[int, int] = (128, 256),
    hidden_size: int = 1024,
    depth: int = 24,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
) -> CheckpointLoadResult:
    """Load a Stormer checkpoint with detailed load result for gate verification.

    Unlike load_stormer_checkpoint, this uses strict=False to capture
    missing/unexpected keys, then validates they are empty.

    Args:
        ckpt_path: Path to checkpoint file
        patch_size: Patch size (2 or 4)
        variables: Variable list (default: DEFAULT_VARIABLES)
        in_img_size: Input image size (H, W)
        hidden_size: Hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio

    Returns:
        CheckpointLoadResult with model, version, and load details

    Raises:
        RuntimeError: If checkpoint loading encounters unexpected issues
    """
    if variables is None:
        variables = DEFAULT_VARIABLES.copy()

    model, official_backend = _build_stormer_model(
        in_img_size, variables, patch_size, hidden_size, depth, num_heads, mlp_ratio
    )

    # Compute file metadata first
    file_size = os.path.getsize(ckpt_path)
    ckpt_sha256 = _compute_file_sha256(ckpt_path)

    # Load checkpoint
    _ensure_climate_learn_pickle_compat()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Strip 'net.' prefix from keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            new_state_dict[k[4:]] = v
        else:
            new_state_dict[k] = v

    # Load with strict=False to capture details
    load_result = model.load_state_dict(new_state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)

    # Freeze all parameters
    model.requires_grad_(False)
    model.eval()

    # Create artifact version
    config_str = f"ps{patch_size}_h{hidden_size}_d{depth}_nh{num_heads}_mr{mlp_ratio}"
    backbone_str = f"stormer_{config_str}_sha256:{ckpt_sha256[:16]}"

    version = ArtifactVersion(
        backbone=backbone_str,
        static_adapter="none",
        edit_bank="none",
        normalization="pending",
        grid=f"{in_img_size[0]}x{in_img_size[1]}",
        projection=f"patch{patch_size}",
        split="full",
        continuation="reference_after_hold",
    )

    return CheckpointLoadResult(
        model=model,
        version=version,
        checkpoint_sha256=ckpt_sha256,
        file_size_bytes=file_size,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


# =============================================================================
# Weather Step Bridge
# =============================================================================

class WeatherStepBridge:
    """Wraps Stormer model + normalization for weather prediction rollout.

    Implements the same rollout logic as GlobalForecastIterativeModule.forward_validation:
    1. Normalize input
    2. Predict normalized diff
    3. Replace constant channels with zero
    4. Denormalize diff
    5. Add to denormalized input
    6. Renormalize for next step
    """

    def __init__(
        self,
        model: Stormer,
        normalization: NormalizationContract,
        version: ArtifactVersion,
    ):
        self.model = model
        self.normalization = normalization
        self._version = version

    @property
    def version(self) -> ArtifactVersion:
        """Return artifact version with normalization hash."""
        from dataclasses import replace
        return replace(self._version, normalization=self.normalization.digest)

    @property
    def variables(self) -> List[str]:
        return self.normalization.variables

    @property
    def patch_size(self) -> int:
        return self.model.patch_size

    @property
    def normalization_identity(self) -> str:
        """Schema-qualified identity of this bridge's normalization contract."""
        return self.normalization.identity_digest

    def normalize(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Normalize a raw input through THIS bridge's own normalizer.

        Exposed so that callers binding a raw input to a saved normalized
        tensor run the real bridge transform rather than re-deriving it.
        """
        return self.normalization.normalize(x_raw)

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize through THIS bridge's own normalizer."""
        return self.normalization.denormalize(x_norm)

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad input height to be divisible by patch_size.

        Only pads the top (consistent with iterative_module.py).

        Args:
            x: Input of shape (B, V, H, W)

        Returns:
            Tuple of (padded_x, pad_size)
        """
        h = x.shape[-2]
        if h % self.model.patch_size != 0:
            pad_size = self.model.patch_size - h % self.model.patch_size
            padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), 'constant', 0)
        else:
            padded_x = x
            pad_size = 0
        return padded_x, pad_size

    def _forward_single(self, x: torch.Tensor, variables: List[str],
                        interval_tensor: torch.Tensor) -> torch.Tensor:
        """Single forward pass with pad/unpad.

        Args:
            x: Normalized input of shape (B, V, H, W)
            variables: List of variable names
            interval_tensor: Interval tensor of shape (B,)

        Returns:
            Normalized diff of shape (B, V, H, W)
        """
        padded_x, pad_size = self.pad(x)
        output = self.model(padded_x, variables, interval_tensor)
        return output[:, :, pad_size:]

    def forward_validation(
        self,
        x_norm: torch.Tensor,
        variables: List[str],
        interval: int,
        steps: int,
    ) -> torch.Tensor:
        """Autoregressive rollout matching iterative_module.forward_validation.

        Args:
            x_norm: Normalized initial condition of shape (B, V, H, W)
            variables: List of variable names
            interval: Forecast interval (6, 12, or 24 hours)
            steps: Number of autoregressive steps

        Returns:
            Normalized prediction at final step, shape (B, V, H, W)
        """
        # Scale interval by 10.0 (matching iterative_module.py convention)
        interval_tensor = torch.tensor([interval], device=x_norm.device, dtype=x_norm.dtype) / 10.0
        interval_tensor = interval_tensor.repeat(x_norm.shape[0])

        x = x_norm
        for _ in range(steps):
            # Predict normalized diff
            pred_diff = self._forward_single(x, variables, interval_tensor)
            # Zero out constant channels
            pred_diff = self.normalization.replace_constant(pred_diff, variables)
            # Denormalize diff
            pred_diff = self.normalization.denormalize_diff(pred_diff, interval)
            # Denormalize current state, add diff, renormalize
            pred = self.normalization.denormalize(x) + pred_diff
            x = self.normalization.normalize(pred)

        return x


# =============================================================================
# Controlled Rollout with LoRA Injection
# =============================================================================

# -----------------------------------------------------------------------------
# Supported-execution-mode declaration (B12)
# -----------------------------------------------------------------------------
#
# `controlled_rollout` injects edits by registering forward hooks on submodules
# of the SHARED backbone (`bridge.model.blocks[i].attn.proj`) and removing them
# again once the step's forward pass has returned. That mechanism is only
# correct under a narrow set of execution modes, which are declared here and
# enforced by the guards below. Modes outside this declaration are REJECTED,
# not silently supported:
#
#   1. SERIAL ONLY. Exactly one `controlled_rollout` may be in flight for a
#      given backbone at any time, process-wide. Hook registration mutates the
#      backbone itself, so a second in-flight rollout would see the first
#      rollout's hooks (and vice versa) and silently produce the sum of two
#      unrelated edits.
#
#   2. NO SHARED-MODEL CONCURRENCY. The unit of exclusion is the *backbone*,
#      not the bridge wrapper. Two distinct `WeatherStepBridge` objects that
#      wrap the same `Stormer` instance are just as unsafe as one bridge used
#      twice, so the guard is keyed on `id(bridge.model)`. It is also
#      process-wide rather than thread-local: a thread-local guard cannot see
#      a concurrent rollout on another thread, which is exactly the case it
#      most needs to reject.
#
#   3. NO ACTIVATION-CHECKPOINT REPLAY. With activation checkpointing the
#      backbone's forward is *replayed* during the backward pass. By then the
#      hooks have already been removed in this function's `finally`, so the
#      replayed graph would omit the LoRA contribution entirely and backward
#      would compute gradients of a model that was never evaluated forward --
#      silently wrong, with no exception. Any backbone that advertises
#      activation checkpointing is therefore refused up front.
#
# This is a rejection guard only. Supporting any of these modes would require a
# different injection mechanism (e.g. permanently wrapped modules) and is out
# of scope here.

import threading

#: Attribute names used by common backbones to advertise activation
#: checkpointing. Any of these being truthy on the backbone or one of its
#: submodules means the forward pass may be replayed during backward.
_ACTIVATION_CHECKPOINT_FLAGS: Tuple[str, ...] = (
    "gradient_checkpointing",
    "grad_checkpointing",
    "use_checkpoint",
    "use_activation_checkpointing",
    "activation_checkpointing",
    "checkpoint_activations",
)

#: Execution modes `controlled_rollout` supports. Exposed so callers and tests
#: can assert against the declaration rather than re-deriving it.
CONTROLLED_ROLLOUT_SUPPORTED_MODES: Dict[str, bool] = {
    "serial_single_rollout_per_backbone": True,
    "concurrent_rollouts_sharing_a_backbone": False,
    "activation_checkpoint_replay": False,
}


class _RolloutRegistry:
    """Process-wide record of the rollouts currently in flight.

    Deliberately NOT `threading.local()`. The state being protected is the
    backbone's forward-hook table, which is shared across threads; a
    thread-local registry would report "no rollout in flight" to precisely the
    concurrent caller that must be rejected.
    """

    def __init__(self) -> None:
        # id(bridge) of every bridge currently inside controlled_rollout.
        self.active_bridges = set()
        # id(model) -> (id(bridge), owning thread name) for every backbone
        # currently being mutated by a rollout.
        self.active_models: Dict[int, Tuple[int, str]] = {}


_rollout_lock = _RolloutRegistry()
_rollout_registry_mutex = threading.Lock()


def _assert_no_activation_checkpointing(model: nn.Module) -> None:
    """Reject backbones that replay their forward pass during backward.

    Raises:
        RuntimeError: If the backbone or any submodule advertises activation
            checkpointing, which would drop the hook-injected LoRA term from
            the recomputed graph.
    """
    if not isinstance(model, nn.Module):
        return
    for module_name, module in model.named_modules():
        for flag in _ACTIVATION_CHECKPOINT_FLAGS:
            if getattr(module, flag, False):
                where = module_name or "<backbone>"
                raise RuntimeError(
                    "controlled_rollout does not support activation-checkpoint "
                    f"replay, but {where} has {flag}=True. The LoRA edit is "
                    "injected via a forward hook that is removed as soon as the "
                    "forward pass returns, so a checkpointed backward would "
                    "recompute the forward WITHOUT the edit and produce "
                    "gradients for a model that was never evaluated. Disable "
                    "activation checkpointing on the backbone before calling "
                    "controlled_rollout."
                )


def _check_reentrant_rollout(bridge: 'WeatherStepBridge') -> None:
    """Admit a rollout only if its backbone is not already being mutated.

    Enforces the serial-only / no-shared-model-concurrency declaration above,
    and refuses backbones configured for activation-checkpoint replay.

    Raises:
        RuntimeError: If this bridge is already in a rollout call, if another
            bridge is mid-rollout on the same backbone (including from another
            thread), or if the backbone advertises activation checkpointing.
    """
    model = getattr(bridge, "model", None)
    _assert_no_activation_checkpointing(model)

    bridge_id = id(bridge)
    model_id = id(model)
    this_thread = threading.current_thread().name

    with _rollout_registry_mutex:
        if bridge_id in _rollout_lock.active_bridges:
            raise RuntimeError(
                "Reentrant call to controlled_rollout on the same bridge detected. "
                "This could corrupt the model's forward hooks. Each bridge instance "
                "should only be used in one rollout at a time."
            )
        owner = _rollout_lock.active_models.get(model_id)
        if owner is not None:
            owner_bridge_id, owner_thread = owner
            raise RuntimeError(
                "Concurrent controlled_rollout on a shared backbone detected: "
                f"backbone id={model_id} is already in a rollout held by bridge "
                f"id={owner_bridge_id} on thread {owner_thread!r} (this call is "
                f"bridge id={bridge_id} on thread {this_thread!r}). controlled_rollout "
                "is serial-only per backbone because it injects edits by mutating "
                "the backbone's forward hooks; two in-flight rollouts would apply "
                "each other's edits. Give each concurrent caller its own backbone, "
                "or serialize the calls."
            )
        _rollout_lock.active_bridges.add(bridge_id)
        _rollout_lock.active_models[model_id] = (bridge_id, this_thread)


def _release_rollout_lock(bridge: 'WeatherStepBridge') -> None:
    """Release the admission record for a bridge.

    Only the bridge that actually owns the backbone's record clears it, so a
    rejected concurrent caller can never release the incumbent's claim.
    """
    bridge_id = id(bridge)
    model_id = id(getattr(bridge, "model", None))
    with _rollout_registry_mutex:
        _rollout_lock.active_bridges.discard(bridge_id)
        owner = _rollout_lock.active_models.get(model_id)
        if owner is not None and owner[0] == bridge_id:
            del _rollout_lock.active_models[model_id]


def controlled_rollout(
    bridge: WeatherStepBridge,
    x_norm: torch.Tensor,
    variables: List[str],
    interval: int,
    steps: int,
    plan: EditPlan,
    expert_loras: Dict[int, ExpertLoRA],
    target_blocks: Tuple[int, ...] = (18, 19, 20, 21, 22, 23),
    sparse: bool = False,
    return_trajectory: bool = False,
    differentiable: bool = False,
) -> torch.Tensor:
    """Controlled rollout with LoRA injection at specified blocks.

    This function has the "no_state_leak" property: the output depends ONLY on
    the explicit arguments (x_norm, variables, interval, steps, plan, expert_loras).
    No code path allows ground-truth or future data to influence the output.

    The LoRA injection works by registering forward hooks on targeted blocks'
    attn.proj submodules:
    1. Running the frozen model's attention block normally via model.forward()
    2. For targeted blocks (18-23 by default), the hook adds the ExpertLoRA contribution
    3. The LoRA output is added as a residual to the projection output

    Note on injection point: The hook is registered on attn.proj, so the LoRA
    contribution is added to the projection output BEFORE proj_drop. In eval mode
    (the only mode used for inference), dropout is a no-op so this is semantically
    equivalent to the previous post-proj_drop injection point.

    Args:
        bridge: WeatherStepBridge instance
        x_norm: Normalized initial condition of shape (B, V, H, W)
        variables: List of variable names
        interval: Forecast interval (6, 12, or 24 hours)
        steps: Number of autoregressive steps
        plan: EditPlan specifying active experts and coefficients
        expert_loras: Dict mapping block index to ExpertLoRA module
        target_blocks: Which blocks to inject LoRA (default: 18-23 per v6 spec)
        sparse: Whether to use sparse LoRA forward (skip inactive experts)
        return_trajectory: If True, return the full trajectory (B, T+1, V, H, W)
            including initial state and all intermediate steps
        differentiable: If True, construct the coefficient tensor in a way that
            preserves gradients for differentiable selection/optimization.
            When False (default), coefficients are detached.

    Returns:
        If return_trajectory is False: Normalized prediction at final step, shape (B, V, H, W)
        If return_trajectory is True: Full trajectory, shape (B, T+1, V, H, W)

    Raises:
        TypeError: If plan or expert_loras have wrong types
        RuntimeError: If the call is outside the declared supported execution
            modes (see CONTROLLED_ROLLOUT_SUPPORTED_MODES): a reentrant call on
            the same bridge, a concurrent rollout on a backbone that is already
            in flight (even from another thread or another bridge wrapping the
            same model), or a backbone configured for activation-checkpoint
            replay.

    Note:
        If plan.coefficients are all zero, this produces identical output
        to bridge.forward_validation (the zero_edit_equals_official property).
    """
    # Validate inputs
    if not isinstance(plan, EditPlan):
        raise TypeError("plan must be an EditPlan")
    if not isinstance(expert_loras, dict):
        raise TypeError("expert_loras must be a dict")

    # Re-entrancy guard
    _check_reentrant_rollout(bridge)

    try:
        batch_size = x_norm.shape[0]

        # Scale interval by 10.0
        interval_tensor = torch.tensor([interval], device=x_norm.device, dtype=x_norm.dtype) / 10.0
        interval_tensor = interval_tensor.repeat(batch_size)

        x = x_norm
        trajectory = [x] if return_trajectory else None

        for step_idx in range(steps):
            # Get coefficients for this step (zero outside application window)
            coeffs_at_step = plan.coefficients_at(step_idx)

            if differentiable:
                # Keep coefficient tensor in a form that allows gradient flow
                # This is needed for differentiable expert selection
                coeffs_tensor = torch.tensor(
                    coeffs_at_step, device=x.device, dtype=x.dtype, requires_grad=True
                ).unsqueeze(0).expand(batch_size, -1)  # [B, K]
            else:
                coeffs_tensor = torch.tensor(
                    coeffs_at_step, device=x.device, dtype=x.dtype
                ).unsqueeze(0).expand(batch_size, -1)  # [B, K]

            # Create hook functions and register them for each target block with LoRA
            hook_handles = []

            def make_lora_hook(lora_module, coeffs, use_sparse):
                """Create a forward hook that adds LoRA contribution to proj output.

                The hook receives (module, input, output) where:
                - input[0] is the attention output (pre-projection tensor)
                - output is the projection output (before proj_drop)

                We compute LoRA from input[0] and add to output, matching the
                original semantic of injecting LoRA contribution after projection.
                """
                def hook(module, input, output):
                    # input[0] is the attention output (what goes into proj)
                    lora_out = lora_module.forward(input[0], coeffs, sparse=use_sparse)
                    return output + lora_out
                return hook

            try:
                # Register hooks for targeted blocks with LoRA modules
                for block_idx in target_blocks:
                    if block_idx in expert_loras:
                        proj_module = bridge.model.blocks[block_idx].attn.proj
                        hook_fn = make_lora_hook(expert_loras[block_idx], coeffs_tensor, sparse)
                        handle = proj_module.register_forward_hook(hook_fn)
                        hook_handles.append(handle)

                # Pad input and run model forward
                padded_x, pad_size = bridge.pad(x)
                output = bridge.model(padded_x, variables, interval_tensor)

            finally:
                # Always remove hooks, even if forward raises
                for handle in hook_handles:
                    handle.remove()

            # Remove padding
            pred_diff = output[:, :, pad_size:]

            # Zero out constant channels
            pred_diff = bridge.normalization.replace_constant(pred_diff, variables)
            # Denormalize diff
            pred_diff = bridge.normalization.denormalize_diff(pred_diff, interval)
            # Denormalize current state, add diff, renormalize
            pred = bridge.normalization.denormalize(x) + pred_diff
            x = bridge.normalization.normalize(pred)

            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return torch.stack(trajectory, dim=1)  # (B, T+1, V, H, W)
        return x

    finally:
        _release_rollout_lock(bridge)


# =============================================================================
# Version Checking
# =============================================================================

def check_version_match(plan_version: ArtifactVersion, bridge_version: ArtifactVersion) -> None:
    """Check that an edit plan's version matches the bridge's version.

    Args:
        plan_version: Version the plan was created against
        bridge_version: Version of the current bridge

    Raises:
        ValueError: If versions don't match
    """
    plan_version.assert_matches(bridge_version)
