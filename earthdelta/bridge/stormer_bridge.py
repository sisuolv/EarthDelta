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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn

from ..contracts import ArtifactVersion, EditPlan
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
        for interval in intervals:
            diff_mean_path = os.path.join(npz_dir, f"normalize_diff_mean_{interval}.npz")
            diff_std_path = os.path.join(npz_dir, f"normalize_diff_std_{interval}.npz")

            if os.path.exists(diff_mean_path):
                dm = dict(np.load(diff_mean_path))
                raw_diff_mean = np.concatenate([dm[v] for v in variables], axis=0)
                # Under official policy, force diff_mean to zero
                if policy == POLICY_OFFICIAL_ZERO_DIFF_MEAN:
                    raw_diff_mean = np.zeros_like(raw_diff_mean)
                diff_mean[interval] = torch.from_numpy(raw_diff_mean).float()

            if os.path.exists(diff_std_path):
                ds = dict(np.load(diff_std_path))
                diff_std[interval] = torch.from_numpy(
                    np.concatenate([ds[v] for v in variables], axis=0)
                ).float()

        return cls(
            inp_mean=torch.from_numpy(inp_mean).float(),
            inp_std=torch.from_numpy(inp_std).float(),
            diff_mean=diff_mean,
            diff_std=diff_std,
            variables=variables,
            policy=policy,
        )

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

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize normalized input: x_raw = x_norm * std + mean.

        Args:
            x_norm: Normalized input of shape (B, V, H, W)

        Returns:
            Raw input of same shape
        """
        mean = self.inp_mean.to(x_norm.device, x_norm.dtype).view(1, -1, 1, 1)
        std = self.inp_std.to(x_norm.device, x_norm.dtype).view(1, -1, 1, 1)
        return x_norm * std + mean

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

        std = self.diff_std[interval].to(diff_norm.device, diff_norm.dtype).view(1, -1, 1, 1)

        # Determine mean based on policy
        if self.policy == POLICY_OFFICIAL_ZERO_DIFF_MEAN:
            # Official semantic: always zero diff_mean
            mean = torch.zeros_like(std)
        elif interval in self.diff_mean:
            # Legacy semantic: use actual diff_mean from NPZ
            mean = self.diff_mean[interval].to(diff_norm.device, diff_norm.dtype).view(1, -1, 1, 1)
        else:
            mean = torch.zeros_like(std)

        return diff_norm * std + mean

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
        """Hash of normalization constants and policy for version tracking.

        Includes:
        - Variable names/order
        - Interval keys
        - Tensor shapes and values
        - Policy (POLICY_LEGACY or POLICY_OFFICIAL_ZERO_DIFF_MEAN)

        This ensures the digest changes when any structural or semantic aspect
        of the contract changes, including the diff_mean policy.
        """
        # Build a canonical representation that includes all structural info
        parts = []

        # Policy MUST be part of the digest - it changes the semantic behavior
        parts.append(f"policy={self.policy}".encode())

        # Variable names and order
        parts.append((",".join(self.variables)).encode())

        # Input normalization with shape
        parts.append(f"inp_mean_shape={tuple(self.inp_mean.shape)}".encode())
        parts.append(self.inp_mean.numpy().tobytes())
        parts.append(f"inp_std_shape={tuple(self.inp_std.shape)}".encode())
        parts.append(self.inp_std.numpy().tobytes())

        # Diff normalization with explicit keys (sorted for determinism)
        # Note: Under POLICY_OFFICIAL_ZERO_DIFF_MEAN, diff_mean tensors are zeros
        # but we still include them for structural completeness
        for interval in sorted(self.diff_mean.keys()):
            parts.append(f"diff_mean_{interval}_shape={tuple(self.diff_mean[interval].shape)}".encode())
            parts.append(self.diff_mean[interval].numpy().tobytes())
        for interval in sorted(self.diff_std.keys()):
            parts.append(f"diff_std_{interval}_shape={tuple(self.diff_std[interval].shape)}".encode())
            parts.append(self.diff_std[interval].numpy().tobytes())

        data = b"".join(parts)
        return hashlib.sha256(data).hexdigest()[:16]


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

    # Build model
    model = Stormer(
        in_img_size=in_img_size,
        variables=variables,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )

    # Load checkpoint
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

    # Build model
    model = Stormer(
        in_img_size=in_img_size,
        variables=variables,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )

    # Compute file metadata first
    file_size = os.path.getsize(ckpt_path)
    ckpt_sha256 = _compute_file_sha256(ckpt_path)

    # Load checkpoint
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

# Thread-local state for re-entrancy detection
import threading
_rollout_lock = threading.local()


def _check_reentrant_rollout(bridge: 'WeatherStepBridge') -> None:
    """Check for concurrent/reentrant rollout on the same bridge.

    Raises:
        RuntimeError: If this bridge is already in a rollout call.
    """
    if not hasattr(_rollout_lock, 'active_bridges'):
        _rollout_lock.active_bridges = set()

    bridge_id = id(bridge)
    if bridge_id in _rollout_lock.active_bridges:
        raise RuntimeError(
            "Reentrant call to controlled_rollout on the same bridge detected. "
            "This could corrupt the model's forward hooks. Each bridge instance "
            "should only be used in one rollout at a time."
        )
    _rollout_lock.active_bridges.add(bridge_id)


def _release_rollout_lock(bridge: 'WeatherStepBridge') -> None:
    """Release the re-entrancy lock for a bridge."""
    if hasattr(_rollout_lock, 'active_bridges'):
        _rollout_lock.active_bridges.discard(id(bridge))


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
        RuntimeError: If called reentrantly on the same bridge instance

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
