"""Canonical metric contract: unified gain, RMSE, and weighting conventions.

This module consolidates the 5 previously inconsistent implementations of the
FSO quadratic gain identity: gain = 2<e0,du> - ||du||^2, and provides explicit
WeightConvention control (WEIGHTED_SUM vs WEIGHTED_MEAN) at every call site.

Design rationale for convention assignment:
- WEIGHTED_MEAN: For loss/error metrics where we want the average error reduction
  over the spatial domain (used in training targets and neural network predictions).
- WEIGHTED_SUM: For coefficient-space operations where weights are already baked
  into benefit/Gram matrices (used in geometry and selection modules).

The RMSE helper enforces correct aggregation order: sqrt(mean(errors^2)), not
mean(sqrt(errors^2)).
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor


class WeightConvention(Enum):
    """Explicit weighting convention for metric reduction.

    WEIGHTED_SUM: gain = sum_i(w_i * (2*e_i*du_i - du_i^2))
        - Use when weights are pre-normalized or represent absolute importance
        - Used by ResponseGeometry (weights baked into Gram/benefit)

    WEIGHTED_MEAN: gain = sum_i(w_i * (2*e_i*du_i - du_i^2)) / sum_i(w_i)
        - Use for loss/error metrics where we want spatial-average reduction
        - Used by paired training targets and neural network predictions
    """
    WEIGHTED_SUM = auto()
    WEIGHTED_MEAN = auto()


@dataclass(frozen=True)
class MetricSpec:
    """Canonical specification for FSO metric computation.

    Captures all metadata needed for consistent metric computation across
    training, inference, and evaluation pipelines.
    """
    variable_order: Tuple[str, ...]
    lead_hours: Tuple[int, ...]
    q_hsf: Optional[Tensor]  # [lat, level, variable] or broadcastable spatial weights
    weight_convention: WeightConvention
    missing_policy: str  # 'error' | 'mask' | 'fill_zero'
    utc_time_convention: bool = True  # All times are UTC
    units: str = 'normalized'  # 'normalized' | 'physical'

    def __post_init__(self):
        if self.missing_policy not in ('error', 'mask', 'fill_zero'):
            raise ValueError(f"missing_policy must be 'error', 'mask', or 'fill_zero', got {self.missing_policy}")
        if self.units not in ('normalized', 'physical'):
            raise ValueError(f"units must be 'normalized' or 'physical', got {self.units}")
        if self.q_hsf is not None:
            if not isinstance(self.q_hsf, Tensor):
                raise ValueError("q_hsf must be a Tensor or None")
            if not self.q_hsf.is_floating_point() or not bool(torch.isfinite(self.q_hsf).all()):
                raise ValueError("q_hsf must be finite floating-point")
            if bool((self.q_hsf < 0).any()):
                raise ValueError("q_hsf weights must be non-negative")
        if not self.variable_order:
            raise ValueError("variable_order must be non-empty")
        if not self.lead_hours:
            raise ValueError("lead_hours must be non-empty")


def _validate_weights(weights: Optional[Tensor], reference_shape: torch.Size,
                      convention: WeightConvention) -> Tensor:
    """Validate and optionally normalize weights.

    Args:
        weights: Optional weight tensor (None = uniform)
        reference_shape: Shape to broadcast/validate against
        convention: Determines if weights are normalized to sum=1

    Returns:
        Validated, possibly normalized weights matching reference_shape
    """
    if weights is None:
        w = torch.ones(reference_shape[-1], device=reference_shape.numel() and 'cpu' or 'cpu')
    else:
        w = weights

    if not w.is_floating_point():
        raise ValueError("weights must be floating-point")
    if not bool(torch.isfinite(w).all()):
        raise ValueError("weights must be finite")
    if bool((w < 0).any()):
        raise ValueError("weights must be non-negative")
    if bool((w.sum() <= 0)):
        raise ValueError("weights must have positive sum")

    return w


def quadratic_gain(
    error: Tensor,
    response: Tensor,
    weights: Optional[Tensor] = None,
    *,
    convention: WeightConvention,
) -> Tensor:
    """Canonical quadratic gain: 2<e0,du> - ||du||^2.

    This is THE single implementation of the FSO gain identity. All call sites
    must use this function with an explicit convention argument.

    Mathematical identity (when D is linear):
        ||e0||^2 - ||e_u||^2 = 2<e0,du> - ||du||^2

    The final feature dimension is reduced according to the specified convention:
    - WEIGHTED_SUM: sum_i(w_i * ...)
    - WEIGHTED_MEAN: sum_i(w_i * ...) / sum_i(w_i)

    Args:
        error: [B,H,S,F] reference errors (truth - reference in summary space)
        response: [B,K,H,S,F] edit responses
        weights: Optional [F] weights for the inner product
        convention: WEIGHTED_SUM or WEIGHTED_MEAN (keyword-only, required)

    Returns:
        [B,K,H,S] quadratic gains
    """
    if error.ndim != 4:
        raise ValueError(f"error must be 4-D [B,H,S,F], got {error.ndim}-D")
    if response.ndim != 5:
        raise ValueError(f"response must be 5-D [B,K,H,S,F], got {response.ndim}-D")
    if response.shape[:1] + response.shape[2:] != error.shape:
        raise ValueError(
            f"response shape {response.shape} inconsistent with error shape {error.shape}; "
            "expected response[0] and response[2:] to match error"
        )
    if not torch.isfinite(error).all():
        raise ValueError("error contains non-finite values")
    if not torch.isfinite(response).all():
        raise ValueError("response contains non-finite values")

    w = _validate_weights(weights, error.shape, convention)
    w = torch.broadcast_to(w.to(error), error.shape)

    # Compute per-element gain contribution: 2*e*du - du^2
    gain_elements = 2 * error[:, None] * response - response.square()

    if convention == WeightConvention.WEIGHTED_MEAN:
        # Normalize weights to sum to 1 along feature dimension
        w_normalized = w / w.sum(-1, keepdim=True)
        return (gain_elements * w_normalized[:, None]).sum(-1)
    elif convention == WeightConvention.WEIGHTED_SUM:
        return (gain_elements * w[:, None]).sum(-1)
    else:
        raise ValueError(f"Unknown convention: {convention}")


def quadratic_gain_from_benefit_gram(
    benefit: Tensor,
    gram: Tensor,
    program: Tensor,
) -> Tensor:
    """Compute gain in coefficient space: 2*b.a - a.T H a.

    This is for cases where the spatial weighting is already baked into the
    benefit (b = R.T Q e) and Gram (H = R.T Q R) matrices. No additional
    weighting convention is applied.

    Supports batched inputs:
    - benefit: [d] or [B, d]
    - gram: [d, d] or [B, d, d]
    - program: [d] or [B, d]

    Args:
        benefit: Predicted/computed benefit vector
        gram: Predicted/computed Gram matrix (symmetric PSD)
        program: Program coefficients

    Returns:
        Scalar or [B] predicted gain
    """
    # Handle shapes
    if benefit.ndim == 1:
        # Unbatched case
        if gram.ndim != 2 or program.ndim != 1:
            raise ValueError("For unbatched benefit [d], gram must be [d,d] and program [d]")
        return 2 * (benefit * program).sum(-1) - torch.einsum('i,ij,j->', program, gram, program)
    else:
        # Batched case [B, d]
        if program.ndim == 1:
            # Broadcast program to batch
            return 2 * (benefit * program).sum(-1) - torch.einsum('i,ij,j->...', program, gram, program)
        else:
            return 2 * (benefit * program).sum(-1) - torch.einsum('...i,...ij,...j->...', program, gram, program)


def quadratic_gain_numpy(
    benefit: np.ndarray,
    gram: np.ndarray,
    program: np.ndarray,
) -> float:
    """Compute gain in coefficient space (numpy): 2*b.a - a.T H a.

    This is the numpy equivalent of quadratic_gain_from_benefit_gram for use
    in scipy optimization loops.

    Args:
        benefit: [d] benefit vector
        gram: [d, d] Gram matrix
        program: [d] program coefficients

    Returns:
        Scalar predicted gain
    """
    if not np.isfinite(benefit).all():
        raise ValueError("benefit contains non-finite values")
    if not np.isfinite(gram).all():
        raise ValueError("gram contains non-finite values")
    if not np.isfinite(program).all():
        raise ValueError("program contains non-finite values")
    return float(2 * benefit @ program - program @ gram @ program)


def weighted_mse(
    prediction: Tensor,
    target: Tensor,
    weights: Optional[Tensor] = None,
    *,
    convention: WeightConvention,
    scale: Optional[Tensor] = None,
) -> Tensor:
    """Compute weighted mean squared error with explicit convention.

    Formula (WEIGHTED_MEAN): L = sum(q * ((Y-F)/s)^2) / sum(q)
    Formula (WEIGHTED_SUM): L = sum(q * ((Y-F)/s)^2)

    Args:
        prediction: Predicted values
        target: Target values (same shape as prediction)
        weights: Optional spatial weights q
        convention: WEIGHTED_SUM or WEIGHTED_MEAN
        scale: Optional scale factors s (for normalization)

    Returns:
        Scalar MSE
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}")

    diff = prediction - target
    if scale is not None:
        scale = torch.broadcast_to(scale.to(diff), diff.shape)
        diff = diff / scale

    sq_err = diff.square()

    if weights is not None:
        weights = torch.broadcast_to(weights.to(sq_err), sq_err.shape)
        sq_err = sq_err * weights
        if convention == WeightConvention.WEIGHTED_MEAN:
            return sq_err.sum() / weights.sum()
        else:
            return sq_err.sum()
    else:
        if convention == WeightConvention.WEIGHTED_MEAN:
            return sq_err.mean()
        else:
            return sq_err.sum()


def weighted_rmse(
    prediction: Tensor,
    target: Tensor,
    weights: Optional[Tensor] = None,
    *,
    convention: WeightConvention,
    scale: Optional[Tensor] = None,
) -> Tensor:
    """Compute weighted RMSE with correct aggregation order.

    CRITICAL: RMSE = sqrt(mean(squared_errors)), NOT mean(sqrt(squared_errors)).
    This function enforces correct aggregation order.

    Args:
        prediction: Predicted values
        target: Target values (same shape as prediction)
        weights: Optional spatial weights
        convention: WEIGHTED_SUM or WEIGHTED_MEAN
        scale: Optional scale factors (for normalization)

    Returns:
        Scalar RMSE
    """
    mse = weighted_mse(prediction, target, weights, convention=convention, scale=scale)
    return torch.sqrt(mse)
