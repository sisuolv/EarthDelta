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

Two families of reducers live here and must not be confused (audit finding B06):

1. DIAGNOSTIC reducers (``quadratic_gain``, ``weighted_mse``, ``weighted_rmse``).
   These use the flattened ``[B,H,S,F]`` summary-space convention and normalize
   weights over the *last* (feature) axis only (``quadratic_gain``) or collapse
   everything to one scalar (``weighted_mse``/``weighted_rmse``).  They are
   useful per-feature / per-lead diagnostics and are kept for backward
   compatibility, but a per-feature normalization cancels any scalar lead/area
   weight that varies only across H or S, so they are NOT the full
   multi-dimensional objective.  (The coefficient-space helpers
   ``quadratic_gain_from_benefit_gram`` / ``quadratic_gain_numpy`` are a third,
   separate family: they operate on a benefit/Gram pair that already has the
   spatial weighting baked in, and apply no further normalization.)

2. The FULL OBJECTIVE contract (``full_objective_loss``, ``full_objective_gain``
   / ``gain_analytic``, ``gain_calibrated``, ``effective_quadratic_weight``).
   These use the native, un-flattened ``[B,(K,)H,V,Lat,Lon]`` convention,
   normalize the weight q exactly ONCE over (H,V,Lat,Lon) while preserving the
   leading batch/candidate axes, and drive both the loss and the analytic gain
   from the same inspectable ``Q_eff = diag(q/s^2) / sum(q)`` object so the
   endpoint identity ``L(F) - L(F+u) == gain`` holds exactly.
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


def _check_weight_values(w: Tensor, name: str = 'weights') -> None:
    """Value-level weight checks: real floating-point, finite, non-negative.

    Split out of ``_validate_weights`` so that every weighted reducer in this
    module (``quadratic_gain``, ``weighted_mse``/``weighted_rmse``, and the full
    objective contract) applies exactly the same rules.  Audit finding B05 was
    precisely that ``weighted_mse`` skipped these checks and therefore returned
    a negative "MSE" for ``weights=[2, -1]``.
    """
    if not isinstance(w, Tensor):
        raise ValueError(f"{name} must be a Tensor")
    if not w.is_floating_point():
        raise ValueError(f"{name} must be floating-point")
    if not bool(torch.isfinite(w).all()):
        raise ValueError(f"{name} must be finite")
    if bool((w < 0).any()):
        raise ValueError(f"{name} must be non-negative")


def _validate_scale(scale: Tensor, name: str = 'scale') -> None:
    """Scale checks: real floating-point, finite, strictly positive.

    A zero scale is a divide-by-zero and a negative scale flips the sign of the
    error, so both are rejected rather than silently propagated (B05).
    """
    if not isinstance(scale, Tensor):
        raise ValueError(f"{name} must be a Tensor")
    if not scale.is_floating_point():
        raise ValueError(f"{name} must be floating-point")
    if not bool(torch.isfinite(scale).all()):
        raise ValueError(f"{name} must be finite")
    if not bool((scale > 0).all()):
        raise ValueError(f"{name} must be strictly positive")


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
        w = torch.ones(reference_shape[-1])
    else:
        w = weights

    _check_weight_values(w, 'weights')
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
    """DIAGNOSTIC per-feature quadratic gain: 2<e0,du> - ||du||^2.

    DIAGNOSTIC ONLY -- NOT the full multi-dimensional objective (B06).  This
    reducer normalizes weights over the LAST (flattened feature) axis only and
    returns a still-unreduced ``[B,K,H,S]`` tensor.  Any scalar weight that
    varies only across H or S (a lead-time or area weight) is therefore divided
    out by the per-slice normalization and has no effect on the result.  Use
    ``full_objective_gain`` / ``full_objective_loss`` when the quantity of
    interest is the full (H,V,Lat,Lon) objective; keep this one for per-feature
    and per-lead diagnostics, and for the paired-training summary target that
    already depends on its ``[B,K,H,S]`` output shape.

    This remains the single implementation of the per-feature FSO gain identity.
    All per-feature call sites must use this function with an explicit
    convention argument.

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
        # Normalize each *actual* reduced slice independently.  A tensor such
        # as [H,S,F] may contain a valid global weight sum while one lead or
        # spatial slice is all zero; dividing that slice used to manufacture
        # NaNs that could silently enter the paired/head path.  Zero-weight
        # slices have no well-defined weighted mean, so fail closed.
        denominator = w.sum(-1, keepdim=True)
        if not bool((denominator > 0).all()):
            bad = (~(denominator > 0)).nonzero(as_tuple=False).detach().cpu().tolist()
            raise ValueError(
                "weights must have a strictly positive sum for every reduced "
                f"feature slice; zero-sum slice(s) at {bad[:8]}"
            )
        w_normalized = w / denominator
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
    """DIAGNOSTIC scalar weighted mean squared error with explicit convention.

    Formula (WEIGHTED_MEAN): L = sum(q * ((Y-F)/s)^2) / sum(q)
    Formula (WEIGHTED_SUM): L = sum(q * ((Y-F)/s)^2)

    DIAGNOSTIC ONLY -- this collapses every axis (including the batch axis) to a
    single scalar, so it cannot express a per-sample or per-candidate objective.
    Use ``full_objective_loss`` for the full ``[B,(K,)H,V,Lat,Lon]`` objective.

    Weight/scale validation (B05): ``weights`` is checked with exactly the same
    rules ``quadratic_gain`` applies -- real floating-point, finite and
    non-negative -- and the effective denominator (the sum of the broadcast
    weights) must be strictly positive.  ``scale`` must be real floating-point,
    finite and strictly positive.  An all-zero weight tensor over a real target
    is a hard error; it is deliberately NOT clamped with ``max(eps, sum(w))``,
    because that would silently turn an empty objective into a huge or zero
    number instead of surfacing the bug.  The positive-sum requirement is
    applied under both conventions, matching ``quadratic_gain``.

    Args:
        prediction: Predicted values
        target: Target values (same shape as prediction)
        weights: Optional spatial weights q
        convention: WEIGHTED_SUM or WEIGHTED_MEAN
        scale: Optional scale factors s (for normalization)

    Returns:
        Scalar MSE

    Raises:
        ValueError: on shape mismatch, or on weights/scale that fail the checks
            described above.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}")
    if convention not in (WeightConvention.WEIGHTED_MEAN, WeightConvention.WEIGHTED_SUM):
        raise ValueError(f"Unknown convention: {convention}")

    diff = prediction - target
    if scale is not None:
        _validate_scale(scale, 'scale')
        scale = torch.broadcast_to(scale.to(diff), diff.shape)
        diff = diff / scale

    sq_err = diff.square()

    if weights is not None:
        _check_weight_values(weights, 'weights')
        weights = torch.broadcast_to(weights.to(sq_err), sq_err.shape)
        # Validate the *effective* denominator, i.e. after broadcasting to the
        # reduction extent, so an all-zero weight tensor is rejected here rather
        # than producing NaN (WEIGHTED_MEAN) or a meaningless 0 (WEIGHTED_SUM).
        if bool((weights.sum() <= 0)):
            raise ValueError("weights must have positive sum")
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
    """DIAGNOSTIC scalar weighted RMSE with correct aggregation order.

    CRITICAL: RMSE = sqrt(mean(squared_errors)), NOT mean(sqrt(squared_errors)).
    This function enforces correct aggregation order.

    DIAGNOSTIC ONLY -- like ``weighted_mse`` it collapses to a single scalar and
    is not the full ``[B,(K,)H,V,Lat,Lon]`` objective.  It inherits the same
    weight/scale validation as ``weighted_mse`` (B05).

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


# ============================================================================
# Full multi-dimensional objective contract (audit finding B06)
#
# Native [B,(K,)H,V,Lat,Lon] scoring.  Unlike the diagnostic reducers above,
# the weight q is normalized exactly ONCE over (H,V,Lat,Lon) and the leading
# batch (B) and edit-candidate (K) axes are preserved, so a lead-time or area
# weight that varies only across H / Lat survives into the result.
# ============================================================================

#: Names of the trailing axes that constitute one scoring target.
OBJECTIVE_DIM_NAMES: Tuple[str, ...] = ('H', 'V', 'Lat', 'Lon')

#: Number of trailing axes reduced (and normalized over) by the full objective.
N_OBJECTIVE_DIMS: int = len(OBJECTIVE_DIM_NAMES)

#: Internal accumulation dtype.  float64 throughout, see ``full_objective_loss``.
_OBJECTIVE_DTYPE = torch.float64


def _check_objective_field(t: Tensor, name: str) -> None:
    """Field checks for prediction/target/error/response: real floating, finite."""
    if not isinstance(t, Tensor):
        raise ValueError(f"{name} must be a Tensor")
    if not t.is_floating_point():
        # torch.is_floating_point() is False for integer AND complex dtypes.
        raise ValueError(f"{name} must be real floating-point, got dtype {t.dtype}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} contains non-finite values")


def _align_objective_weight(t: Tensor, name: str, full_shape: Tuple[int, ...]) -> Tensor:
    """Right-align ``t`` against ``full_shape`` with explicit per-axis checks.

    Broadcasting rule (deliberately narrower than numpy's, so a transposed or
    mis-ranked weight cannot be silently absorbed):

    * ``t.ndim <= N_OBJECTIVE_DIMS`` -- ``t`` is right-aligned against the
      trailing ``(H,V,Lat,Lon)`` axes and shared across every B and K.
    * ``t.ndim == len(full_shape)`` -- ``t`` is fully explicit, including its
      own leading B (and K) axes.
    * Anything in between is REJECTED.  With a 6-D ``[B,K,H,V,Lat,Lon]``
      objective, a 5-D weight is ambiguous: right-aligning it would silently
      match its leading axis against K rather than B, which happens to "work"
      whenever B == K.  The caller must insert the missing axis explicitly.

    Every aligned axis must then be either equal to the objective extent or 1.
    """
    full_ndim = len(full_shape)
    if t.ndim <= N_OBJECTIVE_DIMS:
        aligned_shape = (1,) * (full_ndim - t.ndim) + tuple(t.shape)
    elif t.ndim == full_ndim:
        aligned_shape = tuple(t.shape)
    else:
        raise ValueError(
            f"{name} must have at most {N_OBJECTIVE_DIMS} dims (broadcast over "
            f"{OBJECTIVE_DIM_NAMES}, shared across B/K) or exactly {full_ndim} "
            f"dims (fully explicit against objective shape {tuple(full_shape)}); "
            f"got {t.ndim}-D {tuple(t.shape)}. Insert the missing batch/candidate "
            "axes explicitly rather than relying on implicit broadcasting."
        )
    for i, (a, f) in enumerate(zip(aligned_shape, full_shape)):
        if a != f and a != 1:
            raise ValueError(
                f"{name} shape {tuple(t.shape)} is not broadcastable to objective "
                f"shape {tuple(full_shape)}: axis {i} is {a}, expected {f} or 1"
            )
    return t.reshape(aligned_shape)


def _sum_over_objective_dims(q_aligned: Tensor, full_shape: Tuple[int, ...]) -> Tensor:
    """sum_{H,V,Lat,Lon} q, evaluated over the FULL objective extent.

    ``q_aligned`` may carry size-1 axes that broadcast over the objective; a
    size-1 objective axis contributes ``full_shape[d]`` copies of the same
    value, so the partial sum is scaled by the expansion factor instead of
    materializing the expanded tensor (which would be B*K times larger).

    Returns a tensor of shape ``full_shape[:-N_OBJECTIVE_DIMS]`` (``[B]`` or
    ``[B,K]``): one effective denominator per actual scoring target.
    """
    full_ndim = len(full_shape)
    obj_dims = tuple(range(full_ndim - N_OBJECTIVE_DIMS, full_ndim))
    partial = q_aligned.sum(dim=obj_dims)
    factor = 1
    for d in obj_dims:
        if q_aligned.shape[d] == 1:
            factor *= full_shape[d]
    total = partial * float(factor)
    return torch.broadcast_to(total, full_shape[: full_ndim - N_OBJECTIVE_DIMS])


@dataclass(frozen=True, eq=False)
class EffectiveQuadraticWeight:
    """``Q_eff = diag(q / s^2) / sum_{H,V,Lat,Lon}(q)``, stored factored.

    This is the single weighting object shared by ``full_objective_loss`` and
    ``full_objective_gain``.  Deriving both from the same object is the actual
    content of audit finding B06: previously the loss and the gain each built
    their own weighting and could disagree about normalization.

    ``Q_eff`` is diagonal in the flattened ``(H,V,Lat,Lon)`` objective index, so
    it is represented by that diagonal rather than by a dense
    ``(H*V*Lat*Lon) x (H*V*Lat*Lon)`` matrix, which is the only practical choice
    at weather grid sizes.  The diagonal is further kept in factored form:

        ``diagonal = unnormalized_diagonal / normalizer``

    where ``unnormalized_diagonal`` is ``q / s^2`` right-aligned against the
    objective shape (size-1 axes retained, never expanded) and ``normalizer`` is
    ``sum_{H,V,Lat,Lon} q`` with shape ``[B]`` or ``[B,K]``.  ``quadratic_form``
    consumes the factored form directly, so the full diagonal is never
    materialized during scoring; the ``diagonal`` property materializes it for
    inspection/debugging.

    All fields are float64.
    """

    unnormalized_diagonal: Tensor   # q / s^2, right-aligned to objective_shape
    normalizer: Tensor              # sum_{H,V,Lat,Lon} q; shape [B] or [B,K]
    objective_shape: Tuple[int, ...]

    @property
    def objective_dims(self) -> Tuple[int, ...]:
        """Axis indices of (H,V,Lat,Lon) within ``objective_shape``."""
        n = len(self.objective_shape)
        return tuple(range(n - N_OBJECTIVE_DIMS, n))

    @property
    def leading_shape(self) -> Tuple[int, ...]:
        """``[B]`` or ``[B,K]`` -- the axes preserved by the reduction."""
        return tuple(self.objective_shape[: len(self.objective_shape) - N_OBJECTIVE_DIMS])

    @property
    def diagonal(self) -> Tensor:
        """Materialize the full ``[B,(K,)H,V,Lat,Lon]`` diagonal of ``Q_eff``.

        For inspection and tests.  Scoring uses ``quadratic_form`` instead,
        which never allocates this tensor.
        """
        norm = self.normalizer.reshape(self.leading_shape + (1,) * N_OBJECTIVE_DIMS)
        return torch.broadcast_to(
            self.unnormalized_diagonal / norm, self.objective_shape
        ).contiguous()

    def quadratic_form(self, a: Tensor, b: Tensor) -> Tensor:
        """``a^T Q_eff b``, reduced over (H,V,Lat,Lon).

        Args:
            a, b: float64 tensors broadcastable to ``objective_shape``.

        Returns:
            ``[B]`` or ``[B,K]`` float64.
        """
        numerator = (self.unnormalized_diagonal * a * b).sum(dim=self.objective_dims)
        if tuple(numerator.shape) != self.leading_shape:
            raise ValueError(
                f"quadratic_form operands reduce to {tuple(numerator.shape)}, "
                f"expected {self.leading_shape} for objective shape "
                f"{self.objective_shape}"
            )
        return numerator / self.normalizer


def effective_quadratic_weight(
    q: Optional[Tensor],
    scale: Optional[Tensor],
    objective_shape: Union[torch.Size, Tuple[int, ...]],
    *,
    device: Optional[torch.device] = None,
) -> EffectiveQuadraticWeight:
    """Build the shared ``Q_eff = diag(q/s^2)/sum(q)`` for one objective shape.

    Args:
        q: Non-negative weights, broadcastable over ``(H,V,Lat,Lon)`` (at most
            4-D) or fully explicit against ``objective_shape``.  ``None`` means
            uniform weights.  See ``_align_objective_weight`` for the exact,
            deliberately-narrow broadcasting rule.
        scale: Strictly positive scale factors ``s``, same broadcasting rule.
            ``None`` means ``s = 1``.
        objective_shape: ``[B,H,V,Lat,Lon]`` or ``[B,K,H,V,Lat,Lon]``.
        device: Device for the uniform default when ``q``/``scale`` are None.

    Returns:
        ``EffectiveQuadraticWeight`` in float64.

    Raises:
        ValueError: if q is non-floating / non-finite / negative, if scale is
            non-floating / non-finite / non-positive, if either is not
            broadcastable under the documented rule, or if ``sum(q)`` over
            ``(H,V,Lat,Lon)`` is not strictly positive for EVERY target.  The
            last case is a hard error on purpose: an all-zero weight slice over
            a real target is never clamped with ``max(eps, sum(q))``.
    """
    full_shape = tuple(objective_shape)
    if len(full_shape) < N_OBJECTIVE_DIMS + 1:
        raise ValueError(
            f"objective shape must be at least {N_OBJECTIVE_DIMS + 1}-D "
            f"[B,{','.join(OBJECTIVE_DIM_NAMES)}], got {full_shape}"
        )

    if q is None:
        q_t = torch.ones((1,) * len(full_shape), dtype=_OBJECTIVE_DTYPE, device=device)
    else:
        _check_weight_values(q, 'q')
        q_t = q.to(dtype=_OBJECTIVE_DTYPE)
    q_aligned = _align_objective_weight(q_t, 'q', full_shape)

    if scale is None:
        unnormalized = q_aligned
    else:
        _validate_scale(scale, 'scale')
        s_aligned = _align_objective_weight(
            scale.to(dtype=_OBJECTIVE_DTYPE), 'scale', full_shape
        )
        unnormalized = q_aligned / s_aligned.square()

    normalizer = _sum_over_objective_dims(q_aligned, full_shape)
    if not bool((normalizer > 0).all()):
        raise ValueError(
            "q must have a strictly positive sum over "
            f"{OBJECTIVE_DIM_NAMES} for every target; minimum effective "
            f"denominator is {float(normalizer.min())}. A weight mask that is "
            "zero over some grid points is fine, but a target whose weights are "
            "entirely zero is rejected rather than clamped with an epsilon."
        )

    return EffectiveQuadraticWeight(unnormalized, normalizer, full_shape)


def _resolve_objective_pair(
    primary: Tensor,
    secondary: Tensor,
    primary_name: str,
    secondary_name: str,
) -> Tensor:
    """Align ``secondary`` (a ``[B,H,V,Lat,Lon]`` field) against ``primary``.

    ``primary`` may be 5-D ``[B,H,V,Lat,Lon]`` or 6-D ``[B,K,H,V,Lat,Lon]``.
    ``secondary`` either matches ``primary`` exactly, or -- when ``primary``
    carries a K axis -- is the K-free field, which is unsqueezed at axis 1.
    Any other combination is an error; nothing is broadcast implicitly.
    """
    if tuple(secondary.shape) == tuple(primary.shape):
        return secondary
    if (
        primary.ndim == N_OBJECTIVE_DIMS + 2
        and secondary.ndim == N_OBJECTIVE_DIMS + 1
        and tuple(secondary.shape) == tuple(primary.shape[:1] + primary.shape[2:])
    ):
        return secondary.unsqueeze(1)
    raise ValueError(
        f"{secondary_name} shape {tuple(secondary.shape)} inconsistent with "
        f"{primary_name} shape {tuple(primary.shape)}; expected an exact match, "
        f"or {primary_name}[0] and {primary_name}[2:] to match {secondary_name} "
        "when a K (edit-candidate) axis is present"
    )


def full_objective_loss(
    prediction: Tensor,
    target: Tensor,
    q: Optional[Tensor] = None,
    *,
    scale: Optional[Tensor] = None,
) -> Tensor:
    """Full multi-dimensional objective loss (B06).

        L = sum_{H,V,Lat,Lon}( q * ((Y - F)/s)^2 ) / sum_{H,V,Lat,Lon}( q )

    The weight ``q`` is normalized exactly ONCE, over ``(H,V,Lat,Lon)`` jointly.
    The leading batch (B) and edit-candidate (K) axes are PRESERVED, so this
    returns ``[B]`` or ``[B,K]`` and never collapses to a scalar.  That is what
    distinguishes it from ``weighted_mse``: a per-lead or per-latitude weight
    that varies only across H or Lat survives here, whereas the per-feature
    ``quadratic_gain`` reducer divides it straight back out.

    Precision: the difference, the square and the accumulation are all performed
    in float64 and the RESULT IS RETURNED IN float64 regardless of input dtype.
    Keeping the return in float64 (rather than casting back to the input dtype)
    is deliberate: the endpoint identity ``L(F) - L(F+u) == gain`` is a
    difference of two nearly-equal losses, and rounding the two endpoints back
    to float32 would destroy roughly eight significant digits of the difference.
    Cast at the call site if a float32 loss is needed for a training step;
    autograd carries gradients back through the upcast unchanged.

    Args:
        prediction: ``F``, ``[B,H,V,Lat,Lon]`` or ``[B,K,H,V,Lat,Lon]``.
        target: ``Y``.  Either the exact shape of ``prediction``, or the K-free
            ``[B,H,V,Lat,Lon]`` when ``prediction`` carries a K axis (the only
            broadcast permitted, and it is checked explicitly).
        q: Non-negative weights; see ``effective_quadratic_weight``.
        scale: Strictly positive scales ``s``; see ``effective_quadratic_weight``.

    Returns:
        ``[B]`` or ``[B,K]`` float64 losses.

    Raises:
        ValueError: on non-real/non-finite fields, bad rank, shape mismatch,
            negative or non-finite ``q``, non-positive ``scale``, or a target
            whose weights sum to zero.
    """
    _check_objective_field(prediction, 'prediction')
    _check_objective_field(target, 'target')
    if prediction.ndim not in (N_OBJECTIVE_DIMS + 1, N_OBJECTIVE_DIMS + 2):
        raise ValueError(
            f"prediction must be {N_OBJECTIVE_DIMS + 1}-D "
            f"[B,{','.join(OBJECTIVE_DIM_NAMES)}] or {N_OBJECTIVE_DIMS + 2}-D "
            f"[B,K,{','.join(OBJECTIVE_DIM_NAMES)}], got {prediction.ndim}-D "
            f"{tuple(prediction.shape)}"
        )
    target_aligned = _resolve_objective_pair(prediction, target, 'prediction', 'target')

    q_eff = effective_quadratic_weight(
        q, scale, tuple(prediction.shape), device=prediction.device
    )
    diff = prediction.to(_OBJECTIVE_DTYPE) - target_aligned.to(_OBJECTIVE_DTYPE)
    return q_eff.quadratic_form(diff, diff)


def full_objective_gain(
    error: Tensor,
    response: Tensor,
    q: Optional[Tensor] = None,
    *,
    scale: Optional[Tensor] = None,
) -> Tensor:
    """Analytic full-objective quadratic gain (B06).

        g = 2 * e^T Q_eff u  -  u^T Q_eff u

    with ``Q_eff = diag(q/s^2)/sum_{H,V,Lat,Lon}(q)`` -- the SAME object
    ``full_objective_loss`` builds for the same ``q`` / ``scale``.  Consequently
    the endpoint identity

        full_objective_loss(F, Y, q, scale=s)
          - full_objective_loss(F + u, Y, q, scale=s)
        == full_objective_gain(Y - F, u, q, scale=s)

    holds exactly (to float64 round-off), which is the property B06 asks for.

    This is the ANALYTIC path only: no calibration, no ridge, no cost penalty.
    ``gain_analytic`` is the explicit alias; ``gain_calibrated`` is the separate
    entry point that adds a caller-supplied calibration term, and calibration is
    OFF unless the caller passes one.

    Args:
        error: ``e = Y - F``, ``[B,H,V,Lat,Lon]``.
        response: ``u``, the edit response.  ``[B,H,V,Lat,Lon]`` or
            ``[B,K,H,V,Lat,Lon]`` for K edit candidates.
        q: Non-negative weights; see ``effective_quadratic_weight``.
        scale: Strictly positive scales ``s``; see ``effective_quadratic_weight``.

    Returns:
        ``[B]`` or ``[B,K]`` float64 gains (positive = error reduced).

    Raises:
        ValueError: same conditions as ``full_objective_loss``.
    """
    _check_objective_field(error, 'error')
    _check_objective_field(response, 'response')
    if error.ndim != N_OBJECTIVE_DIMS + 1:
        raise ValueError(
            f"error must be {N_OBJECTIVE_DIMS + 1}-D "
            f"[B,{','.join(OBJECTIVE_DIM_NAMES)}], got {error.ndim}-D "
            f"{tuple(error.shape)}"
        )
    if response.ndim not in (N_OBJECTIVE_DIMS + 1, N_OBJECTIVE_DIMS + 2):
        raise ValueError(
            f"response must be {N_OBJECTIVE_DIMS + 1}-D "
            f"[B,{','.join(OBJECTIVE_DIM_NAMES)}] or {N_OBJECTIVE_DIMS + 2}-D "
            f"[B,K,{','.join(OBJECTIVE_DIM_NAMES)}], got {response.ndim}-D "
            f"{tuple(response.shape)}"
        )
    error_aligned = _resolve_objective_pair(response, error, 'response', 'error')

    q_eff = effective_quadratic_weight(
        q, scale, tuple(response.shape), device=response.device
    )
    e = error_aligned.to(_OBJECTIVE_DTYPE)
    u = response.to(_OBJECTIVE_DTYPE)
    return 2.0 * q_eff.quadratic_form(e, u) - q_eff.quadratic_form(u, u)


def gain_analytic(
    error: Tensor,
    response: Tensor,
    q: Optional[Tensor] = None,
    *,
    scale: Optional[Tensor] = None,
) -> Tensor:
    """Explicitly-named analytic gain path (B06): pure quadratic identity.

    Identical to ``full_objective_gain``.  It exists under this name so that
    call sites state which of the two gain paths they mean; ``gain_analytic``
    never includes a calibration term.
    """
    return full_objective_gain(error, response, q, scale=scale)


def gain_calibrated(
    error: Tensor,
    response: Tensor,
    q: Optional[Tensor] = None,
    *,
    scale: Optional[Tensor] = None,
    calibration: Optional[Tensor] = None,
) -> Tensor:
    """Calibrated gain path (B06): analytic gain plus a caller-supplied term.

    CALIBRATION IS OFF BY DEFAULT.  With ``calibration=None`` this returns
    exactly ``gain_analytic(...)``, bit for bit.  This function owns no
    parameters and creates no ``nn.Module``: any calibration must be computed by
    the caller and passed in as a plain tensor, so "calibrated" can never become
    the silent default the way it did before.

    Args:
        error, response, q, scale: as ``full_objective_gain``.
        calibration: Optional additive term, real floating and finite,
            broadcastable to the ``[B]`` / ``[B,K]`` gain shape under the same
            right-aligned, explicitly-checked rule used for ``q``.

    Returns:
        ``[B]`` or ``[B,K]`` float64 calibrated gains.
    """
    gain = full_objective_gain(error, response, q, scale=scale)
    if calibration is None:
        return gain

    _check_objective_field(calibration, 'calibration')
    if calibration.ndim > gain.ndim:
        raise ValueError(
            f"calibration must have at most {gain.ndim} dims to align with gain "
            f"shape {tuple(gain.shape)}, got {calibration.ndim}-D "
            f"{tuple(calibration.shape)}"
        )
    aligned_shape = (1,) * (gain.ndim - calibration.ndim) + tuple(calibration.shape)
    for i, (a, g) in enumerate(zip(aligned_shape, tuple(gain.shape))):
        if a != g and a != 1:
            raise ValueError(
                f"calibration shape {tuple(calibration.shape)} is not "
                f"broadcastable to gain shape {tuple(gain.shape)}: axis {i} is "
                f"{a}, expected {g} or 1"
            )
    return gain + calibration.reshape(aligned_shape).to(gain)
