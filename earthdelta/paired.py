"""Reference-error/edit-response factorization in an explicitly supplied linear view.

The summary/readout operator D must be linear for the quadratic gain identity to hold exactly.
This is a requirement, not an approximation. If D contains nonlinearities (e.g., ReLU, softmax,
nonlinear normalization), the gain identity 2<e,du> - ||du||^2 does not decompose exactly.

No assertion that a compressed view represents every physical mode, nor that
this quadratic identity holds for AMSE, RMSE, log-energy or ACC objectives.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

from .metrics_contract import quadratic_gain as _canonical_quadratic_gain, WeightConvention


def _assert_linear_operator_assumption():
    """Document the linearity requirement for D.

    The quadratic gain identity e_u^2 - e_0^2 = 2*<e_0, du> - du^2 requires that:
    - e_0 = D(truth) - D(reference) = D(truth - reference) (linearity in prediction)
    - e_u = D(truth) - D(edited) = e_0 - D(edited - reference) = e_0 - du (linearity in edit response)

    If D is nonlinear, e_u != e_0 - du and the gain formula does not hold exactly.
    """
    pass


def edit_responses(reference: Tensor, edited: Tensor) -> Tensor:
    """Compute edit responses: du = edited - reference.

    Args:
        reference: [B,H,S,F] reference forecasts
        edited: [B,K,H,S,F] edited forecasts for K edits

    Returns:
        [B,K,H,S,F] edit responses. No truth is consumed.
    """
    if reference.ndim != 4 or edited.ndim != 5:
        raise ValueError('Expected reference [B,H,S,F], edited [B,K,H,S,F].')
    if edited.shape[:1] + edited.shape[2:] != reference.shape:
        raise ValueError('Reference and edited shapes disagree.')
    if not torch.isfinite(reference).all() or not torch.isfinite(edited).all():
        raise ValueError('Nonfinite forecasts.')
    return edited - reference[:, None]


def quadratic_gain(error: Tensor, response: Tensor, weights: Tensor | None = None) -> Tensor:
    """Compute quadratic gain: 2<error,response> - ||response||^2.

    This is the exact gain identity when D is linear: ||e_0||^2 - ||e_u||^2 = 2<e_0,du> - ||du||^2.
    The final feature dimension is reduced.

    Convention: WEIGHTED_MEAN - weights are normalized to sum to 1 for computing
    the spatially-averaged gain. This is the correct convention for training targets
    where we want the average error reduction per unit of weighted space.

    Args:
        error: [B,H,S,F] reference errors (truth - reference in summary space)
        response: [B,K,H,S,F] edit responses
        weights: Optional [F] weights for the inner product

    Returns:
        [B,K,H,S] quadratic gains
    """
    _assert_linear_operator_assumption()
    # Delegate to canonical implementation with WEIGHTED_MEAN convention
    # This preserves exact bit-identical behavior for all weight configurations
    return _canonical_quadratic_gain(error, response, weights, convention=WeightConvention.WEIGHTED_MEAN)


@dataclass
class PairedTargets:
    """Container for paired training targets (offline use only)."""
    reference_error: Tensor
    edit_response: Tensor
    edited_error: Tensor
    quadratic_gain: Tensor


def make_training_targets(truth: Tensor, reference: Tensor, edited: Tensor,
                          weights: Tensor | None = None) -> PairedTargets:
    """OFFLINE ONLY. Future truth must never be an argument to inference/planning.

    Computes reference error e0 = truth - reference, edit responses du = edited - reference,
    edited errors e_u = e0 - du, and quadratic gains.

    Args:
        truth: [B,H,S,F] ground truth
        reference: [B,H,S,F] reference forecasts
        edited: [B,K,H,S,F] edited forecasts
        weights: Optional [F] weights

    Returns:
        PairedTargets with all training signals
    """
    _assert_linear_operator_assumption()
    if truth.shape != reference.shape or not torch.isfinite(truth).all():
        raise ValueError('Truth/reference shape or finite-value violation.')
    d = edit_responses(reference, edited)
    e = truth - reference
    return PairedTargets(e, d, e[:, None] - d, quadratic_gain(e, d, weights))
