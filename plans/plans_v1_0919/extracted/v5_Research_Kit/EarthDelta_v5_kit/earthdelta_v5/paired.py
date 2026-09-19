"""Reference-error/edit-response factorization, in an explicitly supplied linear view.

No assertion that a compressed view represents every physical mode, nor that
this quadratic identity holds for AMSE, RMSE, log-energy or ACC objectives.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor


def _weights(x: Tensor, weights: Tensor | None) -> Tensor:
    w = torch.ones_like(x) if weights is None else torch.broadcast_to(weights.to(x), x.shape)
    if not torch.isfinite(w).all() or (w < 0).any() or (w.sum(-1) <= 0).any():
        raise ValueError('Finite nonnegative weights with positive feature sums required.')
    return w / w.sum(-1, keepdim=True)


def edit_responses(reference: Tensor, edited: Tensor) -> Tensor:
    """[B,H,S,F], [B,K,H,S,F] -> [B,K,H,S,F]. No truth is consumed."""
    if reference.ndim != 4 or edited.ndim != 5:
        raise ValueError('Expected reference [B,H,S,F], edited [B,K,H,S,F].')
    if edited.shape[:1] + edited.shape[2:] != reference.shape:
        raise ValueError('Reference and edited shapes disagree.')
    if not torch.isfinite(reference).all() or not torch.isfinite(edited).all():
        raise ValueError('Nonfinite forecasts.')
    return edited - reference[:, None]


def quadratic_gain(error: Tensor, response: Tensor, weights: Tensor | None = None) -> Tensor:
    """Gain = 2<error,response> - ||response||^2; final feature dimension is reduced."""
    if error.ndim != 4 or response.ndim != 5 or response.shape[:1]+response.shape[2:] != error.shape:
        raise ValueError('Expected aligned [B,H,S,F] and [B,K,H,S,F].')
    if not torch.isfinite(error).all() or not torch.isfinite(response).all():
        raise ValueError('Nonfinite error/response.')
    w = _weights(response, weights)
    return ((2 * error[:,None] * response - response.square()) * w).sum(-1)

@dataclass
class PairedTargets:
    reference_error: Tensor
    edit_response: Tensor
    edited_error: Tensor
    quadratic_gain: Tensor


def make_training_targets(truth: Tensor, reference: Tensor, edited: Tensor,
                          weights: Tensor | None = None) -> PairedTargets:
    """OFFLINE ONLY. Future truth must never be an argument to inference/planning."""
    if truth.shape != reference.shape or not torch.isfinite(truth).all():
        raise ValueError('Truth/reference shape or finite-value violation.')
    d = edit_responses(reference, edited)
    e = truth - reference
    return PairedTargets(e, d, e[:,None]-d, quadratic_gain(e,d,weights))
