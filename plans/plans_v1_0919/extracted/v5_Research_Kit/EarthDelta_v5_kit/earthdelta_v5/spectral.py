"""Phase-aware scalar-coefficient diagnostics; NOT a spherical transform.

Obtain coefficients from a separately validated grid/normalization-aware SHT.
Half-spectrum multiplicity and band masks must be supplied by the caller.
No planar FFT is silently substituted for a global spherical calculation.
"""
from __future__ import annotations
import torch
from torch import Tensor


def coefficient_diagnostics(pred: Tensor, truth: Tensor, weights: Tensor,
                            eps: float = 1e-10) -> dict[str,Tensor]:
    if pred.shape != truth.shape or pred.ndim < 1 or not pred.is_complex() or not truth.is_complex():
        raise ValueError('Matching complex coefficient arrays required.')
    if eps <= 0:
        raise ValueError('eps must be positive.')
    w = torch.broadcast_to(weights.to(pred.real),pred.shape)
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all() or not torch.isfinite(w).all():
        raise ValueError('Nonfinite coefficients/weights.')
    if (w<0).any() or (w.sum(-1)<=0).any():
        raise ValueError('Invalid coefficient weights.')
    ep = (w*pred.abs().square()).sum(-1)
    ey = (w*truth.abs().square()).sum(-1)
    numerator = (w*(pred*truth.conj()).real).sum(-1)
    denom = torch.sqrt(ep*ey)
    corr = numerator/denom.clamp_min(eps)
    both_empty = (ep<=eps) & (ey<=eps)
    corr = torch.where(both_empty,torch.ones_like(corr),corr).clamp(-1,1)
    return {'energy_pred':ep,'energy_truth':ey,'correlation':corr,
            'log_energy_error':(torch.log(ep+eps)-torch.log(ey+eps)).square()}


def assert_same_latitude_nodes(actual_degrees: Tensor, expected_degrees: Tensor,
                               atol: float = 1e-5) -> None:
    if actual_degrees.shape != expected_degrees.shape or not torch.allclose(
        actual_degrees.double(),expected_degrees.double(),atol=atol,rtol=0):
        raise ValueError('Grid/SHT node mismatch. Regrid the analysis path, not the checkpoint input.')
