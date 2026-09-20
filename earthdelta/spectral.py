"""Phase-aware scalar-coefficient diagnostics; NOT a spherical transform.

Obtain coefficients from a separately validated grid/normalization-aware SHT.
Half-spectrum multiplicity and band masks must be supplied by the caller.
No planar FFT is silently substituted for a global spherical calculation.

NOTE: The torch-harmonics bridge is OUT OF SCOPE for this module. That belongs
to the bridge module being developed separately. This module only provides
diagnostic functions for already-computed spherical harmonic coefficients.
"""
from __future__ import annotations
import torch
from torch import Tensor


def coefficient_diagnostics(pred: Tensor, truth: Tensor, weights: Tensor,
                            eps: float = 1e-10) -> dict[str, Tensor]:
    """Compute spectral diagnostics between predicted and true coefficients.

    Args:
        pred: Complex coefficient array (any shape, last dim is coefficients)
        truth: Complex coefficient array (same shape as pred)
        weights: Coefficient weights (broadcastable to pred shape)
        eps: Small value for numerical stability

    Returns:
        Dict with:
        - energy_pred: Energy of predicted coefficients
        - energy_truth: Energy of true coefficients
        - correlation: Spectral correlation
        - log_energy_error: Log energy error
    """
    if pred.shape != truth.shape or pred.ndim < 1 or not pred.is_complex() or not truth.is_complex():
        raise ValueError('Matching complex coefficient arrays required.')
    if eps <= 0:
        raise ValueError('eps must be positive.')
    w = torch.broadcast_to(weights.to(pred.real), pred.shape)
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all() or not torch.isfinite(w).all():
        raise ValueError('Nonfinite coefficients/weights.')
    if (w < 0).any() or (w.sum(-1) <= 0).any():
        raise ValueError('Invalid coefficient weights.')

    ep = (w * pred.abs().square()).sum(-1)
    ey = (w * truth.abs().square()).sum(-1)
    numerator = (w * (pred * truth.conj()).real).sum(-1)
    denom = torch.sqrt(ep * ey)
    corr = numerator / denom.clamp_min(eps)
    both_empty = (ep <= eps) & (ey <= eps)
    corr = torch.where(both_empty, torch.ones_like(corr), corr).clamp(-1, 1)

    return {
        'energy_pred': ep,
        'energy_truth': ey,
        'correlation': corr,
        'log_energy_error': (torch.log(ep + eps) - torch.log(ey + eps)).square()
    }


def assert_same_latitude_nodes(actual_degrees: Tensor, expected_degrees: Tensor,
                               atol: float = 1e-5) -> None:
    """Assert that two latitude node arrays are equivalent.

    This is important for validating SHT grid alignment. The torch-harmonics
    precompute_latitudes function returns colatitudes, not latitudes, so
    conversions must be validated.

    Args:
        actual_degrees: Actual latitude nodes in degrees
        expected_degrees: Expected latitude nodes in degrees
        atol: Absolute tolerance for comparison

    Raises:
        ValueError: If nodes do not match within tolerance
    """
    if actual_degrees.shape != expected_degrees.shape or not torch.allclose(
            actual_degrees.double(), expected_degrees.double(), atol=atol, rtol=0):
        raise ValueError('Grid/SHT node mismatch. Regrid the analysis path, not the checkpoint input.')


def band_energy(coefficients: Tensor, weights: Tensor, band_l_ranges: list[tuple[int, int]]) -> dict[str, Tensor]:
    """Compute energy in specified spherical harmonic bands.

    Args:
        coefficients: Complex [l_max+1, ...] or [..., l_max+1] coefficient array
        weights: Coefficient weights
        band_l_ranges: List of (l_min, l_max) tuples defining bands

    Returns:
        Dict mapping band names to energy values
    """
    raise NotImplementedError(
        "Band energy computation requires the torch-harmonics bridge, which is "
        "out of scope for this module. Use the bridge module once available."
    )


def single_mode_energy(coefficients: Tensor, l: int, m: int) -> Tensor:
    """Extract energy of a single spherical harmonic mode.

    Args:
        coefficients: Complex coefficient array in standard ordering
        l: Degree
        m: Order

    Returns:
        Energy of mode (l, m)
    """
    raise NotImplementedError(
        "Single mode extraction requires knowing the coefficient ordering, which "
        "depends on the torch-harmonics bridge. Use the bridge module once available."
    )
