"""Frozen direction/rho parameterization helpers."""

from __future__ import annotations

import numpy as np


def signed_rho_grid(base: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0), maximum: float = 4.0,
                    factor: float = 2.0) -> tuple[float, ...]:
    values = list(float(x) for x in base)
    if not values or any(x <= 0 for x in values) or factor <= 1:
        raise ValueError("invalid rho grid")
    current = max(values)
    while current < maximum:
        current = min(maximum, current * factor)
        if current not in values:
            values.append(current)
    return tuple(values)


def epsilon_for_rho(rho: float) -> float:
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive and finite")
    return float(rho) / 4.0


def direction_quality(matrix: np.ndarray, *, energy_floor: float = 0.8, cosine_floor: float = 0.1) -> dict:
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or a.shape[1] == 0:
        raise ValueError("invalid direction matrix")
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    energy = np.cumsum(s * s) / max(float(np.sum(s * s)), 1e-12)
    gram = u.T @ u
    offdiag = gram - np.eye(gram.shape[0])
    return {"status": "OBSERVED", "rank": int(a.shape[1]), "svd_energy": float(energy[min(a.shape[1] - 1, len(energy) - 1)]),
            "max_abs_cosine": float(np.max(np.abs(offdiag))),
            "passes": bool(energy[min(a.shape[1] - 1, len(energy) - 1)] >= energy_floor and
                           np.max(np.abs(offdiag)) <= 1 - cosine_floor)}
