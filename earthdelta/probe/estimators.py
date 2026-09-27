"""Frozen estimands used by the menu probe."""
from __future__ import annotations

import numpy as np


class EstimatorError(ValueError):
    pass


def _check_values(values: np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 3 or not np.isfinite(a).all():
        raise EstimatorError("values must be finite [issue, arm, cell]")
    return a


def per_issue_oracle(values: np.ndarray) -> dict:
    """Choose the lowest L6 arm per issue; arm 0 must be F0."""
    a = _check_values(values)
    selected = np.argmin(a.mean(axis=2), axis=1)
    chosen = a[np.arange(a.shape[0]), selected]
    f0 = a[:, 0]
    improvement = 100.0 * (1.0 - chosen / f0)
    return {"selected_arm": selected.tolist(), "mean_improvement_pct": improvement.mean(axis=0).tolist(),
            "selected_f0_fraction": float(np.mean(selected == 0)),
            "pooled_l6": float(chosen.mean()), "f0_pooled_l6": float(f0[:, :].mean())}


def static_best_arm(values: np.ndarray) -> dict:
    """Choose one arm by pooled L6 and report paired cell improvements."""
    a = _check_values(values)
    arm_scores = a.mean(axis=(0, 2))
    arm = int(np.argmin(arm_scores))
    improvement = 100.0 * (1.0 - a[:, arm] / a[:, 0])
    return {"arm": arm, "arm_scores": arm_scores.tolist(),
            "improvement_pct": improvement.mean(axis=0).tolist(),
            "pooled_l6": float(a[:, arm].mean()), "f0_pooled_l6": float(a[:, 0].mean())}


def ttt_oracle(values: np.ndarray) -> dict:
    """Truth-informed per-issue oracle; label is diagnostic only."""
    result = per_issue_oracle(values)
    result["label"] = "diagnostic_truth_informed_upper_bound"
    return result
