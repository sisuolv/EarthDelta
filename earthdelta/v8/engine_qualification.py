"""Mechanical qualification for an optional no-gradient 5090 engine."""

from __future__ import annotations

import numpy as np


class EngineQualificationError(ValueError):
    pass


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def qualify_5090(*, h100_forecast: np.ndarray, spot_forecast: np.ndarray,
                 h100_l6: float, spot_l6: float, h100_delta_l6: float, spot_delta_l6: float,
                 h100_truth: np.ndarray, thresholds: dict | None = None) -> dict:
    th = thresholds or {"rms_ratio": 1e-3, "l6_relative": 1e-5, "paired_relative": 0.01}
    h, s, y = map(np.asarray, (h100_forecast, spot_forecast, h100_truth))
    if h.shape != s.shape or h.shape != y.shape or not np.isfinite(h).all() or not np.isfinite(s).all() or not np.isfinite(y).all():
        raise EngineQualificationError("invalid engine qualification arrays")
    denom = max(_rms(h - y), 1e-12)
    rms_ratio = _rms(s - h) / denom
    l6_rel = abs(float(spot_l6) - float(h100_l6)) / max(abs(float(h100_l6)), 1e-12)
    delta_rel = abs(float(spot_delta_l6) - float(h100_delta_l6)) / max(abs(float(h100_delta_l6)), 1e-12)
    passed = rms_ratio <= th["rms_ratio"] and l6_rel <= th["l6_relative"] and delta_rel <= th["paired_relative"]
    return {"status": "OBSERVED", "engine_id": "spot_5090", "qualified": bool(passed),
            "probe_metrics": {"rms_ratio": rms_ratio, "l6_relative": l6_rel, "paired_relative": delta_rel},
            "thresholds": th}


def enforce_same_engine(issue_engine_ids: dict[str, str]) -> None:
    ids = set(issue_engine_ids.values())
    if len(ids) > 1:
        raise EngineQualificationError("all candidates for one issue must use one engine")
