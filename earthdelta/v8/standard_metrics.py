"""Canonical WeatherBench-style metrics for the v8 contract.

Arrays are intentionally accepted only by explicit callers.  The metric code
does not mask invalid values: a bad forecast or truth array is a hard error.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .io import read_json, sha256_file


class MetricError(ValueError):
    pass


@dataclass(frozen=True)
class MetricContract:
    variables: tuple[dict, ...]
    leads_hours: tuple[int, ...]
    bootstrap_draws: int
    block_days_primary: int
    block_days_sensitivity: int
    primary_metrics: tuple[str, ...]
    coordinate_check: str
    normalize_std_sha256: str

    @classmethod
    def from_json(cls, path: str | Path) -> "MetricContract":
        d = read_json(path)
        vars_ = tuple(d["variables"])
        if len(vars_) != 6 or tuple(d["leads_hours"]) != (6, 24, 72, 120):
            raise MetricError("contract must contain six variables and four signed leads")
        if int(d["uncertainty"]["B"]) < 10000:
            raise MetricError("bootstrap count below signed minimum")
        for v in vars_:
            if not v.get("source_name") or not v.get("unit") or not isinstance(v.get("channel_index"), int):
                raise MetricError("variable identity is incomplete")
        return cls(vars_, tuple(d["leads_hours"]), int(d["uncertainty"]["B"]),
                   int(d["uncertainty"]["block_days_primary"]), int(d["uncertainty"]["block_days_sensitivity"]),
                   tuple(d["primary_metrics"]), d["coordinate_check"], d["normalize_std"]["sha256"])


def validate_coordinates(actual: Mapping[str, object], expected: Mapping[str, object]) -> None:
    """Compare coordinate/name/level payloads exactly, including array bytes."""
    if set(actual) != set(expected):
        raise MetricError("coordinate identity keys differ")
    for key in expected:
        a, b = actual[key], expected[key]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                raise MetricError(f"coordinate payload differs: {key}")
        elif a != b:
            raise MetricError(f"coordinate payload differs: {key}")


def _weights(latitude: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    lat = np.asarray(latitude, dtype=np.float64)
    if lat.ndim != 1 or len(lat) != shape[0] or not np.isfinite(lat).all():
        raise MetricError("latitude coordinate is invalid")
    w = np.cos(np.deg2rad(lat))[:, None]
    if np.any(w <= 0):
        raise MetricError("latitude weights must be positive")
    w = np.broadcast_to(w, shape).astype(np.float64, copy=False)
    return w / w.sum()


def issue_mse(forecast: np.ndarray, truth: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    f, y = np.asarray(forecast), np.asarray(truth)
    if f.shape != y.shape or f.ndim != 4:
        raise MetricError("expected [issue, lead, lat, lon] arrays with matching shape")
    if not np.isfinite(f).all() or not np.isfinite(y).all():
        raise MetricError("nonfinite forecast or truth")
    w = _weights(np.asarray(latitude), (f.shape[-2], f.shape[-1]))
    return np.sum((f - y) ** 2 * w[None, None, :, :], axis=(-2, -1))


def pooled_rmse(forecast: np.ndarray, truth: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Pool issue MSE first, then take one square root per lead."""
    return np.sqrt(issue_mse(forecast, truth, latitude).mean(axis=0))


def relative_rmse(forecast: np.ndarray, truth: np.ndarray, baseline: np.ndarray,
                  latitude: np.ndarray) -> np.ndarray:
    f = pooled_rmse(forecast, truth, latitude)
    b = pooled_rmse(baseline, truth, latitude)
    if np.any(b <= 0):
        raise MetricError("baseline RMSE must be positive")
    return (b - f) / b * 100.0


def _block_ids(issue_times: Sequence[str], block_days: int) -> np.ndarray:
    from datetime import datetime, timezone
    days = []
    for value in issue_times:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        days.append((dt.date() - datetime(1970, 1, 1).date()).days // block_days)
    return np.asarray(days, dtype=np.int64)


def paired_bootstrap(forecast: np.ndarray, baseline: np.ndarray, truth: np.ndarray, latitude: np.ndarray,
                     issue_times: Sequence[str], *, draws: int = 10000, block_days: int = 7,
                     seed: int = 0) -> dict:
    if draws < 10000:
        raise MetricError("bootstrap draws below signed minimum")
    f, b, y = map(np.asarray, (forecast, baseline, truth))
    if f.shape != b.shape or f.shape != y.shape or f.ndim != 4:
        raise MetricError("paired arrays have incompatible shapes")
    if len(issue_times) != f.shape[0]:
        raise MetricError("issue time count mismatch")
    # A block bootstrap resamples whole chronological blocks, preserving pairing.
    blocks = _block_ids(issue_times, block_days)
    unique = np.unique(blocks)
    members = [np.flatnonzero(blocks == bid) for bid in unique]
    rng = np.random.default_rng(seed)
    values = np.empty((draws, f.shape[1]), dtype=np.float64)
    for i in range(draws):
        chosen = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[j] for j in chosen])
        values[i] = relative_rmse(f[idx], y[idx], b[idx], latitude)
    center = relative_rmse(f, y, b, latitude)
    centered = values - values.mean(axis=0, keepdims=True)
    radius = np.max(np.abs(centered), axis=1)
    halfwidth = np.quantile(radius, 0.995, method="higher")
    return {"estimate_pct": center.tolist(), "simultaneous_halfwidth_pct": float(halfwidth),
            "draws": draws, "block_days": block_days, "seed": seed}


def l69_loss(prediction: np.ndarray, target: np.ndarray, std: np.ndarray) -> float:
    p, t, s = map(np.asarray, (prediction, target, std))
    if p.shape != t.shape or p.shape[-1] != 69 or s.shape[-1] != 69:
        raise MetricError("L69 shape mismatch")
    if not np.isfinite(p).all() or not np.isfinite(t).all() or not np.isfinite(s).all() or np.any(s <= 0):
        raise MetricError("invalid L69 input")
    z = (p - t) / s
    return float(np.mean(z * z))


def l6_loss(predictions: Mapping[str, np.ndarray], truths: Mapping[str, np.ndarray],
            fit_f0_rmse: Mapping[str, float], variables: Sequence[str], leads: Sequence[int]) -> float:
    terms = []
    for name in variables:
        for lead in leads:
            key = f"{name}@{lead}h"
            if key not in predictions or key not in truths or key not in fit_f0_rmse:
                raise MetricError(f"missing L6 key: {key}")
            f = np.asarray(predictions[key]); y = np.asarray(truths[key])
            if f.shape != y.shape or not np.isfinite(f).all() or not np.isfinite(y).all():
                raise MetricError(f"invalid L6 key: {key}")
            denom = float(fit_f0_rmse[key]) ** 2
            if denom <= 0 or not np.isfinite(denom):
                raise MetricError(f"invalid L6 denominator: {key}")
            terms.append(float(np.mean((f - y) ** 2)) / denom)
    return float(np.mean(terms))
