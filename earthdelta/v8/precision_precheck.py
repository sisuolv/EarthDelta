"""Precision guard construction from already exposed legacy endpoint data."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .io import json_hash, sha256_file


def cell_tolerance(h_pred_pct: float) -> float:
    h = float(h_pred_pct)
    if not np.isfinite(h) or h < 0:
        raise ValueError("invalid predicted half-width")
    return max(0.5, h + 0.25)


def compute_cell_tolerances(halfwidths: Mapping[str, float]) -> dict[str, float]:
    return {str(k): cell_tolerance(v) for k, v in sorted(halfwidths.items())}


def receipt(*, source_paths: list[str], halfwidths: Mapping[str, float], access_ledger: list[dict]) -> dict:
    hashes = {}
    for path in source_paths:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(path)
        hashes[str(p)] = sha256_file(p)
    ledger_hash = json_hash(access_ledger)
    return {"status": "OBSERVED", "per_cell_tol": compute_cell_tolerances(halfwidths),
            "source_hash": json_hash(hashes), "source_files": hashes,
            "access_ledger_hash": ledger_hash, "access_ledger": access_ledger}


def load_npz_halfwidths(path: str | Path) -> dict[str, float]:
    """Load only legacy endpoint statistics; callers must record this access."""
    with np.load(path, allow_pickle=False) as data:
        out = {}
        for key in data.files:
            arr = np.asarray(data[key])
            if arr.size == 1 and np.isfinite(arr).all():
                out[key] = float(arr.reshape(-1)[0])
        if not out:
            raise ValueError("legacy endpoint artifact has no scalar half-widths")
        return out


def _block_ids_from_seconds(seconds: np.ndarray, block_days: int = 7) -> np.ndarray:
    if block_days <= 0:
        raise ValueError("block_days must be positive")
    return np.floor((seconds - float(np.min(seconds))) / (block_days * 86400.0)).astype(np.int64)


def formal_legacy_precheck(npz_path: str | Path, *, issue_metadata_root: str | Path,
                           normalize_std_path: str | Path, draws: int = 10000,
                           seed: int = 0) -> dict:
    """Build a precision receipt from the sealed, already exposed 2019H2 table.

    The source contains issue-level MSE rather than model predictions.  We use
    it only to estimate the predeclared guard half-width; it is never reported
    as a new forecast skill result.  Issue metadata are read from the existing
    endpoint receipts, and no weather store is opened.
    """
    if draws < 10000:
        raise ValueError("bootstrap draws below signed minimum")
    npz_path = Path(npz_path)
    root = Path(issue_metadata_root)
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"mse", "ids", "chans", "std"}
        if not required.issubset(data.files):
            raise ValueError("legacy MSE table is missing identity fields")
        mse = np.asarray(data["mse"], dtype=np.float64)
        ids = [str(x) for x in np.asarray(data["ids"]).tolist()]
        channels = [str(x) for x in np.asarray(data["chans"]).tolist()]
        std = np.asarray(data["std"], dtype=np.float64)
    if mse.ndim != 4 or mse.shape[0] != len(ids) or mse.shape[-1] != len(channels) or mse.shape[1:] != (6, 3, 69):
        raise ValueError("legacy MSE shape is not the signed 112 x 6 x 3 x 69 layout")
    if len(set(ids)) != len(ids) or not np.isfinite(mse).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("legacy MSE identity or finite-value check failed")
    expected_names = ("geopotential_500", "temperature_850", "2m_temperature",
                      "mean_sea_level_pressure", "u_component_of_wind_850", "specific_humidity_700")
    positions = []
    for name in expected_names:
        if name not in channels:
            raise ValueError(f"missing signed channel: {name}")
        positions.append(channels.index(name))

    # Match every id to an existing endpoint metadata receipt without opening
    # its array payload.  This also makes the block chronology auditable.
    meta = {}
    for issue_json in root.glob("s*/issue_*/issue.json"):
        import json
        with open(issue_json, "r", encoding="utf-8") as f:
            row = json.load(f)
        if row.get("issue_id") in ids:
            meta[str(row["issue_id"])] = {"issue_time": int(row["issue_time"]),
                                           "shape": row.get("files", {}).get("truth", {}).get("shape")}
    if set(meta) != set(ids):
        raise ValueError("legacy issue metadata do not cover the MSE table")
    times = np.asarray([meta[x]["issue_time"] for x in ids], dtype=np.int64)
    if np.any(np.diff(times[np.argsort(times)]) <= 0):
        raise ValueError("legacy issue chronology is not unique")
    shapes = {tuple(meta[x]["shape"] or ()) for x in ids}
    if shapes != {(3, 69, 128, 256)}:
        raise ValueError("legacy endpoint shape identity mismatch")

    # Issue-level F0 MSE for the six named variables, with the three exposed
    # leads.  Bootstrap whole seven-day blocks and use a shared max-|t| radius.
    cell = mse[:, 0, :, :][:, :, positions]
    blocks = _block_ids_from_seconds(times, 7)
    unique = np.unique(blocks)
    members = [np.flatnonzero(blocks == block) for block in unique]
    rng = np.random.default_rng(seed)
    center = np.sqrt(np.mean(cell, axis=0))
    samples = np.empty((draws, 18), dtype=np.float64)
    for i in range(draws):
        chosen = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[j] for j in chosen])
        samples[i] = np.sqrt(np.mean(cell[indices], axis=0)).reshape(-1)
    centered = samples - center.reshape(1, -1)
    scale = np.maximum(np.abs(center.reshape(-1)), 1e-12)
    radius = np.max(np.abs(centered / scale), axis=1) * 100.0
    shared_radius = float(np.quantile(radius, 0.995, method="higher"))
    half72 = {}
    for var_i, name in enumerate(("Z500", "T850", "T2m", "MSLP", "U850", "Q700")):
        half72[f"{name}@6h"] = shared_radius
        half72[f"{name}@24h"] = shared_radius
        half72[f"{name}@72h"] = shared_radius
        half72[f"{name}@120h"] = 1.5 * shared_radius
    tolerances = {key: cell_tolerance(value) for key, value in half72.items()}
    lat = np.linspace(-89.296875, 89.296875, 128, dtype=np.float64)
    lon = np.arange(256, dtype=np.float64) * (360.0 / 256.0)
    coordinate_hash = json_hash({"lat": lat.tolist(), "lon": lon.tolist(), "shape": [128, 256]})
    channel_hash = json_hash(channels)
    std_hash = sha256_file(normalize_std_path)
    source_hash = json_hash({"npz": sha256_file(npz_path), "normalize_std": std_hash,
                             "channel_hash": channel_hash, "coordinate_hash": coordinate_hash,
                             "issue_ids": json_hash(ids)})
    ledger = [{"path": str(npz_path), "year": 2019, "array_payload": True, "purpose": "sealed endpoint MSE"},
              {"path": str(normalize_std_path), "year": 2019, "array_payload": False, "purpose": "identity only"},
              {"path": str(root), "year": 2019, "array_payload": False, "purpose": "issue metadata"}]
    return {"status": "OBSERVED", "B": draws, "block_days": 7, "seed": seed,
            "per_cell_tol": tolerances, "h_pred_pct": half72,
            "halfwidth_72h_pct": shared_radius, "halfwidth_120h_pct": 1.5 * shared_radius,
            "guard": "tol=max(0.5%,h_pred+0.25%)",
            "coordinate_hash": coordinate_hash, "channel_hash": channel_hash,
            "normalize_std_sha256": std_hash, "source_hash": source_hash,
            "source_files": {str(npz_path): sha256_file(npz_path), str(normalize_std_path): std_hash},
            "n_issues": len(ids), "issue_ids_hash": json_hash(ids),
            "access_ledger": ledger, "access_ledger_hash": json_hash(ledger)}
