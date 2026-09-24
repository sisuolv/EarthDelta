"""Export EarthDelta forecasts to the official WeatherBench forecast layout.

Layout produced (identical naming to the official WB2/WB-X forecast stores):

* data variables named by source variable (``2m_temperature``,
  ``geopotential``, ...); pressure-level variables carry an int ``level`` (hPa)
  dimension, single-level variables do not;
* dims ``time`` (forecast ISSUE/initialization time), ``prediction_timedelta``
  (lead time, ``timedelta64[ns]``), [``level``], ``latitude``, ``longitude``;
* physical units exactly as in the ERA5 truth store (K, m/s, Pa, m^2/s^2,
  kg/kg) -- the local truth zarr is already in raw physical units, and model
  output is brought there by the official inverse normalization.

With this naming both official conventions work unmodified:
``weatherbench2.schema.apply_time_conventions`` renames
``prediction_timedelta -> lead_time`` (and ``time -> init_time`` for
``by_init=True``), and WeatherBench-X's ``xarray_loaders`` "ecmwf" renaming maps
``time -> init_time`` / ``prediction_timedelta -> lead_time`` for forecasts and
``time -> valid_time`` for truth.

IMPORTANT: ``controlled_rollout`` / ``fs_rollout_trajectory`` / the official
``forward_validation`` all return NORMALIZED states. ``trajectory_to_raw_fields``
always applies ``NormalizationContract.denormalize`` (the official inverse
transform) and then checks physical plausibility, so a forgotten or doubled
denormalization fails loudly instead of producing a plausible-looking RMSE.

No rollout logic lives here: F0/Fs/policy predictions come from the existing
``fs_rollout_trajectory`` / ``controlled_rollout`` (``rollout_forecast_record``
is a thin wrapper that calls the former) and are only converted here.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..data.pull_wb2 import (
    CANONICAL_VARIABLES,
    PRESSURE_LEVELS,
    PRESSURE_VARS,
    SINGLE_LEVEL_VARS,
    grid_hash as stormer_grid_hash,
    variable_order_hash,
)
from . import STORMER_EXPORT_PREFIX

EXPORT_SCHEMA = "ed-wbx-forecast/1"
TRUTH_SCHEMA = "ed-wbx-truth/1"
MANIFEST_NAME = "earthdelta_export_manifest.json"
INTERVAL_HOURS = 6
LAT_NAME = "latitude"
LON_NAME = "longitude"
LEVEL_NAME = "level"

#: Global-mean plausibility windows for physical (denormalized) fields.
#: Deliberately wide: they exist to catch "still normalized" (means ~0) and
#: "denormalized twice" (means ~ std*raw), not to judge forecast quality.
_PLAUSIBLE_GLOBAL_MEANS: Dict[str, Tuple[float, float]] = {
    "2m_temperature": (200.0, 320.0),             # K
    "mean_sea_level_pressure": (9.5e4, 1.07e5),   # Pa
    "geopotential_500": (4.8e4, 6.0e4),           # m^2/s^2
    "temperature_850": (230.0, 310.0),            # K
    "specific_humidity_1000": (1e-4, 3e-2),       # kg/kg
}


class ExportContractError(ValueError):
    """A field, coordinate or record violates the export contract."""


class TruthDataGap(ValueError):
    """The truth store has missing (NaN) values or missing times."""


def _xr():
    import xarray as xr  # lazy: keep `earthdelta.wbx` importable without xarray
    return xr


# =============================================================================
# Channel <-> (variable, level)
# =============================================================================

def check_channel_order(channel_names: Sequence[str]) -> str:
    """Require the exact canonical 69-channel order; return its hash.

    The hash is the same one ``pull_wb2`` stamps on every truth store
    (``variable_order_hash`` attr), so forecast and truth are provably in the
    same channel convention.
    """
    names = [str(c) for c in channel_names]
    if len(names) != len(CANONICAL_VARIABLES):
        raise ExportContractError(
            f"expected {len(CANONICAL_VARIABLES)} channels, got {len(names)}")
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    expected = variable_order_hash()
    if digest != expected:
        first_bad = next(i for i, (a, b) in enumerate(zip(names, CANONICAL_VARIABLES)) if a != b)
        raise ExportContractError(
            f"channel order hash {digest[:16]} != canonical {expected[:16]}; first "
            f"difference at index {first_bad}: {names[first_bad]!r} vs "
            f"{CANONICAL_VARIABLES[first_bad]!r}")
    return digest


def _single_index(var: str) -> int:
    return CANONICAL_VARIABLES.index(var)


def _level_indices(var: str) -> List[int]:
    return [CANONICAL_VARIABLES.index(f"{var}_{lvl}") for lvl in PRESSURE_LEVELS]


def _check_grid(lat: np.ndarray, lon: np.ndarray, nlat: int, nlon: int) -> None:
    if lat.ndim != 1 or lon.ndim != 1 or lat.size != nlat or lon.size != nlon:
        raise ExportContractError(
            f"lat/lon sizes ({lat.shape}, {lon.shape}) do not match field grid ({nlat}, {nlon})")
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        raise ExportContractError("latitude and longitude must be strictly increasing")
    if lat.min() < -90 or lat.max() > 90 or lon.min() < 0 or lon.max() >= 360:
        raise ExportContractError("latitude must be in [-90,90] and longitude in [0,360)")


def unflatten_channels(
    data: np.ndarray,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    leading: Sequence[Tuple[str, np.ndarray]] = (),
    channel_names: Sequence[str] = tuple(CANONICAL_VARIABLES),
    attrs: Optional[Mapping[str, Any]] = None,
):
    """``[..., 69, lat, lon]`` array -> per-variable ``xr.Dataset`` with ``level``.

    Args:
        data: array whose last three axes are (channel, lat, lon).
        lat, lon: 1-D increasing coordinates in degrees.
        leading: ``(dim_name, coord_values)`` for every leading axis, in order.
        channel_names: channel labels of ``data``; must be the canonical order.
        attrs: extra dataset attrs.

    No arithmetic is applied, so ``flatten_channels(unflatten_channels(x))`` is
    bitwise ``x``.
    """
    xr = _xr()
    data = np.asarray(data)
    if data.ndim < 3:
        raise ExportContractError(f"data must be [..., C, lat, lon], got shape {data.shape}")
    order_hash = check_channel_order(channel_names)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    _check_grid(lat, lon, data.shape[-2], data.shape[-1])
    if len(leading) != data.ndim - 3:
        raise ExportContractError(
            f"{data.ndim - 3} leading axes but {len(leading)} leading coords given")
    lead_dims: List[str] = []
    coords: Dict[str, Any] = {LAT_NAME: lat, LON_NAME: lon}
    for axis, (name, values) in enumerate(leading):
        values = np.asarray(values)
        if values.ndim != 1 or values.size != data.shape[axis]:
            raise ExportContractError(
                f"leading coord {name!r} has shape {values.shape}, axis {axis} has size {data.shape[axis]}")
        lead_dims.append(name)
        coords[name] = values
    coords[LEVEL_NAME] = np.asarray(PRESSURE_LEVELS, dtype=np.int64)
    data_vars = {}
    for var in SINGLE_LEVEL_VARS:
        arr = np.take(data, _single_index(var), axis=-3)
        data_vars[var] = (tuple(lead_dims) + (LAT_NAME, LON_NAME), arr)
    for var in PRESSURE_VARS:
        arr = np.take(data, _level_indices(var), axis=-3)
        data_vars[var] = (tuple(lead_dims) + (LEVEL_NAME, LAT_NAME, LON_NAME), arr)
    ds_attrs = {"variable_order_hash": order_hash, "channel_convention": "stormer69"}
    if attrs:
        ds_attrs.update(attrs)
    return xr.Dataset(data_vars, coords=coords, attrs=ds_attrs)


def flatten_channels(ds, *, leading_dims: Sequence[str] = ()) -> np.ndarray:
    """Inverse of ``unflatten_channels``: Dataset -> ``[*leading, 69, lat, lon]``."""
    lead = tuple(leading_dims)
    if LEVEL_NAME in ds.coords:
        levels = [int(v) for v in np.asarray(ds[LEVEL_NAME].values)]
        if levels != list(PRESSURE_LEVELS):
            raise ExportContractError(
                f"level coordinate {levels} != canonical {list(PRESSURE_LEVELS)}")
    missing = [v for v in SINGLE_LEVEL_VARS + PRESSURE_VARS if v not in ds.data_vars]
    if missing:
        raise ExportContractError(f"dataset is missing variables {missing}")
    planes = []
    for name in CANONICAL_VARIABLES:
        if name in SINGLE_LEVEL_VARS:
            da = ds[name]
            expected = set(lead) | {LAT_NAME, LON_NAME}
        else:
            var, lvl = name.rsplit("_", 1)
            da = ds[var].sel({LEVEL_NAME: int(lvl)})
            expected = set(lead) | {LAT_NAME, LON_NAME}
        if set(da.dims) != expected:
            raise ExportContractError(
                f"{name}: dims {da.dims} != expected {sorted(expected)}")
        planes.append(np.asarray(da.transpose(*lead, LAT_NAME, LON_NAME).values))
    return np.stack(planes, axis=-3)


# =============================================================================
# Normalized rollout -> physical fields
# =============================================================================

def check_physical_plausibility(raw: np.ndarray, channel_names: Sequence[str] = tuple(CANONICAL_VARIABLES)) -> Dict[str, float]:
    """Fail if denormalized fields do not look like physical ERA5 values.

    ``raw`` is ``[..., C, lat, lon]``. Returns the checked global means.
    """
    raw = np.asarray(raw)
    names = list(channel_names)
    out: Dict[str, float] = {}
    if not np.all(np.isfinite(raw)):
        raise ExportContractError("raw fields contain non-finite values")
    for name, (lo, hi) in _PLAUSIBLE_GLOBAL_MEANS.items():
        if name not in names:
            continue
        c = names.index(name)
        mean = float(np.mean(np.take(raw, c, axis=-3), dtype=np.float64))
        out[name] = mean
        if not (lo <= mean <= hi):
            raise ExportContractError(
                f"{name} global mean {mean:.6g} outside physical range [{lo}, {hi}]; "
                "the field is probably still normalized or was denormalized twice")
    return out


def _denormalize(normalization, state):
    import torch
    if not isinstance(state, torch.Tensor):
        raise ExportContractError("rollout states must be torch tensors (normalized space)")
    if state.ndim != 4:
        raise ExportContractError(f"state must be [B, V, H, W], got {tuple(state.shape)}")
    with torch.no_grad():
        raw = normalization.denormalize(state.detach().cpu())
    return raw.numpy()


def states_to_raw_fields(
    states: Mapping[int, Any],
    normalization,
    *,
    check_plausible: bool = True,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Normalized states keyed by rollout step -> ``[B, H, V, lat, lon]`` raw.

    ``states[step]`` is ``[B, V, lat, lon]`` in normalized space (e.g. the
    official ``forward_validation`` output for that many steps). Returns the
    raw fields ordered by step and the sorted step tuple.
    """
    steps = tuple(sorted(int(s) for s in states))
    if not steps or steps[0] <= 0:
        raise ExportContractError(f"lead steps must be positive, got {steps}")
    raws = [_denormalize(normalization, states[s]) for s in steps]
    shapes = {r.shape for r in raws}
    if len(shapes) != 1:
        raise ExportContractError(f"inconsistent state shapes {shapes}")
    raw = np.stack(raws, axis=1)
    if raw.shape[2] != len(normalization.variables):
        raise ExportContractError("state channel count != normalization variables")
    check_channel_order(normalization.variables)
    if check_plausible:
        check_physical_plausibility(raw)
    return raw, steps


def trajectory_to_raw_fields(
    trajectory,
    normalization,
    lead_steps: Sequence[int],
    *,
    check_plausible: bool = True,
) -> np.ndarray:
    """``controlled_rollout`` trajectory ``[B, T+1, V, H, W]`` -> raw ``[B, H, V, lat, lon]``.

    Step 0 of the trajectory is the (normalized) initial condition and is not
    a forecast; ``lead_steps`` must be positive, unique and within the rollout.
    """
    if trajectory.ndim != 5:
        raise ExportContractError(
            f"trajectory must be [B, T+1, V, H, W], got {tuple(trajectory.shape)}")
    steps = [int(s) for s in lead_steps]
    if not steps or len(set(steps)) != len(steps) or steps != sorted(steps):
        raise ExportContractError(f"lead_steps must be unique and increasing, got {steps}")
    if steps[0] <= 0 or steps[-1] >= trajectory.shape[1]:
        raise ExportContractError(
            f"lead_steps {steps} outside rollout steps 1..{trajectory.shape[1] - 1}")
    raw, _ = states_to_raw_fields(
        {s: trajectory[:, s] for s in steps}, normalization, check_plausible=check_plausible)
    return raw


# =============================================================================
# Forecast records and datasets
# =============================================================================

def _as_datetime64(value) -> np.datetime64:
    """Any timestamp -> naive UTC ``datetime64[ns]`` (tz-aware input is converted)."""
    import pandas as pd
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts.to_datetime64(), "ns")


@dataclass(frozen=True)
class ForecastRecord:
    """One issued forecast in raw physical units, ready for export.

    Attributes:
        model: export model name; Stormer exports must be labelled
            ``Stormer-ps4-6h-path...`` (see ``STORMER_EXPORT_PREFIX``).
        issue_time: UTC initialization time (6-hourly grid).
        lead_hours: forecast lead times in hours, positive multiples of 6.
        fields: ``[len(lead_hours), 69, lat, lon]`` physical fields.
        issue_id: optional admitted-issue identifier (for audit binding).
        provenance: free-form JSON-able provenance (checkpoint, Fs digest, ...).
    """

    model: str
    issue_time: Any
    lead_hours: Tuple[int, ...]
    fields: np.ndarray
    issue_id: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "issue_time", _as_datetime64(self.issue_time))
        object.__setattr__(self, "lead_hours", tuple(int(h) for h in self.lead_hours))
        if not self.model or not isinstance(self.model, str):
            raise ExportContractError("model name must be a non-empty string")
        if "stormer" in self.model.lower() and not self.model.startswith(STORMER_EXPORT_PREFIX):
            raise ExportContractError(
                f"Stormer exports must be named {STORMER_EXPORT_PREFIX!r}...: EarthDelta "
                "runs only the 6h path, not the published 6/12/24h ensemble")
        since_midnight_ns = int(
            (self.issue_time - self.issue_time.astype("datetime64[D]")).astype("timedelta64[ns]").astype(np.int64))
        if since_midnight_ns % (INTERVAL_HOURS * 3600 * 10**9) != 0:
            raise ExportContractError(
                f"issue time {self.issue_time} is not on the {INTERVAL_HOURS}-hourly grid")
        leads = self.lead_hours
        if not leads or any(h <= 0 or h % INTERVAL_HOURS for h in leads):
            raise ExportContractError(
                f"lead hours must be positive multiples of {INTERVAL_HOURS}, got {leads}")
        if list(leads) != sorted(set(leads)):
            raise ExportContractError(f"lead hours must be unique and increasing, got {leads}")
        f = np.asarray(self.fields)
        if f.ndim != 4 or f.shape[0] != len(leads) or f.shape[1] != len(CANONICAL_VARIABLES):
            raise ExportContractError(
                f"fields must be [{len(leads)}, {len(CANONICAL_VARIABLES)}, lat, lon], got {f.shape}")
        if not np.issubdtype(f.dtype, np.floating) or not np.all(np.isfinite(f)):
            raise ExportContractError("fields must be finite floating point")
        object.__setattr__(self, "fields", f)


def record_from_trajectory(
    trajectory,
    normalization,
    *,
    issue_time: Any,
    lead_steps: Sequence[int],
    model: str,
    batch_index: int = 0,
    issue_id: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    interval_hours: int = INTERVAL_HOURS,
) -> ForecastRecord:
    """Any normalized rollout trajectory (F0, Fs or a policy's) -> ForecastRecord.

    Agnostic to how the trajectory was produced: F0/Fs via
    ``static_adapter.fs_rollout_trajectory``, a dynamic policy via
    ``controlled_rollout(..., return_trajectory=True)``. Denormalization is
    applied here, always.
    """
    if int(interval_hours) != INTERVAL_HOURS:
        raise ExportContractError(f"only the {INTERVAL_HOURS}h rollout path is supported")
    raw = trajectory_to_raw_fields(trajectory, normalization, lead_steps)
    return ForecastRecord(
        model=model, issue_time=issue_time,
        lead_hours=tuple(int(s) * INTERVAL_HOURS for s in lead_steps),
        fields=raw[batch_index], issue_id=issue_id, provenance=dict(provenance or {}))


def rollout_forecast_record(
    bridge,
    x_norm,
    *,
    issue_time: Any,
    lead_steps: Sequence[int],
    model: str,
    fs_adapters=None,
    target_blocks: Optional[Sequence[int]] = None,
    issue_id: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> ForecastRecord:
    """Thin F0/Fs wrapper: the EXISTING ``fs_rollout_trajectory`` + export.

    ``fs_adapters=None`` on an unmerged bridge is F0; Fs is either the always-on
    adapters or a bridge with Fs merged into the weights. No rollout logic is
    reimplemented here. ``x_norm`` is ``[1, V, lat, lon]`` normalized.
    """
    import torch
    from ..static_adapter import DEFAULT_TARGET_BLOCKS, fs_rollout_trajectory
    if x_norm.shape[0] != 1:
        raise ExportContractError("rollout_forecast_record exports one issue at a time")
    with torch.no_grad():
        trajectory = fs_rollout_trajectory(
            bridge, x_norm, list(bridge.variables), steps=max(int(s) for s in lead_steps),
            fs_adapters=fs_adapters,
            target_blocks=tuple(target_blocks) if target_blocks is not None else DEFAULT_TARGET_BLOCKS,
            interval_hours=INTERVAL_HOURS)
    return record_from_trajectory(
        trajectory, bridge.normalization, issue_time=issue_time, lead_steps=lead_steps,
        model=model, issue_id=issue_id, provenance=provenance)


def build_forecast_dataset(
    records: Sequence[ForecastRecord],
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    extra_attrs: Optional[Mapping[str, Any]] = None,
):
    """Stack records into one official-layout forecast Dataset.

    All records must share model, lead hours and grid; issue times must be
    unique. The result is sorted by issue time.
    """
    if not records:
        raise ExportContractError("no forecast records")
    models = {r.model for r in records}
    if len(models) != 1:
        raise ExportContractError(f"records mix models {sorted(models)}")
    leads = {r.lead_hours for r in records}
    if len(leads) != 1:
        raise ExportContractError(f"records have different lead hours {sorted(leads)}")
    shapes = {r.fields.shape for r in records}
    if len(shapes) != 1:
        raise ExportContractError(f"records have different field shapes {shapes}")
    order = sorted(range(len(records)), key=lambda i: records[i].issue_time)
    times = np.array([records[i].issue_time for i in order], dtype="datetime64[ns]")
    if len(np.unique(times)) != len(times):
        raise ExportContractError("duplicate issue times in forecast records")
    lead_hours = next(iter(leads))
    deltas = np.array([np.timedelta64(h, "h") for h in lead_hours]).astype("timedelta64[ns]")
    data = np.stack([records[i].fields for i in order], axis=0)
    attrs = {
        "earthdelta_export_schema": EXPORT_SCHEMA,
        "model": next(iter(models)),
        "lead_hours": json.dumps(list(lead_hours)),
        "interval_hours": INTERVAL_HOURS,
        "time_convention": "time=issue/init time (UTC); prediction_timedelta=lead",
        "units_convention": "raw physical units identical to the ERA5 truth store",
        "issue_ids": json.dumps([records[i].issue_id for i in order]),
        "provenance": json.dumps([dict(records[i].provenance) for i in order], sort_keys=True, default=str),
    }
    if extra_attrs:
        attrs.update({k: (v if isinstance(v, (str, int, float)) else json.dumps(v, default=str))
                      for k, v in extra_attrs.items()})
    return unflatten_channels(
        data, lat=lat, lon=lon,
        leading=(("time", times), ("prediction_timedelta", deltas)),
        attrs=attrs,
    )


def verify_issue_times(ds, admitted_issue_times: Sequence[Any]) -> None:
    """Decoded ``time`` coordinate must equal the admitted issue times exactly.

    Binds the exported store to the admission record: a time-zone slip, an
    off-by-one-step, or a missing/extra issue fails here.
    """
    decoded = np.sort(np.asarray(ds["time"].values).astype("datetime64[ns]"))
    admitted = np.sort(np.array([_as_datetime64(t) for t in admitted_issue_times], dtype="datetime64[ns]"))
    if decoded.shape != admitted.shape or not np.array_equal(decoded, admitted):
        extra = sorted(set(decoded.tolist()) - set(admitted.tolist()))
        missing = sorted(set(admitted.tolist()) - set(decoded.tolist()))
        raise ExportContractError(
            f"exported issue times do not match the admission record: "
            f"extra={[str(np.datetime64(x, 'ns')) for x in extra][:5]} "
            f"missing={[str(np.datetime64(x, 'ns')) for x in missing][:5]}")


def dataset_content_sha256(ds) -> str:
    """Order-independent digest of every variable and coordinate's bytes."""
    h = hashlib.sha256()
    for name in sorted(list(ds.data_vars) + list(ds.coords)):
        da = ds[name]
        values = np.ascontiguousarray(np.asarray(da.values))
        h.update(f"{name}|{','.join(map(str, da.dims))}|{values.dtype.str}|{values.shape}\n".encode())
        h.update(values.tobytes())
    return h.hexdigest()


def _datasets_bitwise_equal(a, b) -> bool:
    if set(a.data_vars) != set(b.data_vars) or set(a.coords) != set(b.coords):
        return False
    for name in list(a.data_vars) + list(a.coords):
        x, y = np.asarray(a[name].values), np.asarray(b[name].values)
        if x.dtype != y.dtype or x.shape != y.shape or tuple(a[name].dims) != tuple(b[name].dims):
            return False
        if np.issubdtype(x.dtype, np.floating):
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def write_forecast_zarr(
    ds,
    path: Union[str, Path],
    *,
    overwrite: bool = False,
    manifest_extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write ``ds`` to a zarr store with atomic publish; return the manifest.

    Mirrors ``export_upstream_reference.publish_directory_atomically``: the
    store is written to a sibling staging directory, re-opened and verified
    bitwise against ``ds``, a manifest (with a content digest) is added, and
    only then is the directory renamed into place. A crash leaves only a
    ``.staging-*`` directory that no consumer reads.
    """
    xr = _xr()
    final = Path(path)
    if final.exists() and not overwrite:
        raise FileExistsError(f"{final} already exists (pass overwrite=True to replace)")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".staging-{final.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        encoding = {}
        for name, da in ds.data_vars.items():
            chunks = tuple(1 if d in ("time",) else s for d, s in zip(da.dims, da.shape))
            encoding[name] = {"chunks": chunks}
        ds.to_zarr(staging, mode="w", consolidated=True, encoding=encoding)
        reread = xr.open_zarr(staging, consolidated=True).load()
        if not _datasets_bitwise_equal(ds, reread):
            raise ExportContractError(f"zarr round-trip of {final.name} is not bitwise exact")
        manifest = {
            "schema": (ds.attrs.get("earthdelta_export_schema")
                       or ds.attrs.get("earthdelta_truth_schema") or EXPORT_SCHEMA),
            "path": str(final),
            "content_sha256": dataset_content_sha256(reread),
            "variables": sorted(ds.data_vars),
            "dims": {k: int(v) for k, v in ds.sizes.items()},
            "written_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        if manifest_extra:
            manifest.update(manifest_extra)
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
        staging = None
        return manifest
    finally:
        if staging is not None and Path(staging).exists():
            shutil.rmtree(staging, ignore_errors=True)


# =============================================================================
# Truth
# =============================================================================

def truth_from_store(
    store_path: Union[str, Path],
    valid_times: Sequence[Any],
    *,
    confirm_freeze_path: Optional[Union[str, Path]] = None,
    confirm_freeze_sha256: Optional[str] = None,
    allow_nan: bool = False,
):
    """Read ERA5 truth at exact ``valid_times`` from a local pull_wb2 store.

    Uses the same channel -> (variable, level) mapping as the forecasts. The
    store must carry the canonical ``variable_order_hash`` and (if stamped)
    Stormer ``grid_hash``. Every requested time must be present exactly (no
    nearest-neighbour matching). NaN gaps -- which real pulled years do have --
    raise ``TruthDataGap`` unless ``allow_nan``.

    Reading truth for a year is an exposure: the year must pass
    ``evaluate.assert_years_authorized`` (default: only already-exposed 2020).
    """
    xr = _xr()
    from .evaluate import assert_years_authorized  # local import: no cycle at import time

    times = np.array([_as_datetime64(t) for t in valid_times], dtype="datetime64[ns]")
    if times.size == 0:
        raise TruthDataGap("no valid times requested")
    years = sorted({int(str(t)[:4]) for t in times})
    gate = assert_years_authorized(
        years, confirm_freeze_path=confirm_freeze_path, confirm_freeze_sha256=confirm_freeze_sha256)
    store = xr.open_zarr(str(store_path))
    if "data" not in store or tuple(store["data"].dims) != ("time", "channel", "lat", "lon"):
        raise ExportContractError(f"{store_path} is not a pull_wb2 (time, channel, lat, lon) store")
    order_hash = check_channel_order([str(c) for c in store["channel"].values])
    stamped = store.attrs.get("variable_order_hash")
    if stamped is not None and stamped != order_hash:
        raise ExportContractError(f"store variable_order_hash {stamped} != canonical {order_hash}")
    lat = np.asarray(store["lat"].values, dtype=np.float64)
    lon = np.asarray(store["lon"].values, dtype=np.float64)
    if store.attrs.get("grid_hash") is not None and store.attrs["grid_hash"] != stormer_grid_hash():
        raise ExportContractError("store grid_hash is not the Stormer 128x256 grid")
    store_times = np.asarray(store["time"].values).astype("datetime64[ns]")
    missing = [str(t) for t in times[~np.isin(times, store_times)]]
    if missing:
        raise TruthDataGap(f"truth store lacks valid times {missing[:5]}")
    data = np.asarray(store["data"].sel(time=times).values)
    if not allow_nan and not np.all(np.isfinite(data)):
        bad = np.argwhere(~np.isfinite(data).all(axis=(2, 3)))
        pairs = [(str(times[i]), CANONICAL_VARIABLES[c]) for i, c in bad[:10]]
        raise TruthDataGap(f"truth has NaN/inf for (time, channel) e.g. {pairs}")
    attrs = {
        "earthdelta_truth_schema": TRUTH_SCHEMA,
        "source_store": str(store_path),
        "source_attrs": json.dumps({k: str(v) for k, v in store.attrs.items()}, sort_keys=True),
        "year_gate": json.dumps(gate, sort_keys=True),
    }
    return unflatten_channels(data, lat=lat, lon=lon, leading=(("time", times),), attrs=attrs)


__all__ = [
    "ExportContractError", "TruthDataGap", "ForecastRecord", "EXPORT_SCHEMA",
    "check_channel_order", "unflatten_channels", "flatten_channels",
    "check_physical_plausibility", "states_to_raw_fields", "trajectory_to_raw_fields",
    "record_from_trajectory", "rollout_forecast_record",
    "build_forecast_dataset", "verify_issue_times", "write_forecast_zarr",
    "dataset_content_sha256", "truth_from_store",
]
