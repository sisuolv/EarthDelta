"""Admission contract for pilot data: what a sample must satisfy to be usable.

The pilot pull (`earthdelta.data.pull_wb2`) decides a store is usable from its
SHAPE alone -- `is_pilot_complete` checks the timestep count and the channel
count and nothing else. That is not an admission check. A store with the right
shape and the wrong contents is indistinguishable from a good one, and the
failure only shows up much later as a bad number with no provenance.

This module supplies the content-level admission checks the shape check does
not make, and `scripts/r2_pilot_preflight.py` is the entry point that runs them
against a real store. The four failure modes covered here are the ones that
have actually produced silently-wrong pilot data:

  1. MISSING HISTORY/TARGET ENDPOINTS. A sample at time index `i` needs
     `i - history_steps` and `i + target_steps` to BOTH exist in the store and
     to be spaced at the expected interval. A store that is "complete" by
     timestep count can still be too short to form the sample a caller asks
     for, and `.isel` on an out-of-range index either raises far downstream or,
     for negative indices, silently wraps to the other end of the record.

  2. CHANGED COORDINATE ORDER. The pull flips latitude to increasing before
     regridding, and stamps `grid_hash` into the store's attrs. Nothing ever
     rechecks that the coordinates on disk still agree with either that stamp
     or the canonical target grid. A store written under a different coordinate
     convention loads fine, has the right shape, and is upside down.

  3. NON-FINITE CONTENT. The store is created pre-filled with NaN and each
     source variable overwrites its own channels. Any channel that was never
     written stays NaN at full shape, which passes every existing check.

  4. STALE MARKER POINTING AT A NEW STORE. Resume markers
     (`.markers_<year>/<var>.done`) record only that a variable was pulled at
     some time, with no binding to the store it was pulled into. Delete and
     recreate the store and the markers survive, so the resumed pull skips
     every variable it believes is done and leaves those channels NaN in the
     NEW store. The marker's own recorded timestamp against the store's
     `created` attr is enough to detect this, and both sides already write it.

  5. CONTENT THAT PASSES EVERY STRUCTURAL CHECK AND IS STILL NOT DATA.
     `validate_loaded_sample` admits ONE loaded window. What a training or
     admission run actually consumes is a SUBSET of many windows, and nothing
     re-reads that subset value by value. `verify_content_subset` does: it
     walks the exact indices that will be read, in batches, and checks real
     values -- finite, non-degenerate (a spatially constant global field is not
     weather), and inside a physical band derived from the official Stormer
     normalization constants -- then binds the result into a certificate that
     carries the digest of the bytes it actually read together with the slice
     identity those bytes came from. A marker file or a timestep count cannot
     produce that certificate.

Scope note: these are admission checks for a sample or a store that a caller is
about to use. They are deliberately not a full-corpus scan and they do not
change the downloader. `verify_content_subset` in particular verifies the
subset it is handed and says so in its certificate; it makes no claim about
data it never read.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PilotContractViolation",
    "SliceIndexReport",
    "LoadedSampleReport",
    "ArtifactBindingReport",
    "PilotPreflightReport",
    "ContentCertificate",
    "DataRole",
    "validate_slice_index",
    "validate_loaded_sample",
    "assert_artifact_binding",
    "verify_content_subset",
    "physical_bounds_from_normalization",
    "assert_valid_data_role",
    "run_pilot_preflight",
    "DEFAULT_INTERVAL_HOURS",
    "EXPECTED_CHANNELS",
    "DEFAULT_SIGMA_BOUND",
    "DEFAULT_NORMALIZATION_DIR",
]


DEFAULT_INTERVAL_HOURS = 6
EXPECTED_CHANNELS = 69

#: Official Stormer normalization constants shipped with the repo. The physical
#: plausibility band used by `verify_content_subset` is derived from these, not
#: from hand-written per-variable limits.
DEFAULT_NORMALIZATION_DIR = (
    Path(__file__).resolve().parent.parent
    / "reference" / "stormer" / "normalization_constants"
)

#: Width, in official standard deviations, of the physical plausibility band.
#:
#: Calibrated against real on-disk ERA5 rather than chosen for roundness: over
#: 48 timesteps sampled across `data/era5_1p40625/2020.zarr`, the largest
#: |x - mean| / std seen on any of the 69 channels was 22.09
#: (specific_humidity_100, whose stratospheric std is tiny). A bound at 40 sits
#: comfortably outside the real distribution, so this check does not fire on
#: genuine extremes; it fires on the gross failures -- wrong units, a channel
#: holding another variable's values, a corrupted decode. Constant and all-zero
#: channels are caught by the degeneracy check instead, which is sharper than
#: any sigma band for that failure.
DEFAULT_SIGMA_BOUND = 40.0


# =============================================================================
# Frozen data roles
# =============================================================================

class DataRole(Enum):
    """The three roles a real pilot sample may be admitted for.

    These are frozen at admission, not chosen at consumption time, so that a
    sample cannot migrate between roles once it has been looked at:

    * ``bank_fit``    -- fits Fs / the expert bank.
    * ``policy_dev``  -- cheap-policy HPO and development decisions.
    * ``confirm``     -- the one-time-only final evaluation split. A sample
      admitted as ``confirm`` is spent the first time it is reported on.

    The admission layer only tags and carries the role; enforcing what each
    downstream consumer may read is the consumer's job, and it can filter on
    this tag because the tag travels with the admission record and with the
    content certificate.
    """

    BANK_FIT = "bank_fit"
    POLICY_DEV = "policy_dev"
    CONFIRM = "confirm"


#: Every legal role value, for callers that want a plain container.
DATA_ROLE_VALUES: Tuple[str, ...] = tuple(role.value for role in DataRole)


def assert_valid_data_role(value: Optional[Any]) -> Optional[str]:
    """Return the canonical role string, or raise on anything else.

    `None` is allowed and returned unchanged: a record that carries no role is
    honest about carrying no role. An unrecognised role is refused rather than
    passed through, because a typo'd role silently filters to the empty set at
    consumption time and looks like "no data" instead of "wrong tag".
    """
    if value is None:
        return None
    if isinstance(value, DataRole):
        return value.value
    text = str(value)
    if text not in DATA_ROLE_VALUES:
        raise PilotContractViolation(
            "DATA_ROLE_UNKNOWN",
            f"Unknown data_role {value!r}. The frozen roles are "
            f"{list(DATA_ROLE_VALUES)}.",
            {"data_role": text, "allowed": list(DATA_ROLE_VALUES)},
        )
    return text

#: Tolerance for comparing coordinate values. The coordinates are written as
#: float64 linspaces and read back through zarr/xarray, so exact equality is
#: the wrong test, but anything beyond rounding is a different grid.
COORD_ATOL = 1e-9


class PilotContractViolation(ValueError):
    """A pilot artifact or sample failed an admission check.

    Carries the machine-readable `code` alongside the message so callers can
    branch on the specific violation without parsing prose.
    """

    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.detail: Dict[str, Any] = dict(detail or {})


# =============================================================================
# Reports
# =============================================================================

@dataclass
class SliceIndexReport:
    """Which absolute time indices a validated sample request resolves to."""
    index: int
    history_steps: int
    target_steps: int
    history_index: int
    target_index: int
    n_timesteps: int
    interval_hours: Optional[int] = None


@dataclass
class LoadedSampleReport:
    """What was actually checked about a loaded sample's contents."""
    shape: Tuple[int, ...]
    dtype: str
    n_channels: int
    finite: bool
    coordinates_checked: bool = False
    grid_hash: Optional[str] = None


@dataclass
class ArtifactBindingReport:
    """How a store's markers and stamped identity relate to the store itself."""
    store_path: str
    created_utc: Optional[str]
    grid_hash: Optional[str]
    variable_order_hash: Optional[str]
    marker_dir: Optional[str] = None
    markers_checked: int = 0
    stale_markers: List[str] = field(default_factory=list)


@dataclass
class PilotPreflightReport:
    """Aggregate of every admission check run by the preflight entry point."""
    store_path: str
    passed: bool
    checks: List[Dict[str, Any]] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    content_certificate: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "store_path": self.store_path,
            "passed": self.passed,
            "checks": list(self.checks),
            "violations": list(self.violations),
        }
        if self.content_certificate is not None:
            payload["content_certificate"] = dict(self.content_certificate)
        return payload


# =============================================================================
# 1. Slice index admission
# =============================================================================

def validate_slice_index(
    index: int,
    n_timesteps: int,
    history_steps: int = 1,
    target_steps: int = 1,
    time_coords: Optional[Sequence[Any]] = None,
    interval_hours: Optional[int] = DEFAULT_INTERVAL_HOURS,
) -> SliceIndexReport:
    """Admit a sample request only if both of its endpoints really exist.

    A pilot sample spans `[index - history_steps, index + target_steps]`. Both
    ends must be inside the record, and -- when time coordinates are supplied --
    the spacing across that span must be the expected forecast interval, so a
    store with a gap in its time axis cannot masquerade as a contiguous window.

    Negative indices are rejected outright rather than interpreted: Python's
    wrap-around would silently resolve a missing history endpoint to a sample
    from the far end of the record.

    Args:
        index: Absolute time index of the sample's analysis step.
        n_timesteps: Number of timesteps in the store.
        history_steps: Steps of history the sample needs before `index`.
        target_steps: Steps of target the sample needs after `index`.
        time_coords: Optional time coordinate values, used to check spacing.
        interval_hours: Expected spacing between consecutive steps, or None to
            skip the spacing check.

    Returns:
        SliceIndexReport with the resolved absolute endpoints.

    Raises:
        PilotContractViolation: If any endpoint is missing or the spacing across
            the sample's span is not the expected interval.
    """
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        raise PilotContractViolation(
            "SLICE_INDEX_NOT_INTEGER",
            f"Sample index must be an integer, got {type(index).__name__}.",
            {"index": repr(index)},
        )
    index = int(index)
    n_timesteps = int(n_timesteps)
    history_steps = int(history_steps)
    target_steps = int(target_steps)

    if n_timesteps <= 0:
        raise PilotContractViolation(
            "STORE_EMPTY",
            f"Store reports {n_timesteps} timesteps; no sample can be formed.",
            {"n_timesteps": n_timesteps},
        )
    if history_steps < 0 or target_steps < 0:
        raise PilotContractViolation(
            "SLICE_SPAN_NEGATIVE",
            "history_steps and target_steps must be non-negative, got "
            f"history_steps={history_steps}, target_steps={target_steps}.",
            {"history_steps": history_steps, "target_steps": target_steps},
        )
    if index < 0:
        raise PilotContractViolation(
            "SLICE_INDEX_NEGATIVE",
            f"Sample index {index} is negative. Negative indices are refused "
            "rather than wrapped: wrapping would resolve a missing endpoint to "
            "an unrelated sample from the other end of the record.",
            {"index": index, "n_timesteps": n_timesteps},
        )
    if index >= n_timesteps:
        raise PilotContractViolation(
            "SLICE_INDEX_OUT_OF_RANGE",
            f"Sample index {index} is outside a store of {n_timesteps} timesteps.",
            {"index": index, "n_timesteps": n_timesteps},
        )

    history_index = index - history_steps
    target_index = index + target_steps

    if history_index < 0:
        raise PilotContractViolation(
            "HISTORY_ENDPOINT_MISSING",
            f"Sample at index {index} needs {history_steps} step(s) of history "
            f"(absolute index {history_index}), which is before the start of a "
            f"store holding {n_timesteps} timesteps.",
            {
                "index": index,
                "history_steps": history_steps,
                "history_index": history_index,
                "n_timesteps": n_timesteps,
            },
        )
    if target_index >= n_timesteps:
        raise PilotContractViolation(
            "TARGET_ENDPOINT_MISSING",
            f"Sample at index {index} needs {target_steps} step(s) of target "
            f"(absolute index {target_index}), which is past the end of a store "
            f"holding {n_timesteps} timesteps (last index {n_timesteps - 1}).",
            {
                "index": index,
                "target_steps": target_steps,
                "target_index": target_index,
                "n_timesteps": n_timesteps,
            },
        )

    if time_coords is not None:
        coords = np.asarray(time_coords)
        if coords.shape[0] != n_timesteps:
            raise PilotContractViolation(
                "TIME_COORD_LENGTH_MISMATCH",
                f"Time coordinate has {coords.shape[0]} entries but the store "
                f"reports {n_timesteps} timesteps.",
                {"n_coords": int(coords.shape[0]), "n_timesteps": n_timesteps},
            )
        if interval_hours is not None:
            _assert_uniform_interval(
                coords[history_index:target_index + 1],
                interval_hours,
                span=(history_index, target_index),
            )

    return SliceIndexReport(
        index=index,
        history_steps=history_steps,
        target_steps=target_steps,
        history_index=history_index,
        target_index=target_index,
        n_timesteps=n_timesteps,
        interval_hours=interval_hours,
    )


def _assert_uniform_interval(
    span_coords: np.ndarray, interval_hours: int, span: Tuple[int, int]
) -> None:
    """Refuse a sample whose span is not uniformly spaced at the interval."""
    if span_coords.shape[0] < 2:
        return

    values = np.asarray(span_coords)
    try:
        if np.issubdtype(values.dtype, np.datetime64):
            deltas = np.diff(values.astype("datetime64[s]").astype(np.int64)) / 3600.0
        elif np.issubdtype(values.dtype, np.number):
            deltas = np.diff(values.astype(np.float64))
        else:
            # Object arrays of datetimes (cftime and friends): fall back to a
            # generic difference and only check uniformity, not the unit.
            deltas = np.array([
                (values[i + 1] - values[i]).total_seconds() / 3600.0
                for i in range(values.shape[0] - 1)
            ], dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise PilotContractViolation(
            "TIME_COORD_UNREADABLE",
            f"Could not compute time spacing over span {span}: {exc}",
            {"span": list(span), "dtype": str(values.dtype)},
        ) from exc

    if not np.all(np.isfinite(deltas)):
        raise PilotContractViolation(
            "TIME_COORD_NONFINITE",
            f"Time spacing over span {span} contains non-finite values.",
            {"span": list(span)},
        )

    bad = np.flatnonzero(np.abs(deltas - float(interval_hours)) > 1e-6)
    if bad.size:
        raise PilotContractViolation(
            "TIME_SPACING_MISMATCH",
            f"Sample span {span} is not uniformly spaced at {interval_hours}h: "
            f"{bad.size} of {deltas.size} gap(s) differ, first offending gap is "
            f"{float(deltas[bad[0]])}h at offset {int(bad[0])}.",
            {
                "span": list(span),
                "expected_interval_hours": int(interval_hours),
                "offending_gaps": [float(deltas[i]) for i in bad[:8].tolist()],
            },
        )


# =============================================================================
# 2. Loaded sample admission
# =============================================================================

def validate_loaded_sample(
    sample: Any,
    expected_channels: Optional[int] = EXPECTED_CHANNELS,
    expected_spatial_shape: Optional[Tuple[int, int]] = None,
    lat: Optional[Sequence[float]] = None,
    lon: Optional[Sequence[float]] = None,
    expected_lat: Optional[Sequence[float]] = None,
    expected_lon: Optional[Sequence[float]] = None,
    channel_axis: int = -3,
    name: str = "sample",
) -> LoadedSampleReport:
    """Admit a loaded sample only if its contents are usable, not just shaped.

    Checks, in order: the array is real and numeric, its channel count is the
    expected one, its spatial shape is the expected one, every element is
    finite, and -- when coordinates are supplied -- the coordinate VALUES and
    their ORDER match the expected grid.

    The coordinate check is a value comparison rather than a shape comparison on
    purpose. The pull reverses latitude when the source is decreasing, so a
    store written under the other convention has exactly the right shape and is
    flipped; only comparing values catches that.

    Args:
        sample: Array-like sample, or an object exposing `.values`.
        expected_channels: Required channel count, or None to skip.
        expected_spatial_shape: Required trailing (lat, lon) shape, or None.
        lat: Latitude coordinate values as stored.
        lon: Longitude coordinate values as stored.
        expected_lat: Latitude values the sample must have.
        expected_lon: Longitude values the sample must have.
        channel_axis: Axis holding channels (default -3 for [.., C, H, W]).
        name: Label used in violation messages.

    Returns:
        LoadedSampleReport describing what was checked.

    Raises:
        PilotContractViolation: On any content-level violation.
    """
    values = getattr(sample, "values", sample)
    array = np.asarray(values)

    if array.size == 0:
        raise PilotContractViolation(
            "SAMPLE_EMPTY",
            f"{name} is empty (shape {tuple(array.shape)}).",
            {"shape": list(array.shape)},
        )
    if not np.issubdtype(array.dtype, np.number):
        raise PilotContractViolation(
            "SAMPLE_NOT_NUMERIC",
            f"{name} has non-numeric dtype {array.dtype}.",
            {"dtype": str(array.dtype)},
        )

    if expected_channels is not None:
        if array.ndim < abs(channel_axis):
            raise PilotContractViolation(
                "SAMPLE_RANK_TOO_LOW",
                f"{name} has rank {array.ndim}, too low to index channel axis "
                f"{channel_axis}.",
                {"shape": list(array.shape), "channel_axis": channel_axis},
            )
        n_channels = int(array.shape[channel_axis])
        if n_channels != int(expected_channels):
            raise PilotContractViolation(
                "SAMPLE_CHANNEL_COUNT_MISMATCH",
                f"{name} has {n_channels} channels, expected {expected_channels}.",
                {"n_channels": n_channels, "expected": int(expected_channels)},
            )
    else:
        n_channels = int(array.shape[channel_axis]) if array.ndim >= abs(channel_axis) else -1

    if expected_spatial_shape is not None:
        actual_spatial = tuple(int(s) for s in array.shape[-2:])
        if actual_spatial != tuple(int(s) for s in expected_spatial_shape):
            raise PilotContractViolation(
                "SAMPLE_SPATIAL_SHAPE_MISMATCH",
                f"{name} has spatial shape {actual_spatial}, expected "
                f"{tuple(expected_spatial_shape)}.",
                {
                    "actual": list(actual_spatial),
                    "expected": list(expected_spatial_shape),
                },
            )

    finite_mask = np.isfinite(array)
    if not bool(finite_mask.all()):
        n_bad = int(array.size - int(finite_mask.sum()))
        bad_channels: List[int] = []
        if array.ndim >= abs(channel_axis):
            moved = np.moveaxis(finite_mask, channel_axis, 0)
            per_channel_bad = ~moved.reshape(moved.shape[0], -1).all(axis=1)
            bad_channels = [int(i) for i in np.flatnonzero(per_channel_bad)[:16]]
        raise PilotContractViolation(
            "SAMPLE_NON_FINITE",
            f"{name} contains {n_bad} non-finite value(s) out of {array.size}"
            + (f"; first affected channel indices: {bad_channels}" if bad_channels else "")
            + ". A pilot store is created pre-filled with NaN, so a fully-shaped "
            "store with NaN channels is a store whose pull never completed.",
            {
                "n_non_finite": n_bad,
                "size": int(array.size),
                "bad_channel_indices": bad_channels,
            },
        )

    coordinates_checked = False
    if lat is not None and expected_lat is not None:
        _assert_coordinate_matches(lat, expected_lat, "lat", name)
        coordinates_checked = True
    if lon is not None and expected_lon is not None:
        _assert_coordinate_matches(lon, expected_lon, "lon", name)
        coordinates_checked = True

    return LoadedSampleReport(
        shape=tuple(int(s) for s in array.shape),
        dtype=str(array.dtype),
        n_channels=n_channels,
        finite=True,
        coordinates_checked=coordinates_checked,
    )


def _assert_coordinate_matches(
    actual: Sequence[float], expected: Sequence[float], axis: str, name: str
) -> None:
    """Refuse a coordinate axis whose values or order differ from expected."""
    actual_arr = np.asarray(getattr(actual, "values", actual), dtype=np.float64)
    expected_arr = np.asarray(getattr(expected, "values", expected), dtype=np.float64)

    if actual_arr.shape != expected_arr.shape:
        raise PilotContractViolation(
            "COORDINATE_LENGTH_MISMATCH",
            f"{name} {axis} has {actual_arr.shape} points, expected "
            f"{expected_arr.shape}.",
            {"axis": axis, "actual": list(actual_arr.shape),
             "expected": list(expected_arr.shape)},
        )
    if not np.all(np.isfinite(actual_arr)):
        raise PilotContractViolation(
            "COORDINATE_NON_FINITE",
            f"{name} {axis} contains non-finite coordinate values.",
            {"axis": axis},
        )

    if np.allclose(actual_arr, expected_arr, rtol=0.0, atol=COORD_ATOL):
        return

    # Distinguish a reordering from a genuinely different grid: the reversed
    # case is the one the pull's latitude flip can actually produce, and saying
    # so turns an opaque mismatch into an actionable one.
    if np.allclose(actual_arr[::-1], expected_arr, rtol=0.0, atol=COORD_ATOL):
        raise PilotContractViolation(
            "COORDINATE_ORDER_REVERSED",
            f"{name} {axis} holds the expected values in REVERSED order. The "
            "store was written under the opposite coordinate convention; its "
            "shape is identical and its contents are flipped along this axis.",
            {"axis": axis, "first": float(actual_arr[0]), "last": float(actual_arr[-1]),
             "expected_first": float(expected_arr[0]),
             "expected_last": float(expected_arr[-1])},
        )
    if np.allclose(np.sort(actual_arr), np.sort(expected_arr), rtol=0.0, atol=COORD_ATOL):
        raise PilotContractViolation(
            "COORDINATE_ORDER_PERMUTED",
            f"{name} {axis} holds the expected coordinate values in a different "
            "order.",
            {"axis": axis, "first": float(actual_arr[0]), "last": float(actual_arr[-1])},
        )

    max_diff = float(np.max(np.abs(actual_arr - expected_arr)))
    raise PilotContractViolation(
        "COORDINATE_VALUE_MISMATCH",
        f"{name} {axis} coordinate values differ from the expected grid by up "
        f"to {max_diff:.6g}.",
        {"axis": axis, "max_abs_diff": max_diff,
         "first": float(actual_arr[0]), "last": float(actual_arr[-1]),
         "expected_first": float(expected_arr[0]),
         "expected_last": float(expected_arr[-1])},
    )


# =============================================================================
# 3. Artifact binding admission (stale resume markers)
# =============================================================================

def _parse_marker_timestamp(text: str) -> Optional[datetime.datetime]:
    """Parse the ISO-8601 Z timestamp `mark_source_var_complete` writes."""
    stamp = (text or "").strip().splitlines()
    if not stamp:
        return None
    candidate = stamp[0].strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def assert_artifact_binding(
    store_path: Path,
    marker_dir: Optional[Path] = None,
    expected_grid_hash: Optional[str] = None,
    expected_variable_order_hash: Optional[str] = None,
    store_attrs: Optional[Dict[str, Any]] = None,
    marker_slack_seconds: int = 0,
) -> ArtifactBindingReport:
    """Refuse resume markers that cannot describe the store they sit beside.

    Resume markers record only "this variable was pulled", never "into THIS
    store". Deleting and recreating the store leaves the markers behind, and the
    resumed pull then skips every variable it believes is already done -- into a
    store where those channels were never written.

    The two sides already write enough to detect this: a marker records the wall
    time at which its variable finished, and the store records the wall time at
    which it was created. A marker that finished BEFORE the store was created
    cannot be describing that store, so it is stale by construction.

    The stamped `grid_hash` / `variable_order_hash` attrs are checked against
    the expected values in the same pass, since a store carrying a different
    identity stamp is likewise not the store the caller thinks it has.

    Args:
        store_path: Path to the zarr store.
        marker_dir: Directory holding `<var>.done` markers. Defaults to
            `<store parent>/.markers_<year>` inferred from the store name when
            that is unambiguous; pass explicitly otherwise.
        expected_grid_hash: Grid hash the store must carry, or None to skip.
        expected_variable_order_hash: Variable-order hash the store must carry.
        store_attrs: Pre-read store attrs; read from the store when omitted.
        marker_slack_seconds: Tolerance applied when comparing a marker's
            timestamp to the store's creation time. The pull writes both to
            whole-second resolution, so a small slack avoids flagging a marker
            written in the same second the store was created.

    Returns:
        ArtifactBindingReport describing what was checked.

    Raises:
        PilotContractViolation: If the store is missing, carries the wrong
            identity stamp, or is accompanied by markers that predate it.
    """
    store_path = Path(store_path)
    if not store_path.exists():
        raise PilotContractViolation(
            "STORE_MISSING",
            f"Pilot store does not exist: {store_path}",
            {"store_path": str(store_path)},
        )

    attrs = dict(store_attrs) if store_attrs is not None else _read_store_attrs(store_path)

    created_raw = attrs.get("created")
    grid_hash = attrs.get("grid_hash")
    variable_order_hash = attrs.get("variable_order_hash")

    if expected_grid_hash is not None and grid_hash != expected_grid_hash:
        raise PilotContractViolation(
            "STORE_GRID_HASH_MISMATCH",
            f"Store {store_path} carries grid_hash={grid_hash!r}, expected "
            f"{expected_grid_hash!r}. The store was written against a different "
            "target grid than the one this run uses.",
            {"store_path": str(store_path), "actual": grid_hash,
             "expected": expected_grid_hash},
        )
    if (expected_variable_order_hash is not None
            and variable_order_hash != expected_variable_order_hash):
        raise PilotContractViolation(
            "STORE_VARIABLE_ORDER_HASH_MISMATCH",
            f"Store {store_path} carries variable_order_hash="
            f"{variable_order_hash!r}, expected {expected_variable_order_hash!r}. "
            "Its channel axis does not mean what this run assumes it means.",
            {"store_path": str(store_path), "actual": variable_order_hash,
             "expected": expected_variable_order_hash},
        )

    report = ArtifactBindingReport(
        store_path=str(store_path),
        created_utc=str(created_raw) if created_raw is not None else None,
        grid_hash=grid_hash if isinstance(grid_hash, str) else None,
        variable_order_hash=(
            variable_order_hash if isinstance(variable_order_hash, str) else None
        ),
    )

    resolved_marker_dir = marker_dir if marker_dir is not None else _infer_marker_dir(store_path)
    if resolved_marker_dir is None:
        return report

    resolved_marker_dir = Path(resolved_marker_dir)
    report.marker_dir = str(resolved_marker_dir)
    if not resolved_marker_dir.is_dir():
        return report

    markers = sorted(resolved_marker_dir.glob("*.done"))
    report.markers_checked = len(markers)
    if not markers:
        return report

    created_at = _parse_marker_timestamp(str(created_raw)) if created_raw is not None else None
    if created_at is None:
        raise PilotContractViolation(
            "STORE_CREATION_STAMP_MISSING",
            f"Store {store_path} carries {len(markers)} resume marker(s) but no "
            f"readable `created` attr (got {created_raw!r}), so the markers "
            "cannot be shown to belong to this store. Refusing to treat them as "
            "valid resume state.",
            {"store_path": str(store_path),
             "marker_dir": str(resolved_marker_dir),
             "markers_checked": len(markers),
             "created": str(created_raw)},
        )

    threshold = created_at - datetime.timedelta(seconds=max(0, int(marker_slack_seconds)))
    stale: List[Dict[str, Any]] = []
    for marker in markers:
        try:
            marker_text = marker.read_text()
        except OSError as exc:
            raise PilotContractViolation(
                "MARKER_UNREADABLE",
                f"Resume marker {marker} could not be read: {exc}",
                {"marker": str(marker)},
            ) from exc
        marker_at = _parse_marker_timestamp(marker_text)
        if marker_at is None:
            stale.append({"marker": marker.name, "recorded": marker_text.strip(),
                          "reason": "unparseable timestamp"})
            continue
        if marker_at < threshold:
            stale.append({
                "marker": marker.name,
                "recorded": marker_at.isoformat(),
                "store_created": created_at.isoformat(),
                "reason": "marker predates store creation",
            })

    if stale:
        report.stale_markers = [entry["marker"] for entry in stale]
        raise PilotContractViolation(
            "STALE_MARKER_POINTS_AT_NEW_STORE",
            f"{len(stale)} of {len(markers)} resume marker(s) in "
            f"{resolved_marker_dir} cannot describe the store at {store_path}: "
            f"the store was created at {created_at.isoformat()} and these "
            "markers were written before that. Resuming against them would skip "
            "variables that were never written into THIS store, leaving their "
            "channels at the store's NaN fill. Clear the markers before "
            f"resuming. Offending markers: {[e['marker'] for e in stale][:8]}",
            {
                "store_path": str(store_path),
                "marker_dir": str(resolved_marker_dir),
                "store_created": created_at.isoformat(),
                "markers_checked": len(markers),
                "stale": stale[:16],
            },
        )

    return report


def _infer_marker_dir(store_path: Path) -> Optional[Path]:
    """Infer `.markers_<year>` from a `<year>[_jan].zarr` store name."""
    stem = store_path.name
    for suffix in (".zarr",):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    year_token = stem.split("_")[0]
    if not (len(year_token) == 4 and year_token.isdigit()):
        return None
    return store_path.parent / f".markers_{year_token}"


def _read_store_attrs(store_path: Path) -> Dict[str, Any]:
    """Read a zarr store's group attrs without materializing any data."""
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - zarr is a hard dependency
        raise PilotContractViolation(
            "ZARR_UNAVAILABLE",
            f"zarr is required to read store attrs: {exc}",
            {"store_path": str(store_path)},
        ) from exc
    try:
        group = zarr.open_group(str(store_path), mode="r")
        return dict(group.attrs)
    except Exception as exc:
        raise PilotContractViolation(
            "STORE_ATTRS_UNREADABLE",
            f"Could not read attrs from store {store_path}: "
            f"{type(exc).__name__}: {exc}",
            {"store_path": str(store_path)},
        ) from exc


# =============================================================================
# 4. Content admission for the subset that will really be consumed
# =============================================================================

@dataclass
class ContentCertificate:
    """What was actually read, what it contained, and that it was admitted.

    The certificate is the artifact that replaces "the marker says it is done"
    and "the shape is right". It names the exact indices that were read, the
    exact timestamps those indices resolved to, the digest of the bytes that
    came back, and the realized per-channel statistics. Two certificates over
    the same slice of the same store agree byte for byte on `content_sha256`;
    a certificate cannot be transplanted onto a different slice or a different
    store without `identity_sha256` changing.
    """

    store_path: str
    passed: bool
    indices: List[int] = field(default_factory=list)
    times_utc: List[str] = field(default_factory=list)
    n_timesteps: int = 0
    n_channels: int = 0
    dtype: str = ""
    batch_size: int = 0
    n_batches: int = 0
    n_values: int = 0
    content_sha256: str = ""
    identity_sha256: str = ""
    created_utc: Optional[str] = None
    grid_hash: Optional[str] = None
    variable_order_hash: Optional[str] = None
    sigma_bound: Optional[float] = None
    physical_range_checked: bool = False
    normalization_source: Optional[str] = None
    channel_stats: List[Dict[str, Any]] = field(default_factory=list)
    data_role: Optional[str] = None
    issue_id: Optional[str] = None
    process_group_id: Optional[str] = None
    verified_at: str = ""
    violations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "store_path": self.store_path,
            "passed": self.passed,
            "indices": list(self.indices),
            "times_utc": list(self.times_utc),
            "n_timesteps": self.n_timesteps,
            "n_channels": self.n_channels,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "n_batches": self.n_batches,
            "n_values": self.n_values,
            "content_sha256": self.content_sha256,
            "identity_sha256": self.identity_sha256,
            "created_utc": self.created_utc,
            "grid_hash": self.grid_hash,
            "variable_order_hash": self.variable_order_hash,
            "sigma_bound": self.sigma_bound,
            "physical_range_checked": self.physical_range_checked,
            "normalization_source": self.normalization_source,
            "channel_stats": list(self.channel_stats),
            "data_role": self.data_role,
            "issue_id": self.issue_id,
            "process_group_id": self.process_group_id,
            "verified_at": self.verified_at,
            "violations": list(self.violations),
        }


def physical_bounds_from_normalization(
    channel_names: Sequence[str],
    normalization_dir: Optional[Path] = None,
    sigma_bound: float = DEFAULT_SIGMA_BOUND,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive a per-channel plausibility band from the official constants.

    The band is `mean +- sigma_bound * std` using the SAME per-variable mean and
    std the bridge normalizes with (`normalize_mean.npz` / `normalize_std.npz`),
    so it inherits the official definition of what each channel is instead of
    introducing a second, hand-written notion of physical range that could drift
    from it.

    Args:
        channel_names: Channel names in store order.
        normalization_dir: Directory holding the official npz constants.
        sigma_bound: Half-width of the band in official standard deviations.

    Returns:
        (mean, std, low, high), each a float64 array over the channels.

    Raises:
        PilotContractViolation: If the constants are missing, unreadable, or do
            not cover every channel the store carries.
    """
    if not math.isfinite(sigma_bound) or sigma_bound <= 0:
        raise PilotContractViolation(
            "CONTENT_SIGMA_BOUND_INVALID",
            f"sigma_bound must be a positive finite number, got {sigma_bound!r}.",
            {"sigma_bound": repr(sigma_bound)},
        )

    directory = Path(normalization_dir) if normalization_dir is not None else DEFAULT_NORMALIZATION_DIR
    mean_path = directory / "normalize_mean.npz"
    std_path = directory / "normalize_std.npz"
    if not mean_path.exists() or not std_path.exists():
        raise PilotContractViolation(
            "CONTENT_NORMALIZATION_CONSTANTS_MISSING",
            f"Official normalization constants not found under {directory}: "
            f"expected normalize_mean.npz and normalize_std.npz. The physical "
            "plausibility band is derived from them; without them the band "
            "would be invented, so the check refuses rather than guesses.",
            {"normalization_dir": str(directory)},
        )

    try:
        mean_npz = np.load(mean_path)
        std_npz = np.load(std_path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a violation
        raise PilotContractViolation(
            "CONTENT_NORMALIZATION_CONSTANTS_UNREADABLE",
            f"Could not read normalization constants from {directory}: "
            f"{type(exc).__name__}: {exc}",
            {"normalization_dir": str(directory)},
        ) from exc

    missing = [name for name in channel_names if name not in mean_npz or name not in std_npz]
    if missing:
        raise PilotContractViolation(
            "CONTENT_NORMALIZATION_CHANNEL_MISSING",
            f"{len(missing)} channel(s) have no entry in the official "
            f"normalization constants under {directory}; first missing: "
            f"{missing[:8]}. The store's channel axis does not mean what the "
            "official constants mean.",
            {"normalization_dir": str(directory), "missing": missing[:16]},
        )

    mean = np.array([float(np.ravel(mean_npz[name])[0]) for name in channel_names], dtype=np.float64)
    std = np.array([float(np.ravel(std_npz[name])[0]) for name in channel_names], dtype=np.float64)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise PilotContractViolation(
            "CONTENT_NORMALIZATION_NON_FINITE",
            f"Normalization constants under {directory} contain non-finite "
            "values; no plausibility band can be derived from them.",
            {"normalization_dir": str(directory)},
        )
    if np.any(std <= 0):
        bad = [channel_names[i] for i in np.flatnonzero(std <= 0)[:8]]
        raise PilotContractViolation(
            "CONTENT_NORMALIZATION_ZERO_STD",
            f"Normalization constants under {directory} give a non-positive "
            f"std for {bad}; the band would be degenerate.",
            {"normalization_dir": str(directory), "channels": bad},
        )

    low = mean - float(sigma_bound) * std
    high = mean + float(sigma_bound) * std
    return mean, std, low, high


def _channel_names_for(dataset: Any, n_channels: int) -> Optional[List[str]]:
    """Channel names from the store, falling back to the canonical order."""
    coord = None
    try:
        if dataset is not None and "channel" in getattr(dataset, "coords", {}):
            coord = np.asarray(dataset["channel"].values)
    except Exception:  # noqa: BLE001 - a missing coord is not a violation here
        coord = None
    if coord is not None and coord.shape == (n_channels,) and coord.dtype.kind in "USO":
        return [str(name) for name in coord.tolist()]
    if n_channels == EXPECTED_CHANNELS:
        try:
            from .data.pull_wb2 import CANONICAL_VARIABLES

            return list(CANONICAL_VARIABLES)
        except Exception:  # noqa: BLE001 - optional fallback only
            return None
    return None


def verify_content_subset(
    store_path: Optional[Path] = None,
    indices: Optional[Sequence[int]] = None,
    *,
    dataset: Any = None,
    variable: str = "data",
    batch_size: int = 4,
    expected_channels: Optional[int] = EXPECTED_CHANNELS,
    expected_dtype: Optional[str] = "float32",
    channel_names: Optional[Sequence[str]] = None,
    sigma_bound: float = DEFAULT_SIGMA_BOUND,
    normalization_dir: Optional[Path] = None,
    require_physical_range: bool = True,
    check_degenerate: bool = True,
    data_role: Optional[Any] = None,
    issue_id: Optional[str] = None,
    process_group_id: Optional[str] = None,
    strict: bool = True,
) -> ContentCertificate:
    """Verify, batch by batch, the real values of the subset about to be read.

    This is the content half of the admission gate. `is_year_complete` and
    `is_pilot_complete` decide from a timestep count and a channel count;
    resume markers decide from their own existence. Neither reads a value.
    This function reads exactly the timesteps in `indices`, in batches of
    `batch_size`, and for each batch checks:

      * every value is finite -- the store is created pre-filled with NaN, so a
        never-written channel is NaN at full shape;
      * no (timestep, channel) slab is spatially constant -- a global field that
        takes one value everywhere is not weather, it is fill;
      * every value lies inside `mean +- sigma_bound * std` from the OFFICIAL
        normalization constants, so a channel in the wrong unit or holding
        another variable's values is refused.

    It then binds what it read into a `ContentCertificate`: the digest of the
    exact bytes, the digest of the slice identity those bytes came from, and the
    realized per-channel statistics. Nothing in the certificate is inferred from
    a marker or a shape.

    Args:
        store_path: Zarr store to open, or None when `dataset` is supplied.
        indices: Absolute time indices to verify. These are the indices the
            consumer will really read; an empty subset is refused.
        dataset: Already-open dataset to read from instead of opening one.
        variable: Name of the data variable holding the [time, channel, ...] array.
        batch_size: Timesteps per batch. Bounded so a large subset is verified
            without materializing the whole thing.
        expected_channels: Channel count the store must have, or None to skip.
        expected_dtype: Dtype the stored values must have, or None to skip.
        channel_names: Channel names in store order; read from the store's
            `channel` coordinate when omitted.
        sigma_bound: Half-width, in official standard deviations, of the band.
        normalization_dir: Directory of the official constants.
        require_physical_range: When False, skip the band check and say so in
            the certificate rather than claiming a check that did not run.
        check_degenerate: Whether to refuse spatially constant slabs.
        data_role: Frozen role this subset is being admitted for.
        issue_id: Admission issue id this subset belongs to, if any.
        process_group_id: Admission run id, if any.
        strict: Raise on the first violation (fail-closed). When False, return
            a certificate with `passed=False` and the violations listed.

    Returns:
        ContentCertificate over exactly the indices supplied.

    Raises:
        PilotContractViolation: In strict mode, on any violation. The raised
            error carries the partial certificate under `detail["certificate"]`.
    """
    role = assert_valid_data_role(data_role)
    opened_here = False
    if dataset is None:
        if store_path is None:
            raise PilotContractViolation(
                "CONTENT_NO_SOURCE",
                "verify_content_subset needs either store_path or dataset.",
                {},
            )
        dataset = _open_store(Path(store_path))
        opened_here = True
    resolved_store = str(store_path) if store_path is not None else str(
        getattr(dataset, "encoding", {}).get("source", "<in-memory dataset>")
    )

    certificate = ContentCertificate(
        store_path=resolved_store,
        passed=True,
        batch_size=int(batch_size),
        sigma_bound=float(sigma_bound) if require_physical_range else None,
        data_role=role,
        issue_id=issue_id,
        process_group_id=process_group_id,
        verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    def _fail(code: str, message: str, detail: Dict[str, Any]) -> PilotContractViolation:
        certificate.passed = False
        certificate.violations.append({"code": code, "message": message, "detail": detail})
        return PilotContractViolation(code, message, {**detail, "certificate": certificate.to_dict()})

    try:
        if int(batch_size) <= 0:
            raise _fail(
                "CONTENT_BATCH_SIZE_INVALID",
                f"batch_size must be a positive integer, got {batch_size!r}.",
                {"batch_size": repr(batch_size)},
            )

        index_list = [int(i) for i in (indices or [])]
        if not index_list:
            raise _fail(
                "CONTENT_EMPTY_SUBSET",
                "No indices were supplied, so no content was verified. An empty "
                "subset cannot certify anything; refusing to emit a certificate "
                "that would read as 'verified'.",
                {"store_path": resolved_store},
            )

        attrs = dict(getattr(dataset, "attrs", {}) or {})
        certificate.created_utc = str(attrs["created"]) if "created" in attrs else None
        certificate.grid_hash = attrs.get("grid_hash") if isinstance(attrs.get("grid_hash"), str) else None
        certificate.variable_order_hash = (
            attrs.get("variable_order_hash")
            if isinstance(attrs.get("variable_order_hash"), str) else None
        )

        if variable not in getattr(dataset, "variables", {}) and variable not in getattr(dataset, "data_vars", {}):
            raise _fail(
                "CONTENT_VARIABLE_MISSING",
                f"Store {resolved_store} has no data variable {variable!r}; "
                f"available: {sorted(getattr(dataset, 'data_vars', {}))}.",
                {"store_path": resolved_store, "variable": variable},
            )

        array = dataset[variable]
        n_timesteps = int(dataset.sizes["time"])
        certificate.n_timesteps = n_timesteps
        if n_timesteps <= 0:
            raise _fail(
                "CONTENT_STORE_EMPTY",
                f"Store {resolved_store} holds {n_timesteps} timesteps. A store "
                "can carry a full channel axis, a stamped identity and a "
                "complete set of resume markers and still hold no data at all.",
                {"store_path": resolved_store, "n_timesteps": n_timesteps},
            )

        out_of_range = [i for i in index_list if i < 0 or i >= n_timesteps]
        if out_of_range:
            raise _fail(
                "CONTENT_INDEX_OUT_OF_RANGE",
                f"{len(out_of_range)} requested index/indices fall outside a "
                f"store of {n_timesteps} timesteps (first offending: "
                f"{out_of_range[0]}). Negative indices are refused rather than "
                "wrapped.",
                {"store_path": resolved_store, "n_timesteps": n_timesteps,
                 "out_of_range": out_of_range[:16]},
            )

        if "channel" in array.sizes:
            n_channels = int(array.sizes["channel"])
        elif array.ndim >= 2:
            n_channels = int(array.shape[1])
        else:
            raise _fail(
                "CONTENT_RANK_TOO_LOW",
                f"Variable {variable!r} in {resolved_store} has rank "
                f"{array.ndim} and no channel dimension; expected at least "
                "[time, channel, ...].",
                {"store_path": resolved_store, "variable": variable,
                 "shape": [int(s) for s in array.shape]},
            )
        certificate.n_channels = n_channels
        if expected_channels is not None and n_channels != int(expected_channels):
            raise _fail(
                "CONTENT_CHANNEL_COUNT_MISMATCH",
                f"Store {resolved_store} has {n_channels} channels, expected "
                f"{int(expected_channels)}.",
                {"store_path": resolved_store, "n_channels": n_channels,
                 "expected": int(expected_channels)},
            )

        stored_dtype = str(getattr(array, "dtype", ""))
        certificate.dtype = stored_dtype
        if expected_dtype is not None and stored_dtype != str(expected_dtype):
            raise _fail(
                "CONTENT_DTYPE_UNEXPECTED",
                f"Store {resolved_store} holds dtype {stored_dtype}, expected "
                f"{expected_dtype}.",
                {"store_path": resolved_store, "dtype": stored_dtype,
                 "expected": str(expected_dtype)},
            )

        names = list(channel_names) if channel_names is not None else _channel_names_for(dataset, n_channels)
        low = high = mean = std = None
        if require_physical_range:
            if names is None:
                raise _fail(
                    "CONTENT_CHANNEL_NAMES_UNAVAILABLE",
                    f"Store {resolved_store} carries no usable channel names, so "
                    "the per-channel plausibility band cannot be bound to the "
                    "official constants. Pass channel_names, or set "
                    "require_physical_range=False to record that the band was "
                    "not checked.",
                    {"store_path": resolved_store, "n_channels": n_channels},
                )
            mean, std, low, high = physical_bounds_from_normalization(
                names, normalization_dir=normalization_dir, sigma_bound=sigma_bound
            )
            certificate.physical_range_checked = True
            certificate.normalization_source = str(
                Path(normalization_dir) if normalization_dir is not None else DEFAULT_NORMALIZATION_DIR
            )

        certificate.indices = list(index_list)
        try:
            time_values = np.asarray(dataset["time"].values)[index_list]
            certificate.times_utc = [str(value) for value in time_values.tolist()] \
                if time_values.dtype.kind not in "M" else \
                [str(np.datetime64(value, "s")) for value in time_values]
        except Exception:  # noqa: BLE001 - absent/odd time coord is not fatal here
            certificate.times_utc = []

        identity = {
            "store_path": resolved_store,
            "created_utc": certificate.created_utc,
            "grid_hash": certificate.grid_hash,
            "variable_order_hash": certificate.variable_order_hash,
            "variable": variable,
            "indices": list(index_list),
            "times_utc": list(certificate.times_utc),
            "n_channels": n_channels,
            "channel_names": names,
            "data_role": role,
            "issue_id": issue_id,
        }
        certificate.identity_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        digest = hashlib.sha256()
        digest.update(certificate.identity_sha256.encode("ascii"))

        ch_min = np.full(n_channels, np.inf, dtype=np.float64)
        ch_max = np.full(n_channels, -np.inf, dtype=np.float64)
        ch_sum = np.zeros(n_channels, dtype=np.float64)
        ch_sq = np.zeros(n_channels, dtype=np.float64)
        ch_count = 0
        degenerate: List[Dict[str, Any]] = []

        n_batches = 0
        n_values = 0
        for start in range(0, len(index_list), int(batch_size)):
            batch_indices = index_list[start:start + int(batch_size)]
            n_batches += 1
            batch = np.asarray(array.isel(time=batch_indices).values)
            if batch.ndim < 2:
                raise _fail(
                    "CONTENT_RANK_TOO_LOW",
                    f"Batch {n_batches} has rank {batch.ndim}; expected at least "
                    "[time, channel, ...].",
                    {"shape": list(batch.shape), "batch": n_batches},
                )
            n_values += int(batch.size)
            digest.update(np.ascontiguousarray(batch, dtype="<f4").tobytes())

            flat = batch.reshape(batch.shape[0], batch.shape[1], -1)
            finite = np.isfinite(flat)
            if not bool(finite.all()):
                bad_positions = np.argwhere(~finite.all(axis=2))
                bad_channels = sorted({int(c) for _, c in bad_positions.tolist()})
                raise _fail(
                    "CONTENT_NON_FINITE",
                    f"Batch {n_batches} (time indices {batch_indices}) contains "
                    f"{int(flat.size - finite.sum())} non-finite value(s) across "
                    f"{len(bad_channels)} channel(s); first affected channel "
                    f"indices {bad_channels[:8]}"
                    + (f" ({[names[c] for c in bad_channels[:8]]})" if names else "")
                    + ". A pilot store is created pre-filled with NaN, so this is "
                    "a channel the pull never wrote.",
                    {"batch": n_batches, "indices": batch_indices,
                     "n_non_finite": int(flat.size - finite.sum()),
                     "bad_channel_indices": bad_channels[:16]},
                )

            batch_min = flat.min(axis=2).astype(np.float64)
            batch_max = flat.max(axis=2).astype(np.float64)

            if check_degenerate:
                constant = batch_min == batch_max
                if bool(constant.any()):
                    where = np.argwhere(constant)
                    detail = [
                        {
                            "time_index": int(batch_indices[t]),
                            "channel_index": int(c),
                            "channel": names[int(c)] if names else None,
                            "value": float(batch_min[t, c]),
                        }
                        for t, c in where[:16].tolist()
                    ]
                    degenerate.extend(detail)
                    raise _fail(
                        "CONTENT_DEGENERATE_CHANNEL",
                        f"Batch {n_batches} has {int(constant.sum())} "
                        "(timestep, channel) slab(s) that are spatially constant. "
                        "A global field holding one value everywhere is fill, not "
                        f"weather. First offenders: {detail[:4]}",
                        {"batch": n_batches, "indices": batch_indices,
                         "n_degenerate": int(constant.sum()), "slabs": detail},
                    )

            if low is not None:
                below = batch_min < low[None, :]
                above = batch_max > high[None, :]
                offending = below | above
                if bool(offending.any()):
                    where = np.argwhere(offending)
                    detail = [
                        {
                            "time_index": int(batch_indices[t]),
                            "channel_index": int(c),
                            "channel": names[int(c)] if names else None,
                            "min": float(batch_min[t, c]),
                            "max": float(batch_max[t, c]),
                            "allowed_low": float(low[int(c)]),
                            "allowed_high": float(high[int(c)]),
                            "max_abs_z": float(max(
                                abs(batch_min[t, c] - mean[int(c)]) / std[int(c)],
                                abs(batch_max[t, c] - mean[int(c)]) / std[int(c)],
                            )),
                        }
                        for t, c in where[:16].tolist()
                    ]
                    raise _fail(
                        "CONTENT_OUT_OF_PHYSICAL_RANGE",
                        f"Batch {n_batches} has {int(offending.sum())} "
                        "(timestep, channel) slab(s) outside the official "
                        f"normalization band of +-{sigma_bound} sigma. First "
                        f"offenders: {detail[:4]}",
                        {"batch": n_batches, "indices": batch_indices,
                         "sigma_bound": float(sigma_bound),
                         "n_offending": int(offending.sum()), "slabs": detail},
                    )

            ch_min = np.minimum(ch_min, batch_min.min(axis=0))
            ch_max = np.maximum(ch_max, batch_max.max(axis=0))
            values = flat.astype(np.float64)
            ch_sum += values.sum(axis=(0, 2))
            ch_sq += (values ** 2).sum(axis=(0, 2))
            ch_count += values.shape[0] * values.shape[2]

        certificate.n_batches = n_batches
        certificate.n_values = n_values
        certificate.content_sha256 = digest.hexdigest()

        realized_mean = ch_sum / max(ch_count, 1)
        realized_var = np.maximum(ch_sq / max(ch_count, 1) - realized_mean ** 2, 0.0)
        certificate.channel_stats = [
            {
                "channel_index": int(c),
                "channel": names[c] if names else None,
                "min": float(ch_min[c]),
                "max": float(ch_max[c]),
                "mean": float(realized_mean[c]),
                "std": float(np.sqrt(realized_var[c])),
                "max_abs_z": (
                    float(max(abs(ch_min[c] - mean[c]), abs(ch_max[c] - mean[c])) / std[c])
                    if mean is not None else None
                ),
            }
            for c in range(n_channels)
        ]
        certificate.passed = True
        return certificate
    except PilotContractViolation:
        certificate.passed = False
        if strict:
            raise
        return certificate
    finally:
        if opened_here:
            try:
                dataset.close()
            except Exception:  # noqa: BLE001 - closing must not mask a violation
                pass


# =============================================================================
# 5. Preflight driver (used by scripts/r2_pilot_preflight.py)
# =============================================================================

def run_pilot_preflight(
    store_path: Path,
    index: int = 0,
    history_steps: int = 1,
    target_steps: int = 1,
    interval_hours: Optional[int] = DEFAULT_INTERVAL_HOURS,
    marker_dir: Optional[Path] = None,
    expected_channels: Optional[int] = EXPECTED_CHANNELS,
    check_grid: bool = True,
    marker_slack_seconds: int = 0,
    expected_lat: Optional[Sequence[float]] = None,
    expected_lon: Optional[Sequence[float]] = None,
    verify_content: bool = False,
    content_indices: Optional[Sequence[int]] = None,
    content_batch_size: int = 4,
    content_sigma_bound: float = DEFAULT_SIGMA_BOUND,
    normalization_dir: Optional[Path] = None,
    require_physical_range: bool = True,
    data_role: Optional[Any] = None,
) -> PilotPreflightReport:
    """Run every admission check against a real pilot store and one real sample.

    This is the body of the preflight entry point. It opens the store, binds its
    identity, resolves the requested sample's endpoints, loads exactly that
    sample, and validates its contents and coordinates. Each check is recorded
    on the report; the first violation in each check is captured and the run is
    reported as failed rather than raising, so a single invocation reports every
    independently-checkable fault it can reach.

    Args:
        store_path: Path to the pilot zarr store.
        index: Analysis time index of the sample to admit.
        history_steps: History steps the sample requires.
        target_steps: Target steps the sample requires.
        interval_hours: Expected time spacing, or None to skip spacing checks.
        marker_dir: Explicit marker directory, or None to infer it.
        expected_channels: Required channel count, or None to skip.
        check_grid: Compare coordinates and identity stamps against the
            canonical Stormer target grid from `earthdelta.data.pull_wb2`.
        marker_slack_seconds: Slack for the stale-marker timestamp comparison.
        expected_lat: Latitude values the store must carry. Overrides the
            canonical grid; lets a caller admit a store against a grid other
            than the production one without disabling the coordinate check.
        expected_lon: Longitude values the store must carry, as above.
        verify_content: Also run `verify_content_subset` over the timesteps this
            sample spans (or over `content_indices`) and attach the resulting
            certificate to the report.
        content_indices: Explicit subset to content-verify. Defaults to every
            timestep the admitted sample spans.
        content_batch_size: Timesteps per content-verification batch.
        content_sigma_bound: Half-width of the plausibility band, in official
            standard deviations.
        normalization_dir: Directory of the official normalization constants.
        require_physical_range: Whether the band check must run.
        data_role: Frozen role recorded on the content certificate.

    Returns:
        PilotPreflightReport. `passed` is False if any check violated.
    """
    store_path = Path(store_path)
    report = PilotPreflightReport(store_path=str(store_path), passed=True)

    def _record(check: str, run) -> Optional[Any]:
        try:
            value = run()
        except PilotContractViolation as exc:
            report.passed = False
            report.checks.append({"check": check, "passed": False, "code": exc.code})
            report.violations.append({
                "check": check,
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail,
            })
            return None
        except Exception as exc:  # noqa: BLE001 - surface as a violation, not a crash
            report.passed = False
            report.checks.append({"check": check, "passed": False, "code": "UNEXPECTED_ERROR"})
            report.violations.append({
                "check": check,
                "code": "UNEXPECTED_ERROR",
                "message": f"{type(exc).__name__}: {exc}",
                "detail": {},
            })
            return None
        report.checks.append({"check": check, "passed": True, "code": None})
        return value

    expected_grid_hash = None
    expected_variable_order_hash = None
    if check_grid:
        from .data.pull_wb2 import (
            get_stormer_target_grid,
            grid_hash,
            variable_order_hash,
        )
        expected_grid_hash = grid_hash()
        expected_variable_order_hash = variable_order_hash()
        canonical_lat, canonical_lon = get_stormer_target_grid()
        if expected_lat is None:
            expected_lat = canonical_lat
        if expected_lon is None:
            expected_lon = canonical_lon

    binding = _record("artifact_binding", lambda: assert_artifact_binding(
        store_path,
        marker_dir=marker_dir,
        expected_grid_hash=expected_grid_hash,
        expected_variable_order_hash=expected_variable_order_hash,
        marker_slack_seconds=marker_slack_seconds,
    ))
    if binding is not None:
        report.checks[-1]["detail"] = {
            "created_utc": binding.created_utc,
            "markers_checked": binding.markers_checked,
        }

    dataset = _record("open_store", lambda: _open_store(store_path))
    if dataset is None:
        return report

    try:
        n_timesteps = int(dataset.sizes["time"])
        time_coords = dataset["time"].values

        slice_report = _record("slice_index", lambda: validate_slice_index(
            index=index,
            n_timesteps=n_timesteps,
            history_steps=history_steps,
            target_steps=target_steps,
            time_coords=time_coords,
            interval_hours=interval_hours,
        ))
        if slice_report is None:
            return report

        def _load_and_validate() -> LoadedSampleReport:
            window = dataset["data"].isel(
                time=slice(slice_report.history_index, slice_report.target_index + 1)
            )
            return validate_loaded_sample(
                window.values,
                expected_channels=expected_channels,
                lat=dataset["lat"].values if "lat" in dataset.coords else None,
                lon=dataset["lon"].values if "lon" in dataset.coords else None,
                expected_lat=expected_lat,
                expected_lon=expected_lon,
                name=(
                    f"sample[{slice_report.history_index}:"
                    f"{slice_report.target_index + 1}]"
                ),
            )

        sample_report = _record("loaded_sample", _load_and_validate)
        if sample_report is not None:
            report.checks[-1]["detail"] = {
                "shape": list(sample_report.shape),
                "dtype": sample_report.dtype,
                "coordinates_checked": sample_report.coordinates_checked,
            }

        if verify_content:
            subset = (
                [int(i) for i in content_indices]
                if content_indices is not None
                else list(range(slice_report.history_index, slice_report.target_index + 1))
            )

            def _verify_content() -> ContentCertificate:
                return verify_content_subset(
                    store_path=store_path,
                    indices=subset,
                    dataset=dataset,
                    batch_size=content_batch_size,
                    expected_channels=expected_channels,
                    sigma_bound=content_sigma_bound,
                    normalization_dir=normalization_dir,
                    require_physical_range=require_physical_range,
                    data_role=data_role,
                    strict=True,
                )

            certificate = _record("content_subset", _verify_content)
            if certificate is not None:
                report.content_certificate = certificate.to_dict()
                report.checks[-1]["detail"] = {
                    "indices": list(certificate.indices),
                    "n_batches": certificate.n_batches,
                    "content_sha256": certificate.content_sha256,
                    "identity_sha256": certificate.identity_sha256,
                }
            elif report.violations:
                embedded = report.violations[-1].get("detail", {}).get("certificate")
                if isinstance(embedded, dict):
                    report.content_certificate = embedded
    finally:
        try:
            dataset.close()
        except Exception:  # noqa: BLE001 - closing must not mask a violation
            pass

    return report


def _open_store(store_path: Path):
    """Open a pilot zarr store lazily, as a real consumer would."""
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - xarray is a hard dependency
        raise PilotContractViolation(
            "XARRAY_UNAVAILABLE",
            f"xarray is required to open a pilot store: {exc}",
            {"store_path": str(store_path)},
        ) from exc
    if not Path(store_path).exists():
        raise PilotContractViolation(
            "STORE_MISSING",
            f"Pilot store does not exist: {store_path}",
            {"store_path": str(store_path)},
        )
    try:
        return xr.open_zarr(str(store_path))
    except Exception as exc:
        raise PilotContractViolation(
            "STORE_UNREADABLE",
            f"Could not open pilot store {store_path}: {type(exc).__name__}: {exc}",
            {"store_path": str(store_path)},
        ) from exc
