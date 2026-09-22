"""Admission-contract tests for pilot data, driven through the real entry point.

Each negative case builds a synthetic pilot store that is IDENTICAL to a good
one except for the single property under test, then runs the real
`scripts/r2_pilot_preflight.py` `main()` over it and asserts both a non-zero
exit code and the specific violation code that must fire. The positive fixture
is the control: it must pass the same entry point with exit code 0.

The stores are tiny (4 lat x 8 lon, 6 timesteps) but structurally real: written
through xarray/zarr with the same coordinate names, the same `data` variable
layout and the same `created` / `grid_hash` / `variable_order_hash` attrs the
production pull stamps, so the checks exercised here are the checks that would
run against a real store.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta.pilot_contract import (  # noqa: E402
    PilotContractViolation,
    assert_artifact_binding,
    run_pilot_preflight,
    validate_loaded_sample,
    validate_slice_index,
)

import scripts.r2_pilot_preflight as preflight_cli  # noqa: E402


# =============================================================================
# Synthetic pilot store fixture
# =============================================================================

N_TIME = 6
N_CHANNELS = 5
N_LAT = 4
N_LON = 8
INTERVAL_HOURS = 6
STORE_CREATED = "2026-09-21T12:00:00Z"
GRID_HASH = "pilotgridhash01"
VAR_ORDER_HASH = "pilotvarorderhash01"


def canonical_lat() -> np.ndarray:
    """Increasing latitude cell centres, matching the pull's convention."""
    ddeg = 180.0 / N_LAT
    return np.linspace(-90 + ddeg / 2, 90 - ddeg / 2, num=N_LAT, endpoint=True)


def canonical_lon() -> np.ndarray:
    return np.linspace(0, 360, num=N_LON, endpoint=False)


def build_store(
    path: Path,
    n_time: int = N_TIME,
    data: Optional[np.ndarray] = None,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    time_coords: Optional[np.ndarray] = None,
    created: Optional[str] = STORE_CREATED,
    grid_hash: Optional[str] = GRID_HASH,
    variable_order_hash: Optional[str] = VAR_ORDER_HASH,
) -> Path:
    """Write one synthetic pilot store; every knob defaults to the good value."""
    import xarray as xr

    if lat is None:
        lat = canonical_lat()
    if lon is None:
        lon = canonical_lon()
    if time_coords is None:
        time_coords = np.array(
            [
                np.datetime64("2020-01-01T00:00:00")
                + np.timedelta64(INTERVAL_HOURS * i, "h")
                for i in range(n_time)
            ],
            dtype="datetime64[ns]",
        )
    if data is None:
        rng = np.random.default_rng(20260921)
        data = rng.normal(size=(n_time, N_CHANNELS, len(lat), len(lon))).astype("float32")

    attrs: Dict[str, Any] = {}
    if created is not None:
        attrs["created"] = created
    if grid_hash is not None:
        attrs["grid_hash"] = grid_hash
    if variable_order_hash is not None:
        attrs["variable_order_hash"] = variable_order_hash

    dataset = xr.Dataset(
        {"data": (["time", "channel", "lat", "lon"], data)},
        coords={
            "time": time_coords,
            "channel": [f"var_{i}" for i in range(data.shape[1])],
            "lat": lat,
            "lon": lon,
        },
        attrs=attrs,
    )
    dataset.to_zarr(path, mode="w")
    dataset.close()
    return path


def write_marker(marker_dir: Path, name: str, stamp: str) -> Path:
    """Write a resume marker exactly as `mark_source_var_complete` does."""
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{name}.done"
    marker.write_text(f"{stamp}\n")
    return marker


def run_cli(store: Path, extra: Optional[List[str]] = None) -> int:
    """Invoke the REAL preflight entry point exactly as the CLI does.

    Defaults to index 1 so the default one-step history endpoint exists; tests
    that care about the endpoints pass their own --index.
    """
    argv = [
        "--store", str(store),
        "--expected-channels", str(N_CHANNELS),
        "--no-grid-check",
        "--interval-hours", str(INTERVAL_HOURS),
    ]
    if not (extra and "--index" in extra):
        argv.extend(["--index", "1"])
    if extra:
        argv.extend(extra)
    return preflight_cli.main(argv)


def codes_for(store: Path, **kwargs) -> List[str]:
    """Violation codes the real preflight driver reports for a store."""
    kwargs.setdefault("index", 1)
    report = run_pilot_preflight(
        store_path=store,
        expected_channels=N_CHANNELS,
        check_grid=False,
        interval_hours=INTERVAL_HOURS,
        **kwargs,
    )
    return [violation["code"] for violation in report.violations]


@pytest.fixture
def good_store(tmp_path) -> Path:
    return build_store(tmp_path / "2020_jan.zarr")


# =============================================================================
# Positive control: the entry point admits a well-formed store
# =============================================================================

def test_positive_store_passes_real_entry_point(good_store):
    """A well-formed store passes the real preflight CLI with exit code 0."""
    assert run_cli(good_store) == 0


def test_positive_store_reports_every_check_as_passed(good_store):
    report = run_pilot_preflight(
        store_path=good_store, index=1, expected_channels=N_CHANNELS,
        check_grid=False, interval_hours=INTERVAL_HOURS,
    )
    assert report.passed is True
    assert report.violations == []
    names = {check["check"] for check in report.checks}
    assert {"artifact_binding", "open_store", "slice_index", "loaded_sample"} <= names
    assert all(check["passed"] for check in report.checks)


def test_positive_store_writes_a_json_report(good_store, tmp_path):
    out = tmp_path / "reports" / "preflight.json"
    assert run_cli(good_store, ["--json", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["passed"] is True
    assert payload["violations"] == []


# =============================================================================
# Negative 1: missing history / target endpoints
# =============================================================================

def test_negative_missing_history_endpoint(good_store):
    """Index 0 with one step of history has no history endpoint."""
    assert run_cli(good_store, ["--index", "0", "--history-steps", "1"]) == 1
    assert "HISTORY_ENDPOINT_MISSING" in codes_for(good_store, index=0, history_steps=1)


def test_negative_missing_target_endpoint(good_store):
    """The last index with one step of target has no target endpoint."""
    last = N_TIME - 1
    assert run_cli(good_store, ["--index", str(last), "--target-steps", "1"]) == 1
    assert "TARGET_ENDPOINT_MISSING" in codes_for(
        good_store, index=last, history_steps=0, target_steps=1
    )


def test_negative_target_window_longer_than_record(good_store):
    """A target horizon longer than the record is refused, not truncated."""
    assert run_cli(good_store, ["--index", "1", "--target-steps", str(N_TIME)]) == 1


def test_negative_index_past_end_of_record(good_store):
    assert run_cli(good_store, ["--index", str(N_TIME + 3)]) == 1
    assert "SLICE_INDEX_OUT_OF_RANGE" in codes_for(good_store, index=N_TIME + 3)


def test_negative_negative_index_is_refused_not_wrapped(good_store):
    """A negative index must be refused rather than wrapping to the far end."""
    assert run_cli(good_store, ["--index", "-1"]) == 1
    assert "SLICE_INDEX_NEGATIVE" in codes_for(good_store, index=-1)


def test_negative_time_axis_has_a_gap(tmp_path):
    """A store with a jump in its time axis is not a contiguous window."""
    coords = np.array(
        [
            np.datetime64("2020-01-01T00:00:00") + np.timedelta64(INTERVAL_HOURS * i, "h")
            for i in range(N_TIME)
        ],
        dtype="datetime64[ns]",
    )
    coords[3] = coords[3] + np.timedelta64(30, "h")  # gap in the middle
    store = build_store(tmp_path / "2020_jan.zarr", time_coords=coords)
    assert run_cli(store, ["--index", "2", "--history-steps", "2", "--target-steps", "2"]) == 1
    assert "TIME_SPACING_MISMATCH" in codes_for(
        store, index=2, history_steps=2, target_steps=2
    )


def test_uniform_span_outside_the_gap_is_still_admitted(tmp_path):
    """The spacing check is scoped to the requested span, not the whole store."""
    coords = np.array(
        [
            np.datetime64("2020-01-01T00:00:00") + np.timedelta64(INTERVAL_HOURS * i, "h")
            for i in range(N_TIME)
        ],
        dtype="datetime64[ns]",
    )
    coords[5] = coords[5] + np.timedelta64(30, "h")  # gap only at the tail
    store = build_store(tmp_path / "2020_jan.zarr", time_coords=coords)
    assert run_cli(store, ["--index", "1", "--history-steps", "1", "--target-steps", "1"]) == 0


# =============================================================================
# Negative 2: coordinate order changed
# =============================================================================

def test_negative_latitude_order_reversed_through_the_real_driver(tmp_path):
    """A flipped latitude axis has the right shape and the wrong contents."""
    store = build_store(tmp_path / "2020_jan.zarr", lat=canonical_lat()[::-1].copy())

    report = run_pilot_preflight(
        store_path=store, index=1, expected_channels=N_CHANNELS,
        check_grid=False, interval_hours=INTERVAL_HOURS,
        expected_lat=canonical_lat(), expected_lon=canonical_lon(),
    )
    assert report.passed is False
    codes = [violation["code"] for violation in report.violations]
    assert "COORDINATE_ORDER_REVERSED" in codes


def test_negative_coordinate_order_on_the_real_stormer_grid(tmp_path):
    """The canonical-grid path: a flipped store is rejected by the real CLI.

    Built on the actual `get_stormer_target_grid()` coordinates and stamped
    with the real `grid_hash()` / `variable_order_hash()`, so `--no-grid-check`
    is NOT passed and the production comparison is the one that runs.
    """
    from earthdelta.data.pull_wb2 import (
        get_stormer_target_grid,
        grid_hash,
        variable_order_hash,
    )

    real_lat, real_lon = get_stormer_target_grid()
    n_time, n_channels = 3, 2
    rng = np.random.default_rng(13)
    payload = rng.normal(
        size=(n_time, n_channels, real_lat.size, real_lon.size)
    ).astype("float32")

    def _build(path: Path, lat: np.ndarray) -> Path:
        return build_store(
            path, n_time=n_time, data=payload, lat=lat, lon=real_lon,
            grid_hash=grid_hash(), variable_order_hash=variable_order_hash(),
        )

    aligned = _build(tmp_path / "2020_jan.zarr", real_lat)
    assert preflight_cli.main([
        "--store", str(aligned), "--expected-channels", str(n_channels),
        "--index", "1", "--interval-hours", str(INTERVAL_HOURS),
    ]) == 0

    flipped = _build(tmp_path / "2020_flipped.zarr", real_lat[::-1].copy())
    assert preflight_cli.main([
        "--store", str(flipped), "--expected-channels", str(n_channels),
        "--index", "1", "--interval-hours", str(INTERVAL_HOURS),
    ]) == 1

    report = run_pilot_preflight(
        store_path=flipped, index=1, expected_channels=n_channels, check_grid=True,
        interval_hours=INTERVAL_HOURS,
    )
    assert "COORDINATE_ORDER_REVERSED" in [v["code"] for v in report.violations]


def test_negative_wrong_identity_stamp_on_the_real_stormer_grid(tmp_path):
    """A store stamped with a different grid hash is refused by the real CLI."""
    from earthdelta.data.pull_wb2 import get_stormer_target_grid, variable_order_hash

    real_lat, real_lon = get_stormer_target_grid()
    rng = np.random.default_rng(14)
    payload = rng.normal(size=(3, 2, real_lat.size, real_lon.size)).astype("float32")
    store = build_store(
        tmp_path / "2020_jan.zarr", n_time=3, data=payload, lat=real_lat, lon=real_lon,
        grid_hash="not-the-real-grid", variable_order_hash=variable_order_hash(),
    )
    assert preflight_cli.main([
        "--store", str(store), "--expected-channels", "2",
        "--index", "1", "--interval-hours", str(INTERVAL_HOURS),
    ]) == 1


def test_negative_shifted_longitude_grid(tmp_path):
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_loaded_sample(
            np.zeros((1, N_CHANNELS, N_LAT, N_LON), dtype="float32"),
            expected_channels=N_CHANNELS,
            lon=canonical_lon() + 180.0,
            expected_lon=canonical_lon(),
        )
    assert excinfo.value.code == "COORDINATE_VALUE_MISMATCH"


def test_negative_permuted_coordinates(tmp_path):
    permuted = canonical_lon().copy()
    permuted[[1, 3]] = permuted[[3, 1]]
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_loaded_sample(
            np.zeros((1, N_CHANNELS, N_LAT, N_LON), dtype="float32"),
            expected_channels=N_CHANNELS,
            lon=permuted,
            expected_lon=canonical_lon(),
        )
    assert excinfo.value.code == "COORDINATE_ORDER_PERMUTED"


def test_negative_wrong_coordinate_length(tmp_path):
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_loaded_sample(
            np.zeros((1, N_CHANNELS, N_LAT, N_LON), dtype="float32"),
            expected_channels=N_CHANNELS,
            lat=canonical_lat()[:-1],
            expected_lat=canonical_lat(),
        )
    assert excinfo.value.code == "COORDINATE_LENGTH_MISMATCH"


# =============================================================================
# Negative 3: non-finite content
# =============================================================================

def test_negative_channel_never_written_stays_nan(tmp_path):
    """A never-written channel is full-shaped NaN, which shape checks admit."""
    rng = np.random.default_rng(7)
    data = rng.normal(size=(N_TIME, N_CHANNELS, N_LAT, N_LON)).astype("float32")
    data[:, 2, :, :] = np.nan  # channel 2 was never pulled
    store = build_store(tmp_path / "2020_jan.zarr", data=data)

    assert run_cli(store, ["--index", "1"]) == 1
    codes = codes_for(store, index=1)
    assert "SAMPLE_NON_FINITE" in codes


def test_non_finite_violation_names_the_offending_channel(tmp_path):
    rng = np.random.default_rng(8)
    data = rng.normal(size=(N_TIME, N_CHANNELS, N_LAT, N_LON)).astype("float32")
    data[:, 3, :, :] = np.nan
    store = build_store(tmp_path / "2020_jan.zarr", data=data)
    report = run_pilot_preflight(
        store_path=store, index=1, expected_channels=N_CHANNELS,
        check_grid=False, interval_hours=INTERVAL_HOURS,
    )
    violation = next(v for v in report.violations if v["code"] == "SAMPLE_NON_FINITE")
    assert 3 in violation["detail"]["bad_channel_indices"]


def test_negative_infinite_content(tmp_path):
    rng = np.random.default_rng(9)
    data = rng.normal(size=(N_TIME, N_CHANNELS, N_LAT, N_LON)).astype("float32")
    data[2, 0, 0, 0] = np.inf
    store = build_store(tmp_path / "2020_jan.zarr", data=data)
    assert run_cli(store, ["--index", "2"]) == 1
    assert "SAMPLE_NON_FINITE" in codes_for(store, index=2)


def test_non_finite_outside_the_requested_span_is_not_charged(tmp_path):
    """Admission is about the sample being loaded, not the whole corpus."""
    rng = np.random.default_rng(10)
    data = rng.normal(size=(N_TIME, N_CHANNELS, N_LAT, N_LON)).astype("float32")
    data[5, 0, 0, 0] = np.nan  # last timestep only
    store = build_store(tmp_path / "2020_jan.zarr", data=data)
    assert run_cli(store, ["--index", "1", "--history-steps", "1", "--target-steps", "1"]) == 0


def test_negative_wrong_channel_count(tmp_path):
    rng = np.random.default_rng(11)
    data = rng.normal(size=(N_TIME, N_CHANNELS - 1, N_LAT, N_LON)).astype("float32")
    store = build_store(tmp_path / "2020_jan.zarr", data=data)
    assert run_cli(store, ["--index", "1"]) == 1
    assert "SAMPLE_CHANNEL_COUNT_MISMATCH" in codes_for(store, index=1)


# =============================================================================
# Negative 4: stale marker pointing at a NEW store
# =============================================================================

def test_negative_stale_marker_points_at_a_new_store(tmp_path):
    """Markers written before the store was created cannot describe it."""
    store = build_store(tmp_path / "2020_jan.zarr", created="2026-09-21T12:00:00Z")
    write_marker(tmp_path / ".markers_2020", "geopotential", "2026-09-20T08:00:00Z")

    assert run_cli(store, ["--index", "1"]) == 1
    codes = codes_for(store, index=1)
    assert "STALE_MARKER_POINTS_AT_NEW_STORE" in codes


def test_stale_marker_violation_names_the_offending_markers(tmp_path):
    store = build_store(tmp_path / "2020_jan.zarr", created="2026-09-21T12:00:00Z")
    write_marker(tmp_path / ".markers_2020", "temperature", "2026-09-20T08:00:00Z")
    write_marker(tmp_path / ".markers_2020", "geopotential", "2026-09-21T13:00:00Z")

    with pytest.raises(PilotContractViolation) as excinfo:
        assert_artifact_binding(store)
    assert excinfo.value.code == "STALE_MARKER_POINTS_AT_NEW_STORE"
    assert excinfo.value.detail["markers_checked"] == 2
    stale_names = [entry["marker"] for entry in excinfo.value.detail["stale"]]
    assert stale_names == ["temperature.done"]


def test_markers_written_after_the_store_are_admitted(tmp_path):
    store = build_store(tmp_path / "2020_jan.zarr", created="2026-09-21T12:00:00Z")
    write_marker(tmp_path / ".markers_2020", "geopotential", "2026-09-21T12:30:00Z")
    write_marker(tmp_path / ".markers_2020", "temperature", "2026-09-21T14:00:00Z")

    report = assert_artifact_binding(store)
    assert report.markers_checked == 2
    assert report.stale_markers == []
    assert run_cli(store, ["--index", "1"]) == 0


def test_marker_in_the_same_second_is_admitted_with_slack(tmp_path):
    """Both sides write whole seconds; slack keeps that from being a fault."""
    store = build_store(tmp_path / "2020_jan.zarr", created="2026-09-21T12:00:05Z")
    write_marker(tmp_path / ".markers_2020", "geopotential", "2026-09-21T12:00:03Z")

    with pytest.raises(PilotContractViolation):
        assert_artifact_binding(store)
    report = assert_artifact_binding(store, marker_slack_seconds=5)
    assert report.stale_markers == []


def test_markers_survive_a_store_recreation_and_are_caught(tmp_path):
    """The real sequence: pull, mark, delete the store, recreate it."""
    import shutil

    store = tmp_path / "2020_jan.zarr"
    build_store(store, created="2026-09-20T09:00:00Z")
    markers = tmp_path / ".markers_2020"
    for name in ("geopotential", "temperature", "specific_humidity"):
        write_marker(markers, name, "2026-09-20T10:00:00Z")

    # Store is wiped and rebuilt; the markers are NOT cleared.
    shutil.rmtree(store)
    build_store(store, created="2026-09-21T09:00:00Z")

    assert markers.is_dir() and len(list(markers.glob("*.done"))) == 3
    with pytest.raises(PilotContractViolation) as excinfo:
        assert_artifact_binding(store)
    assert excinfo.value.code == "STALE_MARKER_POINTS_AT_NEW_STORE"
    assert len(excinfo.value.detail["stale"]) == 3


def test_store_without_creation_stamp_cannot_validate_its_markers(tmp_path):
    """Markers with no store creation time to compare against are refused."""
    store = build_store(tmp_path / "2020_jan.zarr", created=None)
    write_marker(tmp_path / ".markers_2020", "geopotential", "2026-09-21T12:30:00Z")

    with pytest.raises(PilotContractViolation) as excinfo:
        assert_artifact_binding(store)
    assert excinfo.value.code == "STORE_CREATION_STAMP_MISSING"


def test_store_with_no_markers_is_admitted(good_store):
    report = assert_artifact_binding(good_store)
    assert report.markers_checked == 0
    assert report.stale_markers == []


def test_negative_wrong_grid_hash_stamp(good_store):
    with pytest.raises(PilotContractViolation) as excinfo:
        assert_artifact_binding(good_store, expected_grid_hash="a-different-grid")
    assert excinfo.value.code == "STORE_GRID_HASH_MISMATCH"


def test_negative_wrong_variable_order_hash_stamp(good_store):
    with pytest.raises(PilotContractViolation) as excinfo:
        assert_artifact_binding(
            good_store, expected_variable_order_hash="a-different-order"
        )
    assert excinfo.value.code == "STORE_VARIABLE_ORDER_HASH_MISMATCH"


def test_negative_missing_store(tmp_path):
    missing = tmp_path / "does_not_exist.zarr"
    with pytest.raises(PilotContractViolation) as excinfo:
        assert_artifact_binding(missing)
    assert excinfo.value.code == "STORE_MISSING"
    assert preflight_cli.main(["--store", str(missing), "--no-grid-check"]) == 1


# =============================================================================
# validate_slice_index unit behaviour
# =============================================================================

def test_slice_index_resolves_absolute_endpoints():
    report = validate_slice_index(
        index=3, n_timesteps=10, history_steps=2, target_steps=4,
        interval_hours=None,
    )
    assert (report.history_index, report.target_index) == (1, 7)


def test_slice_index_allows_zero_width_span():
    report = validate_slice_index(
        index=0, n_timesteps=1, history_steps=0, target_steps=0, interval_hours=None
    )
    assert (report.history_index, report.target_index) == (0, 0)


def test_slice_index_rejects_empty_store():
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_slice_index(index=0, n_timesteps=0)
    assert excinfo.value.code == "STORE_EMPTY"


def test_slice_index_rejects_negative_span():
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_slice_index(index=1, n_timesteps=4, history_steps=-1)
    assert excinfo.value.code == "SLICE_SPAN_NEGATIVE"


def test_slice_index_rejects_non_integer_index():
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_slice_index(index=1.5, n_timesteps=4)
    assert excinfo.value.code == "SLICE_INDEX_NOT_INTEGER"


def test_slice_index_rejects_time_coord_length_mismatch():
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_slice_index(
            index=1, n_timesteps=4, time_coords=np.arange(3), interval_hours=None
        )
    assert excinfo.value.code == "TIME_COORD_LENGTH_MISMATCH"


def test_slice_index_accepts_numeric_hour_coordinates():
    report = validate_slice_index(
        index=2, n_timesteps=5, history_steps=1, target_steps=1,
        time_coords=np.arange(5) * 6.0, interval_hours=6,
    )
    assert report.interval_hours == 6


def test_slice_index_rejects_numeric_coordinates_with_wrong_spacing():
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_slice_index(
            index=2, n_timesteps=5, history_steps=1, target_steps=1,
            time_coords=np.arange(5) * 12.0, interval_hours=6,
        )
    assert excinfo.value.code == "TIME_SPACING_MISMATCH"


def test_slice_index_rejects_non_finite_time_coordinates():
    coords = np.arange(5, dtype=np.float64) * 6.0
    coords[2] = np.nan
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_slice_index(
            index=2, n_timesteps=5, history_steps=1, target_steps=1,
            time_coords=coords, interval_hours=6,
        )
    assert excinfo.value.code == "TIME_COORD_NONFINITE"


# =============================================================================
# validate_loaded_sample unit behaviour
# =============================================================================

def test_loaded_sample_accepts_a_clean_sample():
    report = validate_loaded_sample(
        np.ones((2, N_CHANNELS, N_LAT, N_LON), dtype="float32"),
        expected_channels=N_CHANNELS,
        expected_spatial_shape=(N_LAT, N_LON),
        lat=canonical_lat(), expected_lat=canonical_lat(),
        lon=canonical_lon(), expected_lon=canonical_lon(),
    )
    assert report.finite is True
    assert report.coordinates_checked is True
    assert report.n_channels == N_CHANNELS


def test_loaded_sample_rejects_empty():
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_loaded_sample(np.zeros((0, N_CHANNELS, N_LAT, N_LON)), expected_channels=N_CHANNELS)
    assert excinfo.value.code == "SAMPLE_EMPTY"


def test_loaded_sample_rejects_non_numeric():
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_loaded_sample(np.array([["a", "b"]]), expected_channels=None, channel_axis=-1)
    assert excinfo.value.code == "SAMPLE_NOT_NUMERIC"


def test_loaded_sample_rejects_wrong_spatial_shape():
    with pytest.raises(PilotContractViolation) as excinfo:
        validate_loaded_sample(
            np.ones((2, N_CHANNELS, N_LAT, N_LON + 1), dtype="float32"),
            expected_channels=N_CHANNELS,
            expected_spatial_shape=(N_LAT, N_LON),
        )
    assert excinfo.value.code == "SAMPLE_SPATIAL_SHAPE_MISMATCH"


def test_loaded_sample_accepts_xarray_dataarray(good_store):
    import xarray as xr

    dataset = xr.open_zarr(str(good_store))
    try:
        report = validate_loaded_sample(
            dataset["data"].isel(time=slice(0, 2)),
            expected_channels=N_CHANNELS,
            lat=dataset["lat"].values, expected_lat=canonical_lat(),
        )
    finally:
        dataset.close()
    assert report.finite is True
    assert report.shape[0] == 2


# =============================================================================
# The entry point is a real entry point
# =============================================================================

def test_cli_returns_two_when_it_cannot_run(tmp_path):
    """Bad arguments are distinguished from an admission failure."""
    store = build_store(tmp_path / "2020_jan.zarr")
    assert preflight_cli.main(
        ["--store", str(store), "--no-grid-check", "--history-steps", "-1"]
    ) == 2


def test_cli_parser_exposes_every_documented_switch():
    parser = preflight_cli.build_parser()
    options = {action.dest for action in parser._actions}
    assert {
        "store", "index", "history_steps", "target_steps", "interval_hours",
        "no_interval_check", "marker_dir", "expected_channels", "no_grid_check",
        "marker_slack_seconds", "json",
    } <= options


def test_cli_honours_no_interval_check(tmp_path):
    """Index-only time coordinates pass once the spacing check is waived."""
    store = build_store(
        tmp_path / "2020_jan.zarr",
        time_coords=np.arange(N_TIME).astype("float64"),
    )
    assert run_cli(store, ["--index", "1"]) == 1
    assert preflight_cli.main([
        "--store", str(store), "--expected-channels", str(N_CHANNELS),
        "--no-grid-check", "--no-interval-check", "--index", "1",
    ]) == 0


def test_cli_accepts_an_explicit_marker_dir(tmp_path):
    store = build_store(tmp_path / "pilot.zarr", created="2026-09-21T12:00:00Z")
    markers = tmp_path / "elsewhere"
    write_marker(markers, "geopotential", "2026-09-20T08:00:00Z")

    # Without --marker-dir the store name carries no year, so nothing is found.
    assert run_cli(store, ["--index", "1"]) == 0
    assert run_cli(store, ["--index", "1", "--marker-dir", str(markers)]) == 1


def test_preflight_reports_several_independent_faults_in_one_run(tmp_path):
    """One invocation surfaces every independently-checkable fault it reaches."""
    rng = np.random.default_rng(12)
    data = rng.normal(size=(N_TIME, N_CHANNELS, N_LAT, N_LON)).astype("float32")
    data[:, 1, :, :] = np.nan
    store = build_store(tmp_path / "2020_jan.zarr", data=data, created="2026-09-21T12:00:00Z")
    write_marker(tmp_path / ".markers_2020", "geopotential", "2026-09-19T08:00:00Z")

    codes = codes_for(store, index=1)
    assert "STALE_MARKER_POINTS_AT_NEW_STORE" in codes
    assert "SAMPLE_NON_FINITE" in codes
