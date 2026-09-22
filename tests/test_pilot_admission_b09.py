"""B09: a manifest row is admitted only when its real endpoints exist on disk.

`build_manifest` is calendar arithmetic. It emits a row for every
(issue_time, lead) the schedule would contain, whether or not anything was ever
pulled for those times, and before this module nothing downstream distinguished
"the sample exists" from "the sample was scheduled".

Every negative case here builds a synthetic store that is identical to a good
one except for the single property under test, then drives the real
`admit_real_sample` / `admit_manifest_rows` entry points over it and asserts the
specific violation code. Two cases run against the REAL stores under
`data/era5_1p40625` when they are present: one year that really holds data, and
one year whose store really is empty.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

xr = pytest.importorskip("xarray")

from earthdelta.data.make_splits import (  # noqa: E402
    PLACEHOLDER_NORMALIZATION_HASH,
    AdmissionResult,
    SampleAdmissionError,
    SplitManifestRow,
    admit_manifest_rows,
    admit_real_sample,
    build_manifest,
    compute_issue_id,
    default_store_for_year,
    is_placeholder_normalization_hash,
    new_process_group_id,
)
from earthdelta.pilot_contract import DATA_ROLE_VALUES, DataRole  # noqa: E402

REAL_DATA_ROOT = REPO_ROOT / "data" / "era5_1p40625"

N_CHANNELS = 5
N_LAT = 4
N_LON = 8
INTERVAL_HOURS = 6
GRID_HASH = "admitgridhash01"
REAL_NORM_HASH = "b6c1f0aa6c3f4e1d9a0b7c2d3e4f5061"


# =============================================================================
# Synthetic store fixture
# =============================================================================

def build_store(
    path: Path,
    start: str = "2020-01-01T00:00:00",
    n_time: int = 8,
    interval_hours: int = INTERVAL_HOURS,
    drop_indices: Optional[List[int]] = None,
    n_channels: int = N_CHANNELS,
    grid_hash: Optional[str] = GRID_HASH,
) -> Path:
    """Write one synthetic store; every knob defaults to the good value."""
    times = [
        np.datetime64(start) + np.timedelta64(interval_hours * i, "h")
        for i in range(n_time)
    ]
    if drop_indices:
        times = [t for i, t in enumerate(times) if i not in set(drop_indices)]
    time_coords = np.array(times, dtype="datetime64[ns]")

    rng = np.random.default_rng(20260921)
    data = rng.normal(size=(len(time_coords), n_channels, N_LAT, N_LON)).astype("float32")

    attrs = {"created": "2026-09-21T12:00:00Z"}
    if grid_hash is not None:
        attrs["grid_hash"] = grid_hash

    dataset = xr.Dataset(
        {"data": (["time", "channel", "lat", "lon"], data)},
        coords={
            "time": time_coords,
            "channel": [f"var_{i}" for i in range(n_channels)],
            "lat": np.linspace(-89, 89, N_LAT),
            "lon": np.linspace(0, 360, N_LON, endpoint=False),
        },
        attrs=attrs,
    )
    dataset.to_zarr(path, mode="w")
    dataset.close()
    return path


def make_row(
    issue: str = "2020-01-01T06:00:00",
    lead_hours: int = 6,
    split_id: str = "test",
    normalization_hash: str = REAL_NORM_HASH,
    grid_hash: str = GRID_HASH,
    data_role: Optional[str] = None,
) -> SplitManifestRow:
    issue_dt = datetime.fromisoformat(issue).replace(tzinfo=timezone.utc)
    valid_dt = issue_dt + timedelta(hours=lead_hours)
    issue_time = int(issue_dt.timestamp())
    return SplitManifestRow(
        issue_time=issue_time,
        valid_time=int(valid_dt.timestamp()),
        available_time=int(valid_dt.timestamp()) + 6 * 3600,
        event_id=f'{issue_dt.strftime("%Y%m%d%H")}_L{lead_hours:03d}',
        split_id=split_id,
        normalization_hash=normalization_hash,
        grid_hash=grid_hash,
        availability_source="reanalysis_retrospective",
        issue_id=compute_issue_id(issue_time, split_id),
        data_role=data_role,
    )


@pytest.fixture
def data_root(tmp_path) -> Path:
    root = tmp_path / "stores"
    root.mkdir()
    build_store(root / "2020.zarr")
    return root


def admit(row, root, **kwargs):
    kwargs.setdefault("expected_channels", N_CHANNELS)
    kwargs.setdefault("interval_hours", INTERVAL_HOURS)
    return admit_real_sample(row, root, **kwargs)


# =============================================================================
# Positive control
# =============================================================================

def test_positive_row_is_admitted_with_resolved_real_indices(data_root):
    result = admit(make_row(), data_root)
    assert isinstance(result, AdmissionResult)
    assert result.admitted is True
    # 2020-01-01T00,06,12 -> history 0, issue 1, target 2
    assert (result.history_index, result.issue_index, result.target_index) == (0, 1, 2)
    assert result.n_timesteps == 8
    assert "endpoints_exist_on_disk" in result.checks
    assert "span_uniformly_spaced" in result.checks


def test_admission_stamps_issue_id_and_independent_process_group_id(data_root):
    first = admit(make_row(), data_root)
    second = admit(make_row(), data_root)

    # issue_id is deterministic: the same analysis time in the same split maps
    # to the same id in every process.
    assert first.issue_id == second.issue_id == compute_issue_id(
        first.issue_time, first.split_id
    )
    # process_group_id is NOT: two admission runs over the same row stay
    # distinguishable in the record afterwards.
    assert first.process_group_id != second.process_group_id
    assert first.process_group_id.startswith("pg_")
    assert first.issue_id.startswith("iss_")


def test_issue_id_is_shared_across_leads_and_split_specific():
    """Every lead of one analysis time shares an issue_id, so sharding on it
    cannot split one analysis state across shards."""
    six = make_row(lead_hours=6)
    one_sixty_eight = make_row(lead_hours=168)
    assert six.issue_id == one_sixty_eight.issue_id
    other_split = make_row(split_id="train")
    assert other_split.issue_id != six.issue_id


def test_build_manifest_populates_issue_id_for_every_row():
    manifest = build_manifest(years=[2020], lead_hours=[6, 24])
    assert all(row.issue_id for row in manifest.rows)
    by_issue = {}
    for row in manifest.rows:
        by_issue.setdefault(row.issue_time, set()).add(row.issue_id)
    assert all(len(ids) == 1 for ids in by_issue.values())


# =============================================================================
# Negative: endpoints that the calendar claims and the data does not have
# =============================================================================

def test_negative_missing_target_endpoint(data_root):
    """Lead runs past the end of the store: the calendar says the sample
    exists, the data says it does not."""
    row = make_row(issue="2020-01-02T12:00:00", lead_hours=24)
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(row, data_root)
    assert excinfo.value.code == "ADMISSION_TARGET_ENDPOINT_MISSING"
    assert "target endpoint" in str(excinfo.value)


def test_negative_missing_history_endpoint(tmp_path):
    """The first timestep in the store has no step before it, and the step
    before it is inside the same year -- so the store exists and the endpoint
    does not."""
    root = tmp_path / "nohistory"
    root.mkdir()
    build_store(root / "2020.zarr", start="2020-01-02T00:00:00", n_time=8)
    row = make_row(issue="2020-01-02T00:00:00")
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(row, root, history_steps=1)
    assert excinfo.value.code == "ADMISSION_HISTORY_ENDPOINT_MISSING"
    assert "history endpoint" in str(excinfo.value)


def test_negative_history_endpoint_falls_into_an_unpulled_previous_year(data_root):
    """History crossing back over a year boundary is resolved in the PREVIOUS
    year's store, which here was never pulled."""
    row = make_row(issue="2020-01-01T00:00:00")
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(row, data_root, history_steps=1)
    assert excinfo.value.code == "ADMISSION_STORE_MISSING"
    assert "2019.zarr" in str(excinfo.value)


def test_negative_missing_current_endpoint(tmp_path):
    """An interior timestep is absent, so the issue time itself is missing --
    the store is still 'complete' by any count-based check."""
    root = tmp_path / "gap"
    root.mkdir()
    build_store(root / "2020.zarr", n_time=8, drop_indices=[1])
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(make_row(issue="2020-01-01T06:00:00"), root)
    assert excinfo.value.code == "ADMISSION_CURRENT_ENDPOINT_MISSING"


def test_negative_interior_gap_breaks_the_span(tmp_path):
    """Both endpoints exist and the span between them is not contiguous."""
    root = tmp_path / "interior"
    root.mkdir()
    build_store(root / "2020.zarr", n_time=8, drop_indices=[2])
    row = make_row(issue="2020-01-01T06:00:00", lead_hours=12)
    with pytest.raises(Exception) as excinfo:
        admit(row, root)
    assert getattr(excinfo.value, "code", "") == "TIME_SPACING_MISMATCH"


def test_negative_store_missing_entirely(data_root):
    """A year nobody ever pulled."""
    row = make_row(issue="2021-01-01T06:00:00")
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(row, data_root)
    assert excinfo.value.code == "ADMISSION_STORE_MISSING"
    assert "a calendar entry is not a sample" in str(excinfo.value)


def test_negative_store_present_but_empty(tmp_path):
    """The failure mode the real 2016/2017 stores exhibit: a full channel axis,
    a stamped identity, and zero timesteps."""
    root = tmp_path / "empty"
    root.mkdir()
    build_store(root / "2020.zarr", n_time=0)
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(make_row(), root)
    assert excinfo.value.code == "ADMISSION_STORE_EMPTY"
    assert "0 timesteps" in str(excinfo.value)


def test_negative_cross_year_target_lands_in_a_missing_store(tmp_path):
    """The lead crosses into the next year, whose store was never pulled."""
    root = tmp_path / "crossyear"
    root.mkdir()
    build_store(root / "2019.zarr", start="2019-12-31T00:00:00", n_time=4)
    row = make_row(issue="2019-12-31T12:00:00", lead_hours=12)
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(row, root)
    assert excinfo.value.code == "ADMISSION_STORE_MISSING"
    assert "2020.zarr" in str(excinfo.value)


def test_cross_year_target_is_resolved_in_the_next_years_store(tmp_path):
    """The positive half of the same case: the next year's store exists, and
    the target endpoint is resolved there rather than in the issue store."""
    root = tmp_path / "crossyear_ok"
    root.mkdir()
    build_store(root / "2019.zarr", start="2019-12-31T00:00:00", n_time=4)
    build_store(root / "2020.zarr", start="2020-01-01T00:00:00", n_time=4)
    result = admit(make_row(issue="2019-12-31T18:00:00", lead_hours=6), root)
    assert result.admitted is True
    assert result.issue_store.endswith("2019.zarr")
    assert result.target_store.endswith("2020.zarr")
    assert result.target_index == 0
    # The uniform-spacing check only applies within one store.
    assert "span_uniformly_spaced" not in result.checks


# =============================================================================
# Negative: formal mode refusals
# =============================================================================

def test_negative_formal_mode_rejects_placeholder_normalization_hash(data_root):
    row = make_row(normalization_hash=PLACEHOLDER_NORMALIZATION_HASH,
                   data_role="bank_fit")
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(row, data_root, formal=True)
    assert excinfo.value.code == "ADMISSION_PLACEHOLDER_NORMALIZATION_HASH"
    assert "placeholder_1979_2018" in str(excinfo.value)


def test_non_formal_mode_still_admits_a_placeholder_hash(data_root):
    """The calendar manifest itself stays usable: the refusal is a property of
    formal admission, not a change to the scheduling table."""
    row = make_row(normalization_hash=PLACEHOLDER_NORMALIZATION_HASH)
    assert admit(row, data_root, formal=False).admitted is True


@pytest.mark.parametrize("value", [
    None, "", "   ", "placeholder_1979_2018", "PLACEHOLDER", "todo", "tbd-hash",
    "dummy_norm", "fake", "unknown", "changeme", "default_norm",
])
def test_placeholder_detector_rejects_stand_ins(value):
    assert is_placeholder_normalization_hash(value) is True


@pytest.mark.parametrize("value", [REAL_NORM_HASH, "a" * 64, "ed-norm-identity/1:abc123"])
def test_placeholder_detector_accepts_real_digests(value):
    assert is_placeholder_normalization_hash(value) is False


def test_negative_formal_mode_requires_a_declared_data_role(data_root):
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(make_row(), data_root, formal=True)
    assert excinfo.value.code == "ADMISSION_DATA_ROLE_REQUIRED"
    assert "confirm" in str(excinfo.value)


def test_negative_unknown_data_role_is_refused(data_root):
    with pytest.raises(Exception) as excinfo:
        admit(make_row(), data_root, data_role="bank-fit")
    assert getattr(excinfo.value, "code", "") == "DATA_ROLE_UNKNOWN"


@pytest.mark.parametrize("role", list(DATA_ROLE_VALUES))
def test_each_frozen_data_role_is_carried_onto_the_result(data_root, role):
    result = admit(make_row(normalization_hash=REAL_NORM_HASH), data_root,
                   formal=True, data_role=role)
    assert result.data_role == role


def test_frozen_roles_are_exactly_the_three_specified():
    assert set(DATA_ROLE_VALUES) == {"bank_fit", "policy_dev", "confirm"}
    assert {r.value for r in DataRole} == set(DATA_ROLE_VALUES)


# =============================================================================
# Negative: identity binding
# =============================================================================

def test_negative_grid_hash_mismatch_between_row_and_store(tmp_path):
    root = tmp_path / "grid"
    root.mkdir()
    build_store(root / "2020.zarr", grid_hash="some_other_grid")
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(make_row(), root)
    assert excinfo.value.code == "ADMISSION_GRID_HASH_MISMATCH"


def test_negative_channel_count_mismatch(tmp_path):
    root = tmp_path / "channels"
    root.mkdir()
    build_store(root / "2020.zarr", n_channels=N_CHANNELS - 1)
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(make_row(), root)
    assert excinfo.value.code == "ADMISSION_CHANNEL_COUNT_MISMATCH"


def test_negative_row_that_fails_its_own_validation(data_root):
    row = make_row()
    row.valid_time = row.issue_time - 3600
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(row, data_root)
    assert excinfo.value.code == "ADMISSION_ROW_INVALID"


def test_default_store_resolver_does_not_fall_back_to_intermediate(tmp_path):
    root = tmp_path / "intermediate_only"
    root.mkdir()
    build_store(root / "2020_intermediate.zarr")
    assert default_store_for_year(root, 2020).name == "2020.zarr"
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit(make_row(), root)
    assert excinfo.value.code == "ADMISSION_STORE_MISSING"


# =============================================================================
# Batch admission
# =============================================================================

def test_batch_admission_shares_one_process_group_and_collects_rejections(data_root):
    rows = [
        make_row(issue="2020-01-01T06:00:00"),   # admitted
        make_row(issue="2020-01-01T12:00:00"),   # admitted
        make_row(issue="2020-01-03T00:00:00"),   # past the end of the store
    ]
    report = admit_manifest_rows(rows, data_root, expected_channels=N_CHANNELS,
                                 interval_hours=INTERVAL_HOURS)
    assert report.requested == 3
    assert len(report.admitted) == 2
    assert len(report.rejected) == 1
    assert report.passed is False
    assert report.rejection_codes() == {"ADMISSION_CURRENT_ENDPOINT_MISSING": 1}
    assert len({r.process_group_id for r in report.admitted}) == 1
    assert report.admitted[0].process_group_id == report.process_group_id


def test_batch_admission_report_serializes(data_root):
    report = admit_manifest_rows([make_row()], data_root,
                                 expected_channels=N_CHANNELS,
                                 interval_hours=INTERVAL_HOURS)
    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["n_admitted"] == 1
    assert payload["admitted"][0]["issue_id"].startswith("iss_")
    assert payload["process_group_id"].startswith("pg_")


def test_empty_batch_is_not_reported_as_passing(data_root):
    report = admit_manifest_rows([], data_root)
    assert report.requested == 0
    assert report.passed is False


def test_build_manifest_formal_mode_refuses_a_placeholder_hash():
    with pytest.raises(ValueError, match="requires a real normalization_hash"):
        build_manifest(years=[2020], lead_hours=[6], formal=True)
    manifest = build_manifest(years=[2020], lead_hours=[6], formal=True,
                              normalization_hash=REAL_NORM_HASH)
    assert manifest.rows[0].normalization_hash == REAL_NORM_HASH


def test_manifest_row_round_trips_the_new_fields(tmp_path):
    row = make_row(data_role="policy_dev")
    restored = SplitManifestRow.from_dict(row.to_dict())
    assert restored.issue_id == row.issue_id
    assert restored.data_role == "policy_dev"
    assert restored.validate()


def test_manifest_row_with_unknown_role_fails_validation():
    row = make_row()
    row.data_role = "not_a_role"
    assert row.validate() is False


def test_process_group_ids_are_unique_across_calls():
    ids = {new_process_group_id() for _ in range(32)}
    assert len(ids) == 32


# =============================================================================
# Admission can carry the content certificate with it
# =============================================================================

OFFICIAL_CHANNELS = ["2m_temperature", "mean_sea_level_pressure",
                     "temperature_500", "specific_humidity_850"]
NORM_DIR = REPO_ROOT / "reference" / "stormer" / "normalization_constants"


def build_named_store(path: Path, start: str = "2020-01-01T00:00:00",
                      n_time: int = 8) -> Path:
    """A store whose channels are official variables, so the band applies."""
    from earthdelta.pilot_contract import physical_bounds_from_normalization

    mean, std, _, _ = physical_bounds_from_normalization(OFFICIAL_CHANNELS, NORM_DIR)
    rng = np.random.default_rng(20260922)
    data = (mean[None, :, None, None]
            + std[None, :, None, None]
            * rng.normal(size=(n_time, len(OFFICIAL_CHANNELS), N_LAT, N_LON))
            ).astype("float32")
    times = np.array(
        [np.datetime64(start) + np.timedelta64(INTERVAL_HOURS * i, "h")
         for i in range(n_time)],
        dtype="datetime64[ns]",
    )
    dataset = xr.Dataset(
        {"data": (["time", "channel", "lat", "lon"], data)},
        coords={
            "time": times,
            "channel": OFFICIAL_CHANNELS,
            "lat": np.linspace(-89, 89, N_LAT),
            "lon": np.linspace(0, 360, N_LON, endpoint=False),
        },
        attrs={"created": "2026-09-21T12:00:00Z", "grid_hash": GRID_HASH},
    )
    dataset.to_zarr(path, mode="w")
    dataset.close()
    return path


def test_admission_can_attach_a_content_certificate(tmp_path):
    root = tmp_path / "withcontent"
    root.mkdir()
    build_named_store(root / "2020.zarr")
    result = admit_real_sample(
        make_row(), root, expected_channels=len(OFFICIAL_CHANNELS),
        interval_hours=INTERVAL_HOURS, verify_content=True,
        content_batch_size=2, normalization_dir=NORM_DIR,
        data_role="bank_fit",
    )
    assert "content_verified" in result.checks
    certificate = result.content_certificate
    assert certificate["passed"] is True
    assert certificate["indices"] == [0, 1, 2]
    assert certificate["issue_id"] == result.issue_id
    assert certificate["process_group_id"] == result.process_group_id
    assert certificate["data_role"] == "bank_fit"


def test_content_verification_of_a_cross_store_span_is_refused_not_faked(tmp_path):
    """The span crosses a year boundary, so no single certificate covers it;
    the gate says so instead of certifying the part it can reach."""
    root = tmp_path / "crosscontent"
    root.mkdir()
    build_named_store(root / "2019.zarr", start="2019-12-31T00:00:00", n_time=4)
    build_named_store(root / "2020.zarr", start="2020-01-01T00:00:00", n_time=4)
    with pytest.raises(SampleAdmissionError) as excinfo:
        admit_real_sample(
            make_row(issue="2019-12-31T18:00:00", lead_hours=6), root,
            expected_channels=len(OFFICIAL_CHANNELS),
            interval_hours=INTERVAL_HOURS, verify_content=True,
            normalization_dir=NORM_DIR,
        )
    assert excinfo.value.code == "ADMISSION_CONTENT_SPAN_CROSSES_STORES"
    assert "rather than reported as verified" in str(excinfo.value)


# =============================================================================
# The gate entry point
# =============================================================================

def gate_root(tmp_path: Path) -> Path:
    """A store covering the first manifest row of 2020 and its endpoints.

    The 2020 ('test') split guards the first 24h, so the earliest scheduled
    issue time is 2020-01-02T00:00; with one step of history and a 6h lead the
    sample spans 2020-01-01T18:00 .. 2020-01-02T06:00.
    """
    from earthdelta.data.pull_wb2 import grid_hash as canonical_grid_hash

    root = tmp_path / "gate"
    root.mkdir()
    # `build_manifest` stamps the canonical grid hash on every row, so the
    # store the gate reads has to carry the same one.
    build_store(root / "2020.zarr", start="2020-01-01T00:00:00", n_time=8,
                grid_hash=canonical_grid_hash())
    return root


def test_gate_entry_point_passes_on_admissible_rows(tmp_path):
    import scripts.r2_admission_gate as gate

    out = tmp_path / "record.json"
    exit_code = gate.main([
        "--data-root", str(gate_root(tmp_path)),
        "--years", "2020",
        "--limit", "1",
        "--expected-channels", str(N_CHANNELS),
        "--build-pilot-registry", "3",
        "--json", str(out),
    ])
    assert exit_code == 0
    import json
    record = json.loads(out.read_text())
    assert record["passed"] is True
    assert record["results"]["b09_real_sample_admission"] is True
    assert record["results"]["b15_candidate_registry"] is True
    assert record["admission"]["n_admitted"] == 1
    assert record["admission"]["admitted"][0]["issue_id"].startswith("iss_")


def test_gate_entry_point_fails_when_the_year_was_never_pulled(tmp_path):
    import scripts.r2_admission_gate as gate

    root = tmp_path / "missing"
    root.mkdir()
    exit_code = gate.main([
        "--data-root", str(root),
        "--years", "2020",
        "--limit", "2",
        "--expected-channels", str(N_CHANNELS),
    ])
    assert exit_code == 1


def test_gate_entry_point_requires_a_data_role_in_formal_mode(tmp_path):
    import scripts.r2_admission_gate as gate

    exit_code = gate.main([
        "--data-root", str(gate_root(tmp_path)),
        "--years", "2020",
        "--formal",
    ])
    assert exit_code == 2


# =============================================================================
# Real on-disk data (skipped when the stores are not present)
# =============================================================================

def _real_store(year: int) -> Path:
    return REAL_DATA_ROOT / f"{year}.zarr"


@pytest.mark.skipif(not _real_store(2020).exists(),
                    reason="real ERA5 store data/era5_1p40625/2020.zarr not present")
def test_real_data_rows_are_admitted_against_the_real_store():
    """The real-data path: a real manifest row joined against the real store."""
    manifest = build_manifest(years=[2020], lead_hours=[6])
    report = admit_manifest_rows(manifest.rows, REAL_DATA_ROOT, limit=3)
    assert report.passed is True, report.rejected
    first = report.admitted[0]
    assert first.issue_store.endswith("2020.zarr")
    assert first.target_index == first.issue_index + 1
    assert first.history_index == first.issue_index - 1
    assert first.n_timesteps == 1464  # 2020 is a leap year: 366 * 4


@pytest.mark.skipif(not _real_store(2016).exists(),
                    reason="real ERA5 store data/era5_1p40625/2016.zarr not present")
def test_real_empty_store_rejects_every_calendar_row():
    """2016 is inside the train split and its real store holds 0 timesteps.

    The calendar emits a full year of rows for it regardless; admission against
    the real store rejects all of them. This is the B09 gap on real data, not a
    fixture.
    """
    manifest = build_manifest(years=[2016], lead_hours=[6])
    assert len(manifest.rows) > 1000  # the calendar is happy to schedule them
    report = admit_manifest_rows(manifest.rows, REAL_DATA_ROOT, limit=5)
    assert report.passed is False
    assert set(report.rejection_codes()) == {"ADMISSION_STORE_EMPTY"}
    assert len(report.admitted) == 0
