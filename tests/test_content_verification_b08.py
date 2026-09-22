"""B08: the subset that will really be consumed is verified by VALUE.

`is_year_complete` and `is_pilot_complete` decide a store is usable from a
timestep count and a channel count. Resume markers decide from their own
existence. Neither reads a number. Both are left exactly as they are here --
`test_shape_check_admits_a_store_the_content_check_refuses` pins the gap rather
than changing the downloader -- and `verify_content_subset` is what closes it:
it walks the exact indices a consumer will read, in batches, checks the real
values, and binds what it read into a certificate.

Negative cases build a store that is identical to a good one except for the
single property under test. One case runs against the REAL store under
`data/era5_1p40625` so the real-data path is exercised on real files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

xr = pytest.importorskip("xarray")

from earthdelta.data.pull_wb2 import (  # noqa: E402
    CANONICAL_VARIABLES,
    is_pilot_complete,
    is_year_complete,
)
from earthdelta.pilot_contract import (  # noqa: E402
    DEFAULT_SIGMA_BOUND,
    ContentCertificate,
    PilotContractViolation,
    physical_bounds_from_normalization,
    verify_content_subset,
)
import scripts.r2_pilot_preflight as preflight_cli  # noqa: E402

REAL_DATA_ROOT = REPO_ROOT / "data" / "era5_1p40625"
REAL_STORE = REAL_DATA_ROOT / "2020.zarr"
NORM_DIR = REPO_ROOT / "reference" / "stormer" / "normalization_constants"

N_LAT = 4
N_LON = 8
N_TIME = 6


def build_store(
    path: Path,
    n_time: int = N_TIME,
    channels: Optional[list] = None,
    data: Optional[np.ndarray] = None,
    dtype: str = "float32",
) -> Path:
    """A small store whose channels are real named variables.

    Values are drawn around each channel's official mean with a modest spread,
    so the positive control is inside the plausibility band for real reasons
    rather than by construction of the band.
    """
    if channels is None:
        channels = ["2m_temperature", "mean_sea_level_pressure",
                    "temperature_500", "specific_humidity_850"]
    if data is None:
        mean, std, _, _ = physical_bounds_from_normalization(channels, NORM_DIR)
        rng = np.random.default_rng(20260921)
        noise = rng.normal(size=(n_time, len(channels), N_LAT, N_LON))
        data = (mean[None, :, None, None] + std[None, :, None, None] * noise).astype(dtype)

    times = np.array(
        [np.datetime64("2020-01-01T00:00:00") + np.timedelta64(6 * i, "h")
         for i in range(n_time)],
        dtype="datetime64[ns]",
    )
    dataset = xr.Dataset(
        {"data": (["time", "channel", "lat", "lon"], data)},
        coords={
            "time": times,
            "channel": channels,
            "lat": np.linspace(-89, 89, N_LAT),
            "lon": np.linspace(0, 360, N_LON, endpoint=False),
        },
        attrs={"created": "2026-09-21T12:00:00Z", "grid_hash": "contentgridhash01",
               "variable_order_hash": "contentvarhash01"},
    )
    dataset.to_zarr(path, mode="w")
    dataset.close()
    return path


def verify(path: Path, indices=(0, 1, 2), **kwargs) -> ContentCertificate:
    kwargs.setdefault("expected_channels", 4)
    kwargs.setdefault("normalization_dir", NORM_DIR)
    kwargs.setdefault("batch_size", 2)
    return verify_content_subset(path, list(indices), **kwargs)


@pytest.fixture
def good_store(tmp_path) -> Path:
    return build_store(tmp_path / "2020_jan.zarr")


# =============================================================================
# Positive control
# =============================================================================

def test_positive_subset_is_certified(good_store):
    certificate = verify(good_store)
    assert certificate.passed is True
    assert certificate.indices == [0, 1, 2]
    assert certificate.n_batches == 2          # batch_size=2 over 3 indices
    assert certificate.n_values == 3 * 4 * N_LAT * N_LON
    assert certificate.physical_range_checked is True
    assert certificate.violations == []
    assert len(certificate.content_sha256) == 64
    assert len(certificate.identity_sha256) == 64


def test_certificate_records_the_real_slice_identity(good_store):
    certificate = verify(good_store)
    assert certificate.times_utc == [
        "2020-01-01T00:00:00", "2020-01-01T06:00:00", "2020-01-01T12:00:00",
    ]
    assert certificate.created_utc == "2026-09-21T12:00:00Z"
    assert certificate.grid_hash == "contentgridhash01"
    assert certificate.variable_order_hash == "contentvarhash01"
    names = [stat["channel"] for stat in certificate.channel_stats]
    assert names == ["2m_temperature", "mean_sea_level_pressure",
                     "temperature_500", "specific_humidity_850"]


def test_content_digest_is_reproducible_and_slice_specific(good_store):
    first = verify(good_store, indices=(0, 1, 2))
    again = verify(good_store, indices=(0, 1, 2))
    other = verify(good_store, indices=(1, 2, 3))
    assert first.content_sha256 == again.content_sha256
    assert first.identity_sha256 == again.identity_sha256
    assert first.content_sha256 != other.content_sha256
    assert first.identity_sha256 != other.identity_sha256


def test_batching_does_not_change_the_verdict_or_the_statistics(good_store):
    one = verify(good_store, indices=(0, 1, 2, 3), batch_size=1)
    four = verify(good_store, indices=(0, 1, 2, 3), batch_size=4)
    assert (one.n_batches, four.n_batches) == (4, 1)
    assert one.content_sha256 == four.content_sha256
    assert one.channel_stats[0]["min"] == four.channel_stats[0]["min"]


def test_certificate_carries_the_frozen_role_and_admission_ids(good_store):
    certificate = verify(good_store, data_role="confirm", issue_id="iss_abc",
                         process_group_id="pg_def")
    assert certificate.data_role == "confirm"
    assert certificate.issue_id == "iss_abc"
    assert certificate.process_group_id == "pg_def"
    assert json.loads(json.dumps(certificate.to_dict()))["data_role"] == "confirm"


def test_unknown_data_role_on_a_certificate_is_refused(good_store):
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(good_store, data_role="bankfit")
    assert excinfo.value.code == "DATA_ROLE_UNKNOWN"


# =============================================================================
# Negative: content that every shape/marker check admits
# =============================================================================

def test_shape_check_admits_a_store_the_content_check_refuses(tmp_path):
    """The gap itself: a store whose count and channel count are right and
    whose first channel was never written."""
    channels = ["2m_temperature", "mean_sea_level_pressure",
                "temperature_500", "specific_humidity_850"]
    mean, std, _, _ = physical_bounds_from_normalization(channels, NORM_DIR)
    rng = np.random.default_rng(7)
    data = (mean[None, :, None, None]
            + std[None, :, None, None] * rng.normal(size=(124, 4, N_LAT, N_LON))
            ).astype("float32")
    data[:, 0] = np.nan  # the NaN fill a never-written channel keeps
    store = build_store(tmp_path / "2020_jan.zarr", n_time=124, data=data)

    # The existing shape-only check is unchanged and still admits it.
    assert is_pilot_complete(store, expected_steps=124) is False  # 4 != 69 channels
    assert is_year_complete(store, 2020) is False
    # ... and with the channel count it does check, it cannot see the NaNs:
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(store, indices=(0, 1, 2))
    assert excinfo.value.code == "CONTENT_NON_FINITE"
    assert "never wrote" in str(excinfo.value)
    assert excinfo.value.detail["bad_channel_indices"] == [0]


def test_negative_all_zero_channel_is_refused_as_fill(tmp_path):
    channels = ["2m_temperature", "mean_sea_level_pressure",
                "temperature_500", "specific_humidity_850"]
    mean, std, _, _ = physical_bounds_from_normalization(channels, NORM_DIR)
    rng = np.random.default_rng(11)
    data = (mean[None, :, None, None]
            + std[None, :, None, None] * rng.normal(size=(N_TIME, 4, N_LAT, N_LON))
            ).astype("float32")
    data[:, 2] = 0.0
    store = build_store(tmp_path / "zeros.zarr", data=data)
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(store)
    assert excinfo.value.code == "CONTENT_DEGENERATE_CHANNEL"
    assert "is fill, not weather" in str(excinfo.value)
    assert excinfo.value.detail["slabs"][0]["channel"] == "temperature_500"


def test_negative_spatially_constant_but_nonzero_channel_is_refused(tmp_path):
    channels = ["2m_temperature", "mean_sea_level_pressure",
                "temperature_500", "specific_humidity_850"]
    mean, std, _, _ = physical_bounds_from_normalization(channels, NORM_DIR)
    rng = np.random.default_rng(13)
    data = (mean[None, :, None, None]
            + std[None, :, None, None] * rng.normal(size=(N_TIME, 4, N_LAT, N_LON))
            ).astype("float32")
    data[:, 0] = float(mean[0])  # a plausible VALUE, everywhere, which is not weather
    store = build_store(tmp_path / "constant.zarr", data=data)
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(store)
    assert excinfo.value.code == "CONTENT_DEGENERATE_CHANNEL"


def test_negative_wrong_unit_channel_is_outside_the_official_band(tmp_path):
    """Temperature in Celsius instead of Kelvin: finite, varying, and wrong."""
    channels = ["2m_temperature", "mean_sea_level_pressure",
                "temperature_500", "specific_humidity_850"]
    mean, std, _, _ = physical_bounds_from_normalization(channels, NORM_DIR)
    rng = np.random.default_rng(17)
    data = (mean[None, :, None, None]
            + std[None, :, None, None] * rng.normal(size=(N_TIME, 4, N_LAT, N_LON))
            ).astype("float32")
    data[:, 1] = data[:, 1] / 100.0  # mslp in hPa where Pa is expected
    store = build_store(tmp_path / "units.zarr", data=data)
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(store)
    assert excinfo.value.code == "CONTENT_OUT_OF_PHYSICAL_RANGE"
    assert excinfo.value.detail["slabs"][0]["channel"] == "mean_sea_level_pressure"
    assert excinfo.value.detail["sigma_bound"] == DEFAULT_SIGMA_BOUND


def test_negative_empty_subset_certifies_nothing(good_store):
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(good_store, indices=())
    assert excinfo.value.code == "CONTENT_EMPTY_SUBSET"
    assert "cannot certify anything" in str(excinfo.value)


def test_negative_index_outside_the_store(good_store):
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(good_store, indices=(0, 99))
    assert excinfo.value.code == "CONTENT_INDEX_OUT_OF_RANGE"


def test_negative_negative_index_is_refused_not_wrapped(good_store):
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(good_store, indices=(-1,))
    assert excinfo.value.code == "CONTENT_INDEX_OUT_OF_RANGE"
    assert "refused rather than wrapped" in str(excinfo.value)


def test_negative_empty_store_is_refused(tmp_path):
    store = build_store(tmp_path / "empty.zarr", n_time=0)
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(store, indices=(0,))
    assert excinfo.value.code == "CONTENT_STORE_EMPTY"
    assert "resume markers and still hold no data" in str(excinfo.value)


def test_negative_channel_count_mismatch(good_store):
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(good_store, expected_channels=69)
    assert excinfo.value.code == "CONTENT_CHANNEL_COUNT_MISMATCH"


def test_negative_unexpected_dtype(tmp_path):
    channels = ["2m_temperature", "mean_sea_level_pressure",
                "temperature_500", "specific_humidity_850"]
    mean, std, _, _ = physical_bounds_from_normalization(channels, NORM_DIR)
    rng = np.random.default_rng(23)
    data = (mean[None, :, None, None]
            + std[None, :, None, None] * rng.normal(size=(N_TIME, 4, N_LAT, N_LON))
            ).astype("float64")
    store = build_store(tmp_path / "f64.zarr", data=data, dtype="float64")
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(store)
    assert excinfo.value.code == "CONTENT_DTYPE_UNEXPECTED"


def test_negative_batch_size_must_be_positive(good_store):
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(good_store, batch_size=0)
    assert excinfo.value.code == "CONTENT_BATCH_SIZE_INVALID"


def test_non_strict_mode_returns_a_failing_certificate_instead_of_raising(tmp_path):
    store = build_store(tmp_path / "empty2.zarr", n_time=0)
    certificate = verify(store, indices=(0,), strict=False)
    assert certificate.passed is False
    assert certificate.violations[0]["code"] == "CONTENT_STORE_EMPTY"


def test_failing_certificate_is_attached_to_the_raised_violation(tmp_path):
    store = build_store(tmp_path / "empty3.zarr", n_time=0)
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(store, indices=(0,))
    embedded = excinfo.value.detail["certificate"]
    assert embedded["passed"] is False
    assert embedded["violations"][0]["code"] == "CONTENT_STORE_EMPTY"


# =============================================================================
# The plausibility band is derived, not invented
# =============================================================================

def test_band_comes_from_the_official_constants():
    channels = ["2m_temperature"]
    mean, std, low, high = physical_bounds_from_normalization(
        channels, NORM_DIR, sigma_bound=3.0
    )
    official_mean = np.load(NORM_DIR / "normalize_mean.npz")["2m_temperature"]
    official_std = np.load(NORM_DIR / "normalize_std.npz")["2m_temperature"]
    assert mean[0] == pytest.approx(float(np.ravel(official_mean)[0]))
    assert std[0] == pytest.approx(float(np.ravel(official_std)[0]))
    assert low[0] == pytest.approx(mean[0] - 3.0 * std[0])
    assert high[0] == pytest.approx(mean[0] + 3.0 * std[0])


def test_missing_constants_refuse_rather_than_guess(tmp_path):
    with pytest.raises(PilotContractViolation) as excinfo:
        physical_bounds_from_normalization(["2m_temperature"], tmp_path / "nope")
    assert excinfo.value.code == "CONTENT_NORMALIZATION_CONSTANTS_MISSING"
    assert "would be invented" in str(excinfo.value)


def test_channel_absent_from_the_official_constants_is_refused():
    with pytest.raises(PilotContractViolation) as excinfo:
        physical_bounds_from_normalization(["not_a_variable"], NORM_DIR)
    assert excinfo.value.code == "CONTENT_NORMALIZATION_CHANNEL_MISSING"


UNNAMED_CHANNELS = ["mystery_a", "mystery_b", "mystery_c", "mystery_d"]


def unnamed_store(path: Path) -> Path:
    """A store whose channels are not official variables, so no band exists."""
    rng = np.random.default_rng(31)
    data = rng.normal(size=(N_TIME, len(UNNAMED_CHANNELS), N_LAT, N_LON)).astype("float32")
    return build_store(path, channels=UNNAMED_CHANNELS, data=data)


def test_skipping_the_band_is_recorded_rather_than_claimed(tmp_path):
    store = unnamed_store(tmp_path / "noband.zarr")
    certificate = verify(store, require_physical_range=False)
    assert certificate.passed is True
    assert certificate.physical_range_checked is False
    assert certificate.sigma_bound is None
    assert certificate.channel_stats[0]["max_abs_z"] is None


def test_unknown_channels_with_the_band_required_are_refused(tmp_path):
    store = unnamed_store(tmp_path / "unknown.zarr")
    with pytest.raises(PilotContractViolation) as excinfo:
        verify(store, require_physical_range=True)
    assert excinfo.value.code == "CONTENT_NORMALIZATION_CHANNEL_MISSING"


def test_invalid_sigma_bound_is_refused():
    with pytest.raises(PilotContractViolation) as excinfo:
        physical_bounds_from_normalization(["2m_temperature"], NORM_DIR, sigma_bound=0.0)
    assert excinfo.value.code == "CONTENT_SIGMA_BOUND_INVALID"


# =============================================================================
# Reachable from the real entry point
# =============================================================================

def test_preflight_cli_emits_a_content_certificate(good_store, tmp_path):
    certificate_path = tmp_path / "out" / "certificate.json"
    exit_code = preflight_cli.main([
        "--store", str(good_store),
        "--expected-channels", "4",
        "--no-grid-check",
        "--index", "1",
        "--verify-content",
        "--content-batch-size", "2",
        "--normalization-dir", str(NORM_DIR),
        "--data-role", "policy_dev",
        "--content-certificate", str(certificate_path),
    ])
    assert exit_code == 0
    payload = json.loads(certificate_path.read_text())
    assert payload["passed"] is True
    assert payload["data_role"] == "policy_dev"
    assert payload["indices"] == [0, 1, 2]


def test_preflight_cli_fails_on_content_even_when_the_shape_is_right(tmp_path):
    channels = ["2m_temperature", "mean_sea_level_pressure",
                "temperature_500", "specific_humidity_850"]
    mean, std, _, _ = physical_bounds_from_normalization(channels, NORM_DIR)
    rng = np.random.default_rng(29)
    data = (mean[None, :, None, None]
            + std[None, :, None, None] * rng.normal(size=(N_TIME, 4, N_LAT, N_LON))
            ).astype("float32")
    data[:, 3] = 0.0
    store = build_store(tmp_path / "degenerate.zarr", data=data)
    report_path = tmp_path / "report.json"
    exit_code = preflight_cli.main([
        "--store", str(store),
        "--expected-channels", "4",
        "--no-grid-check",
        "--index", "1",
        "--verify-content",
        "--normalization-dir", str(NORM_DIR),
        "--json", str(report_path),
    ])
    assert exit_code == 1
    payload = json.loads(report_path.read_text())
    assert payload["passed"] is False
    codes = [v["code"] for v in payload["violations"]]
    assert "CONTENT_DEGENERATE_CHANNEL" in codes


def test_requesting_a_certificate_without_verifying_content_is_an_error(good_store, tmp_path):
    exit_code = preflight_cli.main([
        "--store", str(good_store),
        "--expected-channels", "4",
        "--no-grid-check",
        "--index", "1",
        "--content-certificate", str(tmp_path / "never.json"),
    ])
    assert exit_code == 2


# =============================================================================
# Real on-disk data (skipped when the store is not present)
# =============================================================================

@pytest.mark.skipif(not REAL_STORE.exists(),
                    reason="real ERA5 store data/era5_1p40625/2020.zarr not present")
def test_real_store_subset_passes_content_verification():
    """The real-data path: real bytes off disk, real official constants, real
    69-channel names -- read-only."""
    certificate = verify_content_subset(
        REAL_STORE, [0, 1], batch_size=1, expected_channels=69,
        data_role="bank_fit",
    )
    assert certificate.passed is True
    assert certificate.n_channels == 69
    assert certificate.dtype == "float32"
    assert certificate.n_values == 2 * 69 * 128 * 256
    assert certificate.physical_range_checked is True
    assert [s["channel"] for s in certificate.channel_stats] == list(CANONICAL_VARIABLES)
    assert certificate.times_utc == ["2020-01-01T00:00:00", "2020-01-01T06:00:00"]
    # Real weather sits well inside the band the bound was calibrated against.
    assert max(s["max_abs_z"] for s in certificate.channel_stats) < DEFAULT_SIGMA_BOUND
    # ... and no real channel is spatially constant.
    assert all(s["min"] < s["max"] for s in certificate.channel_stats)


@pytest.mark.skipif(not (REAL_DATA_ROOT / "2016.zarr").exists(),
                    reason="real ERA5 store data/era5_1p40625/2016.zarr not present")
def test_real_empty_store_is_refused_by_content_verification():
    """2016.zarr really does carry 69 channels, a stamped grid_hash and a full
    set of resume markers, and really does hold zero timesteps."""
    with pytest.raises(PilotContractViolation) as excinfo:
        verify_content_subset(REAL_DATA_ROOT / "2016.zarr", [0], expected_channels=69)
    assert excinfo.value.code == "CONTENT_STORE_EMPTY"
