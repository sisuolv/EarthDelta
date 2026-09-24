"""earthdelta.wbx.export: WeatherBench-layout export of EarthDelta forecasts (CPU).

Covers the plan's export checklist:
  * flatten/unflatten are bitwise inverses; channel count / level / order hash
    are enforced;
  * the decoded ``time`` coordinate must agree with the admitted issue times,
    and the official WB2 / WB-X time conventions apply unmodified;
  * denormalization is mandatory (normalized or doubly-denormalized fields
    fail the physical plausibility gate);
  * atomic zarr publish; truth extraction with year gate and gap detection;
  * end to end on a tiny synthetic Stormer: official WB-X RMSE == numpy RMSE ==
    EarthDelta ``full_objective_loss`` (and the scale-standardized objective).

Skips cleanly when xarray/zarr (or, for the WB-X parts, jax) are absent, so the
CPU baseline suite is unaffected.
"""
from __future__ import annotations

import os

# Must precede any google.protobuf import (timm -> torchvision -> onnx); see
# earthdelta/wbx/__init__.py. Makes these tests independent of the caller's env.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")
pytest.importorskip("pandas")

from earthdelta.data.pull_wb2 import (  # noqa: E402
    CANONICAL_VARIABLES, PRESSURE_LEVELS, PRESSURE_VARS, SINGLE_LEVEL_VARS, variable_order_hash,
)
from earthdelta.wbx import STORMER_EXPORT_PREFIX, _vendor, export  # noqa: E402

HAVE_REFS = all((_vendor.REFERENCE_ROOT / n / ".git").exists() for n in _vendor.VENDORED_PACKAGES)
NORM_DIR = _vendor.REFERENCE_ROOT / "stormer" / "normalization_constants"
NLAT, NLON = 8, 16
MODEL = f"{STORMER_EXPORT_PREFIX}-F0"


def _grid(nlat=NLAT, nlon=NLON):
    d = 180.0 / nlat
    lat = np.linspace(-90 + d / 2, 90 - d / 2, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    return lat, lon


def _plausible_raw(shape_prefix, rng, nlat=NLAT, nlon=NLON):
    """Physical-looking 69-channel fields (means inside the plausibility windows)."""
    base = np.zeros(len(CANONICAL_VARIABLES))
    for i, name in enumerate(CANONICAL_VARIABLES):
        if name == "2m_temperature" or name.startswith("temperature"):
            base[i] = 270.0
        elif name == "mean_sea_level_pressure":
            base[i] = 101000.0
        elif name.startswith("geopotential"):
            base[i] = 54000.0
        elif name.startswith("specific_humidity"):
            base[i] = 5e-3
        else:
            base[i] = 3.0
    noise = rng.standard_normal(tuple(shape_prefix) + (len(base), nlat, nlon))
    return (base[:, None, None] * (1 + 0.01 * noise)).astype(np.float32)


def _record(issue="2020-01-01T00", leads=(6, 24), seed=0, model=MODEL, **kw):
    rng = np.random.default_rng(seed)
    return export.ForecastRecord(model=model, issue_time=issue, lead_hours=leads,
                                 fields=_plausible_raw((len(leads),), rng), **kw)


# =============================================================================
# flatten / unflatten
# =============================================================================

def test_unflatten_flatten_bitwise_roundtrip():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2, 3, 69, NLAT, NLON)).astype(np.float32)
    lat, lon = _grid()
    times = np.array(["2020-01-01T00", "2020-01-01T12"], dtype="datetime64[ns]")
    leads = (np.array([6, 12, 24]) * np.timedelta64(1, "h")).astype("timedelta64[ns]")
    ds = export.unflatten_channels(data, lat=lat, lon=lon,
                                   leading=(("time", times), ("prediction_timedelta", leads)))
    assert set(ds.data_vars) == set(SINGLE_LEVEL_VARS) | set(PRESSURE_VARS)
    assert ds["geopotential"].dims == ("time", "prediction_timedelta", "level", "latitude", "longitude")
    assert ds["2m_temperature"].dims == ("time", "prediction_timedelta", "latitude", "longitude")
    assert list(ds["level"].values) == list(PRESSURE_LEVELS)
    assert ds.attrs["variable_order_hash"] == variable_order_hash()
    z500 = CANONICAL_VARIABLES.index("geopotential_500")
    assert np.array_equal(ds["geopotential"].sel(level=500).values, data[:, :, z500])
    back = export.flatten_channels(ds, leading_dims=("time", "prediction_timedelta"))
    assert back.dtype == data.dtype and np.array_equal(back, data)


@pytest.mark.parametrize("mutate", ["drop_last", "swap_pair", "rename"])
def test_channel_order_is_enforced(mutate):
    names = list(CANONICAL_VARIABLES)
    if mutate == "drop_last":
        names = names[:-1]
    elif mutate == "swap_pair":
        names[4], names[5] = names[5], names[4]
    else:
        names[0] = "t2m"
    with pytest.raises(export.ExportContractError):
        export.check_channel_order(names)
    lat, lon = _grid()
    data = np.zeros((len(names), NLAT, NLON), np.float32)
    with pytest.raises(export.ExportContractError):
        export.unflatten_channels(data, lat=lat, lon=lon, channel_names=names)


def test_flatten_rejects_wrong_levels_and_missing_variables():
    lat, lon = _grid()
    ds = export.unflatten_channels(np.zeros((69, NLAT, NLON), np.float32), lat=lat, lon=lon)
    bad = ds.assign_coords(level=np.array(PRESSURE_LEVELS[::-1]))
    with pytest.raises(export.ExportContractError):
        export.flatten_channels(bad)
    with pytest.raises(export.ExportContractError):
        export.flatten_channels(ds.drop_vars("specific_humidity"))


def test_unflatten_rejects_bad_grid_and_leading_coords():
    lat, lon = _grid()
    data = np.zeros((2, 69, NLAT, NLON), np.float32)
    with pytest.raises(export.ExportContractError):
        export.unflatten_channels(data, lat=lat[::-1], lon=lon, leading=(("time", np.arange(2)),))
    with pytest.raises(export.ExportContractError):
        export.unflatten_channels(data, lat=lat, lon=lon, leading=(("time", np.arange(3)),))
    with pytest.raises(export.ExportContractError):
        export.unflatten_channels(data, lat=lat, lon=lon)  # missing leading coord


# =============================================================================
# ForecastRecord / dataset / time conventions
# =============================================================================

@pytest.mark.parametrize("kwargs", [
    {"issue": "2020-01-01T03"},                 # not on the 6h grid
    {"issue": "2020-01-01T00:00:30"},           # seconds off the grid
    {"leads": (6, 9)},                          # lead not a multiple of 6
    {"leads": (0, 6)},                          # step 0 is not a forecast
    {"leads": (24, 6)},                         # not increasing
    {"leads": (6, 6)},                          # duplicated
    {"model": "Stormer-F0"},                    # misleading name (plan correction 9)
])
def test_forecast_record_rejects_invalid(kwargs):
    with pytest.raises(export.ExportContractError):
        _record(**kwargs)


def test_forecast_record_rejects_bad_fields():
    rng = np.random.default_rng(0)
    f = _plausible_raw((2,), rng)
    f[0, 0, 0, 0] = np.nan
    with pytest.raises(export.ExportContractError):
        export.ForecastRecord(model=MODEL, issue_time="2020-01-01", lead_hours=(6, 12), fields=f)
    with pytest.raises(export.ExportContractError):
        export.ForecastRecord(model=MODEL, issue_time="2020-01-01", lead_hours=(6, 12),
                              fields=_plausible_raw((2,), rng)[:, :68])


def test_forecast_record_converts_aware_timestamps_to_utc():
    rec = _record(issue="2020-01-01T08:00:00+08:00")
    assert rec.issue_time == np.datetime64("2020-01-01T00:00", "ns")


def _two_issue_dataset():
    lat, lon = _grid()
    recs = [_record("2020-01-01T12", seed=2, issue_id="b"), _record("2020-01-01T00", seed=1, issue_id="a")]
    return export.build_forecast_dataset(recs, lat=lat, lon=lon), recs


def test_build_dataset_uses_official_time_naming():
    ds, recs = _two_issue_dataset()
    assert list(ds["time"].values) == [np.datetime64("2020-01-01T00", "ns"), np.datetime64("2020-01-01T12", "ns")]
    assert [int(v / np.timedelta64(1, "h")) for v in ds["prediction_timedelta"].values] == [6, 24]
    assert json.loads(ds.attrs["issue_ids"]) == ["a", "b"]           # sorted by issue time
    back = export.flatten_channels(ds, leading_dims=("time", "prediction_timedelta"))
    assert np.array_equal(back[0], recs[1].fields) and np.array_equal(back[1], recs[0].fields)
    export.verify_issue_times(ds, ["2020-01-01T12:00", "2020-01-01T00:00"])
    with pytest.raises(export.ExportContractError):
        export.verify_issue_times(ds, ["2020-01-01T00", "2020-01-01T18"])     # off by one step
    with pytest.raises(export.ExportContractError):
        export.verify_issue_times(ds, ["2020-01-01T00"])                      # extra issue exported


@pytest.mark.skipif(not HAVE_REFS, reason="vendored weatherbench2/weatherbenchX clones not present")
def test_official_time_conventions_apply_unmodified():
    ds, _ = _two_issue_dataset()
    schema = _vendor.import_official("weatherbench2.schema")
    by_init = schema.apply_time_conventions(ds, by_init=True)
    expected = (ds["time"] + ds["prediction_timedelta"]).transpose("time", "prediction_timedelta")
    assert by_init["valid_time"].dims == ("init_time", "lead_time")
    assert np.array_equal(by_init["valid_time"].values, expected.values)
    by_valid = schema.apply_time_conventions(ds, by_init=False)
    assert "init_time" in by_valid.coords
    xl = _vendor.import_official("weatherbenchX.data_loaders.xarray_loaders")
    renamed = xl._rename_dataset(ds)
    assert {"init_time", "lead_time"} <= set(renamed.dims) and "time" not in renamed.dims


def test_build_dataset_rejects_inconsistent_records():
    lat, lon = _grid()
    with pytest.raises(export.ExportContractError):
        export.build_forecast_dataset([_record("2020-01-01T00"), _record("2020-01-01T00", seed=3)],
                                      lat=lat, lon=lon)
    with pytest.raises(export.ExportContractError):
        export.build_forecast_dataset([_record("2020-01-01T00"), _record("2020-01-01T06", leads=(6, 12))],
                                      lat=lat, lon=lon)
    with pytest.raises(export.ExportContractError):
        export.build_forecast_dataset([_record("2020-01-01T00"),
                                       _record("2020-01-01T06", model="persistence")], lat=lat, lon=lon)
    with pytest.raises(export.ExportContractError):
        export.build_forecast_dataset([], lat=lat, lon=lon)


# =============================================================================
# Atomic zarr publish
# =============================================================================

def test_write_forecast_zarr_roundtrip_and_manifest(tmp_path):
    ds, _ = _two_issue_dataset()
    path = tmp_path / "fc.zarr"
    manifest = export.write_forecast_zarr(ds, path)
    reread = xr.open_zarr(path).load()
    assert export._datasets_bitwise_equal(ds, reread)
    on_disk = json.loads((path / export.MANIFEST_NAME).read_text())
    assert on_disk["content_sha256"] == manifest["content_sha256"] == export.dataset_content_sha256(ds)
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".staging-")]
    with pytest.raises(FileExistsError):
        export.write_forecast_zarr(ds, path)
    export.write_forecast_zarr(ds, path, overwrite=True)


def test_write_forecast_zarr_failure_leaves_nothing(tmp_path, monkeypatch):
    ds, _ = _two_issue_dataset()

    def boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_zarr", boom)
    with pytest.raises(OSError):
        export.write_forecast_zarr(ds, tmp_path / "fc.zarr")
    assert list(tmp_path.iterdir()) == []


# =============================================================================
# Truth extraction
# =============================================================================

def _make_truth_store(path, times, *, channels=None, nan_at=None):
    lat, lon = _grid()
    rng = np.random.default_rng(7)
    data = _plausible_raw((len(times),), rng)
    if nan_at is not None:
        data[nan_at] = np.nan
    ds = xr.Dataset({"data": (("time", "channel", "lat", "lon"), data)},
                    coords={"time": np.array(times, dtype="datetime64[ns]"),
                            "channel": list(channels or CANONICAL_VARIABLES), "lat": lat, "lon": lon},
                    attrs={"variable_order_hash": variable_order_hash()})
    ds.to_zarr(path, mode="w")
    return data


def test_truth_from_store_exact_times(tmp_path):
    times = np.arange(np.datetime64("2020-01-01T00"), np.datetime64("2020-01-03T00"), np.timedelta64(6, "h"))
    data = _make_truth_store(tmp_path / "t.zarr", times)
    want = [times[1], times[4]]
    truth = export.truth_from_store(tmp_path / "t.zarr", want)
    assert list(truth["time"].values) == [np.datetime64(t, "ns") for t in want]
    assert np.array_equal(export.flatten_channels(truth, leading_dims=("time",)), data[[1, 4]])
    with pytest.raises(export.TruthDataGap):
        export.truth_from_store(tmp_path / "t.zarr", ["2020-01-01T03"])   # no nearest matching


def test_truth_from_store_detects_nan_gaps(tmp_path):
    times = np.array(["2020-02-01T00", "2020-02-01T06"], dtype="datetime64[ns]")
    _make_truth_store(tmp_path / "t.zarr", times, nan_at=(1, 20))
    with pytest.raises(export.TruthDataGap):
        export.truth_from_store(tmp_path / "t.zarr", times)
    ok = export.truth_from_store(tmp_path / "t.zarr", times[:1])
    assert np.isfinite(ok["u_component_of_wind"].values).all()


def test_truth_from_store_rejects_wrong_channel_order(tmp_path):
    names = list(CANONICAL_VARIABLES)
    names[1], names[2] = names[2], names[1]
    _make_truth_store(tmp_path / "t.zarr", ["2020-01-01T00"], channels=names)
    with pytest.raises(export.ExportContractError):
        export.truth_from_store(tmp_path / "t.zarr", ["2020-01-01T00"])


def test_truth_from_store_year_gate(tmp_path):
    from earthdelta.wbx.evaluate import YearNotAuthorized
    _make_truth_store(tmp_path / "t.zarr", ["2022-01-01T00"])
    with pytest.raises(YearNotAuthorized):
        export.truth_from_store(tmp_path / "t.zarr", ["2022-01-01T00"])


# =============================================================================
# Denormalization is mandatory
# =============================================================================

@pytest.fixture
def real_normalization():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    if not (NORM_DIR / "normalize_mean.npz").exists():
        pytest.skip("reference/stormer normalization constants not present")
    from earthdelta.bridge.stormer_bridge import POLICY_OFFICIAL_ZERO_DIFF_MEAN, NormalizationContract
    return torch, NormalizationContract.from_npz_dir(
        str(NORM_DIR), list(CANONICAL_VARIABLES), intervals=(6,), policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN)


def test_trajectory_to_raw_fields_denormalizes(real_normalization):
    torch, norm = real_normalization
    g = torch.Generator().manual_seed(0)
    traj = 0.1 * torch.randn(1, 4, 69, NLAT, NLON, generator=g)
    raw = export.trajectory_to_raw_fields(traj, norm, [1, 3])
    assert raw.shape == (1, 2, 69, NLAT, NLON)
    mean = norm.inp_mean.double().numpy()[:, None, None]
    std = norm.inp_std.double().numpy()[:, None, None]
    for j, s in enumerate([1, 3]):
        expect = traj[0, s].double().numpy() * std + mean
        np.testing.assert_allclose(raw[0, j], expect, rtol=2e-6, atol=1e-6 * np.abs(mean).max())
    # step 0 is the initial condition, out of range / unsorted steps rejected
    for steps in ([0, 1], [1, 4], [3, 1]):
        with pytest.raises(export.ExportContractError):
            export.trajectory_to_raw_fields(traj, norm, steps)


def test_normalized_or_doubly_denormalized_fields_are_rejected(real_normalization):
    torch, norm = real_normalization
    traj = 0.1 * torch.randn(1, 2, 69, NLAT, NLON)
    with pytest.raises(export.ExportContractError):
        export.check_physical_plausibility(traj[:, 1].numpy())                  # still normalized
    raw_state = norm.denormalize(traj[:, 1])
    with pytest.raises(export.ExportContractError):                             # "raw" fed as normalized
        export.states_to_raw_fields({1: raw_state}, norm)
    raw, steps = export.states_to_raw_fields({1: traj[:, 1]}, norm)
    assert steps == (1,) and raw.shape == (1, 1, 69, NLAT, NLON)


# =============================================================================
# End to end on a tiny synthetic Stormer
# =============================================================================

@pytest.mark.skipif(not HAVE_REFS, reason="vendored weatherbench2/weatherbenchX clones not present")
def test_end_to_end_tiny_stormer_three_way_metric_agreement(real_normalization, tmp_path):
    torch, norm = real_normalization
    _vendor.add_wbx_deps_path()
    pytest.importorskip("jax")
    from earthdelta.bridge import Stormer, WeatherStepBridge
    from earthdelta.contracts import ArtifactVersion
    from earthdelta.metrics_contract import full_objective_loss
    from earthdelta.static_adapter import area_weight_q
    from earthdelta.wbx import evaluate as ev

    variables = list(CANONICAL_VARIABLES)
    torch.manual_seed(3)
    model = Stormer(in_img_size=(NLAT, NLON), variables=variables, patch_size=2,
                    hidden_size=32, depth=1, num_heads=4, mlp_ratio=2.0)
    torch.nn.init.normal_(model.head.linear.weight, std=0.02)
    model.requires_grad_(False).eval()
    version = ArtifactVersion(backbone="stormer_test_f0", static_adapter="none", edit_bank="none",
                              normalization="pending", grid=f"{NLAT}x{NLON}", projection="patch2",
                              split="test", continuation="reference_after_hold")
    bridge = WeatherStepBridge(model, norm, version)
    rng = np.random.default_rng(5)
    lat, lon = _grid()
    issues = ["2020-01-01T00", "2020-01-01T12"]
    leads_steps = (1, 2)
    lead_hours = tuple(6 * s for s in leads_steps)
    records, truths = [], {}
    for k, issue in enumerate(issues):
        x_raw = torch.from_numpy(_plausible_raw((1,), rng))
        rec = export.rollout_forecast_record(
            bridge, norm.normalize(x_raw), issue_time=issue, lead_steps=leads_steps, model=MODEL,
            target_blocks=(0,), issue_id=f"i{k}")        # F0: existing fs_rollout_trajectory, no adapters
        assert rec.lead_hours == lead_hours
        raw = rec.fields
        records.append(rec)
        for j, h in enumerate(lead_hours):
            vt = np.datetime64(issue, "ns") + np.timedelta64(h, "h")
            truths[vt] = (raw[j].astype(np.float64)
                          * (1 + 0.003 * rng.standard_normal(raw[j].shape))).astype(np.float32)
    fc = export.build_forecast_dataset(records, lat=lat, lon=lon)
    export.verify_issue_times(fc, issues)
    export.write_forecast_zarr(fc, tmp_path / "fc.zarr")
    vts = sorted(truths)
    truth = export.unflatten_channels(np.stack([truths[t] for t in vts]), lat=lat, lon=lon,
                                      leading=(("time", np.array(vts, dtype="datetime64[ns]")),))
    export.write_forecast_zarr(truth, tmp_path / "truth.zarr")

    res, _ = ev.evaluate(str(tmp_path / "fc.zarr"), str(tmp_path / "truth.zarr"),
                         init_times=issues, lead_times=list(lead_hours),
                         metrics=ev.standard_metrics(wind_vector=False))
    # numpy from scratch: mean over issues of cos-lat weighted spatial MSE
    pred = np.stack([r.fields for r in records]).astype(np.float64)            # [I,H,V,lat,lon]
    tru = np.stack([[truths[np.datetime64(i, "ns") + np.timedelta64(h, "h")] for h in lead_hours]
                    for i in issues]).astype(np.float64)
    w = np.cos(np.deg2rad(lat))
    se = np.einsum("ihvab,a->ihv", (pred - tru) ** 2, w) / (w.sum() * NLON)
    np_mse = se.mean(axis=0)                                                   # [H,V]
    q = area_weight_q(torch.from_numpy(lat))
    wbx_mse = ev.per_channel_values(res, "mse")
    wbx_rmse = ev.per_channel_values(res, "rmse")
    worst = 0.0
    for c, name in enumerate(variables):
        for h in range(len(lead_hours)):
            ed = float(full_objective_loss(torch.from_numpy(pred[:, h:h + 1, c:c + 1]),
                                           torch.from_numpy(tru[:, h:h + 1, c:c + 1]), q).mean())
            for a, b in ((wbx_mse[name][h], np_mse[h, c]), (ed, np_mse[h, c]),
                         (wbx_rmse[name][h], math.sqrt(np_mse[h, c]))):
                worst = max(worst, abs(a - b) / abs(b))
    assert worst < 1e-6, worst
    # the real scale-standardized objective, reconstructed from WB-X MSEs
    scale = norm.inp_std.double().reshape(1, -1, 1, 1)
    ed_obj = full_objective_loss(torch.from_numpy(pred), torch.from_numpy(tru), q, scale=scale).mean()
    recon = np.mean([[wbx_mse[n][h] / float(scale.reshape(-1)[c]) ** 2 for c, n in enumerate(variables)]
                     for h in range(len(lead_hours))])
    assert abs(float(ed_obj) - recon) / recon < 1e-6
