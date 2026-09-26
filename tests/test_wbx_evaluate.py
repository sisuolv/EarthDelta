"""earthdelta.wbx.evaluate / _vendor / baselines: official WB-X driver and gates (CPU).

  * chunked aggregation (any init-chunk size) == single-chunk official result;
  * WB-X RMSE / MSE / Bias == from-scratch numpy, and the EarthDelta
    scale-standardized objective is reconstructed from WB-X MSEs;
  * GridAreaWeighting == EarthDelta cos-lat ``area_weight_q`` (plan item 8);
  * unauthorized evaluation years are rejected (confirm-freeze hash gate);
  * vendored pins fail closed; beam-dependent modules are never imported;
  * baseline registry matches the official configs; 06Z/18Z are unpairable.
"""
from __future__ import annotations

import os

# Must precede any google.protobuf import (timm -> torchvision -> onnx); see
# earthdelta/wbx/__init__.py. Makes these tests independent of the caller's env.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

xr = pytest.importorskip("xarray")
pytest.importorskip("pandas")
from earthdelta.wbx import _vendor  # noqa: E402
from earthdelta.wbx import evaluate as ev  # noqa: E402

HAVE_REFS = all((_vendor.REFERENCE_ROOT / n / ".git").exists() for n in _vendor.VENDORED_PACKAGES)


@pytest.fixture(scope="module")
def wbx():
    if not HAVE_REFS:
        pytest.skip("vendored weatherbench2/weatherbenchX clones not present")
    _vendor.add_wbx_deps_path()
    if tuple(int(x) for x in np.__version__.split(".")[:2]) < (1, 26):
        pytest.skip("jax 0.6 needs numpy>=1.26 (run with PYTHONPATH=.pydeps)")
    pytest.importorskip("jax")
    return _vendor.ensure_vendored()


# =============================================================================
# Year gate (pure python: runs even without jax)
# =============================================================================

def test_default_year_gate_allows_only_exposed_2020():
    assert ev.assert_years_authorized([2020, 2020])["gate"] == "default_exposed_years"
    for years in ([2021], [2022], [2020, 2022], []):
        with pytest.raises(ev.YearNotAuthorized):
            ev.assert_years_authorized(years)


def test_evaluate_enforces_year_gate_before_loader_access():
    with pytest.raises(ev.YearNotAuthorized):
        ev.evaluate("/does/not/exist/prediction.zarr", "/does/not/exist/truth.zarr",
                    init_times=[np.datetime64("2022-01-01T00", "ns")],
                    lead_times=[np.timedelta64(24, "h")])


def test_evaluate_gates_valid_years_that_cross_calendar_boundary():
    with pytest.raises(ev.YearNotAuthorized):
        ev.evaluate("/does/not/exist/prediction.zarr", "/does/not/exist/truth.zarr",
                    init_times=[np.datetime64("2020-12-31T00", "ns")],
                    lead_times=[np.timedelta64(72, "h")])


def test_single_chunk_year_gate_when_issue_times_are_supplied():
    p = {"x": np.zeros((1, 1), dtype=np.float32)}
    t = {"x": np.zeros((1, 1), dtype=np.float32)}
    with pytest.raises(ev.YearNotAuthorized):
        ev.evaluate_single_chunk(p, t)
    with pytest.raises(ev.YearNotAuthorized):
        ev.evaluate_single_chunk(
            p, t, init_times=[np.datetime64("2022-01-01T00", "ns")]
        )
    with pytest.raises(ev.YearNotAuthorized):
        ev.evaluate_single_chunk(
            p, t, init_times=[np.datetime64("2020-12-31T00", "ns")],
            valid_times=[np.datetime64("2021-01-03T00", "ns")]
        )


def test_confirm_freeze_gate(tmp_path):
    freeze = tmp_path / "confirm_freeze.json"
    freeze.write_text(json.dumps({"authorized_evaluation_years": [2022]}))
    good = hashlib.sha256(freeze.read_bytes()).hexdigest()
    rec = ev.assert_years_authorized([2020, 2022], confirm_freeze_path=freeze, confirm_freeze_sha256=good)
    assert rec["gate"] == "confirm_freeze" and rec["confirm_freeze_sha256"] == good
    with pytest.raises(ev.YearNotAuthorized):          # path without hash
        ev.assert_years_authorized([2022], confirm_freeze_path=freeze)
    with pytest.raises(ev.YearNotAuthorized):          # wrong hash
        ev.assert_years_authorized([2022], confirm_freeze_path=freeze, confirm_freeze_sha256="0" * 64)
    with pytest.raises(ev.YearNotAuthorized):          # year not listed in the freeze
        ev.assert_years_authorized([2021], confirm_freeze_path=freeze, confirm_freeze_sha256=good)
    with pytest.raises(ev.YearNotAuthorized):          # missing file
        ev.assert_years_authorized([2022], confirm_freeze_path=tmp_path / "nope.json",
                                   confirm_freeze_sha256=good)
    freeze.write_text(json.dumps({"authorized_evaluation_years": [2021, 2022]}))  # tampered after hashing
    with pytest.raises(ev.YearNotAuthorized):
        ev.assert_years_authorized([2022], confirm_freeze_path=freeze, confirm_freeze_sha256=good)


# =============================================================================
# Vendoring
# =============================================================================

def test_read_git_head_loose_and_packed(tmp_path):
    repo = tmp_path / "r"
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    sha = "a" * 40
    (repo / ".git" / "packed-refs").write_text(f"# pack-refs\n{sha} refs/heads/main\n")
    assert _vendor.read_git_head(repo) == sha
    (repo / ".git" / "refs" / "heads" / "main").write_text("b" * 40 + "\n")
    assert _vendor.read_git_head(repo) == "b" * 40
    (repo / ".git" / "HEAD").write_text("not-a-sha\n")
    with pytest.raises(_vendor.VendorPinError):
        _vendor.read_git_head(repo)


@pytest.mark.skipif(not HAVE_REFS, reason="vendored clones not present")
def test_pin_mismatch_fails_closed():
    manifest = _vendor.load_manifest()
    for name in _vendor.VENDORED_PACKAGES:
        rec = _vendor.verify_vendored_repo(name, manifest=manifest, check_clean=False)
        assert rec.actual_head.startswith(manifest[name]["head"])
        bad = {k: dict(v) for k, v in manifest.items()}
        bad[name]["head"] = "0000000"
        with pytest.raises(_vendor.VendorPinError):
            _vendor.verify_vendored_repo(name, manifest=bad, check_clean=False)
        missing = {k: v for k, v in manifest.items() if k != name}
        with pytest.raises(_vendor.VendorPinError):
            _vendor.verify_vendored_repo(name, manifest=missing, check_clean=False)


def test_modified_tracked_source_is_detected(tmp_path):
    repo = tmp_path / "weatherbench2"
    pkg = repo / "weatherbench2"
    pkg.mkdir(parents=True)
    (pkg / "m.py").write_text("x = 1\n")
    git = ["git", f"--git-dir={repo / '.git'}", f"--work-tree={repo}"]
    try:
        subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
        subprocess.run(git + ["add", "weatherbench2/m.py"], check=True, capture_output=True)
        subprocess.run(git + ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"],
                       check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git unavailable: {exc}")
    assert _vendor.modified_tracked_sources(repo, ["weatherbench2"]) == []
    (pkg / "m.py").write_text("x = 2  # patched\n")
    assert _vendor.modified_tracked_sources(repo, ["weatherbench2"]) == ["weatherbench2/m.py"]
    assert _vendor.worktree_is_clean(repo) is False


def test_pydeps_wbx_is_appended_and_disjoint():
    added = _vendor.add_wbx_deps_path()
    if added is None:
        pytest.skip(".pydeps_wbx not installed")
    assert sys.path.index(str(added)) > 0
    assert not (_vendor._top_level_names(_vendor.PYDEPS_WBX_DIR) & _vendor._top_level_names(_vendor.PYDEPS_DIR))


def test_beam_modules_are_forbidden_and_never_imported(wbx):
    for mod in ("weatherbench2.evaluation", "weatherbench2.metrics", "weatherbenchX.beam_pipeline"):
        with pytest.raises(_vendor.VendorPinError):
            _vendor.import_official(mod)
    ev.standard_metrics()
    ev.make_aggregator(regions=ev.official_regions())
    assert "apache_beam" not in sys.modules
    _vendor.assert_no_beam()


# =============================================================================
# Metric driver
# =============================================================================

NLAT, NLON = 8, 16


def _grid():
    d = 180.0 / NLAT
    return np.linspace(-90 + d / 2, 90 - d / 2, NLAT), np.linspace(0, 360, NLON, endpoint=False)


def _synthetic(n_init=4, leads=(6, 12), seed=0, dtype=np.float64):
    """Forecast + truth datasets (all 69 channels) in the exported layout."""
    from earthdelta.wbx import export
    rng = np.random.default_rng(seed)
    lat, lon = _grid()
    inits = np.array([np.datetime64("2020-03-01T00", "ns") + np.timedelta64(12 * i, "h")
                      for i in range(n_init)])
    deltas = (np.array(leads) * np.timedelta64(1, "h")).astype("timedelta64[ns]")
    pred = (10 + rng.standard_normal((n_init, len(leads), 69, NLAT, NLON))).astype(dtype)
    valid = np.unique((inits[:, None] + deltas[None, :]).reshape(-1))
    truth_by_vt = {t: (10 + rng.standard_normal((69, NLAT, NLON))).astype(dtype) for t in valid}
    fc = export.unflatten_channels(pred, lat=lat, lon=lon,
                                   leading=(("time", inits), ("prediction_timedelta", deltas)))
    truth = export.unflatten_channels(np.stack([truth_by_vt[t] for t in valid]), lat=lat, lon=lon,
                                      leading=(("time", valid),))
    tru = np.stack([[truth_by_vt[i + d] for d in deltas] for i in inits])
    return fc, truth, inits, deltas, pred, tru


def test_chunked_equals_single_chunk(wbx):
    fc, truth, inits, deltas, _, _ = _synthetic()
    metrics = ev.standard_metrics()
    ref, _ = ev.evaluate(fc, truth, init_times=inits, lead_times=deltas, metrics=metrics,
                         init_chunk_size=len(inits))
    for size in (1, 3):
        got, _ = ev.evaluate(fc, truth, init_times=inits, lead_times=deltas, metrics=metrics,
                             init_chunk_size=size)
        for name in ref.data_vars:
            np.testing.assert_allclose(got[name].values, ref[name].values, rtol=1e-12, atol=0)
    xl = _vendor.import_official("weatherbenchX.data_loaders.xarray_loaders")
    p = xl.PredictionsFromXarray(ds=fc).load_chunk(inits, deltas)
    t = xl.TargetsFromXarray(ds=truth).load_chunk(inits, deltas)
    single = ev.evaluate_single_chunk(p, t, metrics=metrics, init_times=inits)
    for name in ref.data_vars:
        np.testing.assert_allclose(single[name].values, ref[name].values, rtol=1e-12, atol=0)


def test_wbx_metrics_equal_numpy_and_earthdelta_objective(wbx):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from earthdelta.data.pull_wb2 import CANONICAL_VARIABLES
    from earthdelta.metrics_contract import full_objective_loss
    from earthdelta.static_adapter import area_weight_q
    fc, truth, inits, deltas, pred, tru = _synthetic()
    res, _ = ev.evaluate(fc, truth, init_times=inits, lead_times=deltas,
                         metrics=ev.standard_metrics(wind_vector=False))
    lat, _ = _grid()
    w = np.cos(np.deg2rad(lat))
    w = w / w.sum()
    d = pred - tru                                                          # [I,H,V,lat,lon]
    np_mse = np.einsum("ihvab,a->hv", d * d, w) / (NLON * len(inits))
    np_bias = np.einsum("ihvab,a->hv", d, w) / (NLON * len(inits))
    mse = ev.per_channel_values(res, "mse")
    rmse = ev.per_channel_values(res, "rmse")
    bias = ev.per_channel_values(res, "bias")
    for c, name in enumerate(CANONICAL_VARIABLES):
        np.testing.assert_allclose(mse[name], np_mse[:, c], rtol=1e-12)
        np.testing.assert_allclose(rmse[name], np.sqrt(np_mse[:, c]), rtol=1e-12)
        np.testing.assert_allclose(bias[name], np_bias[:, c], rtol=1e-9, atol=1e-12)
    scale = torch.linspace(0.5, 3.0, 69, dtype=torch.float64).reshape(1, -1, 1, 1)
    q = area_weight_q(torch.from_numpy(lat))
    ed = full_objective_loss(torch.from_numpy(pred), torch.from_numpy(tru), q, scale=scale).mean()
    s2 = scale.reshape(-1).numpy() ** 2
    recon = np.mean([[mse[n][h] / s2[c] for c, n in enumerate(CANONICAL_VARIABLES)] for h in range(len(deltas))])
    assert abs(float(ed) - recon) / recon < 1e-12


def test_grid_area_weighting_equals_earthdelta_coslat(wbx):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from earthdelta.static_adapter import area_weight_q
    from earthdelta.wbx.regrid import stormer_native_grid
    weighting = _vendor.import_official("weatherbenchX.weighting")
    lat, lon = stormer_native_grid()
    probe = xr.DataArray(np.zeros((len(lat), 1)), dims=("latitude", "longitude"),
                         coords={"latitude": lat, "longitude": lon[:1]})
    w_wbx = np.asarray(weighting.GridAreaWeighting().weights(probe).values)
    w_ed = area_weight_q(torch.from_numpy(lat)).reshape(-1).numpy()
    np.testing.assert_allclose(w_wbx / w_wbx.sum(), w_ed / w_ed.sum(), rtol=1e-12)


def test_official_regions_parsed_not_copied(wbx):
    regions = ev.official_regions()
    assert regions["global"] == ((-90, 90), (0, 360))
    assert regions["tropics"] == ((-20, 20), (0, 360))
    assert regions["north-america"] == ((25, 60), (240, 285))      # "360 - 120" evaluated
    assert len(regions) >= 17


def test_region_binning_global_equals_unbinned(wbx):
    fc, truth, inits, deltas, _, _ = _synthetic(n_init=2)
    m = {"mse": ev.standard_metrics()["mse"]}
    a, _ = ev.evaluate(fc, truth, init_times=inits, lead_times=deltas, metrics=m)
    b, _ = ev.evaluate(fc, truth, init_times=inits, lead_times=deltas, metrics=m,
                       aggregator=ev.make_aggregator(regions=ev.official_regions()))
    for name in a.data_vars:
        np.testing.assert_allclose(b[name].sel(region="global").values, a[name].values, rtol=1e-12)
        assert (b[name].sel(region="tropics").values != a[name].values).any()


def test_missing_lead_or_init_in_truth_fails_loudly(wbx):
    fc, truth, inits, deltas, _, _ = _synthetic(n_init=2)
    short_truth = truth.isel(time=slice(0, -1))
    with pytest.raises(KeyError):
        ev.evaluate(fc, short_truth, init_times=inits, lead_times=deltas,
                    metrics={"mse": ev.standard_metrics()["mse"]})


# =============================================================================
# Process-level protobuf robustness (Tier B shape: wbx/jax + torch/timm)
# =============================================================================

_SUBPROCESS_BODY = r"""
import sys
sys.path.insert(0, {repo!r})
{pre}
import timm                                     # timm -> torchvision -> onnx -> protobuf
import torch
from earthdelta.bridge import Stormer            # real model path
from earthdelta.static_adapter import area_weight_q
from earthdelta.wbx import evaluate as ev, regrid
lat, lon = regrid.stormer_native_grid()
ev.standard_metrics(); regrid.wb2_grid()        # jax-backed official modules
area_weight_q(torch.from_numpy(lat))
Stormer(in_img_size=(8, 16), variables=["2m_temperature"], patch_size=2,
        hidden_size=8, depth=1, num_heads=2, mlp_ratio=2.0)
from google.protobuf.internal import api_implementation
print("OK", api_implementation.Type())
"""


def _run_clean(pre: str):
    import os
    env = {k: v for k, v in os.environ.items() if k != "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"}
    code = _SUBPROCESS_BODY.format(repo=str(REPO_ROOT), pre=pre)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=600)


@pytest.mark.parametrize("order", ["wbx_first", "wbx_and_jax_first"])
def test_wbx_and_real_model_path_share_a_process_without_env_var(wbx, order):
    """A fresh interpreter WITHOUT PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION that
    imports earthdelta.wbx (and jax) first must still load timm/Stormer."""
    pytest.importorskip("timm")
    control = subprocess.run(
        [sys.executable, "-c", "import timm"],
        env={k: v for k, v in __import__("os").environ.items()
             if k != "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"},
        capture_output=True, text=True, timeout=600)
    pre = "import earthdelta.wbx"
    if order == "wbx_and_jax_first":
        pre += "\nfrom earthdelta.wbx import _vendor; _vendor.ensure_vendored(); import jax"
    proc = _run_clean(pre)
    assert proc.returncode == 0 and "OK python" in proc.stdout, (
        f"control(import timm, no env) rc={control.returncode}\n{proc.stderr[-3000:]}")


# =============================================================================
# Baselines stub
# =============================================================================

def test_baseline_registry_and_pairing(wbx):
    from earthdelta.wbx import baselines
    baselines.verify_registry_against_official_configs()
    assert baselines.gs_to_https("gs://weatherbench2/x.zarr") == "https://storage.googleapis.com/weatherbench2/x.zarr"
    baselines.assert_baseline_pairable([np.datetime64("2020-01-01T00"), np.datetime64("2020-01-01T12")])
    with pytest.raises(baselines.BaselineNotPairable):
        baselines.assert_baseline_pairable([np.datetime64("2020-01-01T06")])
    with pytest.raises(ev.YearNotAuthorized):   # year gate precedes any network access
        baselines.probe_public_store("hres_64x32", variable="2m_temperature", time="2022-01-01T00")
