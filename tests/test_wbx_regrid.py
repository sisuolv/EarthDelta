"""earthdelta.wbx.regrid: official WB2 conservative regridding, cross-checked (CPU).

Three implementations must agree: the official ``weatherbench2.regridding.
ConservativeRegridder``, EarthDelta's numpy port in ``pull_wb2`` (which built
the local truth stores), and a from-scratch area-weighted block oracle. All
three are also checked against analytic cell means, so agreement cannot come
from a shared mistake. Skips without xarray/jax or the vendored clones.
"""
from __future__ import annotations

import os

# Must precede any google.protobuf import (timm -> torchvision -> onnx); see
# earthdelta/wbx/__init__.py. Makes these tests independent of the caller's env.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("xarray")
from earthdelta.wbx import _vendor  # noqa: E402

if not all((_vendor.REFERENCE_ROOT / n / ".git").exists() for n in _vendor.VENDORED_PACKAGES):
    pytest.skip("vendored weatherbench2/weatherbenchX clones not present", allow_module_level=True)
_vendor.add_wbx_deps_path()
np_major_minor = tuple(int(x) for x in np.__version__.split(".")[:2])
if np_major_minor < (1, 26):
    pytest.skip("jax 0.6 needs numpy>=1.26 (run with PYTHONPATH=.pydeps)", allow_module_level=True)
pytest.importorskip("jax")
pytest.importorskip("sklearn")

from earthdelta.wbx import regrid  # noqa: E402

SRC_LAT, SRC_LON = regrid.stormer_native_grid()


def _smooth_fields(n=3, seed=0):
    rng = np.random.default_rng(seed)
    la = np.deg2rad(SRC_LAT)[:, None]
    lo = np.deg2rad(SRC_LON)[None, :]
    out = []
    for _ in range(n):
        a, b, c = rng.uniform(-1, 1, 3)
        k, m = rng.integers(1, 6, 2)
        out.append(250 + 30 * a * np.cos(la) ** 2 + 10 * b * np.sin(k * lo) * np.cos(la)
                   + 5 * c * np.sin(m * la) + rng.normal(0, 0.5, (len(SRC_LAT), len(SRC_LON))))
    return np.stack(out)


def test_wb2_64x32_grid_is_the_published_grid():
    lat, lon = regrid.wb2_grid("64x32")
    assert lat.shape == (32,) and lon.shape == (64,)
    assert np.isclose(lat[0], -87.1875) and np.isclose(lat[-1], 87.1875)
    assert lon[0] == 0.0 and np.isclose(lon[-1], 354.375)


def test_three_implementations_agree_float64_and_float32():
    fields = _smooth_fields()
    result = regrid.cross_check_regrid(fields)
    f64, f32 = result["float64"], result["as_shipped"]
    assert f64["official_dtype"] == "float64"
    for key in ("official_vs_numpy", "official_vs_block_oracle", "numpy_vs_block_oracle"):
        assert f64[key]["max_scaled"] < 1e-12, (key, f64[key])
    assert f32["official_dtype"] == "float32"
    for key in ("official_vs_numpy", "official_vs_block_oracle"):
        assert f32[key]["max_scaled"] < 1e-5, (key, f32[key])
    cons = result["global_mean_conservation"]
    assert cons["official_float64"]["max_scaled"] < 1e-13
    assert cons["official_float32"]["max_scaled"] < 1e-6


def _cell_bounds(centres, lo, hi):
    d = centres[1] - centres[0]
    return np.clip(np.concatenate([[centres[0] - d / 2], centres + d / 2]), lo, hi)


def test_all_implementations_match_analytic_cell_means():
    """f(lat, lon) = sin(lat) + g(lon), g piecewise constant on the fine cells.

    Source cells hold the EXACT cell means. Latitude cells nest exactly, and the
    cos-weighted mean of sin(lat) over [a, b] is (sin a + sin b)/2 in closed
    form. In longitude the coarse cell centred on fine cell 4m covers fine
    cells 4m-1..4m+1 fully and half of 4m-2 / 4m+2 (written out explicitly
    below). A conservative regrid must return the exact coarse cell means.
    """
    tgt_lat, tgt_lon = regrid.wb2_grid("64x32")

    def lat_means(lat_c):
        lb = np.deg2rad(_cell_bounds(lat_c, -90, 90))
        return (np.sin(lb[:-1]) + np.sin(lb[1:])) / 2

    g = np.random.default_rng(11).normal(size=len(SRC_LON))
    n = len(SRC_LON)
    g_coarse = np.array([(0.5 * g[(4 * m - 2) % n] + g[(4 * m - 1) % n] + g[4 * m]
                          + g[(4 * m + 1) % n] + 0.5 * g[(4 * m + 2) % n]) / 4
                         for m in range(len(tgt_lon))])
    src = lat_means(SRC_LAT)[:, None] + g[None, :]
    exact = lat_means(tgt_lat)[:, None] + g_coarse[None, :]
    official = regrid.official_regrid_array(src, SRC_LAT, SRC_LON, tgt_lat, tgt_lon, x64=True)
    numpy_ref = regrid.numpy_reference_regrid(src, SRC_LAT, SRC_LON, tgt_lat, tgt_lon, keep_float64=True)
    oracle = regrid.block_mean_oracle(src, SRC_LAT, 4)
    for name, got in (("official", official), ("numpy", numpy_ref), ("oracle", oracle)):
        assert np.max(np.abs(got - exact)) < 1e-12, name


def test_nan_handling_matches_between_official_and_numpy():
    field = _smooth_fields(1)[0]
    field[10:14, 20:30] = np.nan                       # partial holes in several coarse cells
    field[4:8, 2:7] = np.nan                           # exactly covers coarse cell (lat 1, lon 1)
    tgt_lat, tgt_lon = regrid.wb2_grid("64x32")
    official = regrid.official_regrid_array(field, SRC_LAT, SRC_LON, tgt_lat, tgt_lon, x64=True)
    numpy_ref = regrid.numpy_reference_regrid(field, SRC_LAT, SRC_LON, tgt_lat, tgt_lon, keep_float64=True)
    assert np.array_equal(np.isnan(official), np.isnan(numpy_ref))
    assert np.isnan(official[1, 1]) and np.isnan(official).sum() == 1
    ok = ~np.isnan(official)
    assert np.max(np.abs(official[ok] - numpy_ref[ok])) < 1e-9


def test_block_oracle_rejects_unsupported_geometry():
    field = _smooth_fields(1)[0]
    with pytest.raises(ValueError):
        regrid.block_mean_oracle(field, SRC_LAT, 3)          # odd factor
    with pytest.raises(ValueError):
        regrid.block_mean_oracle(field[:, :250], SRC_LAT, 4)  # does not divide
    with pytest.raises(ValueError):
        regrid.block_mean_oracle(field[:120], SRC_LAT[:120], 4)  # does not span the globe


def test_regrid_dataset_keeps_layout_and_matches_array_path():
    pytest.importorskip("zarr")
    from earthdelta.data.pull_wb2 import CANONICAL_VARIABLES
    from earthdelta.wbx import export
    rng = np.random.default_rng(3)
    data = (250 + rng.standard_normal((1, 2, 69, len(SRC_LAT), len(SRC_LON)))).astype(np.float32)
    times = np.array(["2020-01-01T00"], dtype="datetime64[ns]")
    leads = (np.array([6, 24]) * np.timedelta64(1, "h")).astype("timedelta64[ns]")
    ds = export.unflatten_channels(data, lat=SRC_LAT, lon=SRC_LON,
                                   leading=(("time", times), ("prediction_timedelta", leads)))
    ds = ds[["2m_temperature", "geopotential"]]
    out = regrid.regrid_dataset(ds)
    tgt_lat, tgt_lon = regrid.wb2_grid("64x32")
    assert out["geopotential"].dims == ds["geopotential"].dims
    assert np.array_equal(out["latitude"].values, tgt_lat) and np.array_equal(out["longitude"].values, tgt_lon)
    c = CANONICAL_VARIABLES.index("2m_temperature")
    direct = regrid.official_regrid_array(data[0, 1, c], SRC_LAT, SRC_LON, tgt_lat, tgt_lon)
    np.testing.assert_allclose(out["2m_temperature"].values[0, 1], direct, rtol=0, atol=1e-4)
    assert out.attrs["regridder"] == "weatherbench2.regridding.ConservativeRegridder"
