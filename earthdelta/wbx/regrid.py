"""Comparison-only regridding via the official WeatherBench 2 conservative operator.

Stormer's native rollout (128x256, 1.40625 deg, cell-centre grid without pole
points) is NEVER regridded: the native forecast is what EarthDelta's own
objective scores. A regridded copy is only a derived artifact used when scoring
against official WeatherBench baselines, which are published on the
64x32 (and 240x121) grids.

Three independent implementations are available so the operator can be
cross-checked rather than trusted:

1. ``official_regridder`` -- ``weatherbench2.regridding.ConservativeRegridder``
   on ``Grid.from_degrees`` grids (JAX, float32 by default: jax_enable_x64 is
   left untouched).
2. ``numpy_reference_regrid`` -- EarthDelta's existing float64 numpy port in
   ``earthdelta/data/pull_wb2.py`` (the one that produced the local truth
   stores).
3. ``block_mean_oracle`` -- a from-scratch explicit area-weighted block mean
   for integer coarsening ratios. Latitude cells of the 128-row Stormer grid
   nest exactly into the 32-row WB2 grid (both have bounds at -90 + k*d).
   Longitude does NOT nest exactly: both grids are "start at zero" cell-centre
   grids, so a coarse cell centred on fine centre 4m spans fine cells
   4m-1..4m+1 fully plus HALF of 4m-2 and 4m+2. The oracle uses that exact
   [1/2, 1, 1, 1, 1/2]/4 periodic stencil; it shares no code with (1) or (2).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..data import pull_wb2
from ._vendor import import_official

#: Official WB2 coarse grids that published baselines use.
WB2_GRIDS = {
    "64x32": {"nlon": 64, "nlat": 32, "poles": False},
    "240x121": {"nlon": 240, "nlat": 121, "poles": True},
}


def stormer_native_grid() -> Tuple[np.ndarray, np.ndarray]:
    """Stormer native (lat[128], lon[256]) in degrees -- same as the truth stores."""
    return pull_wb2.get_stormer_target_grid()


def wb2_grid(name: str = "64x32") -> Tuple[np.ndarray, np.ndarray]:
    """Official WB2 (lat, lon) for a named grid, via ``regridding.*_values``."""
    regridding = import_official("weatherbench2.regridding")
    spec = WB2_GRIDS[name]
    spacing = (regridding.LatitudeSpacing.EQUIANGULAR_WITH_POLES if spec["poles"]
               else regridding.LatitudeSpacing.EQUIANGULAR_WITHOUT_POLES)
    lat = regridding.latitude_values(spacing, spec["nlat"])
    lon = regridding.longitude_values(regridding.LongitudeScheme.START_AT_ZERO, spec["nlon"])
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def official_regridder(src_lat, src_lon, tgt_lat, tgt_lon):
    """``weatherbench2.regridding.ConservativeRegridder`` between two grids."""
    regridding = import_official("weatherbench2.regridding")
    source = regridding.Grid.from_degrees(lon=np.asarray(src_lon), lat=np.asarray(src_lat))
    target = regridding.Grid.from_degrees(lon=np.asarray(tgt_lon), lat=np.asarray(tgt_lat))
    return regridding.ConservativeRegridder(source, target)


def official_regrid_array(field: np.ndarray, src_lat, src_lon, tgt_lat, tgt_lon,
                          *, x64: bool = False) -> np.ndarray:
    """Regrid ``[..., lat, lon]`` with the official operator; returns ``[..., lat', lon']``.

    WB2's operator works on ``[..., lon, lat]``; the axes are swapped around it.
    ``x64=True`` runs the SAME official code under the scoped
    ``jax.experimental.enable_x64`` context (float64 weights and einsum) so
    algorithmic agreement can be separated from float32 rounding; the global
    jax configuration is not changed.
    """
    regridder = official_regridder(src_lat, src_lon, tgt_lat, tgt_lon)
    arr = np.swapaxes(np.asarray(field), -1, -2)
    if x64:
        from jax.experimental import enable_x64
        with enable_x64():
            out = np.asarray(regridder.regrid_array(arr.astype(np.float64)))
    else:
        out = np.asarray(regridder.regrid_array(arr))
    return np.swapaxes(out, -1, -2)


def regrid_dataset(ds, target: str = "64x32"):
    """Regrid an exported forecast/truth Dataset with the official operator.

    Returns a new Dataset on the official WB2 grid; the input is untouched.
    """
    src_lat = np.asarray(ds["latitude"].values)
    src_lon = np.asarray(ds["longitude"].values)
    tgt_lat, tgt_lon = wb2_grid(target)
    regridder = official_regridder(src_lat, src_lon, tgt_lat, tgt_lon)
    out = regridder.regrid_dataset(ds)
    # The official regrid_dataset transposes every variable to the Dataset-level
    # dim order (level ends up after longitude); restore each variable's layout.
    out = out.assign({name: out[name].transpose(*ds[name].dims) for name in ds.data_vars})
    out.attrs = dict(ds.attrs)
    out.attrs["regridded_to"] = f"wb2_{target}_conservative"
    out.attrs["regridder"] = "weatherbench2.regridding.ConservativeRegridder"
    return out


def numpy_reference_regrid(field: np.ndarray, src_lat, src_lon, tgt_lat, tgt_lon,
                           *, keep_float64: bool = False) -> np.ndarray:
    """EarthDelta's existing numpy conservative regridder, ``[..., lat, lon]`` in/out.

    ``pull_wb2.ConservativeRegridder.regrid`` casts its result to float32 (as
    stored in the truth zarrs). ``keep_float64=True`` applies the same weight
    matrices and the same NaN-aware total/count formula without that cast.
    """
    regridder = pull_wb2.ConservativeRegridder(
        np.asarray(src_lat), np.asarray(src_lon), np.asarray(tgt_lat), np.asarray(tgt_lon))
    arr = np.swapaxes(np.asarray(field, dtype=np.float64), -1, -2)
    if not keep_float64:
        return np.swapaxes(regridder.regrid(arr), -1, -2)
    nulls = np.isnan(arr)
    total = np.einsum("ab,cd,...bd->...ac", regridder.lon_weights, regridder.lat_weights,
                      np.where(nulls, 0.0, arr), optimize=True)
    count = np.einsum("ab,cd,...bd->...ac", regridder.lon_weights, regridder.lat_weights,
                      (~nulls).astype(np.float64), optimize=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.swapaxes(np.where(count > 0, total / count, np.nan), -1, -2)


def _lat_bounds_uniform(lat: np.ndarray) -> np.ndarray:
    """Cell bounds for a uniform cell-centre latitude grid spanning the globe."""
    d = float(lat[1] - lat[0])
    bounds = np.concatenate([[lat[0] - d / 2], lat + d / 2])
    if not (np.isclose(bounds[0], -90.0) and np.isclose(bounds[-1], 90.0)):
        raise ValueError("block_mean_oracle needs a uniform grid whose cells span -90..90")
    return np.clip(bounds, -90.0, 90.0)


def block_mean_oracle(field: np.ndarray, src_lat: np.ndarray, factor: int = 4) -> np.ndarray:
    """From-scratch conservative coarsening by an integer ``factor`` (even).

    ``field`` is ``[..., nlat, nlon]`` on a uniform cell-centre global grid
    whose longitudes start at 0; output is ``[..., nlat/f, nlon/f]`` on the
    coarse cell-centre grid whose longitudes also start at 0.
    """
    field = np.asarray(field, dtype=np.float64)
    nlat, nlon = field.shape[-2:]
    if factor % 2 or nlat % factor or nlon % factor:
        raise ValueError("factor must be even and divide both grid sizes")
    bounds = np.deg2rad(_lat_bounds_uniform(np.asarray(src_lat, dtype=np.float64)))
    area = np.sin(bounds[1:]) - np.sin(bounds[:-1])                  # [nlat]
    blocks = area.reshape(nlat // factor, factor)
    w = blocks / blocks.sum(axis=1, keepdims=True)                    # [nlat/f, f]
    f = field.reshape(field.shape[:-2] + (nlat // factor, factor, nlon))
    lat_mean = np.einsum("...ikx,ik->...ix", f, w)                    # [..., nlat/f, nlon]
    half = factor // 2
    stencil = np.ones(factor + 1)
    stencil[0] = stencil[-1] = 0.5
    stencil /= factor
    acc = np.zeros_like(lat_mean)
    for k, s in zip(range(-half, half + 1), stencil):
        acc += s * np.roll(lat_mean, -k, axis=-1)
    return acc[..., ::factor]


def area_weighted_mean(field: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Global mean over the last two axes with exact cell-area weights."""
    bounds = np.deg2rad(_lat_bounds_uniform(np.asarray(lat, dtype=np.float64)))
    w = np.sin(bounds[1:]) - np.sin(bounds[:-1])
    f = np.asarray(field, dtype=np.float64)
    return np.einsum("...ij,i->...", f, w) / (w.sum() * f.shape[-1])


def compare_fields(candidate: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """Error summary. ``max_scaled`` = max|c-r| / max|r| (robust near zero)."""
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if c.shape != r.shape:
        raise ValueError(f"shape mismatch {c.shape} vs {r.shape}")
    diff = np.abs(c - r)
    scale = float(np.max(np.abs(r))) or 1.0
    nz = np.abs(r) > 1e-6 * scale
    return {
        "max_abs": float(diff.max()),
        "max_scaled": float(diff.max() / scale),
        "max_rel_where_nonnegligible": float((diff[nz] / np.abs(r[nz])).max()) if nz.any() else 0.0,
    }


def _worst(per_field) -> Dict[str, float]:
    keys = per_field[0].keys()
    return {k: max(d[k] for d in per_field) for k in keys}


def compare_per_field(candidate: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """``compare_fields`` applied to each leading 2-D field; worst case returned.

    Scaling per field (not by the global max over all channels) keeps small-
    magnitude variables such as specific humidity from being hidden.
    """
    c = np.asarray(candidate).reshape((-1,) + np.shape(candidate)[-2:])
    r = np.asarray(reference).reshape((-1,) + np.shape(reference)[-2:])
    return _worst([compare_fields(c[i], r[i]) for i in range(c.shape[0])])


def cross_check_regrid(field: np.ndarray, src_lat=None, src_lon=None, target: str = "64x32") -> Dict[str, object]:
    """Run all three regridders on ``[..., lat, lon]`` and compare them.

    Two tiers, both reported per 2-D field (worst case over fields):

    * ``as_shipped``: official WB2 operator at its default float32 precision vs
      the pull_wb2 numpy operator as shipped (float32 output) vs the float64
      block oracle -- the precision actually used for comparisons.
    * ``float64``: the same official code under scoped x64, the numpy weights
      without the float32 cast, and the oracle -- isolates algorithmic
      agreement from storage rounding.

    Also checks conservation of the area-weighted global mean.
    """
    if src_lat is None or src_lon is None:
        src_lat, src_lon = stormer_native_grid()
    tgt_lat, tgt_lon = wb2_grid(target)
    official = official_regrid_array(field, src_lat, src_lon, tgt_lat, tgt_lon)
    reference = numpy_reference_regrid(field, src_lat, src_lon, tgt_lat, tgt_lon)
    official64 = official_regrid_array(field, src_lat, src_lon, tgt_lat, tgt_lon, x64=True)
    reference64 = numpy_reference_regrid(field, src_lat, src_lon, tgt_lat, tgt_lon, keep_float64=True)
    as_shipped: Dict[str, object] = {
        "official_dtype": str(official.dtype),
        "official_vs_numpy": compare_per_field(official, reference),
    }
    f64: Dict[str, object] = {
        "official_dtype": str(official64.dtype),
        "official_vs_numpy": compare_per_field(official64, reference64),
    }
    factor = len(src_lat) // len(tgt_lat)
    if (len(src_lat) == factor * len(tgt_lat) and len(src_lon) == factor * len(tgt_lon)
            and factor % 2 == 0):
        oracle = block_mean_oracle(field, np.asarray(src_lat), factor)
        as_shipped["official_vs_block_oracle"] = compare_per_field(official, oracle)
        as_shipped["numpy_vs_block_oracle"] = compare_per_field(reference, oracle)
        f64["official_vs_block_oracle"] = compare_per_field(official64, oracle)
        f64["numpy_vs_block_oracle"] = compare_per_field(reference64, oracle)
    src_mean = area_weighted_mean(field, np.asarray(src_lat))
    return {
        "target": target,
        "n_fields": int(np.prod(np.shape(field)[:-2], dtype=np.int64)),
        "as_shipped": as_shipped,
        "float64": f64,
        "global_mean_conservation": {
            "official_float32": compare_fields(area_weighted_mean(official, np.asarray(tgt_lat)), src_mean),
            "official_float64": compare_fields(area_weighted_mean(official64, np.asarray(tgt_lat)), src_mean),
        },
    }


__all__ = [
    "stormer_native_grid", "wb2_grid", "official_regridder", "official_regrid_array",
    "regrid_dataset", "numpy_reference_regrid", "block_mean_oracle",
    "area_weighted_mean", "compare_fields", "compare_per_field", "cross_check_regrid",
]
