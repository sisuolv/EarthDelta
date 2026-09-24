"""Official WeatherBench public baseline stores (stub for the Tier A smoke test).

Baselines are read from the official, precomputed public stores -- EarthDelta
never re-runs HRES / IFS-ENS / Pangu. This module currently provides only what
the smoke test's optional network check needs: the store registry (every path
is verified to appear verbatim in the vendored official
``public_benchmark/public_configs.py``), an HTTPS opener through the existing
proxy, and a tiny read probe. The full benchmark runner is a later task.

Pairing rule (plan correction 9): official baselines are initialized only at
00Z/12Z, so EarthDelta issues at 06Z/18Z can be scored against ERA5 but can
never be paired with a baseline; ``assert_baseline_pairable`` enforces that.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ._vendor import REFERENCE_ROOT, ensure_vendored

#: name -> official gs:// path (64x32 conservative, the default comparison grid)
PUBLIC_STORES: Dict[str, str] = {
    "era5_64x32": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
    "hres_64x32": "gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr",
    "ifs_ens_mean_64x32": "gs://weatherbench2/datasets/ifs_ens/2018-2022-64x32_equiangular_conservative_mean.zarr",
    "pangu_64x32": "gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr",
    "era5_climatology_1990_2019_64x32": "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr",
}

BASELINE_INIT_HOURS = (0, 12)


class BaselineNotPairable(ValueError):
    """An EarthDelta issue time has no official baseline counterpart."""


def verify_registry_against_official_configs() -> None:
    """Every registered path must appear verbatim in the official configs."""
    ensure_vendored(need_jax=False)
    text = (REFERENCE_ROOT / "weatherbenchX" / "public_benchmark" / "public_configs.py").read_text()
    missing = [k for k, p in PUBLIC_STORES.items() if p not in text]
    if missing:
        raise RuntimeError(f"baseline paths not found in official public_configs.py: {missing}")


def gs_to_https(path: str) -> str:
    if not path.startswith("gs://"):
        raise ValueError(f"not a gs:// path: {path}")
    return "https://storage.googleapis.com/" + path[len("gs://"):]


def assert_baseline_pairable(issue_times: Sequence[Any]) -> None:
    """Refuse issue times that official baselines do not initialize at."""
    bad = []
    for t in issue_times:
        hour = int((np.datetime64(t, "h") - np.datetime64(t, "D")).astype(int))
        if hour not in BASELINE_INIT_HOURS:
            bad.append(str(np.datetime64(t, "h")))
    if bad:
        raise BaselineNotPairable(
            f"official baselines exist only for {BASELINE_INIT_HOURS}Z inits; "
            f"cannot pair {bad[:5]} (score these against ERA5 only)")


def open_public_store(name: str, *, timeout: float = 600.0):
    """Open an official public store lazily over HTTPS (uses http(s)_proxy)."""
    import fsspec
    import xarray as xr
    verify_registry_against_official_configs()
    mapper = fsspec.get_mapper(gs_to_https(PUBLIC_STORES[name]), timeout=timeout)
    return xr.open_zarr(mapper, consolidated=True)


def probe_public_store(
    name: str,
    *,
    variable: str,
    time: Any,
    lead_hours: Optional[int] = None,
    level: Optional[int] = None,
) -> Dict[str, Any]:
    """Read ONE 2-D slice from a public store and summarise it.

    Year-gated like truth reads: only authorized years may be touched.
    """
    from .evaluate import assert_years_authorized
    t = np.datetime64(time, "ns")
    assert_years_authorized([int(str(t)[:4])])
    ds = open_public_store(name)
    da = ds[variable].sel(time=t)
    if lead_hours is not None:
        da = da.sel(prediction_timedelta=np.timedelta64(int(lead_hours), "h"))
    if level is not None:
        da = da.sel(level=int(level))
    values = np.asarray(da.values, dtype=np.float64)
    return {
        "store": name,
        "path": PUBLIC_STORES[name],
        "variable": variable,
        "time": str(t),
        "lead_hours": lead_hours,
        "level": level,
        "dims": list(da.dims),
        "shape": list(values.shape),
        "latitude_first_last": [float(da["latitude"].values[0]), float(da["latitude"].values[-1])],
        "longitude_first_last": [float(da["longitude"].values[0]), float(da["longitude"].values[-1])],
        "mean": float(np.nanmean(values)),
        "finite": bool(np.all(np.isfinite(values))),
        "values": values,
    }


__all__ = [
    "PUBLIC_STORES", "BASELINE_INIT_HOURS", "BaselineNotPairable",
    "verify_registry_against_official_configs", "gs_to_https",
    "assert_baseline_pairable", "open_public_store", "probe_public_store",
]
