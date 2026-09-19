"""CDS request template for Stormer-compatible 1.40625 deg ERA5 (DRAFT, not yet executed).

Why this exists: WeatherBench-2 ERA5 lives on GCS, which is unreachable from this cluster;
WeatherBench-1 (TUM) lacks mean_sea_level_pressure; ModelScope mirrors are samples/partial.
CDS (cds.climate.copernicus.eu) IS reachable. This template asks MARS to interpolate to the
exact Stormer cell-centre grid so that the 128x256 arrays line up with the official checkpoint.

Grid contract (from stormer/data_preprocessing/regrid_wb2.py):
    new_lat = linspace(-90 + 0.703125, 90 - 0.703125, 128)  -> -89.296875 ... 89.296875
    new_lon = linspace(0, 360, 256, endpoint=False)        ->   0.0 ... 358.59375
MARS 'area' is [North, West, South, East]; with grid 1.40625 this yields exactly those nodes.

Caveat: MARS interpolation is bilinear, Stormer's preprocessing used conservative regridding
from 0.25 deg. Run the parity test described in EarthDelta_v6_Review_and_Plan_CN.md section 5.4
(download a 0.25 deg subset, regrid with regrid_wb2.py, compare fields and zero-edit RMSE).

Requires: `pip install cdsapi` (0.7.7 on PyPI) and ~/.cdsapirc with the user's CDS key.
Dataset names follow the post-2024 CDS ("reanalysis-era5-single-levels",
"reanalysis-era5-pressure-levels"); request keys use the new API spelling
(`data_format`, `download_format`). Verify against the CDS form before the first real call.
"""
from __future__ import annotations

import calendar
import os
from pathlib import Path

# ---- Stormer 69-channel contract (order as in stormer/inference.py) --------------------------
SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
]
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
assert len(SINGLE_LEVEL_VARS) + len(PRESSURE_LEVEL_VARS) * len(PRESSURE_LEVELS) == 69

TIMES = ["00:00", "06:00", "12:00", "18:00"]
GRID = [1.40625, 1.40625]
AREA_CELL_CENTRES = [89.296875, 0.0, -89.296875, 358.59375]  # N, W, S, E  (128 x 256 nodes)


def month_request(dataset: str, variables: list[str], year: int, month: int,
                  levels: list[int] | None = None) -> dict:
    """One month, all 6-hourly times, netCDF, Stormer cell-centre grid."""
    ndays = calendar.monthrange(year, month)[1]
    req = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
        "time": TIMES,
        "grid": GRID,
        "area": AREA_CELL_CENTRES,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if levels is not None:
        req["pressure_level"] = [str(lv) for lv in levels]
    return req


def plan_requests(years: list[int], out_root: str) -> list[tuple[str, dict, str]]:
    """Return (dataset, request, target_path) triples; one file per variable-group per month.

    Size guide at 1.40625 deg: one field = 128*256*4 B ~ 131 kB; one month of 5 pressure
    variables x 13 levels x ~120 times ~ 1.0 GB uncompressed; single-level month ~ 60 MB.
    A year is ~12 GB. Split pressure-level requests per variable if CDS rejects the size.
    """
    triples = []
    for y in years:
        for m in range(1, 13):
            sl = month_request("reanalysis-era5-single-levels", SINGLE_LEVEL_VARS, y, m)
            triples.append(("reanalysis-era5-single-levels", sl,
                            os.path.join(out_root, "single", f"{y}-{m:02d}.nc")))
            for var in PRESSURE_LEVEL_VARS:
                pl = month_request("reanalysis-era5-pressure-levels", [var], y, m, PRESSURE_LEVELS)
                triples.append(("reanalysis-era5-pressure-levels", pl,
                                os.path.join(out_root, "pressure", var, f"{y}-{m:02d}.nc")))
    return triples


def run(years: list[int], out_root: str, dry_run: bool = True) -> None:
    triples = plan_requests(years, out_root)
    print(f"{len(triples)} requests planned for years {years}")
    if dry_run:
        for ds, req, path in triples[:3]:
            print(ds, path, req)
        return
    import cdsapi  # noqa: WPS433  (only needed for real downloads)

    client = cdsapi.Client()
    for ds, req, path in triples:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(path):
            continue
        client.retrieve(ds, req, path)


if __name__ == "__main__":
    # Pilot: validation 2019 + test 2020 first (parity checks need the test year), then 2015-2018.
    run(years=[2020], out_root="/mnt/afs/260010168/EarthDelta/data/era5_1p40625_cds", dry_run=True)

# After download: convert to Stormer's h5 layout with
#   stormer/data_preprocessing/process_one_step_data.py  (expects per-variable yearly .nc under root_dir)
# and reuse stormer/normalization_constants/*.npz (do NOT recompute unless the period changes).
# Latitude must be sorted ascending before saving (process_one_step_data.py does lat.sort()).
