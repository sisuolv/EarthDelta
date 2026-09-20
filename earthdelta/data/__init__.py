"""ERA5 data acquisition and split manifest utilities.

This subpackage provides:
- pull_wb2: Download and regrid ERA5 data from WeatherBench2 GCS zarr stores
- make_splits: Build split manifests with guard windows for train/val/test splits

The canonical variable order (69 channels) matches Stormer's checkpoint:
- 4 single-level: 2m_temperature, 10m_u_component_of_wind, 10m_v_component_of_wind, mean_sea_level_pressure
- 65 pressure-level (5 vars x 13 levels): geopotential, u, v, temperature, specific_humidity
  at levels [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
"""

from earthdelta.data.pull_wb2 import (
    CANONICAL_VARIABLES,
    PRESSURE_LEVELS,
    SINGLE_LEVEL_VARS,
    PRESSURE_VARS,
    variable_order_hash,
    get_stormer_target_grid,
)
from earthdelta.data.make_splits import (
    SplitManifest,
    build_manifest,
    YEAR_SPLITS,
)

__all__ = [
    'CANONICAL_VARIABLES',
    'PRESSURE_LEVELS',
    'SINGLE_LEVEL_VARS',
    'PRESSURE_VARS',
    'variable_order_hash',
    'get_stormer_target_grid',
    'SplitManifest',
    'build_manifest',
    'YEAR_SPLITS',
]
