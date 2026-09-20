"""ERA5 data acquisition from WeatherBench2 GCS zarr stores with conservative regridding.

R5 REDESIGN: Eliminates 13x download amplification by pulling by SOURCE VARIABLE
(not output channel). Pressure-level variables are fetched with all 13 levels per
time batch in one read, avoiding repeated chunk downloads.

Key improvements over previous implementation:
1. Pulls 9 source variables instead of 69 output channels
2. Time batches of 96 (multiple of source chunk size 8)
3. Channel-isolated output chunking (8, 1, 128, 256) for parallel writes
4. 3-process parallel pull with marker-file resume
5. Final disk-only rechunk to training-friendly (1, 69, 128, 256)
6. January pilot mode for fast S0 gate validation

Downloads ERA5 6-hourly data from gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr,
regrids to Stormer's 1.40625 deg 128x256 cell-centre grid (no poles).

Usage:
    # January pilot (unblocks S0 quickly):
    python -m earthdelta.data.pull_wb2 --year 2020 --pilot --https

    # Full year (3 parallel workers):
    python -m earthdelta.data.pull_wb2 --year 2020 --worker-id 0 --https &
    python -m earthdelta.data.pull_wb2 --year 2020 --worker-id 1 --https &
    python -m earthdelta.data.pull_wb2 --year 2020 --worker-id 2 --https &

Environment:
    Requires proxy for GCS access: http_proxy/https_proxy must be set.
"""
from __future__ import annotations

import os
import sys
import gc
import time
import hashlib
import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import warnings

import numpy as np

# Suppress FutureWarnings from google packages
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# Canonical variable order (69 channels) matching Stormer checkpoint
# ============================================================================

SINGLE_LEVEL_VARS = [
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
]

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

PRESSURE_VARS = [
    'geopotential',
    'u_component_of_wind',
    'v_component_of_wind',
    'temperature',
    'specific_humidity',
]

# Source variable to WB2 dataset name mapping
SOURCE_VAR_NAMES = {
    '2m_temperature': '2m_temperature',
    '10m_u_component_of_wind': '10m_u_component_of_wind',
    '10m_v_component_of_wind': '10m_v_component_of_wind',
    'mean_sea_level_pressure': 'mean_sea_level_pressure',
    'geopotential': 'geopotential',
    'u_component_of_wind': 'u_component_of_wind',
    'v_component_of_wind': 'v_component_of_wind',
    'temperature': 'temperature',
    'specific_humidity': 'specific_humidity',
}

# All 9 source variables
ALL_SOURCE_VARS = SINGLE_LEVEL_VARS + PRESSURE_VARS

# Source chunk dimensions: (time=8, level=13, lat=512, lon=256) for pressure vars
SOURCE_TIME_CHUNK = 8
TIME_BATCH_SIZE = 96  # Multiple of 8, aligned to source chunks

# Memory constraints:
# - This container's cgroup hard limit is 8GB (cat /sys/fs/cgroup/memory.max)
# - Baseline overhead (claude harness, mihomo proxy, etc.) is ~3GB
# - Available for pull_wb2 workers: ~5GB total
# - Peak RSS per pressure-var worker at TIME_BATCH_SIZE=96 is ~2-2.3GB
# - NEVER run more than ONE worker process concurrently in this environment
#   (3 concurrent workers x 2GB = 6GB + 3GB baseline = 9GB > 8GB limit => OOM kills)
# - Use --sequential mode for safe single-process full-year pulls
MEMORY_BUDGET_GB = 5.0  # Available for worker(s) after baseline overhead

# Output chunk shape for parallel writes (channel-isolated)
PARALLEL_OUTPUT_CHUNK = (8, 1, 128, 256)

# Final training-friendly chunk shape
FINAL_OUTPUT_CHUNK = (1, 69, 128, 256)

# Worker assignments for 3-process parallelism (balanced by data volume ~GB)
# Worker 0: geopotential, u_component_of_wind (~20 GB)
# Worker 1: v_component_of_wind, temperature (~20 GB)
# Worker 2: specific_humidity + 4 single-level vars (~13 GB)
WORKER_ASSIGNMENTS = {
    0: ['geopotential', 'u_component_of_wind'],
    1: ['v_component_of_wind', 'temperature'],
    2: ['specific_humidity', '2m_temperature', '10m_u_component_of_wind',
        '10m_v_component_of_wind', 'mean_sea_level_pressure'],
}


def build_canonical_variables() -> List[str]:
    """Build the canonical 69-variable list in exact Stormer config order.

    Order: single-level vars first, then pressure vars with level suffix.
    """
    variables = list(SINGLE_LEVEL_VARS)
    for var in PRESSURE_VARS:
        for level in PRESSURE_LEVELS:
            variables.append(f'{var}_{level}')
    assert len(variables) == 69, f'Expected 69 variables, got {len(variables)}'
    return variables


CANONICAL_VARIABLES = build_canonical_variables()


def get_channel_indices_for_source_var(source_var: str) -> List[int]:
    """Get the output channel indices for a source variable.

    Returns:
        List of channel indices in CANONICAL_VARIABLES for this source var.
        Single-level vars return [index], pressure vars return 13 indices.
    """
    if source_var in SINGLE_LEVEL_VARS:
        return [CANONICAL_VARIABLES.index(source_var)]
    else:
        # Pressure variable - 13 levels
        indices = []
        for level in PRESSURE_LEVELS:
            channel_name = f'{source_var}_{level}'
            indices.append(CANONICAL_VARIABLES.index(channel_name))
        return indices


def variable_order_hash() -> str:
    """Compute a stable SHA-256 hash of the canonical variable order.

    This hash can be used to verify consistency across different components.
    """
    data = '\n'.join(CANONICAL_VARIABLES).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


# ============================================================================
# Target grid (Stormer 1.40625 deg, 128x256, no poles)
# ============================================================================

def get_stormer_target_grid() -> Tuple[np.ndarray, np.ndarray]:
    """Get Stormer's target lat/lon grid (cell centres, no poles).

    Returns:
        lat: (128,) array from -89.296875 to 89.296875 deg
        lon: (256,) array from 0 to 358.59375 deg
    """
    ddeg = 1.40625
    lat_start = -90 + ddeg / 2  # -89.296875
    lat_stop = 90 - ddeg / 2     # 89.296875
    new_lat = np.linspace(lat_start, lat_stop, num=128, endpoint=True)
    new_lon = np.linspace(0, 360, num=256, endpoint=False)
    return new_lat, new_lon


def grid_hash() -> str:
    """Compute hash of target grid coordinates for reproducibility tracking."""
    lat, lon = get_stormer_target_grid()
    data = np.concatenate([lat, lon]).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


# ============================================================================
# Conservative regridding (adapted from WeatherBench2/Stormer reference)
# ============================================================================

def _latitude_cell_bounds(lat_rad: np.ndarray) -> np.ndarray:
    """Compute latitude cell bounds from cell centres (radians)."""
    pi_over_2 = np.array([np.pi / 2], dtype=lat_rad.dtype)
    return np.concatenate([-pi_over_2, (lat_rad[:-1] + lat_rad[1:]) / 2, pi_over_2])


def _latitude_overlap(source_lat: np.ndarray, target_lat: np.ndarray) -> np.ndarray:
    """Compute area overlap weights for latitude conservative regridding.

    Args:
        source_lat: Source latitude cell centres (radians), shape (S,)
        target_lat: Target latitude cell centres (radians), shape (T,)

    Returns:
        Overlap weights, shape (T, S), normalized area overlaps
    """
    source_bounds = _latitude_cell_bounds(source_lat)
    target_bounds = _latitude_cell_bounds(target_lat)

    # Compute overlap: integral of cos(lat) from lower to upper bound
    upper = np.minimum(
        target_bounds[1:, np.newaxis], source_bounds[np.newaxis, 1:]
    )
    lower = np.maximum(
        target_bounds[:-1, np.newaxis], source_bounds[np.newaxis, :-1]
    )
    # Area = sin(upper) - sin(lower), only where upper > lower
    overlap = np.where(upper > lower, np.sin(upper) - np.sin(lower), 0.0)
    return overlap


def _conservative_latitude_weights(source_lat: np.ndarray, target_lat: np.ndarray) -> np.ndarray:
    """Compute conservative regridding weights for latitude.

    Args:
        source_lat: Source latitudes in radians (must be increasing)
        target_lat: Target latitudes in radians (must be increasing)

    Returns:
        Weight matrix (T, S) with rows summing to 1
    """
    assert np.all(np.diff(source_lat) > 0), 'source_lat must be increasing'
    assert np.all(np.diff(target_lat) > 0), 'target_lat must be increasing'

    weights = _latitude_overlap(source_lat, target_lat)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights


def _align_phase(x: np.ndarray, target: np.ndarray, period: float) -> np.ndarray:
    """Align phase of periodic values to minimize distance from target."""
    shift_down = x > target + period / 2
    shift_up = x < target - period / 2
    return x + period * shift_up.astype(float) - period * shift_down.astype(float)


def _periodic_bounds(x: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute periodic cell bounds for longitude."""
    x_plus = _align_phase(np.roll(x, -1), x, period)
    x_minus = _align_phase(np.roll(x, 1), x, period)
    upper = (x + x_plus) / 2
    lower = (x_minus + x) / 2
    return lower, upper


def _longitude_overlap(first_points: np.ndarray, second_points: np.ndarray,
                       period: float = 2 * np.pi) -> np.ndarray:
    """Compute longitude overlap weights with periodic boundary handling.

    Args:
        first_points: First grid longitude cell centres (radians) - shape (N1,)
        second_points: Second grid longitude cell centres (radians) - shape (N2,)
        period: Periodicity (2*pi for radians)

    Returns:
        Overlap weights, shape (N1, N2)
    """
    first_points = first_points % period
    second_points = second_points % period

    first_lower, first_upper = _periodic_bounds(first_points, period)
    second_lower, second_upper = _periodic_bounds(second_points, period)

    # Fully vectorized overlap computation
    # Expand to (N1, N2) grids
    first_lower_2d = first_lower[:, np.newaxis]  # (N1, 1)
    first_upper_2d = first_upper[:, np.newaxis]  # (N1, 1)
    second_lower_2d = second_lower[np.newaxis, :]  # (1, N2)
    second_upper_2d = second_upper[np.newaxis, :]  # (1, N2)

    # Align second bounds to first (vectorized)
    # _align_phase: shift periodic value to minimize distance from target
    def align_phase_2d(x, target, period):
        shift_down = x > target + period / 2
        shift_up = x < target - period / 2
        return x + period * shift_up - period * shift_down

    sl = align_phase_2d(second_lower_2d, first_lower_2d, period)  # (N1, N2)
    su = align_phase_2d(second_upper_2d, first_lower_2d, period)  # (N1, N2)

    upper = np.minimum(first_upper_2d, su)
    lower = np.maximum(first_lower_2d, sl)
    overlap = np.maximum(upper - lower, 0.0)

    return overlap


def _conservative_longitude_weights(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Compute conservative regridding weights for longitude.

    Args:
        source_points: Source longitudes in radians (must be increasing within [0, 2pi))
        target_points: Target longitudes in radians (must be increasing within [0, 2pi))

    Returns:
        Weight matrix (T, S) with rows summing to 1, where T=len(target), S=len(source)
    """
    assert np.all(np.diff(source_points) > 0), 'source_points must be increasing'
    assert np.all(np.diff(target_points) > 0), 'target_points must be increasing'

    # _longitude_overlap(first, second) returns (len(first), len(second))
    # We want (target, source) so pass target as first, source as second
    weights = _longitude_overlap(target_points, source_points)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights


class ConservativeRegridder:
    """Conservative regridding from source to target grid.

    This preserves the global mean (area-weighted) of the field.
    Adapted from WeatherBench2 implementation.
    """

    def __init__(self, source_lat: np.ndarray, source_lon: np.ndarray,
                 target_lat: np.ndarray, target_lon: np.ndarray):
        """Initialize regridder with source and target grids.

        Args:
            source_lat, source_lon: Source grid coordinates (degrees)
            target_lat, target_lon: Target grid coordinates (degrees)
        """
        # Convert to radians
        src_lat_rad = np.deg2rad(source_lat)
        src_lon_rad = np.deg2rad(source_lon)
        tgt_lat_rad = np.deg2rad(target_lat)
        tgt_lon_rad = np.deg2rad(target_lon)

        # Precompute weight matrices
        self.lat_weights = _conservative_latitude_weights(src_lat_rad, tgt_lat_rad)
        self.lon_weights = _conservative_longitude_weights(src_lon_rad, tgt_lon_rad)

        self.source_shape = (len(source_lon), len(source_lat))
        self.target_shape = (len(target_lon), len(target_lat))

    def regrid(self, field: np.ndarray) -> np.ndarray:
        """Regrid a field from source to target grid.

        Args:
            field: Array with shape (..., lon, lat). NaNs are handled with nanmean semantics.

        Returns:
            Regridded array with shape (..., target_lon, target_lat)
        """
        # Handle NaNs: use weighted mean excluding NaNs
        nulls = np.isnan(field)
        field_clean = np.where(nulls, 0.0, field)
        valid = np.logical_not(nulls).astype(np.float32)

        # Regrid both data and validity mask
        # CRITICAL: optimize=True for ~100x speedup (55s -> 0.6s per einsum)
        total = np.einsum('ab,cd,...bd->...ac', self.lon_weights, self.lat_weights, field_clean, optimize=True)
        count = np.einsum('ab,cd,...bd->...ac', self.lon_weights, self.lat_weights, valid, optimize=True)

        # NaN where count is 0
        result = np.where(count > 0, total / count, np.nan)
        return result.astype(np.float32)


# ============================================================================
# GCS data access
# ============================================================================

GCS_ZARR_PATH = 'gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr'
# Alternative HTTPS path if needed:
HTTPS_ZARR_PATH = 'https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr/'


def open_wb2_zarr(use_https: bool = False):
    """Open the WeatherBench2 ERA5 zarr store from GCS.

    Args:
        use_https: If True, use HTTPS instead of gs:// protocol

    Returns:
        xarray.Dataset
    """
    import xarray as xr
    import fsspec

    if use_https:
        # HTTPS access via fsspec with extended timeouts for slow proxy
        # Set long request timeout (in seconds) to handle slow network
        storage_options = {
            'timeout': 600,  # 10 minute total timeout per request
        }

        mapper = fsspec.get_mapper(HTTPS_ZARR_PATH, **storage_options)
        return xr.open_zarr(mapper, consolidated=True)
    else:
        # GCS access via gcsfs
        import gcsfs
        fs = gcsfs.GCSFileSystem(token='anon')
        mapper = fs.get_mapper(GCS_ZARR_PATH)
        return xr.open_zarr(mapper, consolidated=True)


# ============================================================================
# Year / timestep helpers
# ============================================================================

def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def expected_timesteps(year: int) -> int:
    """Return expected number of 6-hourly timesteps for a year."""
    days = 366 if is_leap_year(year) else 365
    return days * 4


def is_year_complete(output_path: Path, year: int) -> bool:
    """Check if a zarr store is complete for a given year.

    Completeness requires:
    1. Exact number of timesteps for the year (leap-aware)
    2. All 69 channels present
    """
    import xarray as xr

    if not output_path.exists():
        return False

    try:
        existing = xr.open_zarr(output_path)
        n_time = len(existing.time)
        n_channels = len(existing.channel) if 'channel' in existing.dims else 0
        existing.close()

        expected_time = expected_timesteps(year)
        if n_time != expected_time:
            return False
        if n_channels != 69:
            return False
        return True
    except Exception:
        return False


def is_pilot_complete(output_path: Path, expected_steps: int = 124) -> bool:
    """Check if a pilot zarr store is complete.

    Args:
        output_path: Path to zarr store
        expected_steps: Expected number of timesteps (default 124 for January)

    Returns:
        True if complete with expected steps and 69 channels
    """
    import xarray as xr

    if not output_path.exists():
        return False

    try:
        existing = xr.open_zarr(output_path)
        n_time = len(existing.time)
        n_channels = len(existing.channel) if 'channel' in existing.dims else 0
        existing.close()

        if n_time != expected_steps:
            return False
        if n_channels != 69:
            return False
        return True
    except Exception:
        return False


# ============================================================================
# Marker file logic for resume
# ============================================================================

def get_marker_dir(output_dir: Path, year: int) -> Path:
    """Get the marker directory for a year's pull."""
    return output_dir / f'.markers_{year}'


def mark_source_var_complete(output_dir: Path, year: int, source_var: str) -> None:
    """Mark a source variable as complete for resume."""
    marker_dir = get_marker_dir(output_dir, year)
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_file = marker_dir / f'{source_var}.done'
    marker_file.write_text(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ")}\n')


def is_source_var_complete(output_dir: Path, year: int, source_var: str) -> bool:
    """Check if a source variable is already complete."""
    marker_file = get_marker_dir(output_dir, year) / f'{source_var}.done'
    return marker_file.exists()


def clear_markers(output_dir: Path, year: int) -> None:
    """Clear all markers for a year (fresh start)."""
    marker_dir = get_marker_dir(output_dir, year)
    if marker_dir.exists():
        import shutil
        shutil.rmtree(marker_dir)


def all_source_vars_complete(output_dir: Path, year: int) -> bool:
    """Check if all source variables are complete."""
    for source_var in ALL_SOURCE_VARS:
        if not is_source_var_complete(output_dir, year, source_var):
            return False
    return True


# ============================================================================
# Pull by source variable (core R5 change)
# ============================================================================

def pull_source_variable(
    ds,
    source_var: str,
    year: int,
    regridder: ConservativeRegridder,
    output_path: Path,
    n_timesteps: int,
    time_batch_size: int = TIME_BATCH_SIZE,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Pull and regrid a single source variable, writing directly to zarr.

    For pressure-level variables, fetches all 13 levels in one read per batch,
    avoiding the 13x download amplification of the old per-level approach.

    Args:
        ds: Open xarray Dataset
        source_var: Source variable name (one of 9 source vars)
        year: Year to process
        regridder: ConservativeRegridder instance
        output_path: Path to output zarr store (must exist with correct structure)
        n_timesteps: Number of timesteps for this year
        time_batch_size: Batch size for time dimension
        verbose: Print progress

    Returns:
        Dict with stats (elapsed_s, bytes_fetched_approx, etc.)
    """
    import zarr

    start_time = time.time()

    # Get channel indices for this source variable
    channel_indices = get_channel_indices_for_source_var(source_var)
    is_pressure = source_var in PRESSURE_VARS
    n_levels = 13 if is_pressure else 1

    if verbose:
        print(f'  Processing {source_var} ({n_levels} level{"s" if n_levels > 1 else ""}, '
              f'channels {channel_indices[0]}-{channel_indices[-1] if is_pressure else channel_indices[0]})...',
              flush=True)

    # Open zarr store for writing
    store = zarr.open(str(output_path), mode='r+')
    data_array = store['data']

    # Get source data reference (lazy)
    da = ds[source_var].sel(time=str(year))

    # Process in time batches
    n_batches = (n_timesteps + time_batch_size - 1) // time_batch_size
    bytes_fetched = 0

    for batch_idx in range(n_batches):
        t_start = batch_idx * time_batch_size
        t_end = min((batch_idx + 1) * time_batch_size, n_timesteps)
        batch_len = t_end - t_start

        if verbose and (batch_idx == 0 or batch_idx == n_batches - 1 or batch_idx % 3 == 0):
            print(f'    Batch {batch_idx+1}/{n_batches}: timesteps {t_start}-{t_end}...', flush=True)

        # Load this batch with retry logic
        max_retries = 5
        for retry in range(max_retries):
            try:
                batch_slice = da.isel(time=slice(t_start, t_end))

                if is_pressure:
                    # Fetch ALL 13 levels at once (key optimization)
                    # Shape: (batch, level=13, lat=512, lon=256) from source
                    # After transpose: (batch, level, lon, lat)
                    batch_data = batch_slice.transpose('time', 'level', 'longitude', 'latitude').values
                else:
                    # Single-level: (batch, lat, lon) -> (batch, lon, lat)
                    batch_data = batch_slice.transpose('time', 'longitude', 'latitude').values
                    # Add level dimension for uniform handling
                    batch_data = batch_data[:, np.newaxis, :, :]  # (batch, 1, lon, lat)

                break
            except Exception as e:
                if retry < max_retries - 1:
                    gc.collect()
                    wait_time = min(30 * (2 ** retry), 300)  # Exponential backoff, max 5 min
                    if verbose:
                        print(f'    Network error, retrying in {wait_time}s... ({e})', flush=True)
                    time.sleep(wait_time)
                else:
                    print(f'    Failed after {max_retries} retries: {e}', flush=True)
                    raise

        # Estimate bytes fetched (for progress tracking)
        bytes_fetched += batch_data.nbytes

        # Regrid each level and write to correct channel
        # batch_data shape: (batch, levels, lon, lat)
        for level_idx, channel_idx in enumerate(channel_indices):
            level_data = batch_data[:, level_idx, :, :]  # (batch, lon, lat)

            # Regrid: (batch, lon, lat) -> (batch, target_lon, target_lat)
            regridded = regridder.regrid(level_data)

            # Transpose to (batch, lat, lon) for zarr
            regridded = regridded.transpose(0, 2, 1).astype(np.float32)

            # Write to zarr at correct channel index
            data_array[t_start:t_end, channel_idx, :, :] = regridded

        # Cleanup
        del batch_data
        gc.collect()

    elapsed = time.time() - start_time

    if verbose:
        print(f'    Done in {elapsed:.1f}s ({bytes_fetched / 1e9:.2f} GB fetched)', flush=True)

    return {
        'source_var': source_var,
        'elapsed_s': elapsed,
        'bytes_fetched': bytes_fetched,
        'n_channels': len(channel_indices),
    }


def create_intermediate_zarr(
    output_path: Path,
    time_coords: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    chunk_shape: Tuple[int, int, int, int] = PARALLEL_OUTPUT_CHUNK,
) -> None:
    """Create the zarr store with channel-isolated chunks for parallel writes.

    Args:
        output_path: Path for output zarr
        time_coords: Time coordinate values
        target_lat: Target latitude array
        target_lon: Target longitude array
        chunk_shape: Output chunk shape (default: (8, 1, 128, 256) for parallel)
    """
    import xarray as xr
    import dask.array as da
    import shutil

    # Remove any existing partial output
    if output_path.exists():
        shutil.rmtree(output_path)

    n_timesteps = len(time_coords)

    # Create empty dask array with specified chunks
    empty_data = da.full(
        (n_timesteps, 69, 128, 256),
        fill_value=np.nan,
        dtype=np.float32,
        chunks=chunk_shape,
    )

    # Create dataset
    init_ds = xr.Dataset(
        {
            'data': (['time', 'channel', 'lat', 'lon'], empty_data),
        },
        coords={
            'time': time_coords,
            'channel': CANONICAL_VARIABLES,
            'lat': target_lat,
            'lon': target_lon,
        },
        attrs={
            'source': GCS_ZARR_PATH,
            'regridding': 'conservative',
            'source_grid': '512x256',
            'target_grid': '128x256',
            'variable_order_hash': variable_order_hash(),
            'grid_hash': grid_hash(),
            'intermediate_chunks': str(chunk_shape),
            'created': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        }
    )

    # Write metadata only (compute=False)
    init_ds.to_zarr(output_path, mode='w', compute=False)
    del init_ds, empty_data


# ============================================================================
# Rechunking (disk-only, memory-efficient)
# ============================================================================

def rechunk_to_final(
    intermediate_path: Path,
    final_path: Path,
    final_chunks: Tuple[int, int, int, int] = FINAL_OUTPUT_CHUNK,
    memory_limit_gb: float = 6.0,
) -> Dict[str, Any]:
    """Rechunk from intermediate (8,1,128,256) to final (1,69,128,256) layout.

    This is a disk-only streaming operation that works within the 8GB cgroup.
    Processes one output time slice at a time to bound memory usage.

    Args:
        intermediate_path: Path to intermediate zarr with (8,1,128,256) chunks
        final_path: Path for final zarr with (1,69,128,256) chunks
        final_chunks: Target chunk shape
        memory_limit_gb: Memory limit for processing

    Returns:
        Stats dict with elapsed_s, etc.
    """
    import xarray as xr
    import zarr
    import shutil

    start_time = time.time()

    print(f'Rechunking {intermediate_path} -> {final_path}...', flush=True)

    # Open intermediate store
    src_ds = xr.open_zarr(intermediate_path)
    n_time = len(src_ds.time)
    time_coords = src_ds.time.values
    target_lat = src_ds.lat.values
    target_lon = src_ds.lon.values

    # Remove existing final path
    if final_path.exists():
        shutil.rmtree(final_path)

    # Create final zarr with training-friendly chunks
    import dask.array as da
    empty_data = da.full(
        (n_time, 69, 128, 256),
        fill_value=np.nan,
        dtype=np.float32,
        chunks=final_chunks,
    )

    final_ds = xr.Dataset(
        {
            'data': (['time', 'channel', 'lat', 'lon'], empty_data),
        },
        coords={
            'time': time_coords,
            'channel': CANONICAL_VARIABLES,
            'lat': target_lat,
            'lon': target_lon,
        },
        attrs={
            'source': GCS_ZARR_PATH,
            'regridding': 'conservative',
            'source_grid': '512x256',
            'target_grid': '128x256',
            'variable_order_hash': variable_order_hash(),
            'grid_hash': grid_hash(),
            'chunks': str(final_chunks),
            'created': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        }
    )
    final_ds.to_zarr(final_path, mode='w', compute=False)
    del final_ds, empty_data

    # Open both stores for direct copy
    src_store = zarr.open(str(intermediate_path), mode='r')
    dst_store = zarr.open(str(final_path), mode='r+')
    src_data = src_store['data']
    dst_data = dst_store['data']

    # Process in time slices that fit in memory
    # Each time slice is (69, 128, 256) * 4 bytes = ~9 MB
    # Safe to do many at once, but we'll do small batches for safety
    batch_size = 100  # ~900 MB per batch, well under 6GB limit

    n_batches = (n_time + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        t_start = batch_idx * batch_size
        t_end = min((batch_idx + 1) * batch_size, n_time)

        if batch_idx % 5 == 0:
            print(f'  Rechunk batch {batch_idx+1}/{n_batches}: times {t_start}-{t_end}...', flush=True)

        # Read from intermediate
        batch_data = src_data[t_start:t_end, :, :, :]

        # Write to final (zarr handles chunking)
        dst_data[t_start:t_end, :, :, :] = batch_data

        del batch_data
        gc.collect()

    src_ds.close()

    elapsed = time.time() - start_time
    print(f'Rechunk complete in {elapsed:.1f}s', flush=True)

    return {'elapsed_s': elapsed}


# ============================================================================
# January pilot pull (fast path for S0 gate)
# ============================================================================

def pull_january_pilot(
    output_dir: Path,
    use_https: bool = True,
    year: int = 2020,
    n_days: int = 31,
) -> Dict[str, Any]:
    """Pull January pilot store for fast S0 gate validation.

    Pulls first ~31 days (124 timesteps) of the year with all 69 channels.
    Single-process, direct to final chunk layout (1, 69, 128, 256).

    Args:
        output_dir: Output directory
        use_https: Use HTTPS transport
        year: Year to pull (default 2020)
        n_days: Number of days to pull (default 31 for January)

    Returns:
        Stats dict with shape, means, elapsed time, etc.
    """
    import xarray as xr

    n_timesteps = n_days * 4  # 6-hourly
    output_path = output_dir / f'{year}_jan.zarr'

    if is_pilot_complete(output_path, n_timesteps):
        print(f'Pilot already complete at {output_path}')
        return {'status': 'skipped', 'path': str(output_path)}

    print(f'Pulling January {year} pilot ({n_timesteps} timesteps)...', flush=True)
    start_time = time.time()

    # Open source data
    ds = open_wb2_zarr(use_https=use_https)

    # Get grid info
    source_lat = ds.latitude.values
    source_lon = ds.longitude.values

    # Ensure latitude is increasing
    if source_lat[0] > source_lat[-1]:
        source_lat = source_lat[::-1]
        ds = ds.isel(latitude=slice(None, None, -1))

    # Get target grid and create regridder
    target_lat, target_lon = get_stormer_target_grid()
    regridder = ConservativeRegridder(source_lat, source_lon, target_lat, target_lon)

    # Get time coordinates for January
    time_coords = ds.time.sel(time=str(year)).values[:n_timesteps]

    # Create output zarr with final chunks (1, 69, 128, 256) directly
    create_intermediate_zarr(
        output_path, time_coords, target_lat, target_lon,
        chunk_shape=FINAL_OUTPUT_CHUNK
    )

    # Stats accumulators
    t2m_sum = 0.0
    t2m_count = 0
    mslp_sum = 0.0
    mslp_count = 0

    # Pull all source variables
    for source_var in ALL_SOURCE_VARS:
        stats = pull_source_variable(
            ds, source_var, year, regridder, output_path,
            n_timesteps=n_timesteps,
            time_batch_size=min(TIME_BATCH_SIZE, n_timesteps),
            verbose=True,
        )

    ds.close()

    # Verify and compute sanity stats
    result_ds = xr.open_zarr(output_path)
    data = result_ds.data

    t2m_idx = CANONICAL_VARIABLES.index('2m_temperature')
    mslp_idx = CANONICAL_VARIABLES.index('mean_sea_level_pressure')

    t2m_data = data[:, t2m_idx, :, :].values
    mslp_data = data[:, mslp_idx, :, :].values

    t2m_mean = float(np.nanmean(t2m_data))
    mslp_mean = float(np.nanmean(mslp_data))

    elapsed = time.time() - start_time
    shape = tuple(data.shape)
    result_ds.close()

    # File size
    file_size_gb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024**3)

    stats = {
        'status': 'completed',
        'path': str(output_path),
        'shape': shape,
        't2m_mean_K': t2m_mean,
        'mslp_mean_Pa': mslp_mean,
        'elapsed_s': elapsed,
        'file_size_gb': file_size_gb,
    }

    print(f'\nJanuary pilot complete:')
    print(f'  Path: {output_path}')
    print(f'  Shape: {shape}')
    print(f'  2m_temperature mean: {t2m_mean:.2f} K (expected: 270-290 K)')
    print(f'  MSLP mean: {mslp_mean:.0f} Pa (expected: 100000-102000 Pa)')
    print(f'  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'  File size: {file_size_gb:.2f} GB')

    return stats


# ============================================================================
# Full year pull (parallel workers)
# ============================================================================

def pull_year_worker(
    year: int,
    worker_id: int,
    output_dir: Path,
    use_https: bool = True,
) -> Dict[str, Any]:
    """Pull a subset of source variables for a year (one worker).

    Each worker writes only to its assigned channels (no conflicts).

    Args:
        year: Year to process
        worker_id: Worker ID (0, 1, or 2)
        output_dir: Output directory
        use_https: Use HTTPS transport

    Returns:
        Stats dict
    """
    import xarray as xr

    if worker_id not in WORKER_ASSIGNMENTS:
        raise ValueError(f'Invalid worker_id {worker_id}, must be 0, 1, or 2')

    source_vars = WORKER_ASSIGNMENTS[worker_id]
    intermediate_path = output_dir / f'{year}_intermediate.zarr'

    print(f'Worker {worker_id}: Processing {source_vars}', flush=True)
    start_time = time.time()

    # Check which vars are already done
    remaining_vars = [v for v in source_vars if not is_source_var_complete(output_dir, year, v)]

    if not remaining_vars:
        print(f'Worker {worker_id}: All assigned variables already complete', flush=True)
        return {'status': 'skipped', 'worker_id': worker_id, 'vars': source_vars}

    print(f'Worker {worker_id}: Remaining vars: {remaining_vars}', flush=True)

    # Open source data
    ds = open_wb2_zarr(use_https=use_https)

    # Get grid info
    source_lat = ds.latitude.values
    source_lon = ds.longitude.values

    if source_lat[0] > source_lat[-1]:
        source_lat = source_lat[::-1]
        ds = ds.isel(latitude=slice(None, None, -1))

    # Get target grid and create regridder
    target_lat, target_lon = get_stormer_target_grid()
    regridder = ConservativeRegridder(source_lat, source_lon, target_lat, target_lon)

    # Get time coordinates and count
    time_coords = ds.time.sel(time=str(year)).values
    n_timesteps = len(time_coords)

    # Ensure intermediate zarr exists (worker 0 creates, others wait)
    if worker_id == 0:
        if not intermediate_path.exists():
            print(f'Worker 0: Creating intermediate zarr at {intermediate_path}', flush=True)
            create_intermediate_zarr(
                intermediate_path, time_coords, target_lat, target_lon,
                chunk_shape=PARALLEL_OUTPUT_CHUNK
            )
    else:
        # Wait for worker 0 to create the zarr
        for _ in range(60):  # Wait up to 5 minutes
            if intermediate_path.exists():
                break
            time.sleep(5)
        if not intermediate_path.exists():
            raise RuntimeError(f'Intermediate zarr not created by worker 0 after 5 min')

    # Process each assigned source variable
    var_stats = []
    for source_var in remaining_vars:
        stats = pull_source_variable(
            ds, source_var, year, regridder, intermediate_path,
            n_timesteps=n_timesteps,
            verbose=True,
        )
        var_stats.append(stats)

        # Mark complete
        mark_source_var_complete(output_dir, year, source_var)
        print(f'Worker {worker_id}: Marked {source_var} complete', flush=True)

    ds.close()

    elapsed = time.time() - start_time

    return {
        'status': 'completed',
        'worker_id': worker_id,
        'vars': source_vars,
        'var_stats': var_stats,
        'elapsed_s': elapsed,
    }


def pull_year_sequential(
    year: int,
    output_dir: Path,
    use_https: bool = True,
) -> Dict[str, Any]:
    """Pull all source variables sequentially in a single process.

    This mode is REQUIRED when running in a memory-constrained environment
    (e.g., 8GB cgroup with ~3GB baseline overhead). Running multiple workers
    concurrently will cause OOM kills.

    Peak memory: ~2-2.3GB for one pressure-variable batch + overhead, totaling
    ~5GB max, safely under the ~5GB available budget.

    Args:
        year: Year to process
        output_dir: Output directory
        use_https: Use HTTPS transport

    Returns:
        Stats dict
    """
    import xarray as xr

    intermediate_path = output_dir / f'{year}_intermediate.zarr'

    print(f'Sequential pull: Processing all 9 source variables for {year}', flush=True)
    start_time = time.time()

    # Check which vars are already done
    remaining_vars = [v for v in ALL_SOURCE_VARS if not is_source_var_complete(output_dir, year, v)]

    if not remaining_vars:
        print(f'All source variables already complete for {year}', flush=True)
        return {'status': 'skipped', 'vars': ALL_SOURCE_VARS}

    print(f'Remaining vars to process: {remaining_vars}', flush=True)
    print(f'Already complete: {[v for v in ALL_SOURCE_VARS if v not in remaining_vars]}', flush=True)

    # Open source data
    ds = open_wb2_zarr(use_https=use_https)

    # Get grid info
    source_lat = ds.latitude.values
    source_lon = ds.longitude.values

    if source_lat[0] > source_lat[-1]:
        source_lat = source_lat[::-1]
        ds = ds.isel(latitude=slice(None, None, -1))

    # Get target grid and create regridder
    target_lat, target_lon = get_stormer_target_grid()
    regridder = ConservativeRegridder(source_lat, source_lon, target_lat, target_lon)

    # Get time coordinates and count
    time_coords = ds.time.sel(time=str(year)).values
    n_timesteps = len(time_coords)

    # Ensure intermediate zarr exists (create if needed)
    if not intermediate_path.exists():
        print(f'Creating intermediate zarr at {intermediate_path}', flush=True)
        create_intermediate_zarr(
            intermediate_path, time_coords, target_lat, target_lon,
            chunk_shape=PARALLEL_OUTPUT_CHUNK
        )

    # Process each source variable sequentially
    var_stats = []
    for source_var in remaining_vars:
        print(f'\n=== Processing {source_var} ({remaining_vars.index(source_var)+1}/{len(remaining_vars)}) ===',
              flush=True)

        stats = pull_source_variable(
            ds, source_var, year, regridder, intermediate_path,
            n_timesteps=n_timesteps,
            verbose=True,
        )
        var_stats.append(stats)

        # Mark complete
        mark_source_var_complete(output_dir, year, source_var)
        print(f'Marked {source_var} complete', flush=True)

        # Force garbage collection between variables
        gc.collect()

    ds.close()

    elapsed = time.time() - start_time

    return {
        'status': 'completed',
        'vars': ALL_SOURCE_VARS,
        'var_stats': var_stats,
        'elapsed_s': elapsed,
        'intermediate_path': str(intermediate_path),
    }


def finalize_year(
    year: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Finalize a year's data after all workers complete.

    1. Check all source vars are complete
    2. Rechunk to final layout
    3. Run completeness check and sanity stats

    Args:
        year: Year to finalize
        output_dir: Output directory

    Returns:
        Stats dict
    """
    import xarray as xr

    intermediate_path = output_dir / f'{year}_intermediate.zarr'
    final_path = output_dir / f'{year}.zarr'

    # Check all source vars are complete
    if not all_source_vars_complete(output_dir, year):
        missing = [v for v in ALL_SOURCE_VARS if not is_source_var_complete(output_dir, year, v)]
        return {'status': 'incomplete', 'missing_vars': missing}

    print(f'All source variables complete for {year}, finalizing...', flush=True)

    # Rechunk to final layout
    rechunk_stats = rechunk_to_final(intermediate_path, final_path)

    # Verify completeness
    if not is_year_complete(final_path, year):
        return {'status': 'verification_failed', 'path': str(final_path)}

    # Compute sanity stats
    result_ds = xr.open_zarr(final_path)
    data = result_ds.data

    t2m_idx = CANONICAL_VARIABLES.index('2m_temperature')
    mslp_idx = CANONICAL_VARIABLES.index('mean_sea_level_pressure')

    # Sample a subset for stats (avoid loading all data)
    t2m_sample = data[::10, t2m_idx, :, :].values  # Every 10th timestep
    mslp_sample = data[::10, mslp_idx, :, :].values

    t2m_mean = float(np.nanmean(t2m_sample))
    mslp_mean = float(np.nanmean(mslp_sample))

    shape = tuple(data.shape)
    result_ds.close()

    file_size_gb = sum(f.stat().st_size for f in final_path.rglob('*') if f.is_file()) / (1024**3)

    stats = {
        'status': 'completed',
        'path': str(final_path),
        'shape': shape,
        't2m_mean_K': t2m_mean,
        'mslp_mean_Pa': mslp_mean,
        'file_size_gb': file_size_gb,
        'rechunk_elapsed_s': rechunk_stats['elapsed_s'],
    }

    print(f'\nYear {year} finalized:')
    print(f'  Path: {final_path}')
    print(f'  Shape: {shape}')
    print(f'  2m_temperature mean: {t2m_mean:.2f} K')
    print(f'  MSLP mean: {mslp_mean:.0f} Pa')
    print(f'  File size: {file_size_gb:.2f} GB')

    return stats


# ============================================================================
# Legacy single-process pull (for compatibility)
# ============================================================================

def pull_and_regrid_year(year: int, output_dir: Path,
                          use_https: bool = False,
                          skip_existing: bool = True) -> dict:
    """Pull and regrid one year of ERA5 data (legacy single-process mode).

    This is kept for compatibility but the parallel worker approach is preferred.

    Args:
        year: Year to process
        output_dir: Directory to save output zarr
        use_https: Use HTTPS instead of gs:// for GCS access
        skip_existing: Skip if output already exists

    Returns:
        Dictionary with stats: {file_path, n_timesteps, elapsed_s, throughput_gb_hr, sanity}
    """
    import xarray as xr

    output_path = output_dir / f'{year}.zarr'

    if skip_existing and is_year_complete(output_path, year):
        exp_ts = expected_timesteps(year)
        print(f'Skipping {year}: {output_path} already exists with {exp_ts} timesteps and 69 channels')
        return {'status': 'skipped', 'file_path': str(output_path), 'n_timesteps': exp_ts}

    start_time = time.time()
    print(f'Opening WeatherBench2 zarr store for year {year}...', flush=True)

    ds = open_wb2_zarr(use_https=use_https)

    # Get grid info
    source_lat = ds.latitude.values
    source_lon = ds.longitude.values

    if source_lat[0] > source_lat[-1]:
        source_lat = source_lat[::-1]
        ds = ds.isel(latitude=slice(None, None, -1))

    target_lat, target_lon = get_stormer_target_grid()
    regridder = ConservativeRegridder(source_lat, source_lon, target_lat, target_lon)

    time_coords = ds.time.sel(time=str(year)).values
    n_timesteps = len(time_coords)

    print(f'Processing {n_timesteps} timesteps for year {year}', flush=True)

    # Create output zarr with final chunks directly
    output_dir.mkdir(parents=True, exist_ok=True)
    create_intermediate_zarr(
        output_path, time_coords, target_lat, target_lon,
        chunk_shape=FINAL_OUTPUT_CHUNK
    )

    # Process all source variables
    for source_var in ALL_SOURCE_VARS:
        pull_source_variable(
            ds, source_var, year, regridder, output_path,
            n_timesteps=n_timesteps,
            verbose=True,
        )

    ds.close()

    elapsed = time.time() - start_time
    file_size_gb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024**3)
    throughput = file_size_gb / (elapsed / 3600) if elapsed > 0 else 0

    # Compute sanity stats
    result_ds = xr.open_zarr(output_path)
    t2m_idx = CANONICAL_VARIABLES.index('2m_temperature')
    mslp_idx = CANONICAL_VARIABLES.index('mean_sea_level_pressure')

    t2m_data = result_ds.data[::10, t2m_idx, :, :].values
    mslp_data = result_ds.data[::10, mslp_idx, :, :].values

    t2m_mean = float(np.nanmean(t2m_data))
    mslp_mean = float(np.nanmean(mslp_data))
    result_ds.close()

    stats = {
        'status': 'completed',
        'file_path': str(output_path),
        'n_timesteps': n_timesteps,
        'elapsed_s': elapsed,
        'file_size_gb': file_size_gb,
        'throughput_gb_hr': throughput,
        'sanity': {
            '2m_temperature_mean_K': t2m_mean,
            'mean_sea_level_pressure_mean_Pa': mslp_mean,
        }
    }

    print(f'\nCompleted {year}:')
    print(f'  Timesteps: {n_timesteps}')
    print(f'  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'  File size: {file_size_gb:.2f} GB')
    print(f'  Throughput: {throughput:.2f} GB/hour')
    print(f'  2m_temperature mean: {t2m_mean:.2f} K')
    print(f'  MSLP mean: {mslp_mean:.0f} Pa')

    return stats


# ============================================================================
# CLI
# ============================================================================

def main():
    import sys
    print('Starting pull_wb2 main...', flush=True)

    parser = argparse.ArgumentParser(
        description='Pull ERA5 data from WeatherBench2 GCS (R5 redesign)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # January pilot (fast path for S0 gate):
  python -m earthdelta.data.pull_wb2 --year 2020 --pilot --https

  # Full year SEQUENTIAL mode (REQUIRED for 8GB cgroup to avoid OOM):
  python -m earthdelta.data.pull_wb2 --year 2020 --sequential --https

  # Full year with 3 parallel workers (ONLY if >16GB memory available):
  # WARNING: 3 concurrent workers use ~6GB combined; with baseline overhead
  # this will OOM in 8GB cgroup. Use --sequential instead.
  python -m earthdelta.data.pull_wb2 --year 2020 --worker-id 0 --https &
  python -m earthdelta.data.pull_wb2 --year 2020 --worker-id 1 --https &
  python -m earthdelta.data.pull_wb2 --year 2020 --worker-id 2 --https &

  # Finalize after all workers complete (rechunk to training layout):
  python -m earthdelta.data.pull_wb2 --year 2020 --finalize

  # Legacy single-process mode:
  python -m earthdelta.data.pull_wb2 --year 2020 --https --single
'''
    )
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/afs/260010168/EarthDelta/data/era5_1p40625',
                        help='Output directory for zarr files')
    parser.add_argument('--https', action='store_true', help='Use HTTPS instead of gs://')
    parser.add_argument('--pilot', action='store_true', help='Pull January pilot only (~31 days)')
    parser.add_argument('--worker-id', type=int, choices=[0, 1, 2],
                        help='Worker ID for parallel pull (0, 1, or 2)')
    parser.add_argument('--finalize', action='store_true',
                        help='Finalize year after all workers complete (rechunk + verify)')
    parser.add_argument('--single', action='store_true',
                        help='Legacy single-process mode (not parallel)')
    parser.add_argument('--sequential', action='store_true',
                        help='Sequential mode: process all source vars in one process '
                             '(REQUIRED for 8GB cgroup environments to avoid OOM)')
    parser.add_argument('--fresh', action='store_true',
                        help='Clear markers and start fresh (ignore previous progress)')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        print(f'Clearing markers for year {args.year}...', flush=True)
        clear_markers(output_dir, args.year)

    if args.pilot:
        # January pilot mode
        stats = pull_january_pilot(output_dir, use_https=args.https, year=args.year)
        print(f'\nPilot stats: {json.dumps(stats, indent=2)}', flush=True)
    elif args.sequential:
        # Sequential mode - all vars in one process (safe for 8GB cgroup)
        stats = pull_year_sequential(args.year, output_dir, use_https=args.https)
        print(f'\nSequential stats: {json.dumps(stats, indent=2)}', flush=True)
    elif args.finalize:
        # Finalize mode
        stats = finalize_year(args.year, output_dir)
        print(f'\nFinalize stats: {json.dumps(stats, indent=2)}', flush=True)
    elif args.worker_id is not None:
        # Parallel worker mode
        stats = pull_year_worker(args.year, args.worker_id, output_dir, use_https=args.https)
        print(f'\nWorker stats: {json.dumps(stats, indent=2)}', flush=True)
    elif args.single:
        # Legacy single-process mode
        stats = pull_and_regrid_year(
            year=args.year,
            output_dir=output_dir,
            use_https=args.https,
            skip_existing=True,
        )
        print(f'\nFinal stats: {json.dumps(stats, indent=2)}', flush=True)
    else:
        parser.print_help()
        print('\nError: Specify --pilot, --sequential, --worker-id, --finalize, or --single', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
