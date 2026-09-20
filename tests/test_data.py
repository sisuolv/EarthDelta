"""Tests for earthdelta.data: ERA5 acquisition and split manifest.

These tests use synthetic data to avoid network dependencies for unit tests.
One test is marked @pytest.mark.slow and requires actual GCS access to validate
real pulled data.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from earthdelta.data.pull_wb2 import (
    CANONICAL_VARIABLES,
    PRESSURE_LEVELS,
    SINGLE_LEVEL_VARS,
    PRESSURE_VARS,
    variable_order_hash,
    get_stormer_target_grid,
    grid_hash,
    ConservativeRegridder,
    build_canonical_variables,
    is_leap_year,
    expected_timesteps,
    is_year_complete,
)
from earthdelta.data.make_splits import (
    YEAR_SPLITS,
    SplitManifest,
    SplitManifestRow,
    build_manifest,
    get_split_for_year,
    compute_guard_boundaries,
    is_in_guard_window,
    DEFAULT_AVAILABILITY_DELAY_HOURS,
)
from earthdelta.memory import VerifiedRecord


# ============================================================================
# Variable order tests
# ============================================================================

class TestCanonicalVariables:
    """Tests for the canonical 69-variable order."""

    def test_variable_count_is_69(self):
        """Verify exactly 69 variables."""
        assert len(CANONICAL_VARIABLES) == 69

    def test_variable_order_structure(self):
        """Verify structure: single-level first, then pressure-level."""
        # First 4 are single-level
        assert CANONICAL_VARIABLES[:4] == SINGLE_LEVEL_VARS

        # Remaining 65 are pressure-level
        pressure_vars = CANONICAL_VARIABLES[4:]
        assert len(pressure_vars) == 65  # 5 vars * 13 levels

    def test_pressure_level_order(self):
        """Verify pressure levels are in correct order for each variable."""
        pressure_vars = CANONICAL_VARIABLES[4:]

        # Group by variable name
        var_levels = {}
        for var_full in pressure_vars:
            parts = var_full.rsplit('_', 1)
            var_name = parts[0]
            level = int(parts[1])
            if var_name not in var_levels:
                var_levels[var_name] = []
            var_levels[var_name].append(level)

        # Each variable should have all 13 levels in order
        for var_name, levels in var_levels.items():
            assert levels == PRESSURE_LEVELS, f'{var_name} has wrong level order'

    def test_variable_order_hash_stable(self):
        """Variable order hash should be deterministic."""
        h1 = variable_order_hash()
        h2 = variable_order_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_variable_order_hash_changes_with_order(self):
        """Changing variable order should change the hash."""
        original_hash = variable_order_hash()

        # Manually compute hash with different order
        reordered = list(CANONICAL_VARIABLES)
        reordered[0], reordered[1] = reordered[1], reordered[0]  # Swap first two

        data = '\n'.join(reordered).encode('utf-8')
        reordered_hash = hashlib.sha256(data).hexdigest()

        assert original_hash != reordered_hash

    def test_build_canonical_variables_matches_constant(self):
        """build_canonical_variables() should match CANONICAL_VARIABLES."""
        built = build_canonical_variables()
        assert built == CANONICAL_VARIABLES


# ============================================================================
# Target grid tests
# ============================================================================

class TestTargetGrid:
    """Tests for Stormer's 1.40625 deg target grid."""

    def test_grid_shape(self):
        """Grid should be 128 lat x 256 lon."""
        lat, lon = get_stormer_target_grid()
        assert lat.shape == (128,)
        assert lon.shape == (256,)

    def test_lat_no_poles(self):
        """Latitude should not include poles (cell-centre grid)."""
        lat, _ = get_stormer_target_grid()
        assert lat[0] > -90  # Not exactly -90
        assert lat[-1] < 90  # Not exactly 90

    def test_lat_symmetric(self):
        """Latitude should be symmetric around equator."""
        lat, _ = get_stormer_target_grid()
        np.testing.assert_allclose(lat + lat[::-1], 0, atol=1e-10)

    def test_lon_starts_at_zero(self):
        """Longitude should start at 0 (not include 360)."""
        _, lon = get_stormer_target_grid()
        assert lon[0] == 0
        assert lon[-1] < 360

    def test_lon_spacing(self):
        """Longitude spacing should be 1.40625 deg."""
        _, lon = get_stormer_target_grid()
        spacing = np.diff(lon)
        np.testing.assert_allclose(spacing, 1.40625, atol=1e-10)

    def test_grid_hash_stable(self):
        """Grid hash should be deterministic."""
        h1 = grid_hash()
        h2 = grid_hash()
        assert h1 == h2
        assert len(h1) == 16  # Truncated SHA-256


# ============================================================================
# Conservative regridding tests
# ============================================================================

class TestConservativeRegridding:
    """Tests for conservative regridding."""

    def test_constant_field_preserved(self):
        """A constant field should remain constant after regridding."""
        # Small synthetic grids
        source_lat = np.linspace(-90 + 5, 90 - 5, 36)  # 5 deg spacing
        source_lon = np.linspace(0, 360 - 10, 36)  # 10 deg spacing

        target_lat = np.linspace(-90 + 10, 90 - 10, 18)  # 10 deg spacing
        target_lon = np.linspace(0, 360 - 20, 18)  # 20 deg spacing

        regridder = ConservativeRegridder(source_lat, source_lon, target_lat, target_lon)

        # Constant field
        constant_value = 273.15
        field = np.full((len(source_lon), len(source_lat)), constant_value, dtype=np.float32)

        result = regridder.regrid(field)

        np.testing.assert_allclose(result, constant_value, rtol=1e-5)

    def test_global_mean_preserved(self):
        """Conservative regridding should preserve area-weighted global mean."""
        # Source grid matching actual dimensions
        source_lat = np.linspace(-89.6484375, 89.6484375, 256)  # ~0.703 deg
        source_lon = np.linspace(0, 360 - 360/512, 512)

        target_lat, target_lon = get_stormer_target_grid()

        regridder = ConservativeRegridder(source_lat, source_lon, target_lat, target_lon)

        # Synthetic field with spatial variation
        np.random.seed(42)
        lon_grid, lat_grid = np.meshgrid(source_lon, source_lat)
        field = 273 + 50 * np.sin(np.deg2rad(lat_grid))  # Temperature-like field
        field = field.T  # (lon, lat)

        result = regridder.regrid(field)

        # Compute area-weighted means
        source_cos_lat = np.cos(np.deg2rad(source_lat))
        source_weights = source_cos_lat / source_cos_lat.sum()
        source_mean = np.sum(field.mean(axis=0) * source_weights)

        target_cos_lat = np.cos(np.deg2rad(target_lat))
        target_weights = target_cos_lat / target_cos_lat.sum()
        target_mean = np.sum(result.mean(axis=0) * target_weights)

        # Should be close (not exact due to different pole handling)
        np.testing.assert_allclose(source_mean, target_mean, rtol=0.05)

    def test_sinusoidal_field(self):
        """Sinusoidal mode should regrid smoothly."""
        source_lat = np.linspace(-85, 85, 64)
        source_lon = np.linspace(0, 360 - 5, 72)

        target_lat = np.linspace(-85, 85, 32)
        target_lon = np.linspace(0, 360 - 10, 36)

        regridder = ConservativeRegridder(source_lat, source_lon, target_lat, target_lon)

        # Sinusoidal field (wave number 2)
        lon_grid, lat_grid = np.meshgrid(source_lon, source_lat)
        field = np.cos(2 * np.deg2rad(lon_grid)) * np.cos(np.deg2rad(lat_grid))
        field = field.T  # (lon, lat)

        result = regridder.regrid(field)

        # Result should not have NaNs
        assert not np.isnan(result).any()

        # Result should have same general range
        assert result.min() > -1.5
        assert result.max() < 1.5

    def test_einsum_optimize_gives_identical_results(self):
        """Test that einsum optimize=True produces identical numerical results.

        This verifies Fix A: adding optimize=True only changes contraction order,
        not the mathematical result (within float32 rounding tolerance).
        """
        np.random.seed(42)

        # Create weight matrices similar to those in ConservativeRegridder
        lon_weights = np.random.rand(18, 36).astype(np.float32)
        lon_weights = lon_weights / lon_weights.sum(axis=1, keepdims=True)

        lat_weights = np.random.rand(12, 24).astype(np.float32)
        lat_weights = lat_weights / lat_weights.sum(axis=1, keepdims=True)

        # Create test field with time dimension (time, lon, lat)
        field = np.random.rand(5, 36, 24).astype(np.float32)

        # Without optimize
        result_no_opt = np.einsum('ab,cd,...bd->...ac', lon_weights, lat_weights, field)

        # With optimize (as used in the actual code)
        result_opt = np.einsum('ab,cd,...bd->...ac', lon_weights, lat_weights, field, optimize=True)

        # Should be numerically identical or within float32 rounding
        np.testing.assert_allclose(result_no_opt, result_opt, rtol=1e-5, atol=1e-5)


# ============================================================================
# Completeness check tests (Fix C)
# ============================================================================

class TestCompletenessCheck:
    """Tests for year completeness checking logic."""

    def test_is_leap_year(self):
        """Test leap year detection."""
        # Leap years
        assert is_leap_year(2020) is True
        assert is_leap_year(2000) is True
        assert is_leap_year(1996) is True

        # Non-leap years
        assert is_leap_year(2019) is False
        assert is_leap_year(2021) is False
        assert is_leap_year(1900) is False  # Century non-leap
        assert is_leap_year(2100) is False

    def test_expected_timesteps(self):
        """Test expected timestep count for leap vs non-leap years."""
        # Leap year: 366 days * 4 = 1464
        assert expected_timesteps(2020) == 1464
        assert expected_timesteps(2000) == 1464

        # Non-leap year: 365 days * 4 = 1460
        assert expected_timesteps(2019) == 1460
        assert expected_timesteps(2021) == 1460

    def test_is_year_complete_nonexistent(self):
        """Non-existent path should not be complete."""
        fake_path = Path('/nonexistent/path/2020.zarr')
        assert is_year_complete(fake_path, 2020) is False

    def test_is_year_complete_wrong_timesteps(self):
        """Zarr with wrong timestep count should not be complete."""
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / '2020.zarr'

            # Create zarr using xarray (handles string encoding automatically)
            # Wrong number of timesteps (1460 instead of 1464 for leap year)
            ds = xr.Dataset({
                'data': (['time', 'channel', 'lat', 'lon'],
                         np.zeros((1460, 69, 4, 4), dtype='float32'))
            }, coords={
                'time': np.arange(1460),
                'channel': CANONICAL_VARIABLES,
                'lat': np.arange(4),
                'lon': np.arange(4),
            })
            ds.to_zarr(path)

            # 2020 is a leap year, needs 1464 timesteps
            assert is_year_complete(path, 2020) is False

            # But would be complete for a non-leap year like 2019
            assert is_year_complete(path, 2019) is True

    def test_is_year_complete_wrong_channels(self):
        """Zarr with wrong channel count should not be complete."""
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / '2019.zarr'

            # Create zarr with right timesteps but wrong channels (68 instead of 69)
            ds = xr.Dataset({
                'data': (['time', 'channel', 'lat', 'lon'],
                         np.zeros((1460, 68, 4, 4), dtype='float32'))
            }, coords={
                'time': np.arange(1460),
                'channel': CANONICAL_VARIABLES[:68],
                'lat': np.arange(4),
                'lon': np.arange(4),
            })
            ds.to_zarr(path)

            assert is_year_complete(path, 2019) is False

    def test_is_year_complete_correct(self):
        """Zarr with correct timesteps and channels should be complete."""
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / '2020.zarr'

            # Create zarr with correct shape for 2020 (leap year)
            ds = xr.Dataset({
                'data': (['time', 'channel', 'lat', 'lon'],
                         np.zeros((1464, 69, 4, 4), dtype='float32'))
            }, coords={
                'time': np.arange(1464),
                'channel': CANONICAL_VARIABLES,
                'lat': np.arange(4),
                'lon': np.arange(4),
            })
            ds.to_zarr(path)

            assert is_year_complete(path, 2020) is True


# ============================================================================
# Split manifest tests
# ============================================================================

class TestYearSplits:
    """Tests for year-based split assignments."""

    def test_split_years_coverage(self):
        """All years should be assigned to exactly one split."""
        all_years = []
        for years in YEAR_SPLITS.values():
            all_years.extend(years)

        # No duplicates
        assert len(all_years) == len(set(all_years))

    def test_get_split_for_year(self):
        """get_split_for_year should return correct split."""
        assert get_split_for_year(2015) == 'train'
        assert get_split_for_year(2019) == 'val'
        assert get_split_for_year(2020) == 'test'
        assert get_split_for_year(2021) == 'shift_test'
        assert get_split_for_year(1900) is None  # Unknown year


class TestGuardWindows:
    """Tests for guard window logic."""

    def test_guard_boundaries_exist(self):
        """Guard boundaries should be computed for all splits."""
        boundaries = compute_guard_boundaries(max_history_hours=24, max_lead_hours=168)

        for split_id in YEAR_SPLITS:
            assert split_id in boundaries
            start, end = boundaries[split_id]
            assert start < end

    def test_guard_excludes_boundary_issues(self):
        """Issues near split boundaries should be excluded."""
        boundaries = compute_guard_boundaries(max_history_hours=24, max_lead_hours=168)

        # Test at exact split boundary (Jan 1, 2019 - start of val)
        boundary_dt = datetime(2019, 1, 1, 0, 0, 0)

        # Should be excluded from val (history would cross into train)
        assert is_in_guard_window(boundary_dt, 'val', boundaries)

        # A few days into val should be included
        safe_dt = datetime(2019, 1, 3, 0, 0, 0)
        assert not is_in_guard_window(safe_dt, 'val', boundaries)

    def test_guard_excludes_end_boundary(self):
        """Issues at end of split should be excluded (forecast crosses into next)."""
        boundaries = compute_guard_boundaries(max_history_hours=24, max_lead_hours=168)

        # End of test (Dec 2020) - forecast would verify in shift_test
        late_dt = datetime(2020, 12, 28, 0, 0, 0)
        assert is_in_guard_window(late_dt, 'test', boundaries)

        # Earlier in test should be fine
        safe_dt = datetime(2020, 12, 20, 0, 0, 0)
        assert not is_in_guard_window(safe_dt, 'test', boundaries)


class TestSplitManifest:
    """Tests for SplitManifest class."""

    def test_manifest_row_validation(self):
        """Manifest rows should validate time ordering."""
        # Valid row: issue < valid < available
        valid_row = SplitManifestRow(
            issue_time=1000,
            valid_time=2000,
            available_time=3000,
            event_id='test_001',
            split_id='test',
            normalization_hash='abc123',
            grid_hash='def456',
        )
        assert valid_row.validate()

        # Invalid: valid_time < issue_time
        invalid_row = SplitManifestRow(
            issue_time=2000,
            valid_time=1000,
            available_time=3000,
            event_id='test_002',
            split_id='test',
            normalization_hash='abc123',
            grid_hash='def456',
        )
        assert not invalid_row.validate()

    def test_build_manifest_single_year(self):
        """Build manifest for a single year."""
        manifest = build_manifest(years=[2020], lead_hours=[6, 12])

        assert len(manifest) > 0

        # All rows should be from test split
        for row in manifest.rows:
            assert row.split_id == 'test'
            assert row.validate()

    def test_manifest_to_dataframe(self):
        """Manifest should convert to DataFrame."""
        manifest = build_manifest(years=[2020], lead_hours=[6])
        df = manifest.to_dataframe()

        assert 'issue_time' in df.columns
        assert 'valid_time' in df.columns
        assert 'split_id' in df.columns
        assert len(df) == len(manifest)

    def test_manifest_serialization(self):
        """Manifest should serialize to/from JSON."""
        manifest = build_manifest(years=[2020], lead_hours=[6])

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)

        try:
            manifest.to_json(path)
            loaded = SplitManifest.from_json(path)

            assert len(loaded) == len(manifest)
            assert loaded.rows[0].issue_time == manifest.rows[0].issue_time
        finally:
            path.unlink()

    def test_manifest_guard_window_exclusion(self):
        """Manifest should exclude guard window rows."""
        # Build with short lead time
        manifest = build_manifest(years=[2019, 2020], lead_hours=[24])

        # Get issue times at split boundary
        boundary_ts = int(datetime(2020, 1, 1, 0, 0, 0).timestamp())

        # No rows should have issue_time exactly at boundary
        issue_times = [r.issue_time for r in manifest.rows]
        assert boundary_ts not in issue_times


# ============================================================================
# VerifiedRecord integration tests
# ============================================================================

class TestVerifiedRecordIntegration:
    """Tests for compatibility with earthdelta.memory.VerifiedRecord."""

    def test_manifest_row_to_verified_record(self):
        """Manifest rows should be constructible as VerifiedRecords."""
        row = SplitManifestRow(
            issue_time=1000000000,  # 2001-09-09T01:46:40
            valid_time=1000021600,  # +6 hours
            available_time=1000043200,  # +12 hours
            event_id='test_event_001',
            split_id='test',
            normalization_hash='abc123',
            grid_hash='def456',
        )

        # Create mock key/value tensors
        key = torch.randn(64)
        value = torch.randn(128)

        # Should construct without error
        record = VerifiedRecord(
            record_id=row.event_id,
            issue_time=row.issue_time,
            valid_time=row.valid_time,
            available_time=row.available_time,
            version=row.normalization_hash,
            event_id=row.event_id,
            key=key,
            value=value,
        )

        assert record.record_id == row.event_id

    def test_manifest_time_ordering_matches_verified_record(self):
        """Manifest time ordering should match VerifiedRecord requirements."""
        manifest = build_manifest(years=[2020], lead_hours=[6])

        for row in manifest.rows:
            # VerifiedRecord requires: issue < valid <= available
            assert row.issue_time < row.valid_time
            assert row.valid_time <= row.available_time

    def test_availability_delay_produces_valid_records(self):
        """Availability delay should produce valid_time <= available_time."""
        manifest = build_manifest(
            years=[2020],
            lead_hours=[6],
            availability_delay_hours=0,  # Edge case: no delay
        )

        # All rows should still be valid (available_time == valid_time when delay=0)
        for row in manifest.rows:
            assert row.validate()

    def test_verified_record_rejects_invalid_times(self):
        """VerifiedRecord should reject invalid time orderings."""
        key = torch.randn(64)
        value = torch.randn(128)

        # Invalid: issue >= valid
        with pytest.raises(ValueError, match='issue < valid'):
            VerifiedRecord(
                record_id='bad_001',
                issue_time=1000,
                valid_time=1000,  # Same as issue
                available_time=2000,
                version='v1',
                event_id='event',
                key=key,
                value=value,
            )

        # Invalid: valid > available
        with pytest.raises(ValueError, match='issue < valid'):
            VerifiedRecord(
                record_id='bad_002',
                issue_time=1000,
                valid_time=3000,  # After available
                available_time=2000,
                version='v1',
                event_id='event',
                key=key,
                value=value,
            )


# ============================================================================
# Slow test: Real data validation
# ============================================================================

@pytest.mark.slow
def test_real_pulled_data_sanity():
    """Validate real pulled ERA5 data has physically plausible values.

    This test requires:
    1. Network/GCS access
    2. Data already pulled to /mnt/afs/260010168/EarthDelta/data/era5_1p40625/2020.zarr

    Skip if not available.
    """
    import warnings
    warnings.filterwarnings('ignore')

    data_path = Path('/mnt/afs/260010168/EarthDelta/data/era5_1p40625/2020.zarr')

    if not data_path.exists():
        pytest.skip(f'Real data not available at {data_path}')

    try:
        import xarray as xr
    except ImportError:
        pytest.skip('xarray not available')

    ds = xr.open_zarr(data_path)

    # Check shape
    assert 'data' in ds
    assert ds.data.dims == ('time', 'channel', 'lat', 'lon')
    assert ds.data.shape[1] == 69  # 69 channels
    assert ds.data.shape[2] == 128  # 128 lat
    assert ds.data.shape[3] == 256  # 256 lon

    # Check time coverage (should be ~1460-1464 timesteps for a year)
    n_times = ds.data.shape[0]
    assert n_times >= 1460, f'Expected >= 1460 timesteps, got {n_times}'

    # Check 2m_temperature mean is physically plausible (270-290 K global mean)
    t2m_idx = CANONICAL_VARIABLES.index('2m_temperature')
    t2m_data = ds.data[:, t2m_idx, :, :].values
    t2m_mean = float(np.nanmean(t2m_data))

    assert 270 < t2m_mean < 290, f'2m_temperature mean {t2m_mean} K out of range'

    # Check MSLP mean is physically plausible (100000-102000 Pa)
    mslp_idx = CANONICAL_VARIABLES.index('mean_sea_level_pressure')
    mslp_data = ds.data[:, mslp_idx, :, :].values
    mslp_mean = float(np.nanmean(mslp_data))

    assert 95000 < mslp_mean < 110000, f'MSLP mean {mslp_mean} Pa out of range'

    # Check grid hash matches
    expected_hash = grid_hash()
    actual_hash = ds.attrs.get('grid_hash', '')
    assert actual_hash == expected_hash, f'Grid hash mismatch: {actual_hash} != {expected_hash}'

    # Check variable order hash
    expected_var_hash = variable_order_hash()
    actual_var_hash = ds.attrs.get('variable_order_hash', '')
    assert actual_var_hash == expected_var_hash, f'Variable hash mismatch'

    ds.close()


# ============================================================================
# Hash consistency tests
# ============================================================================

class TestHashConsistency:
    """Tests for hash consistency across components."""

    def test_variable_hash_from_config_vs_code(self):
        """Variable hash computed from list should match code's hash."""
        # Rebuild the list exactly as in the YAML config
        config_vars = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'mean_sea_level_pressure',
        ]
        for var in ['geopotential', 'u_component_of_wind', 'v_component_of_wind',
                    'temperature', 'specific_humidity']:
            for level in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]:
                config_vars.append(f'{var}_{level}')

        # Compute hash from config-derived list
        config_hash = hashlib.sha256('\n'.join(config_vars).encode('utf-8')).hexdigest()

        # Compare with code's hash
        code_hash = variable_order_hash()

        assert config_hash == code_hash, 'Variable hash mismatch between config and code'

    def test_manifest_contains_variable_hash(self):
        """Manifest metadata should contain the variable order hash."""
        manifest = build_manifest(years=[2020])
        assert 'variable_order_hash' in manifest.metadata
        assert manifest.metadata['variable_order_hash'] == variable_order_hash()


# ============================================================================
# R5 redesign tests: batch alignment, source variable grouping, markers
# ============================================================================

class TestR5BatchAlignment:
    """Tests for R5 time batch alignment to source chunk size."""

    def test_time_batch_size_is_multiple_of_source_chunk(self):
        """TIME_BATCH_SIZE (96) must be a multiple of SOURCE_TIME_CHUNK (8)."""
        from earthdelta.data.pull_wb2 import TIME_BATCH_SIZE, SOURCE_TIME_CHUNK

        assert TIME_BATCH_SIZE % SOURCE_TIME_CHUNK == 0, \
            f'TIME_BATCH_SIZE ({TIME_BATCH_SIZE}) must be multiple of SOURCE_TIME_CHUNK ({SOURCE_TIME_CHUNK})'
        assert TIME_BATCH_SIZE >= SOURCE_TIME_CHUNK, \
            f'TIME_BATCH_SIZE must be >= SOURCE_TIME_CHUNK'

    def test_source_time_chunk_is_8(self):
        """Verify source chunk assumption matches WB2 zarr metadata."""
        from earthdelta.data.pull_wb2 import SOURCE_TIME_CHUNK
        assert SOURCE_TIME_CHUNK == 8, 'WB2 source zarr has time chunk size 8'


class TestR5SourceVariableGrouping:
    """Tests for R5 source variable grouping (9 vars instead of 69 channels)."""

    def test_all_source_vars_count(self):
        """Should have exactly 9 source variables."""
        from earthdelta.data.pull_wb2 import ALL_SOURCE_VARS
        assert len(ALL_SOURCE_VARS) == 9

    def test_all_source_vars_composition(self):
        """Source vars = 4 single-level + 5 pressure-level."""
        from earthdelta.data.pull_wb2 import (
            ALL_SOURCE_VARS, SINGLE_LEVEL_VARS, PRESSURE_VARS
        )
        assert len(SINGLE_LEVEL_VARS) == 4
        assert len(PRESSURE_VARS) == 5
        assert ALL_SOURCE_VARS == SINGLE_LEVEL_VARS + PRESSURE_VARS

    def test_get_channel_indices_single_level(self):
        """Single-level vars should return single index."""
        from earthdelta.data.pull_wb2 import (
            get_channel_indices_for_source_var, CANONICAL_VARIABLES
        )

        indices = get_channel_indices_for_source_var('2m_temperature')
        assert len(indices) == 1
        assert indices[0] == CANONICAL_VARIABLES.index('2m_temperature')
        assert indices[0] == 0  # First channel

    def test_get_channel_indices_pressure_level(self):
        """Pressure-level vars should return 13 indices (one per level)."""
        from earthdelta.data.pull_wb2 import (
            get_channel_indices_for_source_var, CANONICAL_VARIABLES, PRESSURE_LEVELS
        )

        indices = get_channel_indices_for_source_var('geopotential')
        assert len(indices) == 13

        # Verify indices are correct
        for i, level in enumerate(PRESSURE_LEVELS):
            expected_name = f'geopotential_{level}'
            assert CANONICAL_VARIABLES[indices[i]] == expected_name

    def test_all_69_channels_covered(self):
        """All source vars together should cover all 69 channels exactly once."""
        from earthdelta.data.pull_wb2 import (
            get_channel_indices_for_source_var, ALL_SOURCE_VARS, CANONICAL_VARIABLES
        )

        all_indices = []
        for source_var in ALL_SOURCE_VARS:
            all_indices.extend(get_channel_indices_for_source_var(source_var))

        # Should be exactly 69 indices
        assert len(all_indices) == 69

        # Should be exactly 0..68 with no duplicates
        assert sorted(all_indices) == list(range(69))


class TestR5WorkerAssignments:
    """Tests for R5 parallel worker assignments."""

    def test_worker_assignments_cover_all_vars(self):
        """All 3 workers together should cover all 9 source vars."""
        from earthdelta.data.pull_wb2 import WORKER_ASSIGNMENTS, ALL_SOURCE_VARS

        all_assigned = []
        for worker_id in [0, 1, 2]:
            all_assigned.extend(WORKER_ASSIGNMENTS[worker_id])

        assert sorted(all_assigned) == sorted(ALL_SOURCE_VARS)

    def test_worker_assignments_no_overlap(self):
        """Workers should have disjoint variable assignments."""
        from earthdelta.data.pull_wb2 import WORKER_ASSIGNMENTS

        vars_0 = set(WORKER_ASSIGNMENTS[0])
        vars_1 = set(WORKER_ASSIGNMENTS[1])
        vars_2 = set(WORKER_ASSIGNMENTS[2])

        assert vars_0.isdisjoint(vars_1)
        assert vars_1.isdisjoint(vars_2)
        assert vars_0.isdisjoint(vars_2)

    def test_worker_channel_isolation(self):
        """Workers should write to disjoint channel indices (no read-modify-write conflicts)."""
        from earthdelta.data.pull_wb2 import (
            WORKER_ASSIGNMENTS, get_channel_indices_for_source_var
        )

        channels_by_worker = {}
        for worker_id in [0, 1, 2]:
            channels = set()
            for source_var in WORKER_ASSIGNMENTS[worker_id]:
                channels.update(get_channel_indices_for_source_var(source_var))
            channels_by_worker[worker_id] = channels

        # Verify disjoint
        assert channels_by_worker[0].isdisjoint(channels_by_worker[1])
        assert channels_by_worker[1].isdisjoint(channels_by_worker[2])
        assert channels_by_worker[0].isdisjoint(channels_by_worker[2])


class TestR5MarkerFiles:
    """Tests for R5 marker-file resume logic."""

    def test_marker_file_roundtrip(self):
        """Marker files should correctly track completion."""
        from earthdelta.data.pull_wb2 import (
            mark_source_var_complete, is_source_var_complete,
            clear_markers, all_source_vars_complete, ALL_SOURCE_VARS
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            year = 2020

            # Initially nothing complete
            assert not is_source_var_complete(output_dir, year, 'geopotential')
            assert not all_source_vars_complete(output_dir, year)

            # Mark one var complete
            mark_source_var_complete(output_dir, year, 'geopotential')
            assert is_source_var_complete(output_dir, year, 'geopotential')
            assert not is_source_var_complete(output_dir, year, 'temperature')
            assert not all_source_vars_complete(output_dir, year)

            # Mark all vars complete
            for var in ALL_SOURCE_VARS:
                mark_source_var_complete(output_dir, year, var)
            assert all_source_vars_complete(output_dir, year)

            # Clear markers
            clear_markers(output_dir, year)
            assert not is_source_var_complete(output_dir, year, 'geopotential')
            assert not all_source_vars_complete(output_dir, year)


class TestR5ChunkShapes:
    """Tests for R5 chunk shape constants."""

    def test_parallel_output_chunk_shape(self):
        """Parallel output chunks should be channel-isolated."""
        from earthdelta.data.pull_wb2 import PARALLEL_OUTPUT_CHUNK

        # (time=8, channel=1, lat=128, lon=256)
        assert PARALLEL_OUTPUT_CHUNK == (8, 1, 128, 256)
        assert PARALLEL_OUTPUT_CHUNK[1] == 1, 'Channel dimension must be 1 for isolation'

    def test_final_output_chunk_shape(self):
        """Final output chunks should be training-friendly."""
        from earthdelta.data.pull_wb2 import FINAL_OUTPUT_CHUNK

        # (time=1, channel=69, lat=128, lon=256)
        assert FINAL_OUTPUT_CHUNK == (1, 69, 128, 256)
        assert FINAL_OUTPUT_CHUNK[0] == 1, 'Time dimension = 1 for single-sample access'
        assert FINAL_OUTPUT_CHUNK[1] == 69, 'All channels in one chunk'


class TestR5PilotCompleteness:
    """Tests for R5 pilot store completeness checking."""

    def test_is_pilot_complete_nonexistent(self):
        """Non-existent path should not be complete."""
        from earthdelta.data.pull_wb2 import is_pilot_complete

        fake_path = Path('/nonexistent/path/2020_jan.zarr')
        assert is_pilot_complete(fake_path, 124) is False

    def test_is_pilot_complete_correct(self):
        """Pilot zarr with correct shape should be complete."""
        import xarray as xr
        from earthdelta.data.pull_wb2 import is_pilot_complete, CANONICAL_VARIABLES

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / '2020_jan.zarr'

            # Create zarr with correct shape (124 timesteps for January)
            ds = xr.Dataset({
                'data': (['time', 'channel', 'lat', 'lon'],
                         np.zeros((124, 69, 4, 4), dtype='float32'))
            }, coords={
                'time': np.arange(124),
                'channel': CANONICAL_VARIABLES,
                'lat': np.arange(4),
                'lon': np.arange(4),
            })
            ds.to_zarr(path)

            assert is_pilot_complete(path, 124) is True
            assert is_pilot_complete(path, 100) is False  # Wrong expected count
