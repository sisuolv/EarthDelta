"""Split manifest generation with guard windows for train/val/test splits.

The split manifest defines which timesteps belong to which split (train/val/test)
and enforces guard windows to prevent data leakage across split boundaries.

Time contract columns (from research_spec_v6.yaml):
- issue_time: When the forecast was issued (analysis time)
- valid_time: When the forecast is valid (verification time)
- available_time: When verification data becomes available (issue_time + delay)
- event_id: Unique identifier for the forecast event
- split_id: train/val/test/shift_test
- normalization_hash: Hash of normalization constants used
- grid_hash: Hash of grid coordinates
- availability_source: How the availability_time was determined
    - reanalysis_retrospective: Data from reanalysis (ERA5), available with ~5 day delay
    - observed_first_seen: Real-time operational NWP analysis availability
    - scenario: Synthetic/scenario-based assumption

Guard windows:
- Exclude issue times near split boundaries whose history/lead-time window
  would cross into an adjacent split.
- Guard width = max_history_hours + max_lead_hours (configurable)

UTC time convention:
- All datetime construction uses timezone-aware UTC (datetime.timezone.utc)
- All .timestamp() calls use timezone-aware datetimes to ensure correctness
  regardless of host timezone settings
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd

from earthdelta.data.pull_wb2 import variable_order_hash, grid_hash


class AvailabilitySource(Enum):
    """Source of data availability time determination.

    This makes explicit whether a data point's availability time reflects:
    - reanalysis_retrospective: ERA5-style reanalysis data with publication delay
    - observed_first_seen: Real-time operational NWP products
    - scenario: Synthetic timing for what-if analysis
    """
    REANALYSIS_RETROSPECTIVE = "reanalysis_retrospective"
    OBSERVED_FIRST_SEEN = "observed_first_seen"
    SCENARIO = "scenario"


# ============================================================================
# Split definitions from research_spec_v6.yaml
# ============================================================================

YEAR_SPLITS: Dict[str, List[int]] = {
    'train': [2015, 2016, 2017, 2018],
    'val': [2019],
    'test': [2020],
    'shift_test': [2021, 2022],
}


def get_split_for_year(year: int) -> Optional[str]:
    """Get the split ID for a given year."""
    for split_id, years in YEAR_SPLITS.items():
        if year in years:
            return split_id
    return None


# ============================================================================
# Availability time modeling
# ============================================================================

# ERA5 reanalysis has ~5 day publication delay, but for research we use a more
# conservative delay to model real-time NWP verification analysis lag.
# This is the delay between valid_time and when verification is available.
DEFAULT_AVAILABILITY_DELAY_HOURS = 6  # Realistic for NWP analysis products


def compute_available_time(valid_time: int, delay_hours: int = DEFAULT_AVAILABILITY_DELAY_HOURS) -> int:
    """Compute available_time from valid_time.

    Args:
        valid_time: UTC timestamp (seconds) when the forecast is valid
        delay_hours: Hours of delay before verification data is available

    Returns:
        available_time: UTC timestamp (seconds) when data becomes available
    """
    return valid_time + delay_hours * 3600


# ============================================================================
# Guard window logic
# ============================================================================

def compute_guard_boundaries(
    max_history_hours: int = 24,
    max_lead_hours: int = 168,
) -> Dict[str, Tuple[datetime, datetime]]:
    """Compute valid datetime boundaries for each split with guard windows.

    The guard window ensures that:
    - A forecast issued at the boundary doesn't use history from the adjacent split
    - A forecast issued at the boundary doesn't verify into the adjacent split

    All datetime objects are timezone-aware (UTC) to ensure correct timestamp
    computation regardless of host timezone.

    Args:
        max_history_hours: Maximum history window used by the model
        max_lead_hours: Maximum forecast lead time

    Returns:
        Dict mapping split_id to (start_datetime, end_datetime) boundaries
        where issue_time must be within these bounds to be included.
    """
    guard_hours = max_history_hours + max_lead_hours

    boundaries = {}

    # Get year ranges for each split
    all_years = sorted(set(y for years in YEAR_SPLITS.values() for y in years))

    for split_id, years in YEAR_SPLITS.items():
        min_year = min(years)
        max_year = max(years)

        # Start: beginning of first year + guard (to not use history from prev split)
        # Use timezone-aware UTC datetime to ensure correct .timestamp() behavior
        start_dt = datetime(min_year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        # Check if there's a previous split
        years_before = [y for y in all_years if y < min_year]
        if years_before:
            # Need guard at start
            start_dt = start_dt + timedelta(hours=max_history_hours)

        # End: end of last year - guard (to not verify into next split)
        # Use timezone-aware UTC datetime
        end_dt = datetime(max_year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # Start of next year

        # Check if there's a next split
        years_after = [y for y in all_years if y > max_year]
        if years_after:
            # Need guard at end
            end_dt = end_dt - timedelta(hours=max_lead_hours)

        boundaries[split_id] = (start_dt, end_dt)

    return boundaries


def is_in_guard_window(
    issue_dt: datetime,
    split_id: str,
    boundaries: Dict[str, Tuple[datetime, datetime]],
) -> bool:
    """Check if an issue time is in the guard window (should be excluded).

    Args:
        issue_dt: Issue datetime
        split_id: The split this issue time nominally belongs to (by year)
        boundaries: Split boundaries from compute_guard_boundaries

    Returns:
        True if this issue time should be EXCLUDED (in guard window)
    """
    if split_id not in boundaries:
        return True  # Unknown split, exclude

    start_dt, end_dt = boundaries[split_id]
    return issue_dt < start_dt or issue_dt >= end_dt


# ============================================================================
# Split manifest
# ============================================================================

@dataclass
class SplitManifestRow:
    """A single row in the split manifest."""
    issue_time: int  # UTC timestamp (seconds)
    valid_time: int  # UTC timestamp (seconds)
    available_time: int  # UTC timestamp (seconds)
    event_id: str
    split_id: str
    normalization_hash: str
    grid_hash: str
    availability_source: str  # One of AvailabilitySource values

    def validate(self) -> bool:
        """Validate time ordering: issue_time < valid_time <= available_time."""
        if not (self.issue_time < self.valid_time <= self.available_time):
            return False
        # Validate availability_source
        valid_sources = {s.value for s in AvailabilitySource}
        if self.availability_source not in valid_sources:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'issue_time': self.issue_time,
            'valid_time': self.valid_time,
            'available_time': self.available_time,
            'event_id': self.event_id,
            'split_id': self.split_id,
            'normalization_hash': self.normalization_hash,
            'grid_hash': self.grid_hash,
            'availability_source': self.availability_source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SplitManifestRow':
        # Handle backward compatibility - default to reanalysis_retrospective if missing
        if 'availability_source' not in d:
            d = dict(d)
            d['availability_source'] = AvailabilitySource.REANALYSIS_RETROSPECTIVE.value
        return cls(**d)


@dataclass
class SplitManifest:
    """Complete split manifest with metadata."""
    rows: List[SplitManifestRow] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.rows])

    def to_json(self, path: Path) -> None:
        """Save manifest to JSON file."""
        data = {
            'metadata': self.metadata,
            'rows': [r.to_dict() for r in self.rows],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def to_parquet(self, path: Path) -> None:
        """Save manifest to Parquet file."""
        df = self.to_dataframe()
        df.to_parquet(path, index=False)

    @classmethod
    def from_json(cls, path: Path) -> 'SplitManifest':
        """Load manifest from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        rows = [SplitManifestRow.from_dict(r) for r in data['rows']]
        return cls(rows=rows, metadata=data.get('metadata', {}))

    @classmethod
    def from_parquet(cls, path: Path) -> 'SplitManifest':
        """Load manifest from Parquet file."""
        df = pd.read_parquet(path)
        rows = [SplitManifestRow.from_dict(row) for _, row in df.iterrows()]
        return cls(rows=rows)

    def filter_split(self, split_id: str) -> 'SplitManifest':
        """Return a new manifest with only rows from the specified split."""
        filtered = [r for r in self.rows if r.split_id == split_id]
        return SplitManifest(rows=filtered, metadata=self.metadata)

    def validate_all(self) -> Tuple[int, int]:
        """Validate all rows. Returns (valid_count, invalid_count)."""
        valid = sum(1 for r in self.rows if r.validate())
        return valid, len(self.rows) - valid

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        df = self.to_dataframe()
        return {
            'total_rows': len(df),
            'splits': df['split_id'].value_counts().to_dict(),
            'time_range': {
                'issue_time_min': int(df['issue_time'].min()),
                'issue_time_max': int(df['issue_time'].max()),
            },
        }


# ============================================================================
# Build manifest
# ============================================================================

def build_manifest(
    years: List[int],
    lead_hours: List[int] = [6, 12, 24, 48, 72, 120, 168],
    time_step_hours: int = 6,
    max_history_hours: int = 24,
    availability_delay_hours: int = DEFAULT_AVAILABILITY_DELAY_HOURS,
    normalization_hash: Optional[str] = None,
    availability_source: str = AvailabilitySource.REANALYSIS_RETROSPECTIVE.value,
) -> SplitManifest:
    """Build a split manifest for the given years.

    All datetime operations use timezone-aware UTC to ensure correct timestamp
    computation regardless of host timezone settings.

    Args:
        years: Years to include
        lead_hours: Forecast lead times to include
        time_step_hours: Time step between issue times (6 for ERA5)
        max_history_hours: Maximum history window (for guard computation)
        availability_delay_hours: Hours between valid_time and available_time
        normalization_hash: Hash of normalization constants (placeholder if None)
        availability_source: How the availability_time was determined.
            Default is 'reanalysis_retrospective' for ERA5-style data.
            Note: 6-hourly ERA5 reanalysis data is retrospective (published with
            ~5 day delay), NOT real-time 'observed_first_seen'. Use 'scenario'
            for synthetic/what-if timing assumptions.

    Returns:
        SplitManifest with all valid issue/valid time combinations
    """
    # Validate availability_source
    valid_sources = {s.value for s in AvailabilitySource}
    if availability_source not in valid_sources:
        raise ValueError(
            f"availability_source must be one of {valid_sources}, got {availability_source}"
        )

    if normalization_hash is None:
        normalization_hash = 'placeholder_1979_2018'

    g_hash = grid_hash()
    var_hash = variable_order_hash()

    # Compute guard boundaries
    max_lead = max(lead_hours)
    boundaries = compute_guard_boundaries(max_history_hours, max_lead)

    rows = []
    excluded_guard = 0
    excluded_no_split = 0

    for year in years:
        split_id = get_split_for_year(year)
        if split_id is None:
            excluded_no_split += 365 * 4  # Approximate
            continue

        # Generate all issue times for this year
        # Use timezone-aware UTC datetime for correct .timestamp() behavior
        start_dt = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        current_dt = start_dt
        while current_dt < end_dt:
            # Check guard window
            if is_in_guard_window(current_dt, split_id, boundaries):
                excluded_guard += 1
                current_dt += timedelta(hours=time_step_hours)
                continue

            # .timestamp() on a timezone-aware datetime gives correct UTC seconds
            issue_time = int(current_dt.timestamp())

            # Generate valid times for each lead hour
            for lead in lead_hours:
                valid_dt = current_dt + timedelta(hours=lead)
                valid_time = int(valid_dt.timestamp())
                available_time = compute_available_time(valid_time, availability_delay_hours)

                # Create event_id: unique identifier
                event_id = f'{current_dt.strftime("%Y%m%d%H")}_L{lead:03d}'

                row = SplitManifestRow(
                    issue_time=issue_time,
                    valid_time=valid_time,
                    available_time=available_time,
                    event_id=event_id,
                    split_id=split_id,
                    normalization_hash=normalization_hash,
                    grid_hash=g_hash,
                    availability_source=availability_source,
                )

                if row.validate():
                    rows.append(row)

            current_dt += timedelta(hours=time_step_hours)

    metadata = {
        'years': years,
        'lead_hours': lead_hours,
        'time_step_hours': time_step_hours,
        'max_history_hours': max_history_hours,
        'availability_delay_hours': availability_delay_hours,
        'availability_source': availability_source,
        'excluded_guard_window': excluded_guard,
        'excluded_no_split': excluded_no_split,
        'variable_order_hash': var_hash,
        'grid_hash': g_hash,
        'normalization_hash': normalization_hash,
        # Use timezone-aware UTC datetime for consistent ISO format
        'created': datetime.now(timezone.utc).isoformat(),
    }

    return SplitManifest(rows=rows, metadata=metadata)


def main():
    """CLI for building split manifests."""
    import argparse

    parser = argparse.ArgumentParser(description='Build split manifest for ERA5 data')
    parser.add_argument('--years', type=int, nargs='+', default=[2020],
                        help='Years to include')
    parser.add_argument('--output', type=str, default='splits.json',
                        help='Output file path (json or parquet)')
    parser.add_argument('--format', choices=['json', 'parquet'], default='json',
                        help='Output format')

    args = parser.parse_args()

    manifest = build_manifest(years=args.years)

    output_path = Path(args.output)
    if args.format == 'json':
        manifest.to_json(output_path)
    else:
        manifest.to_parquet(output_path)

    print(f'Saved manifest to {output_path}')
    print(f'Stats: {manifest.get_stats()}')

    valid, invalid = manifest.validate_all()
    print(f'Validation: {valid} valid, {invalid} invalid')


if __name__ == '__main__':
    main()
