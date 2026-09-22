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

WHAT THE MANIFEST IS, AND WHAT IT IS NOT
----------------------------------------
`build_manifest` is calendar arithmetic. It enumerates the issue/valid/available
times a schedule WOULD have, from year ranges and lead hours, and it is a
legitimate scheduling table in exactly that sense. It is not, and must never be
read as, evidence that a trainable sample exists: no row it emits has ever been
joined against data on disk, so a manifest over a year whose store was never
filled looks identical to a manifest over a year that was.

`admit_real_sample` is the join the manifest does not do. It opens the real
store(s), resolves the row's history, issue and target timestamps against the
real time coordinate, and admits the row only if all three endpoints actually
exist there. It also stamps the two identifiers a downstream training run needs
and the manifest has no way to produce:

  * `issue_id` -- deterministic from (issue_time, split_id), so every lead of
    one analysis time shares it and shard partitioning cannot split an analysis
    state across shards;
  * `process_group_id` -- independently generated per admission run, so two
    processes that admit overlapping issue times stay distinguishable in the
    record afterwards.

In formal mode it additionally refuses a placeholder `normalization_hash`: a row
normalized by a stand-in constant is not a row anything may train on.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd

from earthdelta.data.pull_wb2 import variable_order_hash, grid_hash
from earthdelta.pilot_contract import (
    DATA_ROLE_VALUES,
    DEFAULT_SIGMA_BOUND,
    DataRole,
    PilotContractViolation,
    assert_valid_data_role,
    validate_slice_index,
    verify_content_subset,
)


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


# ============================================================================
# Placeholder normalization hashes
# ============================================================================

#: The sentinel `build_manifest` has always substituted when no normalization
#: hash was supplied. It names no constants; it is a shape for a value.
PLACEHOLDER_NORMALIZATION_HASH = 'placeholder_1979_2018'

#: Substrings that mark a hash as a stand-in rather than a real digest. Kept as
#: substrings on purpose: the failure being prevented is a stand-in reaching a
#: training run, and stand-ins are written by hand under many spellings.
PLACEHOLDER_HASH_MARKERS: Tuple[str, ...] = (
    'placeholder', 'todo', 'tbd', 'dummy', 'fake', 'stub', 'example',
    'unknown', 'none', 'null', 'xxx', 'changeme', 'default',
)


def is_placeholder_normalization_hash(value: Optional[str]) -> bool:
    """Whether a normalization hash is a stand-in rather than a real digest.

    Missing, empty and whitespace-only values count as placeholders: an absent
    hash is the strongest possible stand-in.
    """
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in PLACEHOLDER_HASH_MARKERS)


# ============================================================================
# Identity: issue_id (deterministic) and process_group_id (independent)
# ============================================================================

#: Bumped if the issue_id derivation ever changes, so old ids are never
#: silently reinterpreted under a new rule.
ISSUE_ID_SCHEMA = 'ed-issue-id/1'
PROCESS_GROUP_ID_SCHEMA = 'ed-process-group-id/1'


def compute_issue_id(issue_time: int, split_id: str) -> str:
    """Deterministic id for one analysis time within one split.

    Derived from (issue_time, split_id) and nothing else, so:

      * every lead of the same analysis time maps to the SAME issue_id, which is
        what makes it usable as a shard key -- sharding on it cannot place two
        leads of one analysis state into different shards and call them
        independent;
      * two different splits never collide on it even at the same timestamp;
      * it is reproducible across processes and runs, unlike
        `new_process_group_id`, which is deliberately not.
    """
    payload = f'{ISSUE_ID_SCHEMA}|{int(issue_time)}|{split_id}'
    return 'iss_' + hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def new_process_group_id(label: Optional[str] = None) -> str:
    """A fresh id for ONE admission run, independent of any issue_id.

    Two runs that admit exactly the same issue times must remain
    distinguishable in the record afterwards, so this deliberately mixes in
    entropy that is not a function of the data: a uuid4, the pid, the host and
    a nanosecond clock. Calling it twice in the same process returns two
    different ids; that is the point, not a defect.
    """
    payload = '|'.join([
        PROCESS_GROUP_ID_SCHEMA,
        str(label or ''),
        uuid.uuid4().hex,
        str(os.getpid()),
        socket.gethostname(),
        str(_time.time_ns()),
    ])
    return 'pg_' + hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


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
    """A single row in the split manifest.

    A row is a SCHEDULING entry. It says when a forecast would be issued, when
    it would be valid and when verification would be available. It does not say
    that data exists at any of those times; `admit_real_sample` is what says
    that.
    """
    issue_time: int  # UTC timestamp (seconds)
    valid_time: int  # UTC timestamp (seconds)
    available_time: int  # UTC timestamp (seconds)
    event_id: str
    split_id: str
    normalization_hash: str
    grid_hash: str
    availability_source: str  # One of AvailabilitySource values
    #: Deterministic id for (issue_time, split_id); shared by every lead of one
    #: analysis time. Filled by `build_manifest`; None on legacy rows.
    issue_id: Optional[str] = None
    #: Frozen role this row was admitted for, when one has been assigned. The
    #: calendar itself assigns no role -- roles are an admission decision.
    data_role: Optional[str] = None

    def validate(self) -> bool:
        """Validate time ordering: issue_time < valid_time <= available_time."""
        if not (self.issue_time < self.valid_time <= self.available_time):
            return False
        # Validate availability_source
        valid_sources = {s.value for s in AvailabilitySource}
        if self.availability_source not in valid_sources:
            return False
        if self.data_role is not None and self.data_role not in DATA_ROLE_VALUES:
            return False
        return True

    def resolved_issue_id(self) -> str:
        """This row's issue_id, computing it if the row predates the field."""
        return self.issue_id or compute_issue_id(self.issue_time, self.split_id)

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
            'issue_id': self.issue_id,
            'data_role': self.data_role,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SplitManifestRow':
        # Handle backward compatibility - default to reanalysis_retrospective if missing
        d = dict(d)
        if 'availability_source' not in d:
            d['availability_source'] = AvailabilitySource.REANALYSIS_RETROSPECTIVE.value
        # A parquet round-trip turns a missing optional string into NaN, which
        # is not None and would fail validation as an unknown role.
        for optional in ('issue_id', 'data_role'):
            value = d.get(optional)
            if value is not None and not isinstance(value, str):
                d[optional] = None if pd.isna(value) else str(value)
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
    formal: bool = False,
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
        formal: Build a manifest intended for real consumption. A formal
            manifest may not carry a placeholder normalization_hash; the
            substitution below is a development convenience and is refused
            here rather than being allowed to travel downstream.

    Returns:
        SplitManifest with all valid issue/valid time combinations

    Raises:
        ValueError: On an unknown availability_source, or -- in formal mode --
            on a missing or placeholder normalization_hash.
    """
    # Validate availability_source
    valid_sources = {s.value for s in AvailabilitySource}
    if availability_source not in valid_sources:
        raise ValueError(
            f"availability_source must be one of {valid_sources}, got {availability_source}"
        )

    if formal and is_placeholder_normalization_hash(normalization_hash):
        raise ValueError(
            f"formal=True requires a real normalization_hash, got "
            f"{normalization_hash!r}. A formal manifest names the constants its "
            "rows were normalized with; a placeholder names nothing and would "
            "make every downstream identity check vacuous."
        )

    if normalization_hash is None:
        normalization_hash = PLACEHOLDER_NORMALIZATION_HASH

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
                    issue_id=compute_issue_id(issue_time, split_id),
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


# ============================================================================
# Real-sample admission: join the calendar against data that exists
# ============================================================================

class SampleAdmissionError(PilotContractViolation):
    """A manifest row could not be admitted against real data on disk.

    Subclasses `PilotContractViolation` so a caller can catch either kind of
    admission failure uniformly and still branch on `.code`.
    """


@dataclass
class AdmissionResult:
    """One manifest row, joined against real data and stamped for training."""

    admitted: bool
    event_id: str
    issue_id: str
    process_group_id: str
    split_id: str
    issue_time: int
    valid_time: int
    available_time: int
    lead_hours: float
    normalization_hash: str
    grid_hash: str
    formal: bool
    data_role: Optional[str] = None
    interval_hours: int = 6
    history_steps: int = 1
    history_time: int = 0
    history_store: str = ''
    history_index: int = -1
    issue_store: str = ''
    issue_index: int = -1
    target_store: str = ''
    target_index: int = -1
    n_timesteps: int = 0
    checks: List[str] = field(default_factory=list)
    content_certificate: Optional[Dict[str, Any]] = None
    admitted_at: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'admitted': self.admitted,
            'event_id': self.event_id,
            'issue_id': self.issue_id,
            'process_group_id': self.process_group_id,
            'split_id': self.split_id,
            'data_role': self.data_role,
            'issue_time': self.issue_time,
            'valid_time': self.valid_time,
            'available_time': self.available_time,
            'lead_hours': self.lead_hours,
            'normalization_hash': self.normalization_hash,
            'grid_hash': self.grid_hash,
            'formal': self.formal,
            'interval_hours': self.interval_hours,
            'history_steps': self.history_steps,
            'history_time': self.history_time,
            'history_store': self.history_store,
            'history_index': self.history_index,
            'issue_store': self.issue_store,
            'issue_index': self.issue_index,
            'target_store': self.target_store,
            'target_index': self.target_index,
            'n_timesteps': self.n_timesteps,
            'checks': list(self.checks),
            'content_certificate': self.content_certificate,
            'admitted_at': self.admitted_at,
        }


@dataclass
class AdmissionRunReport:
    """Every row one admission run looked at, and what happened to each."""

    process_group_id: str
    data_root: str
    formal: bool
    data_role: Optional[str]
    requested: int = 0
    admitted: List[AdmissionResult] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    started_at: str = ''
    finished_at: str = ''

    @property
    def passed(self) -> bool:
        """True only if every requested row was admitted against real data."""
        return self.requested > 0 and not self.rejected

    def rejection_codes(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self.rejected:
            counts[entry['code']] = counts.get(entry['code'], 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            'process_group_id': self.process_group_id,
            'data_root': self.data_root,
            'formal': self.formal,
            'data_role': self.data_role,
            'requested': self.requested,
            'n_admitted': len(self.admitted),
            'n_rejected': len(self.rejected),
            'passed': self.passed,
            'rejection_codes': self.rejection_codes(),
            'admitted': [result.to_dict() for result in self.admitted],
            'rejected': list(self.rejected),
            'started_at': self.started_at,
            'finished_at': self.finished_at,
        }


@dataclass
class _StoreTimeIndex:
    """The real time axis of one store, read once and reused."""

    path: Path
    seconds: np.ndarray
    n_timesteps: int
    n_channels: int
    attrs: Dict[str, Any]
    monotonic: bool
    raw_times: Any = None

    def locate(self, timestamp: int) -> int:
        """Index of an EXACT timestamp match, or -1 if the store lacks it."""
        target = np.int64(timestamp)
        if self.n_timesteps == 0:
            return -1
        if self.monotonic:
            position = int(np.searchsorted(self.seconds, target))
            if position < self.n_timesteps and self.seconds[position] == target:
                return position
            return -1
        matches = np.flatnonzero(self.seconds == target)
        return int(matches[0]) if matches.size else -1


def default_store_for_year(data_root: Path, year: int) -> Path:
    """`<data_root>/<year>.zarr` -- the layout the pull writes.

    Deliberately does NOT fall back to `<year>_intermediate.zarr`: the
    intermediate store is a different artifact written under a different chunk
    layout, and silently admitting it would mean the sample a run trained on is
    not the sample the record says it trained on.
    """
    return Path(data_root) / f'{year}.zarr'


def _utc_year(timestamp: int) -> int:
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).year


def _iso(timestamp: int) -> str:
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()


def _to_epoch_seconds(values: Any) -> np.ndarray:
    """Convert a store's time coordinate to int64 UTC seconds."""
    array = np.asarray(values)
    if array.dtype.kind == 'M':
        return array.astype('datetime64[s]').astype(np.int64)
    if array.dtype.kind in 'iu':
        return array.astype(np.int64)
    if array.dtype.kind == 'f':
        if not np.all(np.isfinite(array)):
            raise SampleAdmissionError(
                'ADMISSION_TIME_COORD_NONFINITE',
                'Store time coordinate contains non-finite values.',
                {},
            )
        return array.astype(np.int64)
    try:
        return (pd.to_datetime(array, utc=True).view('int64') // 1_000_000_000).astype(np.int64)
    except Exception as exc:  # noqa: BLE001 - surfaced as an admission failure
        raise SampleAdmissionError(
            'ADMISSION_TIME_COORD_UNUSABLE',
            f'Store time coordinate of dtype {array.dtype} could not be read as '
            f'UTC timestamps: {type(exc).__name__}: {exc}',
            {'dtype': str(array.dtype)},
        ) from exc


def load_store_time_index(
    store_path: Path,
    cache: Optional[Dict[str, '_StoreTimeIndex']] = None,
) -> '_StoreTimeIndex':
    """Read one store's real time axis, channel count and identity attrs.

    Read-only, and the dataset is closed again immediately: this resolves where
    a timestamp lives, it does not load any field values.
    """
    store_path = Path(store_path)
    key = str(store_path)
    if cache is not None and key in cache:
        return cache[key]

    if not store_path.exists():
        raise SampleAdmissionError(
            'ADMISSION_STORE_MISSING',
            f'No store at {store_path}. The manifest row names a time whose '
            'data was never pulled; a calendar entry is not a sample.',
            {'store_path': key},
        )
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - xarray is a hard dependency
        raise SampleAdmissionError(
            'ADMISSION_XARRAY_UNAVAILABLE',
            f'xarray is required to join a manifest row against real data: {exc}',
            {'store_path': key},
        ) from exc

    try:
        dataset = xr.open_zarr(key)
    except Exception as exc:  # noqa: BLE001 - surfaced as an admission failure
        raise SampleAdmissionError(
            'ADMISSION_STORE_UNREADABLE',
            f'Could not open store {store_path}: {type(exc).__name__}: {exc}',
            {'store_path': key},
        ) from exc

    try:
        raw_times = np.asarray(dataset['time'].values)
        seconds = _to_epoch_seconds(raw_times)
        n_timesteps = int(dataset.sizes.get('time', 0))
        n_channels = int(dataset.sizes.get('channel', 0))
        attrs = dict(dataset.attrs)
    finally:
        try:
            dataset.close()
        except Exception:  # noqa: BLE001 - closing must not mask a failure
            pass

    index = _StoreTimeIndex(
        path=store_path,
        seconds=seconds,
        n_timesteps=n_timesteps,
        n_channels=n_channels,
        attrs=attrs,
        monotonic=bool(seconds.size < 2 or np.all(np.diff(seconds) > 0)),
        raw_times=raw_times,
    )
    if cache is not None:
        cache[key] = index
    return index


def admit_real_sample(
    row: SplitManifestRow,
    data_root: Path,
    *,
    formal: bool = False,
    data_role: Optional[Any] = None,
    history_steps: int = 1,
    interval_hours: int = 6,
    expected_channels: Optional[int] = 69,
    process_group_id: Optional[str] = None,
    store_for_year: Optional[Callable[[Path, int], Path]] = None,
    cache: Optional[Dict[str, '_StoreTimeIndex']] = None,
    check_grid_binding: bool = True,
    verify_content: bool = False,
    content_batch_size: int = 4,
    content_sigma_bound: float = DEFAULT_SIGMA_BOUND,
    normalization_dir: Optional[Path] = None,
    require_physical_range: bool = True,
) -> AdmissionResult:
    """Admit a manifest row only if its real endpoints exist on disk.

    The manifest says a sample WOULD span
    `[issue_time - history_steps * interval, issue_time, valid_time]`. This
    resolves each of those three timestamps against the real time coordinate of
    the real store that would hold it -- an exact match, never a nearest-value
    snap -- and refuses the row if any endpoint is absent. Endpoints may land in
    different stores when a lead crosses a year boundary; each is resolved in
    the store that would actually hold it.

    In formal mode it additionally refuses a placeholder `normalization_hash`
    and requires an explicit `data_role`, because a formal admission record
    that cannot say what constants a sample was normalized with, or what the
    sample is allowed to be used for, is not a record.

    Args:
        row: The manifest row to admit.
        data_root: Directory holding the per-year stores.
        formal: Enable the formal-mode refusals described above.
        data_role: Frozen role (`bank_fit` / `policy_dev` / `confirm`).
        history_steps: Steps of history the sample needs before `issue_time`.
        interval_hours: Spacing between consecutive timesteps.
        expected_channels: Channel count each store must carry, or None.
        process_group_id: Id of the admission run; a fresh one is generated
            when omitted.
        store_for_year: `(data_root, year) -> Path` override for the layout.
        cache: Reused `{path: _StoreTimeIndex}` map across many rows.
        check_grid_binding: Compare the row's grid_hash to the store's stamp.
        verify_content: Also content-verify the admitted endpoints and attach
            the certificate to the result.
        content_batch_size: Batch size for that content verification.
        content_sigma_bound: Plausibility band half-width, in official sigmas.
        normalization_dir: Directory of the official normalization constants.
        require_physical_range: Whether the band check must run.

    Returns:
        AdmissionResult with `admitted=True` and every resolved index.

    Raises:
        SampleAdmissionError: On any admission failure. Fail-closed: there is no
            "admitted with warnings" outcome.
    """
    resolver = store_for_year or default_store_for_year
    data_root = Path(data_root)
    process_group = process_group_id or new_process_group_id(label='admit_real_sample')
    role = assert_valid_data_role(data_role if data_role is not None else row.data_role)
    issue_id = row.resolved_issue_id()
    checks: List[str] = []

    if not row.validate():
        raise SampleAdmissionError(
            'ADMISSION_ROW_INVALID',
            f'Row {row.event_id} fails its own validation (time ordering '
            f'{row.issue_time} < {row.valid_time} <= {row.available_time}, '
            f'availability_source={row.availability_source!r}, '
            f'data_role={row.data_role!r}).',
            {'event_id': row.event_id, 'issue_id': issue_id},
        )
    checks.append('row_self_validation')

    if formal:
        if is_placeholder_normalization_hash(row.normalization_hash):
            raise SampleAdmissionError(
                'ADMISSION_PLACEHOLDER_NORMALIZATION_HASH',
                f'Row {row.event_id} carries normalization_hash='
                f'{row.normalization_hash!r}, which is a placeholder. Formal '
                'admission refuses it: a placeholder names no constants, so '
                'nothing downstream can check that the sample was normalized '
                'the way the run assumes it was.',
                {'event_id': row.event_id, 'issue_id': issue_id,
                 'normalization_hash': row.normalization_hash},
            )
        checks.append('normalization_hash_not_placeholder')
        if role is None:
            raise SampleAdmissionError(
                'ADMISSION_DATA_ROLE_REQUIRED',
                f'Row {row.event_id} has no data_role. Formal admission '
                f'requires one of {list(DATA_ROLE_VALUES)}: a sample with no '
                'declared role cannot be kept out of the confirm split later, '
                'because nothing records which split it belonged to.',
                {'event_id': row.event_id, 'issue_id': issue_id,
                 'allowed': list(DATA_ROLE_VALUES)},
            )
        checks.append('data_role_declared')

    if type(history_steps) is not int or history_steps < 0:
        raise SampleAdmissionError(
            'ADMISSION_HISTORY_STEPS_INVALID',
            f'history_steps must be a non-negative integer, got {history_steps!r}.',
            {'history_steps': repr(history_steps)},
        )
    if type(interval_hours) is not int or interval_hours <= 0:
        raise SampleAdmissionError(
            'ADMISSION_INTERVAL_INVALID',
            f'interval_hours must be a positive integer, got {interval_hours!r}.',
            {'interval_hours': repr(interval_hours)},
        )

    history_time = int(row.issue_time) - history_steps * interval_hours * 3600
    endpoints = [
        ('history', history_time),
        ('issue', int(row.issue_time)),
        ('target', int(row.valid_time)),
    ]

    resolved: Dict[str, Tuple[_StoreTimeIndex, int]] = {}
    for name, timestamp in endpoints:
        store_path = Path(resolver(data_root, _utc_year(timestamp)))
        index = load_store_time_index(store_path, cache=cache)

        if index.n_timesteps == 0:
            raise SampleAdmissionError(
                'ADMISSION_STORE_EMPTY',
                f'Store {store_path} holds 0 timesteps, so the {name} endpoint '
                f'of row {row.event_id} at {_iso(timestamp)} cannot exist. The '
                'store still carries a full channel axis and may still have a '
                'complete set of resume markers; neither is data.',
                {'event_id': row.event_id, 'issue_id': issue_id,
                 'endpoint': name, 'store_path': str(store_path),
                 'timestamp': timestamp, 'timestamp_utc': _iso(timestamp)},
            )
        if expected_channels is not None and index.n_channels != int(expected_channels):
            raise SampleAdmissionError(
                'ADMISSION_CHANNEL_COUNT_MISMATCH',
                f'Store {store_path} has {index.n_channels} channels, expected '
                f'{int(expected_channels)}.',
                {'store_path': str(store_path), 'n_channels': index.n_channels,
                 'expected': int(expected_channels)},
            )
        if check_grid_binding:
            stamped = index.attrs.get('grid_hash')
            if isinstance(stamped, str) and row.grid_hash and stamped != row.grid_hash:
                raise SampleAdmissionError(
                    'ADMISSION_GRID_HASH_MISMATCH',
                    f'Store {store_path} carries grid_hash={stamped!r} but row '
                    f'{row.event_id} was scheduled against {row.grid_hash!r}. '
                    'The row and the store do not describe the same grid.',
                    {'event_id': row.event_id, 'store_path': str(store_path),
                     'store_grid_hash': stamped, 'row_grid_hash': row.grid_hash},
                )

        position = index.locate(timestamp)
        if position < 0:
            code = {
                'history': 'ADMISSION_HISTORY_ENDPOINT_MISSING',
                'issue': 'ADMISSION_CURRENT_ENDPOINT_MISSING',
                'target': 'ADMISSION_TARGET_ENDPOINT_MISSING',
            }[name]
            first = _iso(int(index.seconds[0]))
            last = _iso(int(index.seconds[-1]))
            raise SampleAdmissionError(
                code,
                f'Row {row.event_id}: the {name} endpoint at {_iso(timestamp)} '
                f'is not present in {store_path}, which covers {first} .. '
                f'{last} in {index.n_timesteps} timesteps. The calendar says '
                'this sample exists; the data says it does not.',
                {'event_id': row.event_id, 'issue_id': issue_id,
                 'endpoint': name, 'store_path': str(store_path),
                 'timestamp': timestamp, 'timestamp_utc': _iso(timestamp),
                 'store_first_utc': first, 'store_last_utc': last,
                 'n_timesteps': index.n_timesteps},
            )
        resolved[name] = (index, position)
    checks.append('endpoints_exist_on_disk')

    history_index_obj, history_position = resolved['history']
    issue_index_obj, issue_position = resolved['issue']
    target_index_obj, target_position = resolved['target']

    # When all three endpoints live in ONE store, the span between them must
    # also be uniformly spaced: a store that is missing an interior timestep can
    # still hold all three endpoints.
    same_store = (
        str(history_index_obj.path) == str(issue_index_obj.path) == str(target_index_obj.path)
    )
    if same_store:
        validate_slice_index(
            index=issue_position,
            n_timesteps=issue_index_obj.n_timesteps,
            history_steps=issue_position - history_position,
            target_steps=target_position - issue_position,
            time_coords=issue_index_obj.raw_times,
            interval_hours=interval_hours,
        )
        checks.append('span_uniformly_spaced')

    certificate: Optional[Dict[str, Any]] = None
    if verify_content:
        if not same_store:
            raise SampleAdmissionError(
                'ADMISSION_CONTENT_SPAN_CROSSES_STORES',
                f'Row {row.event_id} spans more than one store '
                f'({history_index_obj.path} .. {target_index_obj.path}); '
                'content verification of a cross-store span is not implemented, '
                'so it is refused rather than reported as verified.',
                {'event_id': row.event_id,
                 'stores': sorted({str(history_index_obj.path),
                                   str(issue_index_obj.path),
                                   str(target_index_obj.path)})},
            )
        content = verify_content_subset(
            store_path=issue_index_obj.path,
            indices=list(range(history_position, target_position + 1)),
            batch_size=content_batch_size,
            expected_channels=expected_channels,
            sigma_bound=content_sigma_bound,
            normalization_dir=normalization_dir,
            require_physical_range=require_physical_range,
            data_role=role,
            issue_id=issue_id,
            process_group_id=process_group,
            strict=True,
        )
        certificate = content.to_dict()
        checks.append('content_verified')

    return AdmissionResult(
        admitted=True,
        event_id=row.event_id,
        issue_id=issue_id,
        process_group_id=process_group,
        split_id=row.split_id,
        issue_time=int(row.issue_time),
        valid_time=int(row.valid_time),
        available_time=int(row.available_time),
        lead_hours=(int(row.valid_time) - int(row.issue_time)) / 3600.0,
        normalization_hash=row.normalization_hash,
        grid_hash=row.grid_hash,
        formal=formal,
        data_role=role,
        interval_hours=interval_hours,
        history_steps=history_steps,
        history_time=history_time,
        history_store=str(history_index_obj.path),
        history_index=history_position,
        issue_store=str(issue_index_obj.path),
        issue_index=issue_position,
        target_store=str(target_index_obj.path),
        target_index=target_position,
        n_timesteps=issue_index_obj.n_timesteps,
        checks=checks,
        content_certificate=certificate,
        admitted_at=datetime.now(timezone.utc).isoformat(),
    )


def admit_manifest_rows(
    rows: List[SplitManifestRow],
    data_root: Path,
    *,
    formal: bool = False,
    data_role: Optional[Any] = None,
    limit: Optional[int] = None,
    process_group_id: Optional[str] = None,
    stop_on_first_rejection: bool = False,
    **admit_kwargs: Any,
) -> AdmissionRunReport:
    """Admit many rows under ONE process_group_id, collecting every outcome.

    Rejections are collected rather than raised so a single run reports every
    row it could not admit. The run is only `passed` when nothing was rejected.
    """
    role = assert_valid_data_role(data_role)
    process_group = process_group_id or new_process_group_id(label='admit_manifest_rows')
    selected = list(rows)[: int(limit)] if limit is not None else list(rows)
    report = AdmissionRunReport(
        process_group_id=process_group,
        data_root=str(data_root),
        formal=formal,
        data_role=role,
        requested=len(selected),
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    cache: Dict[str, _StoreTimeIndex] = admit_kwargs.pop('cache', None) or {}

    for row in selected:
        try:
            result = admit_real_sample(
                row,
                data_root,
                formal=formal,
                data_role=role,
                process_group_id=process_group,
                cache=cache,
                **admit_kwargs,
            )
        except PilotContractViolation as exc:
            report.rejected.append({
                'event_id': row.event_id,
                'issue_id': row.resolved_issue_id(),
                'issue_time_utc': _iso(row.issue_time),
                'valid_time_utc': _iso(row.valid_time),
                'code': exc.code,
                'message': exc.message,
            })
            if stop_on_first_rejection:
                break
            continue
        report.admitted.append(result)

    report.finished_at = datetime.now(timezone.utc).isoformat()
    return report


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
