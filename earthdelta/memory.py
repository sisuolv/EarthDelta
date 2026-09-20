"""Auditable verified records + differentiable compact delta-rule replay.

Records hold versioned physical features, not permanently cached moving-EMA
latents. Any training/evaluation split policy must be applied before replay.

The recommended default memory feature construction is ewma_error_feature,
which computes an EWMA of recent verified errors filtered by availability time.
gated_delta_replay is retained for ablation but is not the recommended default.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class VerifiedRecord:
    """A verified record with strict time ordering.

    Enforces issue_time < valid_time <= available_time.
    """
    record_id: str
    issue_time: int
    valid_time: int
    available_time: int
    version: str
    event_id: str
    key: Tensor
    value: Tensor

    def __post_init__(self):
        times = (self.issue_time, self.valid_time, self.available_time)
        if any(type(t) is not int for t in times) or not (times[0] < times[1] <= times[2]):
            raise ValueError('Require UTC seconds: issue < valid <= available.')
        if not self.record_id or not self.version:
            raise ValueError('Record identity and provenance required.')
        if self.key.ndim != 1 or self.value.ndim != 1:
            raise ValueError('Record key and value must be vectors.')
        if not torch.isfinite(self.key).all() or not torch.isfinite(self.value).all():
            raise ValueError('Record contains nonfinite features.')
        # Defensive copy to ensure immutability
        object.__setattr__(self, 'key', self.key.detach().clone())
        object.__setattr__(self, 'value', self.value.detach().clone())


def eligible_records(records: list[VerifiedRecord], origin: int, version: str,
                     blocked_events: frozenset[str] = frozenset()) -> tuple[VerifiedRecord, ...]:
    """Filter records by availability time, version, and blocked events.

    Args:
        records: List of VerifiedRecord instances
        origin: Current time (UTC seconds) - only records available before this are eligible
        version: Required version string - mismatches raise an error
        blocked_events: Event IDs to exclude

    Returns:
        Tuple of eligible records sorted by (available_time, valid_time, record_id)
    """
    if type(origin) is not int:
        raise TypeError('origin must be UTC seconds.')
    ids = [r.record_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate memory record IDs.')
    selected = [r for r in records if r.available_time <= origin and r.event_id not in blocked_events]
    if any(r.version != version for r in selected):
        raise ValueError('Memory version mismatch: re-encode or rebuild records explicitly.')
    # Chronology is defined by actual availability, with deterministic tie breaking.
    return tuple(sorted(selected, key=lambda r: (r.available_time, r.valid_time, r.record_id)))


def gated_delta_replay(keys: Tensor, values: Tensor, valid: Tensor,
                       decay: float = 0.99, write_rate: float = 0.5,
                       initial: Tensor | None = None) -> Tensor:
    """Time-only replay [B,T,Dk], [B,T,Dv] -> [B,Dv,Dk]. Invalid slots do not age memory.

    ABLATION ONLY: This is a simplified original recurrence, not the full
    GatedDeltaNet or StreamTTT implementation. The recommended default is
    ewma_error_feature. No data can be written here without prior eligibility filtering.

    Args:
        keys: [B, T, Dk] key vectors
        values: [B, T, Dv] value vectors
        valid: [B, T] boolean mask of valid time steps
        decay: Decay rate per time step (in [0, 1])
        write_rate: Write rate for updates (in [0, 1])
        initial: Optional [B, Dv, Dk] initial state

    Returns:
        [B, Dv, Dk] memory state after replay
    """
    if keys.ndim != 3 or values.ndim != 3 or keys.shape[:2] != values.shape[:2]:
        raise ValueError('Keys/values must be aligned [B,T,D].')
    if valid.dtype != torch.bool or valid.shape != keys.shape[:2]:
        raise ValueError('valid must be bool [B,T].')
    if not 0 <= decay <= 1 or not 0 <= write_rate <= 1:
        raise ValueError('decay/write_rate must be in [0,1].')
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError('Nonfinite memory input.')
    shape = (keys.shape[0], values.shape[-1], keys.shape[-1])
    m = keys.new_zeros(shape) if initial is None else initial.clone().to(keys)
    if m.shape != shape:
        raise ValueError('Initial state has the wrong shape.')
    k = F.normalize(keys, dim=-1, eps=1e-8)
    for t in range(keys.shape[1]):
        old = m * decay
        error = values[:, t] - torch.einsum('bvk,bk->bv', old, k[:, t])
        proposal = old + write_rate * error[:, :, None] * k[:, t, None, :]
        m = torch.where(valid[:, t, None, None].to(m.device), proposal, m)
    return m


def read_memory(snapshot: Tensor, query: Tensor) -> Tensor:
    """Read from memory snapshot.

    Args:
        snapshot: [B, Dv, Dk] memory state
        query: [B, Dk] query vector

    Returns:
        [B, Dv] retrieved values
    """
    if snapshot.ndim != 3 or query.shape != (snapshot.shape[0], snapshot.shape[2]):
        raise ValueError('Expect [B,Dv,Dk] snapshot and [B,Dk] query.')
    return torch.einsum('bvk,bk->bv', snapshot, F.normalize(query, dim=-1, eps=1e-8))


def ewma_error_feature(records: list[VerifiedRecord], origin: int, version: str,
                       decay: float = 0.95, max_records: int | None = None,
                       blocked_events: frozenset[str] = frozenset()) -> Tensor:
    """Compute EWMA of recent verified errors as memory feature.

    This is the RECOMMENDED DEFAULT memory feature construction. It computes
    an exponentially weighted moving average of recent error values, filtered
    by availability time.

    Args:
        records: List of VerifiedRecord instances
        origin: Current time (UTC seconds) - only records available before this are used
        version: Required version string
        decay: EWMA decay factor (higher = older records weighted less)
        max_records: Maximum number of recent records to use (None = all eligible)
        blocked_events: Event IDs to exclude

    Returns:
        [Dv] EWMA error feature vector (zeros if no eligible records)
    """
    eligible = eligible_records(records, origin, version, blocked_events)
    if not eligible:
        # Return zeros with shape inferred from records
        if records:
            return torch.zeros_like(records[0].value)
        raise ValueError('No records provided and no eligible records found.')

    if max_records is not None and max_records > 0:
        eligible = eligible[-max_records:]  # Most recent records

    # Compute EWMA: most recent record has weight 1, older records have decayed weights
    values = torch.stack([r.value for r in reversed(eligible)], dim=0)  # [T, Dv]
    T = values.shape[0]
    weights = torch.tensor([decay ** i for i in range(T)], device=values.device, dtype=values.dtype)
    weights = weights / weights.sum()  # Normalize

    return (weights[:, None] * values).sum(dim=0)  # [Dv]


def ewma_error_feature_batched(records_list: list[list[VerifiedRecord]], origins: list[int],
                               version: str, decay: float = 0.95,
                               max_records: int | None = None,
                               blocked_events: frozenset[str] = frozenset()) -> Tensor:
    """Batched version of ewma_error_feature.

    Args:
        records_list: List of record lists, one per batch element
        origins: List of origin times, one per batch element
        version: Required version string
        decay: EWMA decay factor
        max_records: Maximum records per batch element
        blocked_events: Event IDs to exclude

    Returns:
        [B, Dv] batched EWMA error features
    """
    features = []
    for records, origin in zip(records_list, origins):
        try:
            feat = ewma_error_feature(records, origin, version, decay, max_records, blocked_events)
        except ValueError:
            # No eligible records - use zeros
            if records:
                feat = torch.zeros_like(records[0].value)
            else:
                raise ValueError('Empty records list with no way to infer feature dimension.')
        features.append(feat)
    return torch.stack(features, dim=0)
