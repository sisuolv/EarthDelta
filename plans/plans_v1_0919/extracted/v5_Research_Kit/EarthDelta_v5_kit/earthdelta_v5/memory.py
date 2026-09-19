"""Auditable verified records + differentiable compact delta-rule replay.

Records hold versioned physical features, not permanently cached moving-EMA
latents. Any training/evaluation split policy must be applied before replay.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn import functional as F

@dataclass(frozen=True)
class VerifiedRecord:
    record_id: str
    issue_time: int
    valid_time: int
    available_time: int
    version: str
    event_id: str
    key: Tensor
    value: Tensor

    def __post_init__(self):
        times = (self.issue_time,self.valid_time,self.available_time)
        if any(type(t) is not int for t in times) or not (times[0] < times[1] <= times[2]):
            raise ValueError('Require UTC seconds: issue < valid <= available.')
        if not self.record_id or not self.version:
            raise ValueError('Record identity and provenance required.')
        if self.key.ndim != 1 or self.value.ndim != 1:
            raise ValueError('Record key and value must be vectors.')
        if not torch.isfinite(self.key).all() or not torch.isfinite(self.value).all():
            raise ValueError('Record contains nonfinite features.')
        object.__setattr__(self,'key',self.key.detach().clone())
        object.__setattr__(self,'value',self.value.detach().clone())


def eligible_records(records: list[VerifiedRecord], origin: int, version: str,
                     blocked_events: frozenset[str] = frozenset()) -> tuple[VerifiedRecord,...]:
    if type(origin) is not int:
        raise TypeError('origin must be UTC seconds.')
    ids = [r.record_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate memory record IDs.')
    selected = [r for r in records if r.available_time <= origin and r.event_id not in blocked_events]
    if any(r.version != version for r in selected):
        raise ValueError('Memory version mismatch: re-encode or rebuild records explicitly.')
    # Chronology is defined by actual availability, with deterministic tie breaking.
    return tuple(sorted(selected,key=lambda r:(r.available_time,r.valid_time,r.record_id)))


def gated_delta_replay(keys: Tensor, values: Tensor, valid: Tensor,
                       decay: float = 0.99, write_rate: float = 0.5,
                       initial: Tensor | None = None) -> Tensor:
    """Time-only replay [B,T,Dk], [B,T,Dv] -> [B,Dv,Dk]. Invalid slots do not age memory.

    This is a simplified original recurrence, not the full GatedDeltaNet or
    StreamTTT implementation. No data can be written here without prior eligibility filtering.
    """
    if keys.ndim != 3 or values.ndim != 3 or keys.shape[:2] != values.shape[:2]:
        raise ValueError('Keys/values must be aligned [B,T,D].')
    if valid.dtype != torch.bool or valid.shape != keys.shape[:2]:
        raise ValueError('valid must be bool [B,T].')
    if not 0 <= decay <= 1 or not 0 <= write_rate <= 1:
        raise ValueError('decay/write_rate must be in [0,1].')
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError('Nonfinite memory input.')
    shape = (keys.shape[0],values.shape[-1],keys.shape[-1])
    m = keys.new_zeros(shape) if initial is None else initial.clone().to(keys)
    if m.shape != shape:
        raise ValueError('Initial state has the wrong shape.')
    k = F.normalize(keys,dim=-1,eps=1e-8)
    for t in range(keys.shape[1]):
        old = m*decay
        error = values[:,t] - torch.einsum('bvk,bk->bv',old,k[:,t])
        proposal = old + write_rate*error[:,:,None]*k[:,t,None,:]
        m = torch.where(valid[:,t,None,None].to(m.device),proposal,m)
    return m


def read_memory(snapshot: Tensor, query: Tensor) -> Tensor:
    if snapshot.ndim != 3 or query.shape != (snapshot.shape[0],snapshot.shape[2]):
        raise ValueError('Expect [B,Dv,Dk] snapshot and [B,Dk] query.')
    return torch.einsum('bvk,bk->bv',snapshot,F.normalize(query,dim=-1,eps=1e-8))
