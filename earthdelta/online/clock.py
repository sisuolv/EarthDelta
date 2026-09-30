"""Private pending feedback and immutable released snapshots.

Times are six-hour 2020 indices. This is an API boundary, not an adversarial
Python sandbox: the actor is never given the queue, scorer, or pending list.
"""
from dataclasses import dataclass, replace
import numpy as np
from .timeindex import checked_index


def immutable_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError('finite numeric payload required')
    # A bytes owner prevents callers from re-enabling WRITEABLE with setflags.
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class SupervisionRecord:
    record_id: str
    issue_index: int
    label_available: int
    artifact_available: int
    version: str
    lineage: frozenset
    value: np.ndarray
    channel: str = ''
    lead_step: int = 0

    def __post_init__(self):
        for x in (self.issue_index, self.label_available, self.artifact_available):
            checked_index(x)
        if any(not isinstance(x, str) or not x for x in (self.record_id, self.version)) or self.label_available < self.issue_index:
            raise ValueError('invalid feedback identity or time')
        if isinstance(self.lineage, str) or not self.lineage or any(not isinstance(x, str) or not x for x in self.lineage):
            raise ValueError('raw supervision lineage required')
        object.__setattr__(self, 'lineage', frozenset(self.lineage))
        object.__setattr__(self, 'value', immutable_array(self.value))

    @property
    def usable_after(self):
        return max(self.label_available, self.artifact_available)


class SupervisionQueue:
    def __init__(self):
        self.__pending = []
        self.__released = []
        self.__ids = set()
        self.__cutoff = -1

    def enqueue(self, record):
        # Worker-only method. Duplicate identity validation is deferred until
        # the record becomes observable, so pending IDs cannot leak via errors.
        if not isinstance(record, SupervisionRecord):
            raise TypeError('worker must supply validated feedback')
        self.__pending.append(replace(record, value=record.value))

    def release(self, cutoff):
        checked_index(cutoff)
        if cutoff < self.__cutoff:
            raise ValueError('release clock cannot move backwards')
        eligible = [r for r in self.__pending if r.usable_after <= cutoff]
        seen = set(self.__ids)
        for r in eligible:
            if r.record_id in seen:
                raise ValueError('released feedback identity conflict')
            seen.add(r.record_id)
        self.__pending = [r for r in self.__pending if r.usable_after > cutoff]
        self.__released.extend(eligible)
        self.__ids = seen
        self.__cutoff = cutoff
        return tuple(eligible)

    def released(self):
        return tuple(self.__released)


@dataclass(frozen=True)
class FeedbackBundle:
    issue_index: int
    records: tuple
    variables: tuple

    def __post_init__(self):
        checked_index(self.issue_index)
        records = tuple(self.records)
        if any(not isinstance(r, SupervisionRecord) for r in records):
            raise TypeError('bundle requires validated supervision records')
        expected = {(v, k) for v in self.variables for k in (1, 2, 3, 4)}
        if len(set(self.variables)) != 6 or len(records) != 24:
            raise ValueError('six-variable four-step feedback required')
        if {(r.channel, r.lead_step) for r in records} != expected:
            raise ValueError('incomplete or duplicate short-feedback cells')
        if any(r.issue_index != self.issue_index or r.label_available != self.issue_index+r.lead_step for r in records):
            raise ValueError('feedback issue/valid-time binding mismatch')
        if len({r.version for r in records}) != 1:
            raise ValueError('mixed feedback versions')
        object.__setattr__(self, 'records', records)
        object.__setattr__(self, 'variables', tuple(self.variables))

    @property
    def usable_after(self):
        return max(self.issue_index + 4, *(r.usable_after for r in self.records))

    @property
    def lineage(self):
        return frozenset().union(*(r.lineage for r in self.records))

    def visible(self, cutoff):
        if cutoff < self.usable_after:
            raise PermissionError('feedback bundle not released')
        return self


@dataclass(frozen=True)
class Derived:
    version: str
    artifact_available: int
    source_roles: tuple
    source_lineage: frozenset

    def __post_init__(self):
        checked_index(self.artifact_available)
        if (not isinstance(self.version, str) or not self.version or
                isinstance(self.source_roles, str) or isinstance(self.source_lineage, str) or
                not self.source_roles or not self.source_lineage or
                any(not isinstance(x, str) or not x for x in (*self.source_roles, *self.source_lineage))):
            raise ValueError('complete fitted artifact provenance required')
        object.__setattr__(self, 'source_roles', tuple(self.source_roles))
        object.__setattr__(self, 'source_lineage', frozenset(self.source_lineage))

    def require_online(self, cutoff):
        checked_index(cutoff)
        if cutoff < self.artifact_available:
            raise PermissionError('fitted artifact cannot publish retroactively')


def require_fresh_lineage(bundle, gradient_lineage, cutoff):
    bundle.visible(cutoff)
    if bundle.issue_index != cutoff-4:
        raise ValueError('fresh must mean exact previous calendar day')
    if not gradient_lineage or not frozenset(gradient_lineage).issubset(bundle.lineage):
        raise ValueError('output feedback omits raw labels used by the gradient')
