"""Immutable execution/cache contracts. Timestamps are integer UTC seconds."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math

@dataclass(frozen=True)
class ArtifactVersion:
    backbone: str
    static_adapter: str
    edit_bank: str
    normalization: str
    grid: str
    projection: str
    split: str
    continuation: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError('Every provenance field must be a nonempty content/version identifier.')

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def assert_matches(self, other: 'ArtifactVersion') -> None:
        if self != other:
            differing = [k for k, v in asdict(self).items() if v != asdict(other)[k]]
            raise ValueError('Stale/incompatible artifact: ' + ', '.join(differing))

@dataclass(frozen=True)
class EditPlan:
    plan_id: str
    groups: tuple[bool, ...]
    coefficients: tuple[float, ...]
    hold_steps: int = 4
    interval_hours: int = 6
    continuation: str = 'reference_after_hold'

    def __post_init__(self):
        if not self.plan_id or not self.groups or len(self.groups) != len(self.coefficients):
            raise ValueError('Plan needs an ID and aligned group/coefficient tuples.')
        if any(type(m) is not bool for m in self.groups):
            raise TypeError('Group masks must be boolean.')
        if any(not math.isfinite(a) for a in self.coefficients):
            raise ValueError('Coefficients must be finite.')
        if any((not m) and a != 0 for m, a in zip(self.groups, self.coefficients)):
            raise ValueError('Inactive groups must have zero coefficients.')
        if type(self.hold_steps) is not int or self.hold_steps <= 0 or self.interval_hours != 6:
            raise ValueError('This pilot uses a positive integer hold length and a 6h interval.')
        if self.continuation != 'reference_after_hold':
            raise ValueError('Pilot continuation is fixed: use reference after the edit window.')

    def active_at(self, step: int) -> tuple[bool, ...]:
        if type(step) is not int or step < 0:
            raise ValueError('step must be a nonnegative integer.')
        return self.groups if step < self.hold_steps else (False,) * len(self.groups)

    def descriptor(self) -> tuple[float, ...]:
        return tuple(map(float, self.groups)) + self.coefficients + (self.hold_steps / 4.0,)


def pilot_plans() -> tuple[EditPlan, ...]:
    masks = [(0,0,0,0), (1,0,0,0), (0,1,0,0), (1,1,0,0), (0,0,1,1), (1,1,1,1)]
    ids = ['reference','g0','g1','g01','g23','all']
    return tuple(EditPlan(i, tuple(map(bool, m)), tuple(float(x) for x in m)) for i,m in zip(ids,masks))
