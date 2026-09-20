"""Immutable execution/cache contracts and program specifications.

Timestamps are integer UTC seconds. EditPlan generalizes from rank-group masks
to expert-dictionary coefficients with an application window.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections.abc import Sequence
import hashlib
import json
import math
import torch
from torch import Tensor


@dataclass(frozen=True)
class ArtifactVersion:
    """Version fingerprint for provenance tracking and mismatch rejection."""
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
    """An edit plan specifying which experts are active, their coefficients, and window.

    Generalizes from rank-group masks to K-expert dictionary coefficients in [-rho, rho]
    with an application window (hold_steps). coefficients[k] == 0 implies expert k is inactive.
    """
    plan_id: str
    num_experts: int
    coefficients: tuple[float, ...]
    hold_steps: int = 4
    interval_hours: int = 6
    continuation: str = 'reference_after_hold'
    rho: float = 0.25

    def __post_init__(self):
        if not self.plan_id:
            raise ValueError('Plan needs an ID.')
        if type(self.num_experts) is not int or self.num_experts <= 0:
            raise ValueError('num_experts must be a positive integer.')
        if len(self.coefficients) != self.num_experts:
            raise ValueError('coefficients length must equal num_experts.')
        if any(not math.isfinite(a) for a in self.coefficients):
            raise ValueError('Coefficients must be finite.')
        if not math.isfinite(self.rho) or self.rho <= 0:
            raise ValueError('rho must be a positive finite value.')
        if any(abs(a) > self.rho + 1e-7 for a in self.coefficients):
            raise ValueError(f'Coefficients must be in [-{self.rho}, {self.rho}].')
        if type(self.hold_steps) is not int or self.hold_steps <= 0:
            raise ValueError('hold_steps must be a positive integer.')
        if self.interval_hours != 6:
            raise ValueError('This pilot uses a 6h interval.')
        if self.continuation != 'reference_after_hold':
            raise ValueError('Pilot continuation is fixed: use reference after the edit window.')

    @property
    def active_mask(self) -> tuple[bool, ...]:
        """Boolean mask of which experts have nonzero coefficients."""
        return tuple(a != 0 for a in self.coefficients)

    def active_at(self, step: int) -> tuple[bool, ...]:
        """Return which experts are active at a given rollout step."""
        if type(step) is not int or step < 0:
            raise ValueError('step must be a nonnegative integer.')
        return self.active_mask if step < self.hold_steps else (False,) * self.num_experts

    def coefficients_at(self, step: int) -> tuple[float, ...]:
        """Return coefficient values at a given rollout step (zero outside window)."""
        if type(step) is not int or step < 0:
            raise ValueError('step must be a nonnegative integer.')
        return self.coefficients if step < self.hold_steps else (0.0,) * self.num_experts

    def descriptor(self) -> tuple[float, ...]:
        """Dense descriptor for the plan (for model input)."""
        active_floats = tuple(float(a != 0) for a in self.coefficients)
        return active_floats + self.coefficients + (self.hold_steps / 4.0,)


def reference_plan(num_experts: int, rho: float = 0.25) -> EditPlan:
    """Return the no-edit reference plan."""
    return EditPlan('reference', num_experts, (0.0,) * num_experts, rho=rho)


def single_expert_plans(num_experts: int, coefficient: float = 1.0, rho: float = 0.25) -> tuple[EditPlan, ...]:
    """Return K plans, each activating a single expert at full strength."""
    plans = []
    for k in range(num_experts):
        coeffs = [0.0] * num_experts
        coeffs[k] = min(coefficient, rho)
        plans.append(EditPlan(f'expert_{k}', num_experts, tuple(coeffs), rho=rho))
    return tuple(plans)


def pilot_plans() -> tuple[EditPlan, ...]:
    """Legacy pilot plans (4 experts, rank-group style masks)."""
    masks = [(0,0,0,0), (1,0,0,0), (0,1,0,0), (1,1,0,0), (0,0,1,1), (1,1,1,1)]
    ids = ['reference','g0','g1','g01','g23','all']
    return tuple(EditPlan(i, 4, tuple(float(x) for x in m), rho=1.0) for i, m in zip(ids, masks))


@dataclass(frozen=True)
class Slot:
    """Intervention slot metadata."""
    name: str
    step: int       # zero-based FORECAST step; not observed time
    layer: int
    rank_group: int
    region: str = 'global'


@dataclass(frozen=True)
class ProgramSpec:
    """Program specification for intervention slots.

    Defines which (step, layer, rank_group, region) coordinates can be edited.
    coefficients_for builds dense coefficients and DOES NOT skip adapter matmuls.
    Runtime must use the declared support to skip whole inactive groups.
    """
    slots: tuple[Slot, ...]
    horizon_steps: int
    total_layers: int
    rank_groups: int
    bound: float = 0.25

    def __post_init__(self):
        object.__setattr__(self, 'slots', tuple(self.slots))
        if any(type(v) is not int or v <= 0 for v in (self.horizon_steps, self.total_layers, self.rank_groups)):
            raise ValueError('sizes must be positive integers')
        if not math.isfinite(self.bound) or self.bound <= 0 or not self.slots:
            raise ValueError('nonempty program and positive bound required')
        names = set()
        locations = set()
        for slot in self.slots:
            if not slot.name or slot.name in names:
                raise ValueError('duplicate/empty slot name')
            if any(type(v) is not int for v in (slot.step, slot.layer, slot.rank_group)):
                raise ValueError('slot indices must be integers')
            if not 0 <= slot.step < self.horizon_steps or not 0 <= slot.layer < self.total_layers or not 0 <= slot.rank_group < self.rank_groups:
                raise ValueError('slot is outside rollout/layer/group support')
            key = (slot.step, slot.layer, slot.rank_group, slot.region)
            if key in locations:
                raise ValueError('duplicate intervention coordinate')
            names.add(slot.name)
            locations.add(key)

    def coefficients_for(self, values: Tensor, *, step: int, layer: int,
                         masks: dict[str, Tensor], tokens: int) -> Tensor:
        """[B,d] -> [B,N,G]. Uses values additively; outside support = 0.

        This builds dense coefficients and DOES NOT skip adapter matmuls.
        The dense path is intentional for gradient correctness, not an optimization bug.
        Runtime must use the declared support to skip whole inactive groups.
        Disjoint/overlapping region masks are an explicit experiment choice.
        """
        if values.ndim != 2 or values.shape[1] != len(self.slots) or not values.is_floating_point():
            raise ValueError('values must be [batch,total_program_dimension]')
        if not bool(torch.isfinite(values).all()) or bool((values.abs() > self.bound + 1e-7).any()):
            raise ValueError('coefficient outside declared box')
        if type(tokens) is not int or tokens <= 0 or not 0 <= step < self.horizon_steps or not 0 <= layer < self.total_layers:
            raise ValueError('invalid query')
        out = values.new_zeros((values.shape[0], tokens, self.rank_groups))
        for j, s in enumerate(self.slots):
            if s.step != step or s.layer != layer:
                continue
            if s.region not in masks:
                raise ValueError('missing region mask')
            m = masks[s.region].to(values)
            if m.shape != (tokens,) or not bool(torch.isfinite(m).all()) or bool(((m < 0) | (m > 1)).any()):
                raise ValueError('invalid fixed mask')
            out[:, :, s.rank_group] = out[:, :, s.rank_group] + values[:, j, None] * m[None, :]
        return out
