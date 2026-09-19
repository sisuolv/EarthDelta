"""Small immutable intervention metadata; no global hooks or hidden state."""
from dataclasses import dataclass
from collections.abc import Sequence
import math
import torch
from torch import Tensor


@dataclass(frozen=True)
class Slot:
    name: str
    step: int       # zero-based FORECAST step; not observed time
    layer: int
    rank_group: int
    region: str = 'global'


@dataclass(frozen=True)
class ProgramSpec:
    slots: tuple[Slot,...]
    horizon_steps: int
    total_layers: int
    rank_groups: int
    bound: float = .25

    def __post_init__(self):
        object.__setattr__(self,'slots',tuple(self.slots))
        if any(type(v) is not int or v<=0 for v in (self.horizon_steps,self.total_layers,self.rank_groups)):
            raise ValueError('sizes must be positive integers')
        if not math.isfinite(self.bound) or self.bound<=0 or not self.slots:
            raise ValueError('nonempty program and positive bound required')
        names=set(); locations=set()
        for slot in self.slots:
            if not slot.name or slot.name in names: raise ValueError('duplicate/empty slot name')
            if any(type(v) is not int for v in (slot.step,slot.layer,slot.rank_group)):
                raise ValueError('slot indices must be integers')
            if not 0<=slot.step<self.horizon_steps or not 0<=slot.layer<self.total_layers or not 0<=slot.rank_group<self.rank_groups:
                raise ValueError('slot is outside rollout/layer/group support')
            key=(slot.step,slot.layer,slot.rank_group,slot.region)
            if key in locations: raise ValueError('duplicate intervention coordinate')
            names.add(slot.name); locations.add(key)

    def coefficients_for(self, values: Tensor, *, step: int, layer: int,
                         masks: dict[str,Tensor], tokens: int) -> Tensor:
        """[B,d] -> [B,N,G]. Uses values additively; outside support = 0.

        This builds dense coefficients and DOES NOT skip adapter matmuls.
        Runtime must use the declared support to skip whole inactive groups.
        Disjoint/overlapping region masks are an explicit experiment choice.
        """
        if values.ndim!=2 or values.shape[1]!=len(self.slots) or not values.is_floating_point():
            raise ValueError('values must be [batch,total_program_dimension]')
        if not bool(torch.isfinite(values).all()) or bool((values.abs()>self.bound+1e-7).any()):
            raise ValueError('coefficient outside declared box')
        if type(tokens) is not int or tokens<=0 or not 0<=step<self.horizon_steps or not 0<=layer<self.total_layers:
            raise ValueError('invalid query')
        out=values.new_zeros((values.shape[0],tokens,self.rank_groups))
        for j,s in enumerate(self.slots):
            if s.step!=step or s.layer!=layer: continue
            if s.region not in masks: raise ValueError('missing region mask')
            m=masks[s.region].to(values)
            if m.shape!=(tokens,) or not bool(torch.isfinite(m).all()) or bool(((m<0)|(m>1)).any()):
                raise ValueError('invalid fixed mask')
            out[:,:,s.rank_group]=out[:,:,s.rank_group]+values[:,j,None]*m[None,:]
        return out
