"""Original small grouped residual implementation; not a port of FLA/PEFT.

Branches are evaluated only for selected batch rows. This is a correctness-first
Python implementation; sparse arithmetic does NOT establish a GPU speedup.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn

class GroupedLowRankResidual(nn.Module):
    def __init__(self, in_features: int, out_features: int, groups: int = 4,
                 rank_per_group: int = 4, scale: float = 1.0):
        super().__init__()
        if min(in_features,out_features,groups,rank_per_group) <= 0 or not math.isfinite(scale):
            raise ValueError('Invalid feature/group/rank/scale configuration.')
        self.in_features,self.out_features = in_features,out_features
        self.groups,self.rank_per_group,self.scale = groups,rank_per_group,scale
        self.down = nn.ModuleList([nn.Linear(in_features,rank_per_group,bias=False) for _ in range(groups)])
        self.up = nn.ModuleList([nn.Linear(rank_per_group,out_features,bias=False) for _ in range(groups)])
        for a,b in zip(self.down,self.up):
            nn.init.kaiming_uniform_(a.weight,a=math.sqrt(5))
            nn.init.zeros_(b.weight)

    def forward(self, x: Tensor, active: Tensor, coefficients: Tensor | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.in_features:
            raise ValueError('x must be [B,N,in_features].')
        if active.dtype != torch.bool or active.shape != (x.shape[0],self.groups):
            raise ValueError('active must be bool [B,groups].')
        active = active.to(x.device)
        c = active.to(x.dtype) if coefficients is None else coefficients.to(x)
        if c.shape != active.shape or not torch.isfinite(c).all():
            raise ValueError('coefficients must be finite [B,groups].')
        if ((~active) & (c != 0)).any():
            raise ValueError('Inactive coefficients must equal zero.')
        out = x.new_zeros(x.shape[0],x.shape[1],self.out_features)
        for g,(a,b) in enumerate(zip(self.down,self.up)):
            rows = active[:,g].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            y = b(a(x.index_select(0,rows))) * c[rows,g,None,None] * self.scale
            out = out.index_add(0,rows,y)
        return out

    def rank_upper_bound(self, active: Tensor) -> Tensor:
        return (active.sum(-1)*self.rank_per_group).clamp(max=min(self.in_features,self.out_features))
