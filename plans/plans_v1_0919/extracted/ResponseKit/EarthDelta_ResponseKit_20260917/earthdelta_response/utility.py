"""Deployable LOW-DIMENSIONAL surrogate planner: no truth or response probe.

Input b_hat, H_hat must be PREDICTED from legal history by a trained student.
This module does not certify actual forecast improvement. Additive costs here
are an abstract unit-cost model; they must not be reported as measured FLOPs.
"""
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
from scipy.optimize import minimize
import torch
from torch import Tensor,nn
import torch.nn.functional as F


class InteractionUtilityHead(nn.Module):
    """Predict b:[B,d], H:[B,d,d] for gain(a)=2b.a-a.T H a.

    H is PSD via L L.T + eps I. Off-diagonals describe response overlap,
    NOT exact nonlinear mixed derivatives or atmosphere causality. d=8 uses
    8+36=44 outputs. Inputs must not contain future residual/teacher labels.
    """
    def __init__(self,features:int,dimension:int,width:int=64,eps:float=1e-5):
        super().__init__()
        if any(type(v) is not int or v<=0 for v in (features,dimension,width)) or not math.isfinite(eps) or eps<=0:
            raise ValueError('invalid dimensions/eps')
        self.features=features; self.dimension=dimension; self.eps=float(eps)
        rows,cols=torch.tril_indices(dimension,dimension)
        self.register_buffer('rows',rows); self.register_buffer('cols',cols)
        self.net=nn.Sequential(nn.Linear(features,width),nn.GELU(),
            nn.Linear(width,dimension+len(rows)))
    def forward(self,features:Tensor)->tuple[Tensor,Tensor]:
        if features.ndim!=2 or features.shape[-1]!=self.features:
            raise ValueError('features must be [batch,features]')
        out=self.net(features); d=self.dimension
        b=out[:,:d]; raw=out[:,d:]
        values=torch.where((self.rows==self.cols)[None,:],F.softplus(raw),raw)
        lower=out.new_zeros((out.shape[0],d,d))
        lower[:,self.rows,self.cols]=values
        identity=torch.eye(d,device=out.device,dtype=out.dtype)
        return b,lower@lower.transpose(-1,-2)+self.eps*identity


def quadratic_gain(benefit:Tensor,gram:Tensor,program:Tensor)->Tensor:
    """Batched differentiable surrogate; program can be [B,d] or [d]."""
    return 2*(benefit*program).sum(-1)-torch.einsum('...i,...ij,...j->...',program,gram,program)


@dataclass(frozen=True)
class SurrogatePlan:
    coefficients:Tensor
    support:tuple[int,...]
    predicted_gain:float
    objective_gain:float
    declared_cost:float
    solves:int
    solver_failures:int


@torch.no_grad()
def plan_from_prediction(benefit:Tensor,gram:Tensor,*,bound:float=.25,
                         max_active:int=2,costs:tuple[float,...]|None=None,
                         budget:float=2.,ridge:float=1e-4,max_candidates:int=4096)->SurrogatePlan:
    """Enumerate small supports, solve convex box QP using SciPy L-BFGS-B.

    Does NOT call a weather model, access labels, or run online teacher probes.
    Requires one sample [d], [d,d]. Includes no-op and ignores failed local
    numerical solves, reporting the failures. Larger support spaces rejected.
    """
    if benefit.ndim!=1 or benefit.numel()==0 or gram.shape!=(benefit.numel(),benefit.numel()):
        raise ValueError('expected [d] and [d,d]')
    if any(not x.is_floating_point() or not bool(torch.isfinite(x).all()) for x in (benefit,gram)):
        raise ValueError('nonfinite/nonfloating inputs')
    d=benefit.numel()
    if type(max_active) is not int or not 0<=max_active<=d: raise ValueError('invalid support size')
    if not math.isfinite(bound) or bound<=0 or not math.isfinite(ridge) or ridge<0 or not math.isfinite(budget) or budget<0:
        raise ValueError('invalid bounds/ridge/budget')
    if sum(math.comb(d,k) for k in range(max_active+1))>max_candidates:
        raise ValueError('support space too large')
    b=benefit.detach().double().cpu().numpy(); h=gram.detach().double().cpu().numpy()
    if not np.allclose(h,h.T,atol=1e-8,rtol=1e-8) or np.linalg.eigvalsh(h).min() < -1e-8:
        raise ValueError('predicted gram must be symmetric PSD')
    c=np.ones(d) if costs is None else np.asarray(costs,dtype=float)
    if c.shape!=(d,) or not np.isfinite(c).all() or (c<=0).any(): raise ValueError('invalid costs')
    best=np.zeros(d); best_gain=0.; best_predicted=0.; best_support=(); best_cost=0.; solves=0; failures=0
    for k in range(1,max_active+1):
        for support in combinations(range(d),k):
            idx=np.asarray(support); cost=float(c[idx].sum())
            if cost>budget+1e-12: continue
            a=h[np.ix_(idx,idx)]+ridge*np.eye(k); z=b[idx]
            sol=minimize(lambda x:float(x@a@x-2*z@x),np.zeros(k),
                         jac=lambda x:2*(a@x-z),bounds=[(-bound,bound)]*k,
                         method='L-BFGS-B',options={'ftol':1e-13,'gtol':1e-10,'maxiter':200})
            solves+=1
            if not sol.success or not np.isfinite(sol.x).all(): failures+=1; continue
            gain=-float(sol.fun)
            if gain>best_gain+1e-12:
                best=np.zeros(d); best[idx]=sol.x; best_support=support; best_cost=cost
                best_gain=gain; best_predicted=float(2*b@best-best@h@best)
    return SurrogatePlan(torch.from_numpy(best).to(benefit),best_support,best_predicted,
                         best_gain,best_cost,solves,failures)
