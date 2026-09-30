"""Conditional fit-derived power proxies with the registered inference rule."""
import numpy as np
from .coverage import ARMS


def centered_null_source(mse):
    """Positive empirical null preserving the paired first-order influences.

    Divide each arm/cell by its source pooled MSE. All population arm means
    are then exactly one. This is not removal of unknown seasonal drift.
    """
    a=np.asarray(mse,dtype=np.float64)
    if a.ndim!=3 or a.shape[1:]!=(9,24) or len(a)<2 or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('complete paired fit cells required')
    pooled=a.mean(0)
    if np.any(pooled<=0):raise ValueError('nonpositive source MSE')
    null=a/pooled
    influence=-50*(null-null[:,0:1,:])
    return null,influence


def resample_source(source,n,seed,block_days=14):
    if block_days>len(source):raise ValueError('source shorter than one registered dependence block')
    rng=np.random.default_rng(seed)
    start=rng.integers(len(source),size=(n+block_days-1)//block_days)
    ids=((start[:,None]+np.arange(block_days))%len(source)).ravel()[:n]
    return source[ids]


def power_trial(plan,mse,factor,grid):
    """Each marginal contrast has its own coherent alternative: improve arm a.

    For the full Gate all required a's are lag1c, hence one common improved
    lag1c supplies the entire conjunction. F_O and F_M marginal power is
    never stitched into a single fictitious all-family model.
    """
    center,ratios=plan.ratios(mse)
    ra=ratios[:,plan.a,plan.c];rb=ratios[:,plan.b,plan.c]
    ca=center[plan.a,plan.c];cb=center[plan.b,plan.c]
    thresholds=[];gate=[]
    for family,row in plan.members:
        q=plan.contract['gate']['harm_floor_pct'] if family=='F_H' else (
            plan.contract['gate']['delta_pp'] if family=='F_B' and row['b'] in ('static','ewma_g') else 0.)
        thresholds.append(q);gate.append(family in ('F_B','F_H','F_C'))
    if any(row['a']!='lag1c' for flag,(_,row) in zip(gate,plan.members) if flag):raise ValueError('Gate alternative no longer coherent')
    thresholds=np.array(thresholds);gate=np.array(gate);passes=[];halves=[];joint=[]
    for effect in grid:
        if not 0<=effect<100:raise ValueError('invalid injected RMSE improvement')
        theta=100*(cb-ca)+effect*ca
        sample=100*(rb-ra)+effect*ra
        lo=np.empty(len(plan.members));hi=lo.copy();offset=0
        for name,members in plan.families.items():
            stop=offset+len(members);tail=plan.contract['inference']['family_alpha']/(2*len(members))
            lower,upper=np.quantile(sample[:,offset:stop],[tail,1-tail],axis=0)
            lo[offset:stop]=theta[offset:stop]-factor*np.maximum(theta[offset:stop]-lower,0)
            hi[offset:stop]=theta[offset:stop]+factor*np.maximum(upper-theta[offset:stop],0)
            offset=stop
        success=lo>thresholds
        passes.append(success);halves.append((hi-lo)/2);joint.append(bool(success[gate].all()))
    return np.array(passes),np.array(halves),np.array(joint)
