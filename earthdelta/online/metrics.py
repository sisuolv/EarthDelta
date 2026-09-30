"""Paired pooled RMSE and calendar-block confidence intervals, NumPy only."""
from dataclasses import dataclass
import numpy as np
from numbers import Integral
from .timeindex import index_of, checked_index, calendar_pairs


def _validate(mse, mask):
    m=np.asarray(mse,dtype=np.float64); valid=np.asarray(mask)
    if m.ndim!=3 or valid.shape!=(m.shape[0],m.shape[2]) or valid.dtype!=bool:
        raise ValueError('MSE [issue,arm,cell] and one common bool mask [issue,cell] required')
    if np.any(~np.isfinite(m) & valid[:,None,:]) or np.any((m<0)&valid[:,None,:]):
        raise ValueError('a failed method cannot disappear from the denominator')
    if np.any(valid.sum(axis=0)==0):
        raise ValueError('empty registered scoring cell')
    return np.where(valid[:,None,:],m,0.0),valid


def relative_rmse_from_mse(mse,mask,*,f0_index=0):
    m,v=_validate(mse,mask)
    return _improvement(m.sum(0),v.sum(0),f0_index)


def _improvement(sums,counts,f0):
    if isinstance(f0, bool) or not isinstance(f0, Integral) or not 0 <= f0 < sums.shape[-2]:
        raise ValueError('F0 column outside registered arms')
    if np.any(counts<=0):
        raise ValueError('bootstrap sample has an undefined scoring denominator')
    pooled=sums/counts[...,None,:]
    baseline=pooled[...,f0:f0+1,:]
    if np.any(baseline<=0) or not np.isfinite(pooled).all():
        raise ValueError('nonpositive F0 MSE or nonfinite statistic')
    return 100*(1-np.sqrt(pooled/baseline))


@dataclass(frozen=True)
class WeeklyStats:
    sums: np.ndarray
    counts: np.ndarray
    first_week: int
    anchor_index: int


def weekly_sufficient_stats(mse,mask,indices,*,anchor='2020-03-30T00:00:00Z'):
    m,v=_validate(mse,mask)
    idx=np.array([checked_index(i) for i in indices],dtype=np.int64)
    if len(idx)!=len(m) or idx.tolist()!=sorted(set(idx.tolist())):
        raise ValueError('unique chronological issue indices required')
    anchor=index_of(anchor); week=(idx-anchor)//28
    w0,w1=int(week.min()),int(week.max())
    sums=np.zeros((w1-w0+1,m.shape[1],m.shape[2]),dtype=np.float64)
    counts=np.zeros((w1-w0+1,m.shape[2]),dtype=np.int64)
    np.add.at(sums,week-w0,m);np.add.at(counts,week-w0,v.astype(np.int64))
    return WeeklyStats(sums,counts,w0,anchor)


def circular_indices(n,block_weeks,draws,*,seed=20260930):
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (n, block_weeks, draws)) or min(n,block_weeks,draws)<=0 or block_weeks>n:
        raise ValueError('invalid circular block dimensions')
    rng=np.random.default_rng(seed)
    starts=rng.integers(n,size=(draws,(n+block_weeks-1)//block_weeks))
    return ((starts[:,:,None]+np.arange(block_weeks))%n).reshape(draws,-1)[:,:n]


def widen_percentile(estimate,lower,upper,factor):
    """Apply the predeclared STATISTICAL_PLAN section 2 formula exactly.

    This is an estimate-anchored expanded percentile interval, not an
    unmodified percentile interval. Even factor=1 includes the estimate.
    Its coverage must be measured; the name does not certify coverage.
    """
    if not np.isfinite([estimate,lower,upper,factor]).all() or lower>upper or factor<1:
        raise ValueError('invalid percentile width contract')
    return float(estimate-factor*max(estimate-lower,0)),float(estimate+factor*max(upper-estimate,0))


def paired_intervals(stats,arms,cells,families,*,draws=50000,block_weeks=2,seed=20260930,alpha=.05,width_factor=1.,batch=256):
    if len(set(arms))!=len(arms) or 'f0' not in arms or len(set(cells))!=len(cells):
        raise ValueError('unique arms/cells including F0 required')
    if (stats.sums.shape[1:]!=(len(arms),len(cells)) or
            stats.counts.shape!=(len(stats.sums),len(cells)) or
            not np.isfinite([alpha,width_factor]).all() or not 0<alpha<1 or width_factor<1 or
            isinstance(batch,bool) or not isinstance(batch,Integral) or batch<=0):
        raise ValueError('invalid interval contract')
    if not np.isfinite(stats.sums).all() or np.any(stats.sums<0) or np.any(stats.counts<0):
        raise ValueError('invalid weekly sufficient statistics')
    sample=circular_indices(len(stats.sums),block_weeks,draws,seed=seed)
    center=_improvement(stats.sums.sum(0),stats.counts.sum(0),arms.index('f0'))
    values=np.empty((draws,len(arms),len(cells)),dtype=np.float64)
    for start in range(0,draws,batch):
        ids=sample[start:start+batch]
        values[start:start+len(ids)]=_improvement(stats.sums[ids].sum(1),stats.counts[ids].sum(1),arms.index('f0'))
    result={}
    for name,members in families.items():
        if not members or len({(x['a'],x['b'],x['cell']) for x in members})!=len(members):
            raise ValueError('duplicate or empty comparison family')
        rows=[];tail=alpha/(2*len(members))
        for item in members:
            a,b,k=arms.index(item['a']),arms.index(item['b']),cells.index(item['cell'])
            theta=float(center[a,k]-center[b,k]);samples=values[:,a,k]-values[:,b,k]
            lo,hi=np.quantile(samples,[tail,1-tail])
            marginal=np.quantile(samples,[alpha/2,1-alpha/2])
            lower,upper=widen_percentile(theta,lo,hi,width_factor)
            rows.append(dict(item,estimate=theta,lower=lower,upper=upper,marginal=marginal.tolist()))
        result[name]=rows
    return {'families':result,'draws':draws,'block_weeks':block_weeks,'seed':seed,'width_factor':width_factor,'coverage_claim':'not certified by successful execution'}


def threshold_state(lower,upper,threshold):
    if not np.isfinite([lower,upper,threshold]).all() or lower>upper:
        raise ValueError('invalid interval')
    return 'PASS' if lower>threshold else 'NOT_MET' if upper<=threshold else 'INCONCLUSIVE'


def gate_status(intervals,contract,*,engineering_pass,coverage_pass,authorization):
    if not authorization:return 'AWAIT_AUTHORIZATION'
    if not engineering_pass:return 'BLOCKED'
    if not coverage_pass:return 'STATISTICAL_INCONCLUSIVE'
    expected=contract['inference']['families']
    actual=intervals['families']
    if set(actual)!=set(expected):raise ValueError('missing or extra registered family')
    states=[]
    for family,members in expected.items():
        rows=actual[family]
        key=lambda x:(x['a'],x['b'],x['cell'])
        if len(rows)!=len(members) or {key(x) for x in rows}!={key(x) for x in members}:
            raise ValueError('registered family membership changed')
        for row in rows:
            if family not in ('F_B','F_C','F_H'):continue
            q=contract['gate']['harm_floor_pct'] if family=='F_H' else contract['gate']['delta_pp'] if family=='F_B' and row['b'] in ('static','ewma_g') else contract['gate']['benefit_floor_pct']
            states.append(threshold_state(row['lower'],row['upper'],q))
    if all(s=='PASS' for s in states):return 'GO_RECOMMENDATION_ONLY'
    return 'NOT_MET' if 'NOT_MET' in states else 'INCONCLUSIVE'
