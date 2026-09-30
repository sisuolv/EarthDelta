"""Preregistered full-gradient direction families and fit-only raw centering."""
import numpy as np
from .grad_utils import unit
from .roles import require_fit_rows
from .timeindex import checked_index
from .dabc import calendar_ewma
from .clock import Derived


def raw_gradient_mean(records):
    if not records:raise ValueError('estimate gradient set is empty')
    require_fit_rows([r.issue_index for r in records])
    if len({r.version for r in records})!=1:raise ValueError('mixed gradient identities')
    shape=records[0].value.shape
    if len(shape)!=1 or any(r.value.shape!=shape for r in records):raise ValueError('full gradient vector shape mismatch')
    total=np.zeros(shape,dtype=np.float64)
    for r in records:total+=r.value.astype(np.float64)
    return total/len(records)


def direction(name,cutoff,released,mean,*,mean_provenance,tau_days=None,offline_fit=False,artifact_context=None):
    checked_index(cutoff);mu=np.asarray(mean,dtype=np.float64)
    if mu.ndim!=1 or not np.isfinite(mu).all():raise ValueError('finite raw estimate gradient mean required')
    if offline_fit:
        if artifact_context is None:raise ValueError('offline calibration must name artifact context')
        checked_index(artifact_context)
    context=artifact_context if offline_fit else cutoff
    if not isinstance(mean_provenance,Derived) or set(mean_provenance.source_roles)!={'estimate'}:
        raise ValueError('raw mean must have estimate-only provenance')
    mean_provenance.require_online(context)
    records=[r for r in released if r.label_available<=cutoff and r.artifact_available<=context]
    if len({r.version for r in records})>1 or len({r.issue_index for r in records})!=len(records):raise ValueError('mixed/duplicate gradient history')
    if any(r.value.shape!=mu.shape for r in records):raise ValueError('gradient/mean shape mismatch')
    if name=='static':return None if unit(mu) is None else -unit(mu)
    if name=='ewma_g':
        if tau_days is None:raise ValueError('registered EWMA time constant required')
        g=calendar_ewma(records,cutoff,tau_days,offline_fit=offline_fit,artifact_context=artifact_context)
    elif name in ('lag1','lag1c','delay30'):
        lag=30 if name=='delay30' else 1
        exact=[r for r in records if r.issue_index==cutoff-4*lag]
        g=None if not exact else exact[0].value.astype(np.float64)
        if g is not None and name in ('lag1c','delay30'):g=g-mu
    else:raise ValueError('unregistered direction family')
    if g is None:return None
    u=unit(g);return None if u is None else -u
