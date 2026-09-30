"""Fit-only EOF transforms and a single-split Ridge output baseline."""
from dataclasses import dataclass
import numpy as np
from .clock import immutable_array,require_fresh_lineage


@dataclass(frozen=True)
class EOF:
    mean: np.ndarray
    basis: np.ndarray
    sqrt_weights: np.ndarray
    shape: tuple

    def encode(self,fields):
        x=np.asarray(fields,dtype=np.float64)
        if x.shape[-2:]!=self.shape or not np.isfinite(x).all():raise ValueError('invalid EOF field')
        return ((x.reshape(-1,self.mean.size)-self.mean)*self.sqrt_weights)@self.basis.T

    def decode(self,coefficients):
        x=np.asarray(coefficients,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(self.basis) or not np.isfinite(x).all():raise ValueError('invalid EOF coefficients')
        return (self.mean+(x@self.basis)/self.sqrt_weights).reshape(len(x),*self.shape)


def fit_eof(fields,k,latitude):
    x=np.asarray(fields,dtype=np.float64)
    lat=np.asarray(latitude,dtype=np.float64)
    if x.ndim!=3 or x.shape[1]!=len(lat) or not np.isfinite(x).all() or not np.isfinite(lat).all() or np.any(abs(lat)>=90):raise ValueError('invalid EOF training field')
    if not 1<=k<=min(x.shape[0]-1,np.prod(x.shape[1:])):raise ValueError('registered EOF dimension is not estimable')
    w=np.broadcast_to(np.cos(np.deg2rad(lat))[:,None],x.shape[1:]);sw=np.sqrt((w/w.sum()).ravel())
    matrix=x.reshape(len(x),-1);mean=matrix.mean(0)
    _,s,basis=np.linalg.svd((matrix-mean)*sw,full_matrices=False)
    # A degenerate EOF basis is deterministic up to sign but still a valid
    # fixed-size transform. Do not silently lower the registered dimension.
    basis=basis[:k].copy()
    signs=np.sign(basis[np.arange(k),np.abs(basis).argmax(1)]);basis*=np.where(signs==0,1,signs)[:,None]
    return EOF(immutable_array(mean),immutable_array(basis),immutable_array(sw),tuple(x.shape[1:]))


def make_fresh_features(bundle,gradient_lineage,cutoff,eofs,same_lead_history,target_eof):
    if bundle is None or same_lead_history is None:return None
    require_fresh_lineage(bundle,gradient_lineage,cutoff)
    expected={(v,k) for v in bundle.variables for k in (1,2,3,4)}
    if set(eofs)!=expected:raise ValueError('all six-variable/four-step EOF blocks required')
    records={(r.channel,r.lead_step):r for r in bundle.records}
    values=[eofs[v,k].encode(records[v,k].value)[0] for v in bundle.variables for k in (1,2,3,4)]
    values.append(target_eof.encode(same_lead_history)[0])
    return np.concatenate(values)


@dataclass
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    estimator: object

    def predict(self,features):
        x=np.asarray(features,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all():raise ValueError('finite complete prediction features required')
        return self.estimator.predict((x-self.mean)/self.scale)


def fit_ridge(features,target,regularization):
    from sklearn.linear_model import Ridge
    x,y=np.asarray(features,dtype=np.float64),np.asarray(target,dtype=np.float64)
    if x.ndim!=2 or len(x)<2 or y.ndim not in (1,2) or len(y)!=len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(regularization) or regularization<=0:raise ValueError('complete finite fit rows and positive ridge penalty required')
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>0,scale,1.)
    estimator=Ridge(alpha=len(x)*regularization,solver='svd').fit((x-mean)/scale,y)
    return RidgeModel(immutable_array(mean),immutable_array(scale),estimator)


def select_fit_config(scores,candidate_order):
    if set(scores)!=set(candidate_order) or len(candidate_order)!=len(set(candidate_order)):
        raise ValueError('every preregistered candidate needs a score')
    values=np.asarray([scores[k] for k in candidate_order],dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('failed fit candidate cannot be omitted')
    return next(k for k in candidate_order if scores[k]<=values.min()+1e-12)


def predict_correction(model,features,target_eof):
    if features is None:return None
    return target_eof.decode(model.predict(np.asarray(features)[None,:]))[0]
