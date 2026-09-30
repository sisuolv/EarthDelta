"""Immutable F0 bank layout and physical scoring, independent of model weights."""
import numpy as np
from .io import sha256_file
from .losses import cells

SHORT_AND_ENDPOINT_STEPS=(1,2,3,4,12,20)


def physical_cells(predictions,truths,latitude):
    p=np.asarray(predictions);y=np.asarray(truths)
    if p.ndim!=4 or p.shape!=y.shape or p.shape[:2]!=(6,6):
        raise ValueError('six registered steps and six variables required')
    return cells(p[None],y[None],latitude)[0]


def write_prediction(path,prediction,*,norm69_step4=None):
    p=np.asarray(prediction)
    if p.ndim!=4 or p.shape[:2]!=(6,6) or not np.isfinite(p).all():
        raise ValueError('invalid physical F0 field')
    fields={'physical':p.astype(np.float32)}
    if not np.isfinite(fields['physical']).all():raise ValueError('physical field overflows cache dtype')
    if norm69_step4 is not None:
        n=np.asarray(norm69_step4)
        if n.shape!=(69,*p.shape[-2:]) or not np.isfinite(n).all():raise ValueError('invalid all-channel calibration state')
        fields['norm69_step4']=n.astype(np.float32)
        if not np.isfinite(fields['norm69_step4']).all():raise ValueError('calibration field overflows cache dtype')
    with open(path,'xb') as f:np.savez(f,**fields)
    return {'path':str(path),'sha256':sha256_file(path),'dtype':'float32',
            'shape':list(p.shape),'lead_steps':list(SHORT_AND_ENDPOINT_STEPS),'all69_step4':norm69_step4 is not None}


def fit_denominators(mse,indices,roles):
    from .roles import role
    a=np.asarray(mse,dtype=np.float64)
    if a.shape!=(len(indices),6,6) or len(indices)!=len(roles) or len(indices)!=len(set(indices)):
        raise ValueError('bound [issue,variable,six_steps] MSE required')
    if any(role(i)!=r for i,r in zip(indices,roles)):
        raise ValueError('target role contradicts the registered calendar')
    chosen=[i for i,r in enumerate(roles) if r=='estimate']
    if not chosen or not np.isfinite(a[chosen]).all() or np.any(a[chosen]<0):raise ValueError('invalid estimate target losses')
    pooled=a[chosen].mean(0)
    if np.any(pooled<=0):raise ValueError('nonpositive fitted MSE denominator')
    return {'source_indices':[indices[i] for i in chosen],
            'short_mse':pooled[:,:4].T.tolist(),
            'selection_mse':pooled[:,[0,3,4,5]].T.tolist(),
            'unit':'physical MSE, not RMSE','fit_role':'estimate_only'}
