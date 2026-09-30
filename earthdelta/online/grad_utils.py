"""Safe ownership of temporary weight edits and full-scale raw gradients."""
from contextlib import contextmanager
from pathlib import Path
import threading
import weakref
import numpy as np
import torch
from earthdelta.probe.edits import DirectWeightEditor, pristine_state_digest
from .losses import area_weights
from .io import sha256_file

_OWNERS=weakref.WeakKeyDictionary()
_LOCK=threading.Lock()


def target_metadata(model,blocks):
    blocks=tuple(blocks)
    if not blocks or len(set(blocks))!=len(blocks):raise ValueError('unique target blocks required')
    result=[]
    for b in blocks:
        if isinstance(b,bool) or not isinstance(b,int) or not 0<=b<len(model.blocks):raise ValueError('invalid target block')
        w=model.blocks[b].attn.proj.weight
        if w.ndim!=2:raise ValueError('target must be a matrix')
        result.append({'block':b,'shape':tuple(w.shape),'numel':w.numel()})
    return result


@contextmanager
def model_guard(model,*,verify_digest=True):
    modes=[(m,m.training) for m in model.modules()]
    flags=[(p,p.requires_grad) for p in model.parameters()]
    versions=[(v,v._version) for v in list(model.parameters())+list(model.buffers())]
    before=pristine_state_digest(model) if verify_digest else None
    cpu_rng=torch.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model.eval()
        yield
    finally:
        for p,flag in flags:p.requires_grad_(flag)
        for m,mode in modes:m.training=mode
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        if any(v._version!=old for v,old in versions):raise RuntimeError('pristine model state mutated')
        if before is not None and pristine_state_digest(model)!=before:raise RuntimeError('pristine model digest changed')


@contextmanager
def safe_weight_editor(model,deltas,*,blocks,verify_digest=True):
    meta=target_metadata(model,blocks)
    if set(deltas)-set(blocks):raise ValueError('delta for non-target block')
    # Validate every delta before registering the first hook.
    for row in meta:
        b=row['block']
        if b not in deltas:continue
        d=deltas[b];w=model.blocks[b].attn.proj.weight
        if d.shape!=w.shape or d.dtype!=w.dtype or d.device!=w.device or not bool(torch.isfinite(d).all()):
            raise ValueError('delta shape/device/dtype/value mismatch')
    with _LOCK:
        if model in _OWNERS:raise RuntimeError('shared-model concurrent editing refused')
        _OWNERS[model]=True
    editor=DirectWeightEditor(model,deltas,blocks=blocks)
    try:
        with model_guard(model,verify_digest=verify_digest):
            try:
                editor.__enter__()
                yield editor
            finally:
                editor.__exit__(None,None,None)
    finally:
        with _LOCK:_OWNERS.pop(model,None)


def weight_gradient(model,blocks,loss_fn,*,verify_digest=True):
    meta=target_metadata(model,blocks)
    with model_guard(model,verify_digest=verify_digest):
        for p in model.parameters():p.requires_grad_(False)
        params=[model.blocks[row['block']].attn.proj.weight for row in meta]
        for p in params:p.requires_grad_(True)
        loss=loss_fn()
        if loss.ndim!=0 or not bool(torch.isfinite(loss)):raise ValueError('finite scalar loss required')
        grad=torch.autograd.grad(loss,params,allow_unused=False)
        vector=torch.cat([g.detach().reshape(-1) for g in grad]).float()
        if not bool(torch.isfinite(vector).all()):raise ValueError('nonfinite gradient')
        return vector,float(loss.detach().cpu())


def unit(value):
    x=np.asarray(value,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('nonfinite direction')
    norm=float(np.linalg.norm(x.ravel()))
    return None if norm==0 else x/norm


def cosine(a,b):
    a,b=unit(a),unit(b)
    if a is None or b is None:return None
    if a.shape!=b.shape:raise ValueError('direction dimensions differ')
    return float(np.clip(np.dot(a.ravel(),b.ravel()),-1,1))


def split_direction(model,blocks,vector,amplitude):
    v=torch.as_tensor(vector)
    meta=target_metadata(model,blocks)
    if v.ndim!=1 or len(v)!=sum(x['numel'] for x in meta) or not np.isfinite(amplitude):raise ValueError('invalid edit vector')
    result={};offset=0
    for row in meta:
        w=model.blocks[row['block']].attn.proj.weight
        result[row['block']]=v[offset:offset+row['numel']].reshape(row['shape']).to(w)*amplitude
        offset+=row['numel']
    return result


def response_effect(edited,f0,truth_norm,latitude,*,step=4):
    if edited.shape!=f0.shape or edited.shape!=truth_norm.shape or edited.ndim!=5 or edited.shape[0]!=1 or not 0<=step<edited.shape[1]:
        raise ValueError('matching single-issue trajectories containing response step required')
    if not all(bool(torch.isfinite(v).all()) for v in (edited,f0,truth_norm)):raise ValueError('nonfinite response field')
    w=area_weights(latitude,edited.shape[-1],dtype=edited.dtype,device=edited.device)
    num=((edited[:,step]-f0[:,step]).square()*w).sum()
    den=((f0[:,step]-truth_norm[:,step]).square()*w).sum()
    if not bool(den>0):raise ValueError('zero response denominator')
    return float(torch.sqrt(num/den).detach().cpu())


def calibrate_amplitude(evaluate,weight_norm,*,n_issues=16,target=.1,bracket=(1e-6,.1),iterations=12,relative_tolerance=.05,monotonicity_tolerance=1e-4,all_directions_zero=False):
    if (not np.isfinite([target,relative_tolerance,monotonicity_tolerance]).all() or
            target<=0 or relative_tolerance<=0 or monotonicity_tolerance<0 or
            not isinstance(iterations,int) or iterations<1 or not isinstance(n_issues,int) or n_issues<1):
        raise ValueError('invalid response calibration contract')
    if all_directions_zero:return {'status':'F0_EQUIVALENT','amplitude':0.0,'multiplier':1,'trace':[]}
    if not np.isfinite(weight_norm) or weight_norm<=0:raise ValueError('invalid target weight norm')
    trace=[]
    def f(a):
        x=np.asarray(evaluate(a),dtype=np.float64)
        if x.shape!=(n_issues,) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('calibration cannot shrink registered issue set')
        median=float(np.median(x));trace.append({'amplitude':float(a),'median':median,'responses':x.tolist()})
        values=sorted((row['amplitude'],row['median']) for row in trace)
        if any(b[1]+monotonicity_tolerance<a[1] for a,b in zip(values,values[1:])):raise ValueError('nonmonotonic calibration')
        return median
    lo,hi=np.asarray(bracket)*weight_norm
    if not 0<lo<hi:raise ValueError('invalid calibration bracket')
    a,b=f(lo),f(hi)
    if not a<=target<=b:raise ValueError('response target not bracketed')
    for amp,median in [(lo,a),(hi,b)]:
        if abs(median-target)<=target*relative_tolerance:return {'status':'CALIBRATED','amplitude':float(amp),'trace':trace}
    for _ in range(iterations):
        mid=float(np.sqrt(lo*hi));value=f(mid)
        if abs(value-target)<=target*relative_tolerance:return {'status':'CALIBRATED','amplitude':mid,'trace':trace}
        if value<target:lo=mid
        else:hi=mid
    raise ValueError('calibration tolerance not reached')


def serialize_gradient(path,gradient):
    x=gradient.detach().cpu().numpy() if hasattr(gradient,'detach') else np.asarray(gradient)
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('finite raw gradient vector required')
    x32=x.astype(np.float32)
    alignment=cosine(x,x32)
    if not np.isfinite(x32).all() or (np.any(x!=0) and alignment is None) or (alignment is not None and alignment<=.99999):raise ValueError('gradient FP32 direction not preserved')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as out:np.save(out,x32,allow_pickle=False)
    return {'path':str(path),'sha256':sha256_file(path),'dtype':'float32','norm_fp64':float(np.linalg.norm(x32.astype(np.float64))),
            'roundtrip_cosine':alignment,'fp16_underflow_fraction':float(np.mean((x32!=0)&(x32.astype(np.float16)==0))),'numel':len(x32)}
