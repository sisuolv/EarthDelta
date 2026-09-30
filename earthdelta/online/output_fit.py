"""Offline fit/selection only. These retrospective fits are not online forecasts."""
from dataclasses import dataclass
import numpy as np
from .bank import SHORT_AND_ENDPOINT_STEPS
from .roles import role,require_fit_rows
from .losses import cells
from .outcorr import fit_eof,fit_ridge,select_fit_config
from .timeindex import index_of


def label_ids(issue,variables,steps=(1,2,3,4)):
    return frozenset(f'2020:{issue+k}:{v}' for v in variables for k in steps)


class OutputFit:
    """Fit-only cache. Never pass this object or its full residuals to an actor."""
    def __init__(self,residuals,indices,latitude,denominator,variables):
        r=np.asarray(residuals,dtype=np.float64)
        self.indices=np.asarray(indices,dtype=np.int64);self.lat=np.asarray(latitude)
        self.d=np.asarray(denominator,dtype=np.float64);self.variables=tuple(variables)
        if r.ndim!=5 or r.shape[:3]!=(len(indices),6,6) or r.shape[3]!=len(self.lat) or self.d.shape!=(4,6):raise ValueError('output fit layout mismatch')
        if not np.isfinite(r).all() or not np.isfinite(self.d).all() or np.any(self.d<=0):raise ValueError('invalid fit residuals or denominator')
        if list(indices)!=sorted(set(indices)):raise ValueError('chronological unique fit issues required')
        self.roles=[role(int(i)) for i in indices]
        if set(self.roles)-{'estimate','gap','select'}:raise PermissionError('output fitter cannot read warmup/analysis targets')
        self.r=r;self.positions={int(i):p for p,i in enumerate(indices)}
        self.score_steps=(0,3,4,5)
        self.weights=np.broadcast_to(np.cos(np.deg2rad(self.lat))[:,None],r.shape[-2:]).copy()
        self.weights/=self.weights.sum()

    def training_positions(self,final_refit=False):
        wanted={'estimate','select'} if final_refit else {'estimate'}
        pos=np.array([p for p,r in enumerate(self.roles) if r in wanted],dtype=int)
        require_fit_rows(self.indices[pos].tolist(),final_refit=final_refit)
        return pos

    def history(self,step_position,var,tau):
        result=[];step=SHORT_AND_ENDPOINT_STEPS[step_position]
        for i in self.indices:
            positions=np.flatnonzero(self.indices+step<=i)
            if not len(positions):result.append(None);continue
            logw=(self.indices[positions]+step-i)/(4*tau);w=np.exp(logw-logw.max());w/=w.sum()
            result.append(np.tensordot(w,self.r[positions,step_position,var],axes=(0,0)))
        return result

    def transforms(self,k,final_refit=False):
        pos=self.training_positions(final_refit)
        return {(s,v):fit_eof(self.r[pos,s,v],k,self.lat) for s in range(6) for v in range(6)}

    def fresh(self,eofs):
        encoded=np.concatenate([eofs[s,v].encode(self.r[:,s,v]) for v in range(6) for s in range(4)],axis=1)
        fresh={};lineage={}
        for p,i in enumerate(self.indices):
            previous=self.positions.get(int(i)-4)
            if previous is None:continue
            fresh[p]=encoded[previous].copy()
            # Six variables x four raw valid-time labels, same source ids as gradients.
            lineage[str(int(i))]=sorted(label_ids(int(i)-4,self.variables))
        return fresh,lineage

    def features(self,fresh,eof,s,v):
        history=self.history(s,v,7)
        available=np.array([p for p in range(len(self.indices)) if p in fresh and history[p] is not None],dtype=int)
        if not len(available):raise ValueError('no complete fresh training feature rows')
        x=np.stack([np.concatenate([fresh[p],eof.encode(history[p])[0]]) for p in available])
        return available,x

    def score(self,corrections,positions):
        # Missing feature is explicitly represented by None and falls back to F0.
        errors=[]
        for p in positions:
            cs=corrections[p]
            if cs is None:errors.append(self.r[p,self.score_steps,:,:,:])
            else:
                if cs.shape!=(4,6,*self.r.shape[-2:]) or not np.isfinite(cs).all():raise ValueError('invalid candidate correction')
                errors.append(self.r[p,self.score_steps,:,:,:]-cs)
        a=np.stack(errors)
        mse=cells(a,np.zeros_like(a),self.lat)
        return float(np.mean(mse/self.d.T)),mse

    def obc(self,strength,final_refit=False):
        mean=self.r[self.training_positions(final_refit)].mean(0)[list(self.score_steps)]
        return [strength*mean for _ in self.indices]

    def dabc(self,tau,strength):
        correction=np.zeros((len(self.indices),4,6,*self.r.shape[-2:]),dtype=np.float64);fallback=np.zeros((len(self.indices),4,6),bool)
        for l,s in enumerate(self.score_steps):
            for v in range(6):
                history=self.history(s,v,tau)
                for p,h in enumerate(history):
                    if h is None:fallback[p,l,v]=True
                    else:correction[p,l,v]=strength*h
        return correction,fallback

    def ocl(self,k,regularization,*,eofs=None,final_refit=False):
        eofs=eofs or self.transforms(k,final_refit);fresh,lineage=self.fresh(eofs)
        correction=np.zeros((len(self.indices),4,6,*self.r.shape[-2:]),dtype=np.float64)
        fallback=np.ones((len(self.indices),4,6),bool);models={};counts={}
        training=set(self.training_positions(final_refit).tolist())
        for l,s in enumerate(self.score_steps):
            for v in range(6):
                eof=eofs[s,v];available,x=self.features(fresh,eof,s,v)
                fit=np.array([j for j,p in enumerate(available) if int(p) in training],dtype=int)
                if len(fit)<2:raise ValueError('registered Ridge fit cannot be estimated')
                target=eof.encode(self.r[available[fit],s,v])
                model=fit_ridge(x[fit],target,regularization)
                correction[available,l,v]=eof.decode(model.predict(x));fallback[available,l,v]=False
                name=f'{self.variables[v]}@{SHORT_AND_ENDPOINT_STEPS[s]*6}h'
                models[name]=model;counts[name]={'n_train':len(fit),'source_indices':self.indices[available[fit]].tolist(),'alpha':len(fit)*regularization}
        return correction,fallback,models,counts,eofs,lineage


def select_outputs(work,selection,output_dir):
    """Return fit-only scores and fixed choices; final refit is a separate call."""
    pos=np.array([p for p,r in enumerate(work.roles) if r=='select'],dtype=int)
    if not len(pos):raise ValueError('selection period missing')
    scores={};choices={};diagnostics={}
    for name in ('obc','dabc','ocl_fresh'):scores[name]={}
    for strength in selection['obc_lambda']:
        key=f'lambda={strength}';scores['obc'][key]=work.score(work.obc(strength),pos)[0]
    for tau in selection['tau_days']:
        for strength in selection['dabc_lambda']:
            key=f'tau={tau},lambda={strength}';cs,mask=work.dabc(tau,strength)
            scores['dabc'][key]=work.score(cs,pos)[0]
    for k in selection['ocl_eof_k']:
        eofs=work.transforms(k)
        for regularization in selection['ocl_ridge_lambda']:
            key=f'k={k},lambda={regularization}'
            cs,mask,models,counts,_,lineage=work.ocl(k,regularization,eofs=eofs)
            scores['ocl_fresh'][key]=work.score(cs,pos)[0]
            diagnostics[key]={'train':counts,'selection_fallback_cells':int(mask[pos].sum())}
    for name,values in scores.items():choices[name]=select_fit_config(values,list(values))
    return {'scores':scores,'choices':choices,'ocl_diagnostics':diagnostics,
            'selection_source_indices':work.indices[pos].tolist(),'fit_scope':'estimate; selection used only to choose grid entry',
            'interpretation':'selection losses are not held-out research evidence'}
