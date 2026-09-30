"""Finite-sample coverage qualification; all generators are synthetic.

The optimized bootstrap uses exactly the same circular indices and pooled
statistics as metrics.paired_intervals. It never treats simulation success
as coverage certification for the unknown weather-error process.
"""
from itertools import product
import math
import numpy as np
from .metrics import circular_indices, weekly_sufficient_stats, _improvement
from .timeindex import index_of

ARMS = ('f0','obc','dabc','ocl_fresh','lag1c','lag1','static','ewma_g','delay30')


def cell_names(contract):
    return [f"{v['name']}@{6*k}h" for v in contract['metrics']['variables']
            for k in contract['metrics']['lead_steps']]


def scoring_mask(roster, manifest, contract):
    """Uses QC metadata only, never opens analysis inputs/truth arrays."""
    indices=np.array([index_of(t) for t in roster['dates']['analysis']],dtype=np.int64)
    eligible=lambda i: manifest['indices'].get(str(int(i)),{}).get('eligible',False)
    mask=np.array([[eligible(i) and eligible(i+k) for _ in contract['metrics']['variables']
                   for k in contract['metrics']['lead_steps']] for i in indices],dtype=bool)
    return indices,mask


def fit_rho(bank, denominator, roster):
    """Estimate dependence before simulations, from estimate F0 cells only.

    Use the daily mean of 24 MSE/D_est cells; adjacent calendar pairs only.
    Short winter fit is a scale/dependence proxy, not evidence on analysis.
    """
    allowed={index_of(t) for t in roster['dates']['estimate']}
    d=np.asarray(denominator['selection_mse'],dtype=np.float64).T
    rows={r['index']:float(np.mean(np.asarray(r['mse'])[:,[0,3,4,5]]/d))
          for r in bank['slots'] if r['status']=='PUBLISHED' and r['index'] in allowed}
    pairs=[(rows[i],rows[i+4]) for i in sorted(rows) if i+4 in rows]
    if len(pairs)<20:raise ValueError('too few adjacent estimate pairs for dependence proxy')
    a=np.asarray(pairs);rho=float(np.corrcoef(a.T)[0,1])
    if not np.isfinite(rho):raise ValueError('undefined fit autocorrelation')
    return {'raw':rho,'clipped':float(np.clip(rho,0,.95)),
            'source_indices':sorted(rows),'adjacent_pairs':len(pairs),
            'method':'lag-1 correlation of daily mean 24 normalized F0 MSE cells; estimate only',
            'limitation':'not an estimate of all paired-arm influence dependencies'}


def scenarios(contract, estimated_rho):
    cfg=contract['inference']['coverage_calibration'];out=[]
    for r,innovation,drift,scale in product(cfg['daily_loss_rho'],cfg['innovations'],
                                          (False,True),('equal','ordered')):
        out.append({'id':len(out),'rho':estimated_rho if isinstance(r,str) else r,
                    'rho_source':r,'innovation':innovation,'drift':drift,'scale':scale})
    return out


def synthetic_mse(scenario, seed, indices, denominators, *, correlation=.5,burnin=4096):
    """Unit-variance AR with exact finite 4096-step burn-in reduction.

    The weighted sum is algebraically the same AR recursion for burn-in;
    all 4096 innovations are generated. Multivariate t uses a common chi
    scale for the entire arm x cell vector at each time.
    """
    rng=np.random.default_rng(seed);n=len(indices);d=np.asarray(denominators)
    if d.ndim!=1 or np.any(d<=0) or not np.isfinite(d).all():raise ValueError('invalid D_est')
    dim=len(ARMS)*len(d);phi=math.sqrt(scenario['rho'])
    if not 0<=phi<1 or not 0<=correlation<1:raise ValueError('invalid synthetic dependence')
    z0=math.sqrt(correlation)*rng.normal()+math.sqrt(1-correlation)*rng.normal(size=dim)
    e=math.sqrt(correlation)*rng.normal(size=(burnin+n,1))+math.sqrt(1-correlation)*rng.normal(size=(burnin+n,dim))
    if scenario['innovation']=='student_t_df5':e*=np.sqrt(3/rng.chisquare(5,size=(burnin+n,1)))
    elif scenario['innovation']!='gaussian':raise ValueError('unregistered innovation')
    weights=math.sqrt(1-phi*phi)*phi**np.arange(burnin-1,-1,-1,dtype=np.float64)
    z=phi**burnin*z0+weights@e[:burnin]
    values=np.empty((n,len(ARMS),len(d)),dtype=np.float64)
    for t in range(n):
        z=phi*z+math.sqrt(1-phi*phi)*e[burnin+t]
        values[t]=z.reshape(len(ARMS),len(d))
    day=np.asarray(indices)/4+1
    drift=.5*np.sin(2*np.pi*day/365) if scenario['drift'] else np.zeros(n)
    scale=np.array([1,1.02,1,.98,.96,.97,1.01,.99,1.03]) if scenario['scale']=='ordered' else np.ones(len(ARMS))
    mse=d[None,None,:]*scale[None,:,None]*(.05+(values+drift[:,None,None])**2)
    truth=d[None,None,:]*scale[None,:,None]*(1.05+drift[:,None,None]**2)
    return mse,np.broadcast_to(truth,mse.shape)


class BootstrapPlan:
    def __init__(self,indices,mask,contract,draws):
        self.indices=np.asarray(indices);self.mask=np.asarray(mask);self.contract=contract
        self.cells=cell_names(contract);self.families=contract['inference']['families']
        # v9.2 stores the ISO anchor followed by explanatory prose.
        anchor=contract['inference']['week_anchor'].split()[0]
        s=weekly_sufficient_stats(np.ones((len(indices),len(ARMS),len(self.cells))),mask,indices,
                                  anchor=anchor)
        ids=circular_indices(len(s.sums),contract['inference']['block_days_primary']//7,draws,
                             seed=contract['inference']['seed'])
        self.weights=np.zeros((draws,len(s.sums)),dtype=np.float64)
        np.add.at(self.weights,(np.arange(draws)[:,None],ids),1)
        self.counts=self.weights@s.counts
        if np.any(self.counts<=0):raise ValueError('undefined bootstrap denominator')
        self.week=(self.indices-index_of(anchor))//28-s.first_week
        self.nweek=len(s.sums)
        self.members=[(name,row) for name,rows in self.families.items() for row in rows]
        self.a=np.array([ARMS.index(row['a']) for name,row in self.members])
        self.b=np.array([ARMS.index(row['b']) for name,row in self.members])
        self.c=np.array([self.cells.index(row['cell']) for name,row in self.members])

    def contrasts(self,improvement):
        return improvement[...,self.a,self.c]-improvement[...,self.b,self.c]

    def ratios(self,mse):
        m=np.asarray(mse,dtype=np.float64)
        if m.shape!=(len(self.indices),len(ARMS),len(self.cells)) or not np.isfinite(m).all() or np.any(m<0):
            raise ValueError('invalid synthetic MSE layout')
        sums=np.zeros((self.nweek,len(ARMS),len(self.cells)),dtype=np.float64)
        np.add.at(sums,self.week,np.where(self.mask[:,None,:],m,0))
        total=sums.sum(0)
        center=np.sqrt(total/total[0:1])
        pooled=(self.weights@sums.reshape(self.nweek,-1)).reshape(-1,len(ARMS),len(self.cells))
        if np.any(pooled[:,0,:]<=0):raise ValueError('nonpositive F0 bootstrap denominator')
        samples=np.sqrt(pooled/pooled[:,0:1,:])
        return center,samples

    def bounds(self,mse):
        center,ratios=self.ratios(mse)
        center=self.contrasts(100*(1-center))
        samples=self.contrasts(100*(1-ratios))
        lower=np.empty(len(self.members));upper=lower.copy();offset=0
        for name,members in self.families.items():
            size=len(members);tail=self.contract['inference']['family_alpha']/(2*size)
            lower[offset:offset+size],upper[offset:offset+size]=np.quantile(samples[:,offset:offset+size],[tail,1-tail],axis=0)
            offset+=size
        return center,lower,upper

    def truth(self,expected_mse):
        sums=np.where(self.mask[:,None,:],expected_mse,0).sum(0)
        return self.contrasts(_improvement(sums,self.mask.sum(0),0))


def wilson(successes,n):
    if n<=0 or successes<0 or successes>n:raise ValueError('invalid Monte Carlo count')
    z=1.959963984540054;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def coverage_rows(theta,lower,upper,truth,contract):
    """All factors retained; no changing factor after validation failure."""
    result=[];offset=0
    for name,members in contract['inference']['families'].items():
        stop=offset+len(members);sl=slice(offset,stop)
        for factor in contract['inference']['width_factors']:
            lo=theta[:,sl]-factor*np.maximum(theta[:,sl]-lower[:,sl],0)
            hi=theta[:,sl]+factor*np.maximum(upper[:,sl]-theta[:,sl],0)
            covered=np.all((lo<=truth[:,sl])&(truth[:,sl]<=hi),axis=1)
            count=int(covered.sum());n=len(covered);ci=wilson(count,n)
            result.append({'family':name,'factor':factor,'mc_n':n,'successes':count,
                'coverage':count/n,'wilson95':ci,'pass':count/n>=.95 and ci[0]>=.93,
                'mean_interval_width_pp':float(np.mean(hi-lo)),
                'mc_se':math.sqrt((count/n)*(1-count/n)/n)})
        offset=stop
    return result
