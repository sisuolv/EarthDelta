import numpy as np, json, datetime as dt, collections
Z=np.load('/mnt/afs/260010168/earthdelta_independent_review_20260926/per_channel_mse.npz'); mse=Z['mse']; ids=list(Z['ids']); ch=list(Z['chans']); s=Z['std']
R='/mnt/afs/260010168/EarthDelta/plans/plan_v4_0923/run_20260924T104725Z_fp05b'
rows=json.load(open(f'{R}/cache/dev/candidate_results.json'))['rows']
bg=json.load(open(f'{R}/cache/dev/background_f0.json'))['issues']
pred=json.load(open(f'{R}/policies/oof_predictions.json'))['predictions']
cl={}; T={}
for r in rows: cl[(r['issue_id'],r['candidate_id'],r['lead_hours'])]=r['cpu_loss']; T[r['issue_id']]=r['issue_time']
cands=['reference','expert_0','expert_1','expert_2','expert_3']; names=['F0','Fs','e0','e1','e2','e3']
leads=[6,24,72]
# 1) validate: objective = mean_c mse/std^2 must equal stored cpu_loss
obj=(mse/(s**2)[None,None,None,:]).mean(-1)
err=[]
for k,i in enumerate(ids):
    for l,h in enumerate(leads):
        err.append(abs(obj[k,0,l]-bg[i]['f0_cpu'][str(h)])/bg[i]['f0_cpu'][str(h)])
        for c in range(5): err.append(abs(obj[k,c+1,l]-cl[(i,cands[c],h)])/cl[(i,cands[c],h)])
print('max rel diff of my objective vs stored cpu_loss:',max(err))
t0=dt.datetime(2019,7,6,12,tzinfo=dt.timezone.utc).timestamp()
blk=np.array([int((T[i]-t0)//(7*86400)) for i in ids]); ub=sorted(set(blk)); byb={b:np.where(blk==b)[0] for b in ub}
rng=np.random.default_rng(1); draws=[np.concatenate([byb[b] for b in rng.choice(ub,len(ub))]) for _ in range(4000)]
m1=[cands.index(pred[i]['M1'])+1 for i in ids]; m3=[cands.index(pred[i]['M3'])+1 for i in ids]
key=['geopotential_500','temperature_850','2m_temperature','mean_sea_level_pressure','u_component_of_wind_850','10m_u_component_of_wind','specific_humidity_700','temperature_500','geopotential_50','specific_humidity_50','u_component_of_wind_50','temperature_50']
def rmse_change(sel_idx, c_of_issue, j, l):
    # % change of RMSE (sqrt of mean MSE over issues) vs F0; + = better
    base=mse[sel_idx,0,l,j]; alt=np.array([mse[k,c_of_issue[k],l,j] for k in sel_idx])
    return 100*(1-np.sqrt(alt.mean()/base.mean()))
def ci(c_of_issue,j,l):
    pt=rmse_change(np.arange(len(ids)),c_of_issue,j,l)
    bs=[rmse_change(d,c_of_issue,j,l) for d in draws[:1000]]
    return pt,np.percentile(bs,2.5),np.percentile(bs,97.5)
Fs=[1]*len(ids)
print('\nRMSE of F0 (physical units) and % RMSE change vs F0 (+ = better), paired 7-day block bootstrap 95% CI')
for v in key:
    j=ch.index(v)
    line=f'{v:<26}'
    for l,h in enumerate(leads):
        r0=np.sqrt(mse[:,0,l,j].mean())
        a=ci(Fs,j,l); b=ci(m1,j,l)
        line+=f' | {h}h F0 {r0:9.4g}  Fs {a[0]:+.2f}[{a[1]:+.2f},{a[2]:+.2f}] M1 {b[0]:+.2f}[{b[1]:+.2f},{b[2]:+.2f}]'
    print(line)
# 2) which channels drive the aggregate objective difference (M1 vs F0, 24h / 72h)
for l,h in enumerate(leads):
    d=np.array([(mse[k,0,l]-mse[k,m1[k],l])/s**2 for k in range(len(ids))]).mean(0)/69   # contribution to objective gain
    tot=d.sum(); order=np.argsort(-np.abs(d))
    print(f'\n{h}h objective gain M1 vs F0 = {tot:.3e} (rel {100*tot/obj[:,0,l].mean():+.4f}%); top channel contributions:')
    for j in order[:8]: print(f'   {ch[j]:<28} {d[j]:+.3e}  ({100*d[j]/abs(tot) if tot else 0:+.0f}% of |total|)')
    # fraction of channels improved
    rel=np.array([mse[:,m1[k] if False else 0,l,j].mean() for j in range(69)])
    impr=np.array([(mse[:,0,l,j].mean()-np.mean([mse[k,m1[k],l,j] for k in range(len(ids))]))>0 for j in range(69)])
    print(f'   channels where M1 RMSE < F0 RMSE: {impr.sum()}/69')
# 3) share of objective from each variable group for F0
grp=collections.defaultdict(float)
for j,c in enumerate(ch): grp[c.rsplit('_',1)[0] if c[-1].isdigit() and not c.startswith(('2m','10m')) else c]+= (mse[:,0,1,j]/s[j]**2).mean()/69
print('\nF0 24h objective share by variable group:',{k:round(v/obj[:,0,1].mean(),3) for k,v in grp.items()})
