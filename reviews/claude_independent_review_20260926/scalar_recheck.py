"""Independent read-only recheck of FP-05b DEV scalars (no arrays, no model)."""
import json, collections, datetime as dt
import numpy as np
R='/mnt/afs/260010168/EarthDelta/plans/plan_v4_0923/run_20260924T104725Z_fp05b'
rows=json.load(open(f'{R}/cache/dev/candidate_results.json'))['rows']
bg=json.load(open(f'{R}/cache/dev/background_f0.json'))['issues']
pred=json.load(open(f'{R}/policies/oof_predictions.json'))['predictions']
cands=['reference','expert_0','expert_1','expert_2','expert_3']
L=collections.defaultdict(dict); T={}
for r in rows:
    L[(r['issue_id'],r['lead_hours'])][r['candidate_id']]=r['cpu_loss']; T[r['issue_id']]=r['issue_time']
ids=sorted(T,key=lambda i:T[i]); n=len(ids)
t0=dt.datetime(2019,7,6,12,tzinfo=dt.timezone.utc).timestamp()
blk=np.array([int((T[i]-t0)//(7*86400)) for i in ids])
leads=[6,24,72]
A={}  # lead -> [n, 6] columns: F0, Fs, e0..e3
for h in leads:
    A[h]=np.array([[bg[i]['f0_cpu'][str(h)]]+[L[(i,h)][c] for c in cands] for i in ids])
names=['F0','Fs','e0','e1','e2','e3']
print('n issues',n,'blocks',len(set(blk)))
for h in leads:
    s=A[h].sum(0)
    print(f'lead {h}h: mean L', dict(zip(names,np.round(A[h].mean(0),6))))
    print('   rel. loss change vs F0 (+ = better than F0) %:', {k: round(100*(s[0]-s[j])/s[0],4) for j,k in enumerate(names)})
    win=(A[h][:,2:]<A[h][:,[0]]).mean(0); print('   frac issues expert beats F0:',np.round(win,3), ' Fs beats F0:',round(float((A[h][:,1]<A[h][:,0]).mean()),3))
# policies
act={m:[pred[i][m] for i in ids] for m in ['M1','M2','M3']}
print('M1 actions',collections.Counter(act['M1']),'M3',collections.Counter(act['M3']),'M2',collections.Counter(act['M2']))
ci={c:j+1 for j,c in enumerate(cands)}
def pol(m,h): return np.array([A[h][k,ci[a]] for k,a in enumerate(act[m])])
orc24=A[24][:,1:].argmin(1)+1          # plan's oracle: 5 candidates, 24h
orcF0=A[24].argmin(1)                  # oracle incl. F0 as a candidate
rng=np.random.default_rng(0); ub=sorted(set(blk)); B=10000
idx_by_b={b:np.where(blk==b)[0] for b in ub}
draws=[np.concatenate([idx_by_b[b] for b in rng.choice(ub,len(ub))]) for _ in range(B)]
def boot(lb,lm):   # G = (sum Lb - sum Lm)/sum Lb, paired block bootstrap
    pt=(lb.sum()-lm.sum())/lb.sum(); g=np.array([(lb[d].sum()-lm[d].sum())/lb[d].sum() for d in draws])
    return 100*pt, 100*np.percentile(g,2.5), 100*np.percentile(g,97.5)
print('\n=== vs F0 (percent of F0 loss; + = better than raw Stormer) ===')
for h in leads:
    F0=A[h][:,0]
    rows_=[('Fs',A[h][:,1]),('M1',pol('M1',h)),('M3',pol('M3',h)),
           ('oracle5@24h',A[h][np.arange(n),orc24]),('oracle6(incl F0)@24h',A[h][np.arange(n),orcF0]),
           ('best-of-5 at THIS lead',A[h][:,1:].min(1))]
    for nm,v in rows_:
        p,lo,hi=boot(F0,v); print(f'  {h:>2}h {nm:<24} {p:+.4f}  [{lo:+.4f}, {hi:+.4f}]')
print('oracle6 picks F0 on', int((orcF0==0).sum()),'of',n,'issues (24h)')
# expert gain structure vs Fs at 24h
G=(A[24][:,[1]]-A[24][:,2:])   # gain vs Fs per expert
print('\nper-issue 24h gain vs Fs: mean',G.mean(0),' sd',G.std(0))
print('corr between experts gain vs Fs:\n',np.round(np.corrcoef(G.T),3))
D=G-G[:,[0]]
print('gain relative to expert_0: mean',D.mean(0)[1:],' sd',D.std(0)[1:])
print('scale: sd of per-issue F0 loss 24h',A[24][:,0].std(),' mean',A[24][:,0].mean())
# ratio L_c/L_Fs per issue
rat=A[24][:,2:]/A[24][:,[1]]; print('L_e/L_Fs 24h: min',rat.min(0),' max',rat.max(0))
rat0=A[24][:,1:]/A[24][:,[0]]; print('L_c/L_F0 24h (Fs,e0..e3) mean',rat0.mean(0),' min',rat0.min(0),' max',rat0.max(0))
