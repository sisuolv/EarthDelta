"""Read-only per-channel breakdown of the sealed FP-05b DEV endpoints (2019H2 policy_dev, already scored).
Computes area-weighted MSE per (issue, candidate, lead, channel) for F0, Fs, e0..e3. No model, no new data."""
import json, sys, numpy as np, time
R='/mnt/afs/260010168/EarthDelta'
rows=json.load(open(f'{R}/plans/plan_v4_0923/run_20260924T104725Z_fp05b/cache/dev/candidate_results.json'))['rows']
ep={}; T={}
for r in rows: ep[r['issue_id']]=r['endpoint_file']; T[r['issue_id']]=r['issue_time']
ids=sorted(ep,key=lambda i:T[i])
lat=np.linspace(-89.296875,89.296875,128); q=np.cos(np.deg2rad(lat)); q=q/q.mean()
std=np.load(f'{R}/reference/stormer/normalization_constants/normalize_std.npz')
chans=[str(c) for c in __import__('zarr').open(f'{R}/data/era5_1p40625/2019.zarr','r')['channel'][:]]
s=np.array([float(std[c]) for c in chans])
out=np.zeros((len(ids),6,3,69)); t=time.time()
for k,i in enumerate(ids):
    d=ep[i].rsplit('/',1)[0]
    E=np.load(f'{d}/endpoints.npy',mmap_mode='r'); F=np.load(f'{d}/f0_endpoints.npy',mmap_mode='r'); Y=np.load(f'{d}/truth.npy',mmap_mode='r')
    for l in range(3):
        y=np.asarray(Y[l],dtype=np.float64)
        for c in range(6):
            x=np.asarray(F[l] if c==0 else E[c-1,l],dtype=np.float64)
            out[k,c,l]=((x-y)**2*q[None,:,None]).mean(axis=(1,2))
    if k%10==0: print(k,round(time.time()-t,1),flush=True)
np.savez('/mnt/afs/260010168/earthdelta_independent_review_20260926/per_channel_mse.npz',mse=out,ids=np.array(ids),chans=np.array(chans),std=s)
print('done',time.time()-t)
