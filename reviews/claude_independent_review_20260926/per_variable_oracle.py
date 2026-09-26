import numpy as np
Z=np.load('/mnt/afs/260010168/earthdelta_independent_review_20260926/per_channel_mse.npz'); mse=Z['mse']; ch=list(Z['chans'])
key=['geopotential_500','temperature_850','2m_temperature','mean_sea_level_pressure','u_component_of_wind_850','specific_humidity_700','v_component_of_wind_50']
print('per-variable hindsight oracle (choose candidate per issue by THAT variable at THAT lead) -- % RMSE change vs F0, + = better')
for v in key:
    j=ch.index(v); s=f'{v:<26}'
    for l,h in enumerate([6,24,72]):
        base=mse[:,0,l,j].mean()
        o5=mse[:,1:,l,j].min(1).mean(); o6=mse[:,:,l,j].min(1).mean()
        s+=f' | {h}h oracle5 {100*(1-np.sqrt(o5/base)):+.2f}  oracle6 {100*(1-np.sqrt(o6/base)):+.2f}'
    print(s)
