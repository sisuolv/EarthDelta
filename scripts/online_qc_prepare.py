#!/usr/bin/env python3
"""Only the preregistered 17 new rows and 8 content-overlap checks on ACP CPU."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from earthdelta.online.io import checked_json,sha256_file,decoded_sha256,write_json_once
from earthdelta.online.qc_reuse import verify_qc_sources,compose_index_manifest
from earthdelta.online.runtime import load_runtime,ledger


def inspect_row(x,index,mean,std,shape,expected_epoch):
    if x.shape!=tuple(shape) or x.dtype!=np.float32:
        raise ValueError('physical state shape/dtype mismatch')
    if mean.shape!=(shape[0],) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('normalization does not bind these channels')
    finite=bool(np.isfinite(x).all())
    z=np.abs((x.astype(np.float64)-mean[:,None,None])/std[:,None,None])
    arg=int(np.argmax(z)) if finite else None
    return {'index':index,'finite':finite,'max_abs_z':float(z.flat[arg]) if finite else None,
            'argmax_flat':arg,'decoded_sha256':decoded_sha256(x),'expected_epoch':expected_epoch}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'BLOCKED','array_scope':'2020 declared 17+8 only','science_support':False}
    start=time.perf_counter()
    try:
        cfg,contract,roster=load_runtime(args.config)
        old=cfg['old_qc'];oldspec=cfg['old_spec']
        spec=checked_json(oldspec['path'],oldspec['sha256']);binding=spec['bindings']
        store=contract['data']['store']
        if Path(store).resolve()!=Path(binding['data_store']['path']).resolve():raise ValueError('store mismatch')
        meta={k[:-7]:v for k,v in binding['data_store'].items() if k.endswith('_sha256')}
        qc,_=verify_qc_sources(old['path'],old['sha256'],oldspec['path'],oldspec['sha256'],store=store,metadata_bindings=meta)
        from earthdelta.contracts import compute_normalization_asset_sha256
        normhash=compute_normalization_asset_sha256(cfg['norm_dir'])
        if normhash!=binding['normalization']['npz_content_sha256']:raise ValueError('normalization asset identity mismatch')
        from earthdelta.bridge import NormalizationContract,DEFAULT_VARIABLES,POLICY_OFFICIAL_ZERO_DIFF_MEAN
        norm=NormalizationContract.from_npz_dir(cfg['norm_dir'],variables=list(DEFAULT_VARIABLES),intervals=(6,),policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN)
        channels=qc['metadata']['channels']
        if list(DEFAULT_VARIABLES)!=channels:raise ValueError('bridge/QC channel order mismatch')
        for var in contract['metrics']['variables']:
            if channels[var['index']]!=var['channel']:raise ValueError('registered variable index mismatch')
        # Old QC used the original NPZ precision, not the public FP32 bridge tensors.
        with np.load(Path(cfg['norm_dir'])/'normalize_mean.npz',allow_pickle=False) as means:
            mean=np.asarray([means[name].reshape(-1)[0] for name in channels],dtype=np.float64)
        with np.load(Path(cfg['norm_dir'])/'normalize_std.npz',allow_pickle=False) as stds:
            std=np.asarray([stds[name].reshape(-1)[0] for name in channels],dtype=np.float64)
        c=qc['coordinate_cache'];cp=Path(c['path'])
        if sha256_file(cp)!=c['sha256']:raise ValueError('coordinate cache byte identity mismatch')
        with np.load(cp,allow_pickle=False) as archive:coords={k:archive[k] for k in archive.files}
        import zarr
        group=zarr.open_group(store,mode='r')
        for name in ('lat','lon','time'):
            ledger(out/'ACCESS_LEDGER.jsonl',{'stage':'qc','coordinate':name,'store':store})
            actual=np.asarray(group[name][:])
            if name in ('lat','lon'):actual=actual.astype(np.float64)
            if decoded_sha256(actual)!=qc['metadata'][name+'_sha256'] or not np.array_equal(actual,coords[name]):
                raise ValueError('current coordinate differs from bound QC')
        if group['data'].shape!=tuple(qc['metadata']['shape']):raise ValueError('data shape identity mismatch')
        ledger(out/'ACCESS_LEDGER.jsonl',{'stage':'qc','coordinate':'channel','store':store})
        source_channels=[x.decode() if isinstance(x,bytes) else str(x) for x in group['channel'][:]]
        if source_channels!=channels or source_channels!=list(coords['channels']):raise ValueError('source channel metadata mismatch')
        if group['time'].attrs.get('units')!='hours since 2020-01-01 00:00:00' or group['time'].attrs.get('calendar')!='proleptic_gregorian':raise ValueError('time coordinate units/calendar mismatch')
        new_indices=roster['pending_qc_indices'];overlap=roster['overlap_checks']
        if len(new_indices)!=17 or len(overlap)!=8 or set(new_indices)&set(overlap):raise ValueError('QC scope changed')
        rows={};(out/'cache').mkdir()
        for i in sorted(new_indices+overlap):
            ledger(out/'ACCESS_LEDGER.jsonl',{'stage':'qc','store':store,'indices':[i],'purpose':'new_qc' if i in new_indices else 'overlap_content_check'})
            x=np.asarray(group['data'][i])
            expected=1577836800+i*21600
            if float(coords['time'][i])!=i*6:raise ValueError('time grid identity mismatch')
            row=inspect_row(x,i,mean,std,qc['metadata']['shape'][1:],expected)
            p=out/'cache'/f'index_{i:04d}.npy'
            with p.open('xb') as f:np.save(f,x,allow_pickle=False)
            rows[i]={'qc':row,'cache':{'path':str(p),'sha256':sha256_file(p),'shape':list(x.shape),'dtype':str(x.dtype)}}
            write_json_once(out/f'index_{i:04d}.json',rows[i])
        manifest=compose_index_manifest(qc,rows,support=range(roster['nominal_support_bounds'][0],roster['nominal_support_bounds'][1]+1),missing=roster['missing_indices_in_support'],overlap=overlap,normalization_identity=normhash,z_threshold=contract['data']['qc_z_threshold'])
        rejected=[int(i) for i,r in manifest['indices'].items() if not r['eligible']]
        unexpected=set(rejected)-{712}
        write_json_once(out/'INDEX_MANIFEST.json',manifest)
        receipt.update(status='BLOCKED' if unexpected else 'OBSERVED',unexpected_rejections=sorted(unexpected),rejected_indices=rejected,new_count=len(new_indices),overlap_count=len(overlap),reused_count=len(manifest['indices'])-len(new_indices),normalization_sha256=normhash,coordinate_cache=c,manifest={'path':str(out/'INDEX_MANIFEST.json'),'sha256':sha256_file(out/'INDEX_MANIFEST.json')},new_rows={str(i):rows[i]['qc'] for i in new_indices})
    except Exception as exc:
        receipt['error']=f'{type(exc).__name__}: {exc}'
    receipt['wall_seconds']=time.perf_counter()-start
    write_json_once(out/'QC_RECEIPT.json',receipt)
    print(json.dumps(receipt,indent=2,allow_nan=False))
    return 0 if receipt['status']=='OBSERVED' else 1


if __name__=='__main__':raise SystemExit(main())
