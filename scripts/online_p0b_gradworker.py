#!/usr/bin/env python3
"""Raw FP32 fit gradients; original-scale estimate-only mean in FP64."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from earthdelta.online.io import checked_json,sha256_file,write_json_once
from earthdelta.online.runtime import load_runtime,fit_cache_row
from earthdelta.online.grad_utils import model_guard,weight_gradient,target_metadata,serialize_gradient
from earthdelta.online.losses import normalized_step_mse
from earthdelta.online.roles import role,require_fit_rows
from earthdelta.online.timeindex import index_of
from earthdelta.online.output_fit import label_ids
from earthdelta.probe.rollout import load_bridge,rollout_trajectory,TARGET_BLOCKS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'BLOCKED','scope':'fit gradients only','scientific_support':False,'records':[]};start=time.perf_counter()
    try:
        cfg,contract,roster=load_runtime(args.config)
        output=checked_json(cfg['output_fit']['path'],cfg['output_fit']['sha256'])
        if output['status']!='OBSERVED':raise RuntimeError('T3A output fit gate is not complete')
        bank=checked_json(cfg['f0_bank']['path'],cfg['f0_bank']['sha256'])
        den=checked_json(bank['denominators']['path'],bank['denominators']['sha256']);require_fit_rows(den['source_indices'])
        d=np.asarray(den['short_mse'],dtype=np.float64)
        qc=checked_json(cfg['qc_receipt']['path'],cfg['qc_receipt']['sha256']);manifest=checked_json(qc['manifest']['path'],qc['manifest']['sha256'])
        spec=checked_json(cfg['old_spec']['path'],cfg['old_spec']['sha256']);binding=spec['bindings']
        coordinate=qc['coordinate_cache']
        if sha256_file(coordinate['path'])!=coordinate['sha256']:raise RuntimeError('coordinate identity changed')
        with np.load(coordinate['path'],allow_pickle=False) as a:lat=a['lat']
        from earthdelta.contracts import compute_normalization_asset_sha256
        if compute_normalization_asset_sha256(cfg['norm_dir'])!=binding['normalization']['npz_content_sha256']:raise RuntimeError('normalization changed')
        if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):raise RuntimeError('gradient worker requires H100')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.device('cuda');loaded=load_bridge(binding['checkpoint']['path'],cfg['norm_dir'],device)
        if loaded.checkpoint_sha256!=binding['checkpoint']['sha256']:raise RuntimeError('checkpoint changed')
        model,bridge=loaded.model,loaded.bridge;model.requires_grad_(False)
        channels=[v['index'] for v in contract['metrics']['variables']];variables=[v['name'] for v in contract['metrics']['variables']]
        mean=None;mean_indices=[];artifact_date=max(den['source_indices'])+20
        version=bank['denominators']['sha256']
        with model_guard(model):
            for slot in bank['slots']:
                if slot['status']=='NO_FORECAST':continue
                i=slot['index'];raw=fit_cache_row(manifest,i,out/'ACCESS_LEDGER.jsonl')
                truth=np.stack([fit_cache_row(manifest,i+k,out/'ACCESS_LEDGER.jsonl')[channels] for k in (1,2,3,4)])
                x=bridge.normalize(torch.from_numpy(np.array(raw,copy=True)[None]).to(device))
                y=torch.from_numpy(truth[None]).to(device)
                def objective():
                    trajectory=rollout_trajectory(bridge,x,steps=4,differentiable=True,checkpoint_steps=False)
                    physical=bridge.denormalize(trajectory[:,1:].reshape(-1,69,128,256)).reshape(1,4,69,128,256)
                    physical=physical.index_select(2,torch.tensor(channels,device=device))
                    return normalized_step_mse(physical,y,lat,d)
                torch.cuda.synchronize();t=time.perf_counter()
                g,loss=weight_gradient(model,TARGET_BLOCKS,objective,verify_digest=False)
                torch.cuda.synchronize();elapsed=time.perf_counter()-t
                reference_loss=float(np.mean(np.asarray(slot['mse'])[:,:4]/d.T))
                if not np.isclose(loss,reference_loss,rtol=5e-5,atol=1e-6):
                    raise RuntimeError('gradient forward objective disagrees with the cached F0 four-step metric')
                info=serialize_gradient(out/f'gradient_{i:04d}.npy',g)
                row={'issue_index':i,'role':role(i),'label_available':i+4,'artifact_available':max(i+4,artifact_date),'version':version,'raw_label_ids':sorted(label_ids(i,variables)),'gradient':info,'loss':loss,'reference_fp64_loss':reference_loss,'grad4_seconds':elapsed}
                receipt['records'].append(row);write_json_once(out/f'gradient_{i:04d}.json',row)
                if role(i)=='estimate':
                    array=g.detach().cpu().numpy().astype(np.float64)
                    if mean is None:mean=np.zeros_like(array)
                    mean+=array;mean_indices.append(i)
                print(json.dumps({'issue':i,'loss':loss,'norm':info['norm_fp64'],'grad4_seconds':elapsed}),flush=True)
                del g,x,y
        if mean_indices!=den['source_indices']:raise RuntimeError('mean gradient population differs from estimate denominator population')
        mean/=len(mean_indices);p=out/'RAW_MEAN_ESTIMATE_FP64.npy'
        with p.open('xb') as f:np.save(f,mean,allow_pickle=False)
        receipt.update(status='OBSERVED',mean={'path':str(p),'sha256':sha256_file(p),'dtype':'float64','source_indices':mean_indices,'artifact_available':artifact_date},target_metadata=target_metadata(model,TARGET_BLOCKS),denominators=bank['denominators'],gradient_count=len(receipt['records']),model_sha256=loaded.checkpoint_sha256,contract_sha256=cfg['contract']['sha256'])
    except Exception as exc:
        receipt['error']=f'{type(exc).__name__}: {exc}'
        import traceback;traceback.print_exc()
    receipt['wall_seconds']=time.perf_counter()-start
    write_json_once(out/'GRADIENT_RECEIPT.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='records'},indent=2))
    return 0 if receipt['status']=='OBSERVED' else 1


if __name__=='__main__':raise SystemExit(main())
