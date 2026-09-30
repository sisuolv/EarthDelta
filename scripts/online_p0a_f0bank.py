#!/usr/bin/env python3
"""Build the authorized fit-only F0 bank; no analysis mode before its gates."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from earthdelta.online.io import checked_json,sha256_file,write_json_once
from earthdelta.online.runtime import load_runtime,fit_cache_row
from earthdelta.online.bank import SHORT_AND_ENDPOINT_STEPS,physical_cells,write_prediction,fit_denominators
from earthdelta.online.roles import role
from earthdelta.online.grad_utils import model_guard
from earthdelta.online.cost import projected_card_hours
from earthdelta.probe.rollout import load_bridge,rollout_trajectory


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True);ap.add_argument('--phase',choices=['fit'],required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'BLOCKED','phase':args.phase,'scientific_support':False,'slots':[]};start=time.perf_counter()
    try:
        cfg,contract,roster=load_runtime(args.config)
        engine=checked_json(cfg['engine_receipt']['path'],cfg['engine_receipt']['sha256'])
        if engine['status']!='OBSERVED' or not engine.get('engineering_pass'):raise RuntimeError('numerical engine gate missing')
        projection=projected_card_hours(engine['p95'],contract,charged_fit_hours=cfg['preflight_charged_fit_upper_bound'])
        receipt['cost_admission']=projection
        if not projection['admitted']:raise RuntimeError('measured budget gate did not admit fit')
        qc=checked_json(cfg['qc_receipt']['path'],cfg['qc_receipt']['sha256'])
        if qc['status']!='OBSERVED':raise RuntimeError('QC gate missing')
        manifest=checked_json(qc['manifest']['path'],qc['manifest']['sha256'])
        oldspec=checked_json(cfg['old_spec']['path'],cfg['old_spec']['sha256']);binding=oldspec['bindings']
        from earthdelta.contracts import compute_normalization_asset_sha256
        if compute_normalization_asset_sha256(cfg['norm_dir'])!=binding['normalization']['npz_content_sha256']:raise RuntimeError('normalization identity changed')
        coord=qc['coordinate_cache']
        if sha256_file(coord['path'])!=coord['sha256']:raise RuntimeError('coordinate cache changed')
        with np.load(coord['path'],allow_pickle=False) as a:lat=a['lat']
        if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):raise RuntimeError('only qualified H100 engine admitted')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.device('cuda');loaded=load_bridge(binding['checkpoint']['path'],cfg['norm_dir'],device)
        if loaded.checkpoint_sha256!=binding['checkpoint']['sha256']:raise RuntimeError('checkpoint changed')
        model,bridge=loaded.model,loaded.bridge;model.requires_grad_(False)
        channels=[v['index'] for v in contract['metrics']['variables']];steps=list(SHORT_AND_ENDPOINT_STEPS)
        dates=roster['dates'];from earthdelta.online.timeindex import index_of
        issues=[index_of(s) for r in ('estimate','gap','select') for s in dates[r]]
        if len(issues)!=86:raise RuntimeError('fit roster changed')
        calibration={index_of(s) for s in contract['arms']['calibration']['issues']}
        losses=[];valid=[];roles=[]
        with model_guard(model):
            for i in issues:
                item={'index':i,'role':role(i)};raw=fit_cache_row(manifest,i,out/'ACCESS_LEDGER.jsonl')
                if raw is None:
                    item.update(status='NO_FORECAST',reason='current_input_qc');receipt['slots'].append(item)
                    write_json_once(out/f'slot_{i:04d}.json',item);continue
                x=torch.from_numpy(np.array(raw,copy=True)[None]).to(device);x=bridge.normalize(x)
                torch.cuda.synchronize();t=time.perf_counter()
                with torch.no_grad():
                    trajectory=rollout_trajectory(bridge,x,steps=20)
                    pred=bridge.denormalize(trajectory[:,steps].reshape(-1,69,128,256)).reshape(1,6,69,128,256)[0]
                    pred=pred.index_select(1,torch.tensor(channels,device=device))
                torch.cuda.synchronize();seconds=time.perf_counter()-t
                prediction=pred.detach().cpu().numpy()
                full=trajectory[0,4].detach().cpu().numpy() if i in calibration else None
                info=write_prediction(out/f'prediction_{i:04d}.npz',prediction,norm69_step4=full)
                # Truth is obtained after F0 publication, only by this scorer side.
                truths=[]
                for k in steps:
                    row=fit_cache_row(manifest,i+k,out/'ACCESS_LEDGER.jsonl')
                    if row is None:raise RuntimeError('unexpected missing fit truth; keep slot and fail closed')
                    truths.append(row[channels])
                mse=physical_cells(prediction,np.stack(truths),lat)
                item.update(status='PUBLISHED',prediction=info,mse=mse.tolist(),roll20_seconds=seconds)
                receipt['slots'].append(item);losses.append(mse);valid.append(i);roles.append(role(i))
                write_json_once(out/f'slot_{i:04d}.json',item)
                print(json.dumps({'issue':i,'role':role(i),'roll20_seconds':seconds}),flush=True)
                del trajectory,pred,x
        denominator=fit_denominators(losses,valid,roles)
        write_json_once(out/'DENOMINATORS.json',denominator)
        receipt.update(status='OBSERVED',denominators={'path':str(out/'DENOMINATORS.json'),'sha256':sha256_file(out/'DENOMINATORS.json')},published=len(valid),nominal=len(issues),model_sha256=loaded.checkpoint_sha256,normalization_sha256=binding['normalization']['npz_content_sha256'],contract_sha256=cfg['contract']['sha256'],variables=contract['metrics']['variables'],lead_steps=steps,coordinate_cache=coord)
    except Exception as exc:
        receipt['error']=f'{type(exc).__name__}: {exc}'
        import traceback;traceback.print_exc()
    receipt['wall_seconds']=time.perf_counter()-start
    write_json_once(out/'F0_BANK_RECEIPT.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='slots'},indent=2))
    return 0 if receipt['status']=='OBSERVED' else 1


if __name__=='__main__':raise SystemExit(main())
