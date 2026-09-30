#!/usr/bin/env python3
"""Common-date response calibration and fit-only parameter selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from earthdelta.online.io import checked_json,sha256_file,write_json_once
from earthdelta.online.runtime import load_runtime,fit_cache_row,ledger
from earthdelta.online.clock import SupervisionRecord,Derived
from earthdelta.online.directions import direction,raw_gradient_mean
from earthdelta.online.grad_utils import model_guard,safe_weight_editor,split_direction,calibrate_amplitude,response_effect,target_metadata
from earthdelta.online.losses import cells
from earthdelta.online.outcorr import select_fit_config
from earthdelta.online.timeindex import index_of
from earthdelta.probe.rollout import load_bridge,rollout_trajectory,TARGET_BLOCKS


def load_gradients(receipt):
    records=[]
    for row in receipt['records']:
        info=row['gradient'];p=Path(info['path'])
        if sha256_file(p)!=info['sha256']:raise RuntimeError('gradient cache changed')
        value=np.load(p,allow_pickle=False)
        if value.dtype!=np.float32 or value.shape!=(info['numel'],):raise RuntimeError('raw FP32 full gradient contract violated')
        records.append(SupervisionRecord(f"g:{row['issue_index']}",row['issue_index'],row['label_available'],row['artifact_available'],row['version'],frozenset(row['raw_label_ids']),value))
    info=receipt['mean'];p=Path(info['path'])
    if sha256_file(p)!=info['sha256']:raise RuntimeError('estimate mean changed')
    mean=np.load(p,allow_pickle=False)
    if mean.dtype!=np.float64:raise RuntimeError('estimate mean must be FP64')
    estimate=[r for r in records if r.issue_index in info['source_indices']]
    if [r.issue_index for r in estimate]!=info['source_indices']:raise RuntimeError('estimate mean source roster differs')
    np.testing.assert_array_equal(raw_gradient_mean(estimate),mean,err_msg='raw estimate gradient mean arithmetic mismatch')
    provenance=Derived(info['sha256'],info['artifact_available'],('estimate',),frozenset().union(*(r.lineage for r in records if r.issue_index in info['source_indices'])))
    return records,mean,provenance


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    result={'status':'BLOCKED','scope':'fit calibration/selection only','scientific_support':False,'calibration':{},'selection':[]};start=time.perf_counter()
    try:
        cfg,contract,roster=load_runtime(args.config)
        grad=checked_json(cfg['gradients']['path'],cfg['gradients']['sha256'])
        if grad['status']!='OBSERVED' or grad['contract_sha256']!=cfg['contract']['sha256']:raise RuntimeError('gradient receipt not bound')
        bank=checked_json(cfg['f0_bank']['path'],cfg['f0_bank']['sha256']);slots={s['index']:s for s in bank['slots'] if s['status']=='PUBLISHED'}
        den=checked_json(bank['denominators']['path'],bank['denominators']['sha256']);d=np.asarray(den['selection_mse']).T
        if grad['denominators']!=bank['denominators']:raise RuntimeError('gradient/scoring denominator versions differ')
        qc=checked_json(cfg['qc_receipt']['path'],cfg['qc_receipt']['sha256']);manifest=checked_json(qc['manifest']['path'],qc['manifest']['sha256'])
        spec=checked_json(cfg['old_spec']['path'],cfg['old_spec']['sha256']);binding=spec['bindings']
        coordinate=qc['coordinate_cache']
        if sha256_file(coordinate['path'])!=coordinate['sha256']:raise RuntimeError('coordinate cache changed')
        with np.load(coordinate['path'],allow_pickle=False) as a:lat=a['lat']
        from earthdelta.contracts import compute_normalization_asset_sha256
        if compute_normalization_asset_sha256(cfg['norm_dir'])!=binding['normalization']['npz_content_sha256']:raise RuntimeError('normalization changed')
        if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):raise RuntimeError('qualified H100 required')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.device('cuda');loaded=load_bridge(binding['checkpoint']['path'],cfg['norm_dir'],device)
        if loaded.checkpoint_sha256!=binding['checkpoint']['sha256']:raise RuntimeError('checkpoint changed')
        model,bridge=loaded.model,loaded.bridge;model.requires_grad_(False)
        if json.loads(json.dumps(target_metadata(model,TARGET_BLOCKS)))!=grad['target_metadata']:
            raise RuntimeError('gradient target matrices differ from the admitted model')
        records,mean,provenance=load_gradients(grad)
        norm=float(torch.sqrt(sum(model.blocks[b].attn.proj.weight.detach().double().square().sum() for b in TARGET_BLOCKS)).cpu())
        cal=contract['arms']['calibration'];selection=contract['arms']['selection']
        issues=[index_of(s) for s in cal['issues']]
        variants=[(n,None) for n in ('lag1c','lag1','static','delay30')]+[('ewma_g',t) for t in selection['tau_days']]
        names=[n if tau is None else f'{n}_tau{tau}' for n,tau in variants]
        result['registered_variants']=names
        cached={}
        channels=[v['index'] for v in contract['metrics']['variables']]
        with model_guard(model):
            for i in issues:
                if i not in slots:raise RuntimeError('registered calibration issue missing')
                rows=[fit_cache_row(manifest,i+k,out/'ACCESS_LEDGER.jsonl') for k in range(5)]
                if any(x is None for x in rows):raise RuntimeError('calibration truth incomplete')
                raw=torch.from_numpy(np.stack(rows)).to(device)
                truth=bridge.normalize(raw).unsqueeze(0);x=truth[:,0]
                with torch.no_grad():f0=rollout_trajectory(bridge,x,steps=4)
                info=slots[i]['prediction']
                if sha256_file(info['path'])!=info['sha256']:raise RuntimeError('F0 prediction changed')
                with np.load(info['path'],allow_pickle=False) as a:saved=a['norm69_step4']
                if not np.array_equal(f0[0,4].cpu().numpy(),saved):raise RuntimeError('calibration F0 differs from immutable bank')
                cached[i]=(x,truth,f0)
            for (family,tau),name in zip(variants,names):
                result['active_variant']=name
                ds=[direction(family,i,records,mean,mean_provenance=provenance,tau_days=tau,offline_fit=True,artifact_context=provenance.artifact_available) for i in issues]
                ds=[None if x is None else torch.from_numpy(x.copy()).float().to(device) for x in ds]
                def evaluate(amplitude):
                    ledger(out/f'calibration_{name}.jsonl',{'event':'begin_evaluation','amplitude':float(amplitude),'indices':issues})
                    responses=[]
                    for i,v in zip(issues,ds):
                        if v is None:responses.append(0.);continue
                        x,truth,f0=cached[i]
                        with torch.no_grad(),safe_weight_editor(model,split_direction(model,TARGET_BLOCKS,v,amplitude),blocks=TARGET_BLOCKS,verify_digest=False):
                            edited=rollout_trajectory(bridge,x,steps=4)
                        try:responses.append(response_effect(edited,f0,truth,lat,step=cal['step']))
                        except Exception as exc:raise RuntimeError(f'calibration {name}, issue {i}, amplitude {amplitude}: {exc}') from exc
                    ledger(out/f'calibration_{name}.jsonl',{'amplitude':float(amplitude),'indices':issues,'responses':responses})
                    return responses
                print(json.dumps({'calibration_started':name}),flush=True)
                fit=calibrate_amplitude(evaluate,norm,n_issues=len(issues),target=cal['response_target'],bracket=cal['bracket_relative_weight_norm'],iterations=cal['bisection_iterations'],relative_tolerance=cal['response_tolerance_relative'],monotonicity_tolerance=cal['monotonicity_tolerance_absolute'],all_directions_zero=all(x is None for x in ds))
                fit.update(family=family,tau_days=tau,weight_norm=norm,indices=issues)
                result['calibration'][name]=fit;write_json_once(out/f'CALIBRATION_{name}.json',fit)
                print(json.dumps({'calibrated':name,'amplitude':fit['amplitude'],'evaluations':len(fit['trace'])}),flush=True)
            del cached,ds
            selected_issues=[index_of(s) for s in roster['dates']['select']]
            configs=[(name,family,tau,mult,result['calibration'][name]['amplitude']*mult) for (family,tau),name in zip(variants,names) for mult in selection['amplitude_multipliers']]
            if len(configs)!=24:raise RuntimeError('registered parameter candidate count changed')
            config_cells={f'{name}_x{mult}':[] for name,family,tau,mult,amp in configs}
            for i in selected_issues:
                raw=fit_cache_row(manifest,i,out/'ACCESS_LEDGER.jsonl')
                if raw is None:raise RuntimeError('missing registered selection input')
                x=bridge.normalize(torch.from_numpy(np.array(raw,copy=True)[None]).to(device))
                truth=np.stack([fit_cache_row(manifest,i+k,out/'ACCESS_LEDGER.jsonl')[channels] for k in (1,4,12,20)])[None]
                directions={name:direction(family,i,records,mean,mean_provenance=provenance,tau_days=tau) for (family,tau),name in zip(variants,names)}
                for name,family,tau,mult,amp in configs:
                    key=f'{name}_x{mult}';v=directions[name]
                    if v is None or amp==0:
                        mse=np.asarray(slots[i]['mse'])[:,[0,3,4,5]];fallback=True
                    else:
                        with torch.no_grad(),safe_weight_editor(model,split_direction(model,TARGET_BLOCKS,v,amp),blocks=TARGET_BLOCKS,verify_digest=False):
                            tr=rollout_trajectory(bridge,x,steps=20)
                            p=bridge.denormalize(tr[:,[1,4,12,20]].reshape(-1,69,128,256)).reshape(1,4,69,128,256)
                            p=p.index_select(2,torch.tensor(channels,device=device)).cpu().numpy()
                        mse=cells(p,truth,lat)[0];fallback=False
                    row={'index':i,'candidate':key,'mse':mse.tolist(),'loss':float(np.mean(mse/d)),'fallback':fallback}
                    ledger(out/'selection_cells.jsonl',row);result['selection'].append(row);config_cells[key].append(mse)
                print(json.dumps({'selection_issue_complete':i}),flush=True)
            candidate_scores={key:float(np.mean(np.asarray(value)/d)) for key,value in config_cells.items()}
            choices={};resolved={}
            for family in ('lag1c','lag1','static','ewma_g','delay30'):
                group={f'{name}_x{mult}':candidate_scores[f'{name}_x{mult}'] for name,f,tau,mult,amp in configs if f==family}
                all_zero=all(result['calibration'][name]['amplitude']==0 for (f,tau),name in zip(variants,names) if f==family)
                choice=next(k for k in group if k.endswith('_x1')) if all_zero else select_fit_config(group,list(group))
                choices[family]=choice
                row=next((name,tau,mult,amp) for name,f,tau,mult,amp in configs if f==family and f'{name}_x{mult}'==choice)
                resolved[family]={'variant':row[0],'tau_days':row[1],'multiplier':row[2],'amplitude':row[3],'all_zero_family':all_zero}
            write_json_once(out/'CONFIG_RESOLVED.json',{'arms':resolved,'primary':'lag1c','mean':grad['mean'],'denominators':bank['denominators'],'contract_sha256':cfg['contract']['sha256'],'analysis_run':False})
            result.update(status='OBSERVED',candidate_scores=candidate_scores,choices=choices,resolved={'path':str(out/'CONFIG_RESOLVED.json'),'sha256':sha256_file(out/'CONFIG_RESOLVED.json')},model_sha256=loaded.checkpoint_sha256,contract_sha256=cfg['contract']['sha256'])
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        import traceback;traceback.print_exc()
    result['wall_seconds']=time.perf_counter()-start
    write_json_once(out/'PARAMETER_FIT_RECEIPT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('selection','calibration')},indent=2))
    return 0 if result['status']=='OBSERVED' else 1


if __name__=='__main__':raise SystemExit(main())
