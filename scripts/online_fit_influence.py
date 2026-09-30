#!/usr/bin/env python3
"""Fit-only paired cells for power planning, never held-out skill evidence."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from earthdelta.online.io import checked_json,sha256_file,write_json_once
from earthdelta.online.runtime import load_runtime,fit_cache_row,ledger
from earthdelta.online.bank import SHORT_AND_ENDPOINT_STEPS,physical_cells
from earthdelta.online.output_fit import OutputFit
from earthdelta.online.coverage import ARMS,cell_names
from earthdelta.online.directions import direction
from earthdelta.online.grad_utils import model_guard,safe_weight_editor,split_direction,target_metadata
from earthdelta.online.losses import cells
from earthdelta.probe.rollout import load_bridge,rollout_trajectory,TARGET_BLOCKS
from online_p0b_edits import load_gradients


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result={'status':'BLOCKED','scope':'fit influence only','scientific_support':False,'analysis_arrays_opened':False}
    try:
        cfg,c,roster=load_runtime(args.config)
        get=lambda k:checked_json(cfg[k]['path'],cfg[k]['sha256'])
        bank=get('f0_bank');pf=get('parameter_fit');of=get('output_fit');grad=get('gradients')
        if any(r['status']!='OBSERVED' for r in (bank,pf,of,grad)):raise ValueError('all fit receipts required')
        if pf['contract_sha256']!=cfg['contract']['sha256'] or of['contract_sha256']!=cfg['contract']['sha256']:raise ValueError('fit contract mismatch')
        param=checked_json(pf['resolved']['path'],pf['resolved']['sha256'])
        output=checked_json(of['resolved']['path'],of['resolved']['sha256'])
        if param['denominators']!=bank['denominators'] or output['denominators']!=bank['denominators']:raise ValueError('denominator identity differs')
        qc=get('qc_receipt');manifest=checked_json(qc['manifest']['path'],qc['manifest']['sha256'])
        den=checked_json(bank['denominators']['path'],bank['denominators']['sha256'])
        coordinate=bank['coordinate_cache']
        if sha256_file(coordinate['path'])!=coordinate['sha256']:raise ValueError('coordinate cache changed')
        with np.load(coordinate['path'],allow_pickle=False) as a:lat=a['lat']
        channels=[v['index'] for v in c['metrics']['variables']];variables=[v['name'] for v in c['metrics']['variables']]
        residuals=[];indices=[];truths={};slots={}
        for slot in bank['slots']:
            if slot['status']=='NO_FORECAST':continue
            i=slot['index'];info=slot['prediction']
            if sha256_file(info['path'])!=info['sha256']:raise ValueError('F0 prediction changed')
            with np.load(info['path'],allow_pickle=False) as a:p=a['physical'].astype(np.float64)
            y=np.stack([fit_cache_row(manifest,i+k,out/'ACCESS_LEDGER.jsonl')[channels] for k in SHORT_AND_ENDPOINT_STEPS]).astype(np.float64)
            np.testing.assert_allclose(physical_cells(p,y,lat),slot['mse'],rtol=1e-12,atol=1e-15)
            residuals.append(p-y);indices.append(i);truths[i]=y[[0,3,4,5]];slots[i]=slot
        work=OutputFit(np.stack(residuals),indices,lat,den['selection_mse'],variables);del residuals
        pos=work.training_positions(True);ids=work.indices[pos]
        paired=np.full((len(pos),len(ARMS),24),np.nan,dtype=np.float64)
        paired[:,0]=np.array([np.asarray(slots[int(i)]['mse'])[:,[0,3,4,5]].reshape(-1) for i in ids])
        corrections=[work.obc(output['obc_lambda']),
                     work.dabc(output['dabc_tau_days'],output['dabc_lambda'])[0],
                     work.ocl(output['ocl_k'],output['ocl_lambda'],final_refit=False)[0]]
        for j,correction in enumerate(corrections,1):paired[:,j]=work.score(correction,pos)[1].reshape(len(pos),24)
        del corrections,work
        original=checked_json(of['selection_scores']['path'],of['selection_scores']['sha256'])
        selected=set(original['selection_source_indices']);selection_rows=np.array([int(i) in selected for i in ids])
        denominator=np.asarray(den['selection_mse']).T.reshape(-1)
        for j,name in enumerate(('obc','dabc','ocl_fresh'),1):
            score=float(np.mean(paired[selection_rows,j]/denominator))
            np.testing.assert_allclose(score,original['scores'][name][of['choices'][name]],rtol=1e-12,atol=1e-15)
        stored={(r['index'],r['candidate']):r for r in pf['selection']}
        for p,i in enumerate(ids):
            if int(i) in selected:
                for j,name in enumerate(ARMS[4:],4):paired[p,j]=np.asarray(stored[int(i),pf['choices'][name]]['mse']).reshape(-1)
        binding=get('old_spec')['bindings']
        from earthdelta.contracts import compute_normalization_asset_sha256
        if compute_normalization_asset_sha256(cfg['norm_dir'])!=binding['normalization']['npz_content_sha256']:raise ValueError('normalization changed')
        if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):raise RuntimeError('qualified H100 required')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.device('cuda');loaded=load_bridge(binding['checkpoint']['path'],cfg['norm_dir'],device)
        if loaded.checkpoint_sha256!=binding['checkpoint']['sha256']:raise ValueError('checkpoint changed')
        model,bridge=loaded.model,loaded.bridge;model.requires_grad_(False)
        if json.loads(json.dumps(target_metadata(model,TARGET_BLOCKS)))!=grad['target_metadata']:raise ValueError('gradient target layout changed')
        records,mean,provenance=load_gradients(grad);nroll=0;fallback=0
        with model_guard(model):
            for p,i in enumerate(ids):
                i=int(i)
                if i in selected:continue
                raw=fit_cache_row(manifest,i,out/'ACCESS_LEDGER.jsonl')
                x=bridge.normalize(torch.from_numpy(np.array(raw,copy=True)[None]).to(device))
                for j,name in enumerate(ARMS[4:],4):
                    config=param['arms'][name]
                    v=direction(name,i,records,mean,mean_provenance=provenance,tau_days=config['tau_days'],
                                offline_fit=True,artifact_context=provenance.artifact_available)
                    if v is None or config['amplitude']==0:
                        paired[p,j]=paired[p,0];fallback+=1
                    else:
                        with torch.no_grad(),safe_weight_editor(model,split_direction(model,TARGET_BLOCKS,v,config['amplitude']),blocks=TARGET_BLOCKS,verify_digest=False):
                            tr=rollout_trajectory(bridge,x,steps=20)
                            pred=bridge.denormalize(tr[:,[1,4,12,20]].reshape(-1,69,128,256)).reshape(1,4,69,128,256)
                            pred=pred.index_select(2,torch.tensor(channels,device=device)).cpu().numpy()
                        paired[p,j]=cells(pred,truths[i][None],lat)[0].reshape(-1);nroll+=1
                    ledger(out/'FIT_CELLS.jsonl',{'index':i,'arm':name,'mse':paired[p,j].tolist(),'role':'estimate_retrospective_for_power_only'})
                if (p+1)%10==0:print(json.dumps({'estimate_issues_completed':p+1,'rollouts':nroll}),flush=True)
        if not np.isfinite(paired).all() or np.any(paired<0):raise ValueError('missing or invalid paired fit denominator')
        artifact=out/'FIT_PAIRED_CELLS.npz'
        with artifact.open('xb') as f:np.savez(f,mse=paired,indices=ids,is_select=selection_rows)
        result.update(status='OBSERVED',artifact={'path':str(artifact),'sha256':sha256_file(artifact)},
            arms=list(ARMS),cells=cell_names(c),estimate_count=int((~selection_rows).sum()),selection_count=int(selection_rows.sum()),
            extra_rollouts=nroll,extra_forecast_slots=270,estimate_fallback_slots=fallback,
            reused_selection_slots=130,contract_sha256=cfg['contract']['sha256'],
            warning='retrospective fitted estimate cells and hyperparameter-selected selection cells; centered power proxy only, neither is held-out evidence')
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        import traceback;traceback.print_exc()
    result['wall_seconds']=time.perf_counter()-start;write_json_once(out/'FIT_INFLUENCE_RECEIPT.json',result)
    print(json.dumps(result,indent=2),flush=True);return 0 if result['status']=='OBSERVED' else 1


if __name__=='__main__':raise SystemExit(main())
