#!/usr/bin/env python3
"""Estimate/select OBC, DABC, OCL-fresh and refit once; never read analysis data."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from earthdelta.online.io import checked_json,sha256_file,write_json_once
from earthdelta.online.runtime import load_runtime,fit_cache_row
from earthdelta.online.bank import SHORT_AND_ENDPOINT_STEPS,physical_cells
from earthdelta.online.output_fit import OutputFit,select_outputs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'BLOCKED','scope':'fit and selection only; analysis sealed','scientific_support':False};start=time.perf_counter()
    try:
        cfg,contract,roster=load_runtime(args.config)
        bank=checked_json(cfg['f0_bank']['path'],cfg['f0_bank']['sha256'])
        if bank['status']!='OBSERVED' or bank['contract_sha256']!=cfg['contract']['sha256']:raise RuntimeError('F0 bank not bound to this contract')
        qc=checked_json(cfg['qc_receipt']['path'],cfg['qc_receipt']['sha256'])
        manifest=checked_json(qc['manifest']['path'],qc['manifest']['sha256'])
        den=checked_json(bank['denominators']['path'],bank['denominators']['sha256'])
        coordinate=bank['coordinate_cache']
        if sha256_file(coordinate['path'])!=coordinate['sha256']:raise RuntimeError('coordinates changed')
        with np.load(coordinate['path'],allow_pickle=False) as a:lat=a['lat']
        channels=[v['index'] for v in contract['metrics']['variables']];variables=[v['name'] for v in contract['metrics']['variables']]
        residuals=[];indices=[]
        for slot in bank['slots']:
            if slot['status']=='NO_FORECAST':continue
            i=slot['index'];info=slot['prediction']
            if sha256_file(info['path'])!=info['sha256']:raise RuntimeError('F0 cache changed')
            with np.load(info['path'],allow_pickle=False) as a:p=a['physical'].astype(np.float64)
            truths=[]
            for step in SHORT_AND_ENDPOINT_STEPS:
                row=fit_cache_row(manifest,i+step,out/'ACCESS_LEDGER.jsonl')
                if row is None:raise RuntimeError('missing fit truth')
                truths.append(row[channels])
            y=np.stack(truths).astype(np.float64)
            recomputed=physical_cells(p,y,lat)
            np.testing.assert_allclose(recomputed,slot['mse'],rtol=1e-12,atol=1e-15,err_msg='F0 cell reproduction failed')
            residuals.append(p-y);indices.append(i)
        work=OutputFit(np.stack(residuals),indices,lat,den['selection_mse'],variables)
        del residuals
        selection=contract['arms']['selection']
        selected=select_outputs(work,selection,out)
        write_json_once(out/'SELECTION_SCORES.json',selected)
        print(json.dumps({'selected':selected['choices']}),flush=True)
        obc_grid={f'lambda={x}':x for x in selection['obc_lambda']}
        dabc_grid={f'tau={t},lambda={x}':(t,x) for t in selection['tau_days'] for x in selection['dabc_lambda']}
        ocl_grid={f'k={k},lambda={l}':(k,l) for k in selection['ocl_eof_k'] for l in selection['ocl_ridge_lambda']}
        strength=obc_grid[selected['choices']['obc']];tau,dstrength=dabc_grid[selected['choices']['dabc']];k,regularization=ocl_grid[selected['choices']['ocl_fresh']]
        cs,fallback,models,counts,eofs,lineage=work.ocl(k,regularization,final_refit=True)
        del cs
        fields={}
        for (s,v),eof in eofs.items():
            prefix=f'eof_s{s}_v{v}_'
            fields.update({prefix+'mean':eof.mean,prefix+'basis':eof.basis,prefix+'sqrt_weights':eof.sqrt_weights})
        for name,model in models.items():
            fields.update({name+'_xmean':model.mean,name+'_xscale':model.scale,name+'_coef':model.estimator.coef_,name+'_intercept':model.estimator.intercept_})
        fields['obc_bias']=work.r[work.training_positions(True)].mean(0)[list(work.score_steps)]
        p=out/'FINAL_OUTPUT_MODELS.npz'
        with p.open('xb') as f:np.savez(f,**fields)
        resolved={'obc_lambda':strength,'dabc_tau_days':tau,'dabc_lambda':dstrength,
                  'ocl_k':k,'ocl_lambda':regularization,'ocl_history_tau_days':7,
                  'artifact_available':'2020-03-31T00:00:00Z','forecast_publish_from':'2020-04-01T00:00:00Z',
                  'refit_target_indices':work.indices[work.training_positions(True)].tolist(),
                  'train_counts':counts,'models':{'path':str(p),'sha256':sha256_file(p)},
                  'denominators':bank['denominators'],'contract_sha256':cfg['contract']['sha256']}
        write_json_once(out/'CONFIG_RESOLVED.json',resolved);write_json_once(out/'FEEDBACK_LINEAGE.json',lineage)
        receipt.update(status='OBSERVED',choices=selected['choices'],candidate_counts={k:len(v) for k,v in selected['scores'].items()},selection_scores={'path':str(out/'SELECTION_SCORES.json'),'sha256':sha256_file(out/'SELECTION_SCORES.json')},resolved={'path':str(out/'CONFIG_RESOLVED.json'),'sha256':sha256_file(out/'CONFIG_RESOLVED.json')},refit_target_count=len(resolved['refit_target_indices']),model_sha256=bank['model_sha256'],contract_sha256=cfg['contract']['sha256'])
    except Exception as exc:
        receipt['error']=f'{type(exc).__name__}: {exc}'
        import traceback;traceback.print_exc()
    receipt['wall_seconds']=time.perf_counter()-start
    write_json_once(out/'OUTPUT_FIT_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='OBSERVED' else 1


if __name__=='__main__':raise SystemExit(main())
