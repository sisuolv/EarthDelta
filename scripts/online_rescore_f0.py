#!/usr/bin/env python3
"""Re-score immutable forecasts in FP64, preserving the initial FP32 receipt."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from earthdelta.online.io import checked_json,sha256_file,write_json_once
from earthdelta.online.runtime import load_runtime,fit_cache_row
from earthdelta.online.bank import SHORT_AND_ENDPOINT_STEPS,physical_cells,fit_denominators
from earthdelta.v8.standard_metrics import issue_mse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cfg,contract,roster=load_runtime(args.config)
    original=checked_json(cfg['f0_bank']['path'],cfg['f0_bank']['sha256'])
    if original['status']!='OBSERVED':raise RuntimeError('source F0 bank is incomplete')
    qc=checked_json(cfg['qc_receipt']['path'],cfg['qc_receipt']['sha256']);manifest=checked_json(qc['manifest']['path'],qc['manifest']['sha256'])
    coord=original['coordinate_cache']
    if sha256_file(coord['path'])!=coord['sha256']:raise RuntimeError('coordinate cache changed')
    with np.load(coord['path'],allow_pickle=False) as a:lat=a['lat']
    bank=copy.deepcopy(original);indices=[];roles=[];all_mse=[];differences=[]
    channels=[v['index'] for v in contract['metrics']['variables']]
    for slot in bank['slots']:
        if slot['status']=='NO_FORECAST':continue
        info=slot['prediction'];i=slot['index']
        if sha256_file(info['path'])!=info['sha256']:raise RuntimeError('forecast bytes changed')
        with np.load(info['path'],allow_pickle=False) as a:p=a['physical']
        y=np.stack([fit_cache_row(manifest,i+k,out/'ACCESS_LEDGER.jsonl')[channels] for k in SHORT_AND_ENDPOINT_STEPS])
        # First reproduce the old arithmetic exactly; no tolerance relaxation
        # conceals a cache/source mismatch.
        old=np.stack([issue_mse(p[None,:,v],y[None,:,v],lat)[0] for v in range(6)])
        np.testing.assert_array_equal(old,np.asarray(slot['mse']))
        new=physical_cells(p,y,lat)
        differences.append({'index':i,'max_relative_change':float(np.max(np.abs(new-old)/np.maximum(old,1e-300)))})
        slot['mse']=new.tolist();indices.append(i);roles.append(slot['role']);all_mse.append(new)
    den=fit_denominators(all_mse,indices,roles);write_json_once(out/'DENOMINATORS.json',den)
    bank['denominators']={'path':str(out/'DENOMINATORS.json'),'sha256':sha256_file(out/'DENOMINATORS.json')}
    bank['scoring_revision']={'source_receipt':cfg['f0_bank'],'rule':'cast forecast and truth to FP64 before subtraction and square','original_arithmetic_reproduced_bitwise':True,'per_issue_changes':differences,'max_relative_mse_change':max(x['max_relative_change'] for x in differences),'forecasts_rerun':False}
    write_json_once(out/'F0_BANK_RECEIPT.json',bank)
    print(json.dumps({'status':'OBSERVED','published':len(indices),'max_relative_mse_change':bank['scoring_revision']['max_relative_mse_change'],'old_receipt_unchanged':True}))


if __name__=='__main__':main()
