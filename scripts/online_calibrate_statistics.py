#!/usr/bin/env python3
"""CPU-only fit-input coverage simulation on ACP, restartable by MC chunk."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from earthdelta.online.coverage import (BootstrapPlan,synthetic_mse,scenarios,
    scoring_mask,fit_rho,coverage_rows)
from earthdelta.online.io import checked_json,sha256_file,write_json_once
from earthdelta.online.runtime import load_runtime

CONTEXT=None
PLAN=None


def initialize(context):
    global CONTEXT,PLAN
    CONTEXT=context
    PLAN=BootstrapPlan(context['indices'],np.asarray(context['mask'],bool),context['contract'],context['draws'])


def chunk(task):
    scenario,first,n=task;c=CONTEXT;rows=[];started=time.perf_counter()
    for mc in range(first,first+n):
        seed=np.random.SeedSequence([c['seed'],scenario['id'],mc])
        mse,truth=synthetic_mse(scenario,seed,np.asarray(c['indices']),np.asarray(c['denominators']))
        center,lo,hi=PLAN.bounds(mse)
        x=mse[:,0,:];a=x[:-1]-x[:-1].mean(0);b=x[1:]-x[1:].mean(0)
        rho=float(np.mean(np.sum(a*b,0)/np.sqrt(np.sum(a*a,0)*np.sum(b*b,0))))
        rows.append((center,lo,hi,PLAN.truth(truth),rho))
    return {'scenario':scenario['id'],'first':first,'n':n,
            'theta':np.array([r[0] for r in rows]),'lower':np.array([r[1] for r in rows]),
            'upper':np.array([r[2] for r in rows]),'truth':np.array([r[3] for r in rows]),
            'rho':np.array([r[4] for r in rows]),'seconds':time.perf_counter()-started}


def inputs(path,phase):
    cfg,contract,roster=load_runtime(path)
    bank=checked_json(cfg['f0_bank']['path'],cfg['f0_bank']['sha256'])
    if bank['status']!='OBSERVED':raise ValueError('fit F0 bank not observed')
    for row in bank['slots']:
        if row['index']>360:raise PermissionError('analysis source in fit statistics')
    den=checked_json(bank['denominators']['path'],bank['denominators']['sha256'])
    if den['fit_role']!='estimate_only':raise PermissionError('D_est is not estimate-only')
    qc=checked_json(cfg['qc_receipt']['path'],cfg['qc_receipt']['sha256'])
    manifest=checked_json(qc['manifest']['path'],qc['manifest']['sha256'])
    ids,mask=scoring_mask(roster,manifest,contract)
    rho=fit_rho(bank,den,roster)
    if phase!='preflight':
        for key in ('parameter_fit','output_fit'):
            result=checked_json(cfg[key]['path'],cfg[key]['sha256'])
            if result['status']!='OBSERVED':raise ValueError('all fit gates required before qualification')
    return cfg,contract,ids,mask,den,rho


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--phase',choices=('preflight','calibration','validation'),required=True)
    ap.add_argument('--workers',type=int,default=12);ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    cfg,c,ids,mask,den,rho=inputs(args.config,args.phase);rules=c['inference']['coverage_calibration']
    if not 1<=args.workers<=rules['max_cpu_workers'] or not 0<=args.shard<args.shards:raise ValueError('invalid sharding')
    d=np.asarray(den['selection_mse']).T.reshape(-1)
    all_scenarios=scenarios(c,rho['clipped'])
    metadata={'config_sha256':sha256_file(args.config),'contract_sha256':cfg['contract']['sha256'],
        'phase':args.phase,'rho':rho,'analysis_arrays_opened':False,'source_bank':cfg['f0_bank'],
        'indices':ids.tolist(),'mask':mask.tolist(),'valid_counts':mask.sum(0).tolist(),
        'scenarios':all_scenarios,'workers':args.workers,'shard':args.shard,'shards':args.shards,
        'interval_formula':'theta-c*max(theta-L,0), theta+c*max(U-theta,0); STATISTICAL_PLAN section 2',
        'bootstrap_randomization':'fixed seed shared circular index multiplicities; equivalent to the direct registered method',
        'created_utc':datetime.now(timezone.utc).isoformat()}
    meta_path=out/'INPUTS.json'
    if meta_path.exists():
        old=json.loads(meta_path.read_text())
        for key in ('config_sha256','phase','shard','shards'):
            if old[key]!=metadata[key]:raise ValueError('incompatible resume identity')
        metadata=old
    else:write_json_once(meta_path,metadata)
    context={'indices':ids.tolist(),'mask':mask.tolist(),'contract':c,'denominators':d.tolist()}
    if args.phase=='preflight':
        timings=[]
        for draws in (rules['bootstrap_screen_draws'],rules['bootstrap_validation_draws']):
            initialize(dict(context,draws=draws,seed=90123))
            result=chunk((all_scenarios[10],0,2))
            timings.append({'draws':draws,'mc_reps':2,'seconds':result['seconds'],'seconds_per_mc':result['seconds']/2})
        projection=sum(t['seconds_per_mc']*len(all_scenarios)*n/args.workers for t,n in zip(timings,(rules['mc_calibration_reps'],rules['mc_validation_reps'])))
        write_json_once(out/'PREFLIGHT.json',{'status':'OBSERVED','timings':timings,'serial_worker_projection_seconds':projection,
            'wall_seconds':time.perf_counter()-started,'qualification_claim':False,'rho':rho})
        print(json.dumps({'preflight':timings,'wall_projection_seconds':projection}),flush=True);return 0
    phase=args.phase;validation=phase=='validation'
    if validation:
        approval=checked_json(cfg['coverage_calibration']['path'],cfg['coverage_calibration']['sha256'])
        if approval['status']!='OBSERVED' or approval['selected_factor'] not in c['inference']['width_factors']:
            raise ValueError('calibration did not release independent validation')
        factor=approval['selected_factor']
    else:factor=None
    draws=rules['bootstrap_validation_draws'] if validation else rules['bootstrap_screen_draws']
    reps=rules['mc_validation_reps'] if validation else rules['mc_calibration_reps']
    seed=rules['validation_seed'] if validation else rules['calibration_seed']
    context.update(draws=draws,seed=seed)
    scenario_list=[s for s in all_scenarios if s['id']%args.shards==args.shard]
    tasks=[];chunk_dir=out/'chunks';chunk_dir.mkdir(exist_ok=True)
    for s in scenario_list:
        for first in range(0,reps,10):
            path=chunk_dir/f"s{s['id']:02d}_m{first:04d}.npz"
            side=path.with_suffix('.json')
            if path.exists() or side.exists():
                if not(path.exists() and side.exists()):raise RuntimeError('incomplete chunk needs quarantine review, not silent overwrite')
                info=json.loads(side.read_text())
                if info['inputs_sha256']!=sha256_file(meta_path) or info['sha256']!=sha256_file(path):raise RuntimeError('chunk identity changed')
            else:tasks.append((s,first,min(10,reps-first)))
    end=datetime.fromisoformat(cfg['statistics_deadline_utc'])
    remaining=(end-datetime.now(timezone.utc)).total_seconds()
    if remaining<=0:raise RuntimeError('registered common calibration/validation timebox exhausted')
    deadline=time.monotonic()+min(remaining,rules['clock_budget_hours']*3600)
    pool=ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(context,))
    futures={pool.submit(chunk,t):t for t in tasks};completed=0
    try:
        for future in as_completed(futures,timeout=max(1,deadline-time.monotonic())):
            r=future.result();path=chunk_dir/f"s{r['scenario']:02d}_m{r['first']:04d}.npz"
            with path.open('xb') as f:np.savez(f,**{k:r[k] for k in ('theta','lower','upper','truth','rho')})
            write_json_once(path.with_suffix('.json'),{k:r[k] for k in ('scenario','first','n','seconds')}|
                {'sha256':sha256_file(path),'inputs_sha256':sha256_file(meta_path)})
            completed+=1
            if completed%20==0:print(json.dumps({'phase':phase,'completed_chunks':completed,'scheduled_chunks':len(tasks),'seconds':time.perf_counter()-started}),flush=True)
    finally:pool.shutdown(wait=True,cancel_futures=True)
    summaries=[]
    for s in scenario_list:
        collected={k:[] for k in ('theta','lower','upper','truth','rho')}
        for first in range(0,reps,10):
            with np.load(chunk_dir/f"s{s['id']:02d}_m{first:04d}.npz",allow_pickle=False) as a:
                for k in collected:collected[k].append(a[k])
        collected={k:np.concatenate(v) for k,v in collected.items()}
        if len(collected['theta'])!=reps:raise RuntimeError('missing Monte Carlo denominator')
        rows=coverage_rows(*(collected[k] for k in ('theta','lower','upper','truth')),c)
        if validation:rows=[r for r in rows if r['factor']==factor]
        measured=collected['rho']
        summaries.append({'scenario':s,'families':rows,'measured_daily_loss_rho':float(measured.mean()),
                          'measured_rho_mc_se':float(measured.std(ddof=1)/np.sqrt(len(measured)))})
    receipt={'status':'OBSERVED','phase':phase,'shard':args.shard,'shards':args.shards,
        'draws':draws,'mc_per_scenario':reps,'selected_factor':factor,'scenarios':summaries,
        'inputs_sha256':sha256_file(meta_path),'wall_seconds':time.perf_counter()-started,
        'scientific_support':False,'qualification_complete':False,
        'requires_all_shards_and_independent_validation':True}
    write_json_once(out/'SHARD_RECEIPT.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='scenarios'}),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
