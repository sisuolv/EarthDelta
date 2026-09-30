#!/usr/bin/env python3
"""Fit-derived conditional power and MDE, computed on an ACP CPU worker."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from earthdelta.online.io import checked_json,sha256_file,write_json_once
from earthdelta.online.runtime import load_runtime
from earthdelta.online.coverage import BootstrapPlan,scoring_mask,wilson,ARMS,cell_names
from earthdelta.online.power import centered_null_source,resample_source,power_trial
from earthdelta.online.roles import require_fit_rows

STATE=None
PLAN=None


def initialize(state):
    global STATE,PLAN
    STATE=state;PLAN=BootstrapPlan(state['indices'],np.array(state['mask'],bool),state['contract'],state['draws'])


def trial(task):
    source_id,mc=task;s=STATE
    source=np.array(s['sources'][source_id])
    sample=resample_source(source,len(s['indices']),np.random.SeedSequence([18501,source_id,mc]))
    passed,half,joint=power_trial(PLAN,sample,s['factor'],s['grid'])
    return source_id,mc,passed,half,joint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--workers',type=int,default=12);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    cfg,c,roster=load_runtime(args.config)
    receipt=checked_json(cfg['fit_influence']['path'],cfg['fit_influence']['sha256'])
    coverage=checked_json(cfg['coverage_validation']['path'],cfg['coverage_validation']['sha256'])
    if receipt['status']!='OBSERVED' or not coverage.get('qualified'):raise RuntimeError('coverage qualification and paired fit cells required')
    if receipt['arms']!=list(ARMS) or receipt['cells']!=cell_names(c):raise ValueError('paired cell identity mismatch')
    info=receipt['artifact']
    if sha256_file(info['path'])!=info['sha256']:raise ValueError('fit cells changed')
    with np.load(info['path'],allow_pickle=False) as a:mse=a['mse'];ids=a['indices'];select=a['is_select']
    require_fit_rows(ids.tolist(),final_refit=True)
    qc=checked_json(cfg['qc_receipt']['path'],cfg['qc_receipt']['sha256'])
    manifest=checked_json(qc['manifest']['path'],qc['manifest']['sha256'])
    target,mask=scoring_mask(roster,manifest,c)
    sources=[];influences={};descriptive={};names=['estimate','selection_sensitivity']
    for name,pick in zip(names,(~select,select)):
        if np.any(np.diff(ids[pick])!=4):raise ValueError('power source must retain contiguous registered dates')
        source,influence=centered_null_source(mse[pick]);sources.append(source.tolist())
        pooled=mse[pick].mean(0)
        descriptive[name]={'n':int(pick.sum()),'pooled_physical_rmse':np.sqrt(pooled).tolist(),
            'improvement_pct_F0_denominator':(100*(1-np.sqrt(pooled/pooled[0:1]))).tolist(),
            'interpretation':'retrospective training or hyperparameter-selected fit split; not held-out skill'}
        influences[name]={'indices':ids[pick].tolist(),'days':int(pick.sum()),
                         'nonoverlapping_14day_blocks':int(pick.sum())/14,
                         'influence_std_pp':influence.std(0,ddof=1).tolist()}
    grid=c['inference']['mde']['effect_grid_pp'];n=500
    state={'sources':sources,'indices':target.tolist(),'mask':mask.tolist(),'contract':c,
           'draws':c['inference']['replicates'],'factor':coverage['selected_factor'],'grid':grid}
    metadata={'status':'TO_BE_RUN','config_sha256':sha256_file(args.config),'script_sha256':sha256_file(__file__),
        'fit_input':cfg['fit_influence'],'coverage_input':cfg['coverage_validation'],
        'mc_reps_per_source':n,'seed_rule':'SeedSequence([18501, source_id, mc])','source_roles':names,
        'source_normalization':'each arm/cell divided by its own fit pooled MSE, giving empirical null means 1; paired centered influence -50*(x_arm-x_F0)',
        'resampling':'paired circular 14-calendar-day source blocks projected to fixed 270 analysis dates and QC masks',
        'injected_effect':'scale arm-a MSE by (1-d/100)^2 per marginal comparison; same lag1c alternative across entire Gate',
        'limitations':['short retrospective winter fit; selected configuration optimism','seasonal covariance transfer unproved','F_O/F_M marginal alternatives are distinct, not a fictitious single-model conjunction'],
        'analysis_arrays_opened':False,'sources':influences,'effect_grid_pp':grid,
        'bootstrap_draws':state['draws'],'width_factor':state['factor']}
    meta_path=out/'MDE_INPUTS.json'
    if meta_path.exists():
        if json.loads(meta_path.read_text())!=metadata:raise ValueError('incompatible power resume identity')
    else:
        write_json_once(meta_path,metadata)
        write_json_once(out/'FIT_DESCRIPTIVE.json',{'arms':list(ARMS),'cells':cell_names(c),'sources':descriptive,
            'scientific_support':False,'analysis_arrays_opened':False})
    values={j:{} for j in range(2)}
    chunk_dir=out/'chunks';chunk_dir.mkdir(exist_ok=True)
    for path in sorted(chunk_dir.glob('*.npz')):
        info=json.loads(path.with_suffix('.json').read_text())
        if info['sha256']!=sha256_file(path) or info['inputs_sha256']!=sha256_file(meta_path):raise ValueError('power checkpoint identity changed')
        with np.load(path,allow_pickle=False) as a:
            for p,mc in enumerate(a['mc_ids']):
                j=info['source_id'];mc=int(mc)
                if j not in values or not 0<=mc<n or mc in values[j]:raise ValueError('duplicate power checkpoint')
                values[j][mc]=(a['passed'][p],a['half_width'][p],a['joint'][p])
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(state,)) as pool:
        futures=[pool.submit(trial,(j,mc)) for j in range(2) for mc in range(n) if mc not in values[j]]
        pending={0:[],1:[]}
        for k,future in enumerate(as_completed(futures)):
            j,mc,passed,half,joint=future.result();values[j][mc]=(passed,half,joint)
            pending[j].append(mc)
            if len(pending[j])==10:
                ids_chunk=sorted(pending[j]);path=chunk_dir/f's{j}_m{min(ids_chunk):04d}.npz'
                with path.open('xb') as f:np.savez(f,mc_ids=ids_chunk,
                    passed=np.array([values[j][q][0] for q in ids_chunk]),
                    half_width=np.array([values[j][q][1] for q in ids_chunk]),
                    joint=np.array([values[j][q][2] for q in ids_chunk]))
                write_json_once(path.with_suffix('.json'),{'source_id':j,'sha256':sha256_file(path),'inputs_sha256':sha256_file(meta_path)})
                pending[j]=[]
            if (k+1)%50==0:print(json.dumps({'mc_complete':k+1,'total':2*n,'seconds':time.perf_counter()-started}),flush=True)
    plan=BootstrapPlan(target,mask,c,10);summaries=[]
    for j,name in enumerate(names):
        passed=np.array([values[j][mc][0] for mc in range(n)])
        half=np.array([values[j][mc][1] for mc in range(n)])
        joint=np.array([values[j][mc][2] for mc in range(n)])
        with (out/f'MC_{name}.npz').open('xb') as f:np.savez(f,passed=passed,half_width=half,joint=joint)
        rows=[]
        for k,(family,member) in enumerate(plan.members):
            counts=passed[:,:,k].sum(0).astype(int)
            powers=[{'effect_pp':d,'power':int(count)/n,'wilson95':wilson(int(count),n),
                     'mean_half_width_pp':float(half[:,g,k].mean())} for g,(d,count) in enumerate(zip(grid,counts))]
            qualifying=[d for d,count in zip(grid,counts) if count/n>=c['inference']['mde']['power']]
            rows.append({'family':family,**member,'mde_pp':min(qualifying) if qualifying else None,'power_curve':powers})
        counts=joint.sum(0).astype(int)
        qualifying=[d for d,count in zip(grid,counts) if count/n>=c['inference']['mde']['power']]
        summaries.append({'source':name,'comparisons':rows,'gate_mde_pp':min(qualifying) if qualifying else None,
            'gate_power':[{'effect_pp':d,'power':int(count)/n,'wilson95':wilson(int(count),n)} for d,count in zip(grid,counts)]})
    qualified=all(row['gate_mde_pp'] is not None for row in summaries)
    result={'status':'OBSERVED','scope':'conditional fit power only','summaries':summaries,
        'gate_detectable_within_preregistered_grid_for_both_sources':qualified,
        'recommendation':'REVIEW_POWER_BEFORE_ANALYSIS' if qualified else 'STATISTICAL_INCONCLUSIVE',
        'analysis_release':False,'scientific_support':False,'novelty_support':False,
        'wall_seconds':time.perf_counter()-started,'inputs_sha256':sha256_file(out/'MDE_INPUTS.json')}
    write_json_once(out/'MDE.json',result);print(json.dumps({k:v for k,v in result.items() if k!='summaries'}),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
