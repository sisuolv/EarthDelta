#!/usr/bin/env python3
"""H100-only official 20-step, hook, finite-difference, and cost preflight."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
from earthdelta.online.io import checked_json,sha256_file,write_json_once
from earthdelta.online.runtime import load_runtime,fit_cache_row
from earthdelta.online.grad_utils import model_guard,safe_weight_editor,weight_gradient,split_direction,unit
from earthdelta.online.losses import normalized_step_mse
from earthdelta.online.timeindex import index_of
from earthdelta.probe.rollout import load_bridge,rollout_trajectory,TARGET_BLOCKS


def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
    return value,time.perf_counter()-start


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'BLOCKED','scientific_support':False,'timings':[]};start=time.perf_counter()
    try:
        cfg,contract,roster=load_runtime(args.config)
        qc=checked_json(cfg['qc_receipt']['path'],cfg['qc_receipt']['sha256'])
        if qc['status']!='OBSERVED':raise RuntimeError('QC gate not passed')
        manifest=checked_json(qc['manifest']['path'],qc['manifest']['sha256'])
        spec=checked_json(cfg['old_spec']['path'],cfg['old_spec']['sha256'])
        bindings=spec['bindings'];coord=qc['coordinate_cache']
        if sha256_file(coord['path'])!=coord['sha256']:raise RuntimeError('coordinate cache changed')
        with np.load(coord['path'],allow_pickle=False) as archive:lat=archive['lat']
        if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):raise RuntimeError('H100 is required for gradient qualification')
        import xformers
        receipt['environment']={'torch':torch.__version__,'xformers':xformers.__version__,'device':torch.cuda.get_device_name(0)}
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(20260930)
        device=torch.device('cuda')
        from earthdelta.contracts import compute_normalization_asset_sha256
        if compute_normalization_asset_sha256(cfg['norm_dir'])!=bindings['normalization']['npz_content_sha256']:raise RuntimeError('normalization changed')
        issues=[index_of(s) for s in roster['dates']['estimate'] if manifest['indices'].get(str(index_of(s)),{}).get('eligible')][:8]
        if len(issues)!=8:raise RuntimeError('eight registered estimate timing issues unavailable')
        write_json_once(out/'TIMING_DATES.json',{'indices':issues,'selection':'first 8 calendar-ordered eligible estimate issues'})
        loaded,load_seconds=timed(lambda:load_bridge(bindings['checkpoint']['path'],cfg['norm_dir'],device))
        if loaded.checkpoint_sha256!=bindings['checkpoint']['sha256']:raise RuntimeError('checkpoint identity changed')
        for relative,h in bindings['bridge_sources_sha256'].items():
            if sha256_file(Path(__file__).resolve().parents[1]/relative)!=h:raise RuntimeError('frozen bridge source changed')
        model,bridge=loaded.model,loaded.bridge;model.requires_grad_(False)
        receipt['load_seconds']=load_seconds
        # Load the actual official implementation with independently constructed transforms.
        source=Path(__file__).with_name('export_upstream_reference.py')
        module_spec=importlib.util.spec_from_file_location('online_official_reference',source)
        official_api=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(official_api)
        inp,diff=official_api.build_official_transforms(Path(cfg['norm_dir']),official_api.OFFICIAL_VARIABLES,intervals=(6,))
        official=official_api.load_official_module(Path(bindings['checkpoint']['path']),4,official_api.OFFICIAL_VARIABLES,inp,diff,device=device)
        channels=[v['index'] for v in contract['metrics']['variables']]
        # Engineering-only loss scaling, never reused as D_est for research fits.
        d=bridge.normalization.inp_std[channels].detach().cpu().numpy().astype(np.float64)**2
        d=np.broadcast_to(d,(4,len(channels))).copy()
        receipt['gradient_check_objective']='six-variable physical MSE/state-std^2 at steps 1..4; engineering only, not the later fitted D_est'
        weight_norm=float(torch.sqrt(sum(model.blocks[b].attn.proj.weight.detach().double().square().sum() for b in TARGET_BLOCKS)).cpu())
        receipt['numerical_rule']=cfg['numerical_precheck']
        with model_guard(model,verify_digest=True):
            for position,i in enumerate(issues):
                io_start=time.perf_counter()
                raw=fit_cache_row(manifest,i,out/'ACCESS_LEDGER.jsonl')
                truths=np.stack([fit_cache_row(manifest,i+k,out/'ACCESS_LEDGER.jsonl') for k in (1,2,3,4)])
                xraw=torch.from_numpy(np.array(raw,copy=True)[None]).to(device)
                truth=torch.from_numpy(truths[:,channels][None]).to(device)
                x=bridge.normalize(xraw);io_seconds=time.perf_counter()-io_start
                with torch.no_grad():
                    f0,t20=timed(lambda:rollout_trajectory(bridge,x,steps=20))
                    short,t4=timed(lambda:rollout_trajectory(bridge,x,steps=4))
                if not bool(torch.isfinite(f0).all()) or not torch.equal(short,f0[:,:5]):raise RuntimeError('short/long prefix mismatch or nonfinite rollout')
                if position==0:
                    with torch.no_grad():
                        reference,tref=timed(lambda:official.forward_validation(inp(xraw),official_api.OFFICIAL_VARIABLES,6,20))
                        maxdiff=float((reference-f0[:,-1]).abs().max().cpu())
                    receipt['official20']={'max_abs_diff':maxdiff,'bitwise_equal':torch.equal(reference,f0[:,-1]),'wall_seconds':tref}
                    if maxdiff!=0:raise RuntimeError('official 20-step forward is not bitwise equal')
                    del official,reference;gc.collect();torch.cuda.empty_cache()
                    zero={b:torch.zeros_like(model.blocks[b].attn.proj.weight) for b in TARGET_BLOCKS}
                    with torch.no_grad(),safe_weight_editor(model,zero,blocks=TARGET_BLOCKS,verify_digest=False):
                        z,tzero=timed(lambda:rollout_trajectory(bridge,x,steps=20))
                    receipt['zero_hook20']={'max_abs_diff':float((z-f0).abs().max().cpu()),'wall_seconds':tzero}
                    if not torch.equal(z,f0):raise RuntimeError('zero-hook forward differs')
                    del z,zero
                def loss_fn(steps):
                    tr=rollout_trajectory(bridge,x,steps=steps,differentiable=True,checkpoint_steps=False)
                    raw_pred=bridge.denormalize(tr[:,1:].reshape(-1,69,128,256)).reshape(1,steps,69,128,256)
                    return normalized_step_mse(raw_pred[:,:,channels],truth[:,:steps],lat,d[:steps])
                (g,loss),tg4=timed(lambda:weight_gradient(model,TARGET_BLOCKS,lambda:loss_fn(4),verify_digest=False))
                (g2,loss2),tg2=timed(lambda:weight_gradient(model,TARGET_BLOCKS,lambda:loss_fn(2),verify_digest=False))
                if position==0:
                    direction=unit(g.detach().cpu().numpy())
                    if direction is None:raise RuntimeError('zero probe gradient cannot qualify directional derivative')
                    gradnorm=float(torch.linalg.vector_norm(g.double()).cpu());checks=[]
                    for relative in cfg['numerical_precheck']['fd_relative_weight_norm']:
                        amp=weight_norm*relative;values=[]
                        for sign in (1,-1):
                            with torch.no_grad(),safe_weight_editor(model,split_direction(model,TARGET_BLOCKS,direction,sign*amp),blocks=TARGET_BLOCKS,verify_digest=False):
                                values.append(float(loss_fn(4).detach().cpu()))
                        fd=(values[0]-values[1])/(2*amp);error=abs(fd-gradnorm)/max(abs(fd),gradnorm,1e-30)
                        checks.append({'relative_weight_norm':relative,'autograd':gradnorm,'finite_difference':fd,'relative_error':error})
                    receipt['gradient_direction']=checks
                    if any(c['relative_error']>cfg['numerical_precheck']['fd_relative_error_max'] or c['finite_difference']<=0 for c in checks):raise RuntimeError('gradient direction finite-difference gate failed')
                    with torch.no_grad(),safe_weight_editor(model,split_direction(model,TARGET_BLOCKS,direction,weight_norm*cfg['numerical_precheck']['fd_relative_weight_norm'][-1]),blocks=TARGET_BLOCKS,verify_digest=False):
                        edited,te=timed(lambda:rollout_trajectory(bridge,x,steps=20))
                    receipt['nonzero_hook20']={'finite':bool(torch.isfinite(edited).all()),'max_abs_change':float((edited-f0).abs().max().cpu()),'wall_seconds':te}
                    if not receipt['nonzero_hook20']['finite'] or receipt['nonzero_hook20']['max_abs_change']==0:raise RuntimeError('nonzero edit forward invalid')
                    del edited
                if any(model.blocks[b].attn.proj._forward_hooks for b in TARGET_BLOCKS):raise RuntimeError('edit hook leaked')
                row={'index':i,'io':io_seconds,'roll20':t20,'roll4':t4,'grad4':tg4,'grad2':tg2,'grad_norm':float(torch.linalg.vector_norm(g.double()).cpu()),'engineering_loss':loss,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
                receipt['timings'].append(row);write_json_once(out/f'timing_{i:04d}.json',row)
                del g,g2,f0,short,truth,x,xraw
        receipt['p50']={k:float(np.median([x[k] for x in receipt['timings']])) for k in ('roll20','roll4','grad4','grad2','io')}
        receipt['p95']={k:float(np.quantile([x[k] for x in receipt['timings']],.95)) for k in receipt['p50']}
        receipt.update(status='OBSERVED',engineering_pass=True,pristine_digest_unchanged=True,checkpoint_sha256=loaded.checkpoint_sha256)
    except Exception as exc:
        receipt['error']=f'{type(exc).__name__}: {exc}'
        import traceback;traceback.print_exc()
    receipt['wall_seconds']=time.perf_counter()-start
    write_json_once(out/'ENGINE_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='OBSERVED' else 1


if __name__=='__main__':raise SystemExit(main())
