#!/usr/bin/env python3
"""P3: construct fixed gradient/random directions and calibrate amplitudes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from earthdelta.bridge.stormer_bridge import DEFAULT_VARIABLES
from earthdelta.probe.contracts import issue_index, issue_sets, load_probe_spec
from earthdelta.probe.edits import pristine_state_digest
from earthdelta.probe.rollout import area_weighted_mse, l6_per_issue, load_bridge, rollout_trajectory


TARGET_BLOCKS = (18, 19, 20, 21, 22, 23)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_row(qc: dict, index: int, *, cache_dir: Path, ledger: Path,
              run_id: str, job: str, stage: str) -> np.ndarray:
    row = qc.get("cache_files", {}).get(str(index))
    if not row:
        raise RuntimeError(f"P1 cache missing index {index}")
    path = Path(row["path"])
    if not path.is_absolute():
        path = cache_dir / path
    if digest(path) != row.get("sha256"):
        raise RuntimeError(f"P1 cache file hash mismatch at {index}")
    value = np.asarray(np.load(path, allow_pickle=False))
    decoded = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    if decoded != qc.get("per_index", {}).get(str(index), {}).get("decoded_sha256"):
        raise RuntimeError(f"P1 decoded hash mismatch at {index}")
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"schema":"earthdelta.probe.access.v1","run_id":run_id,
                            "path":str(path),"indices":[index],"purpose":"cache_read",
                            "stage":stage,"job":job,"array_payload":True,
                            "decoded_sha256":decoded,"shape":list(value.shape)},
                           sort_keys=True,separators=(",", ":")) + "\n")
    return value


def load_issue(qc: dict, issue: str, *, cache_dir: Path, ledger: Path,
               run_id: str, job: str, stage: str) -> np.ndarray:
    i0 = issue_index(issue)
    return np.stack([cache_row(qc, i, cache_dir=cache_dir, ledger=ledger,
                               run_id=run_id, job=job, stage=stage)
                     for i in range(i0, i0 + 21)], axis=0)


def target_params(model: torch.nn.Module):
    out = []
    for b in TARGET_BLOCKS:
        out.append((f"blocks.{b}.attn.proj.weight", model.blocks[b].attn.proj.weight))
    return out


def flat_delta(vector: np.ndarray, params, device: torch.device) -> dict[int, torch.Tensor]:
    out = {}
    cursor = 0
    for b, (_, p) in zip(TARGET_BLOCKS, params):
        n = p.numel()
        out[b] = torch.from_numpy(np.asarray(vector[cursor:cursor+n], dtype=np.float32).reshape(tuple(p.shape))).to(device)
        cursor += n
    if cursor != len(vector):
        raise RuntimeError("direction length does not match target parameter count")
    return out


def cell_mse(pred_norm: torch.Tensor, truth_raw: torch.Tensor, bridge, lat: np.ndarray,
             variables, leads) -> np.ndarray:
    pred_raw = bridge.denormalize(pred_norm.reshape(-1,69,128,256)).reshape_as(pred_norm)
    out = np.empty((len(variables),len(leads)),dtype=np.float64)
    for vi,v in enumerate(variables):
        ch=int(v["channel_index"])
        for li,lead in enumerate(leads):
            step=int(lead//6)
            out[vi,li]=float(area_weighted_mse(pred_raw[:,[step]],truth_raw[:,[step]],lat,ch).mean().detach().cpu())
    return out


def normalized_rms(pred_norm: torch.Tensor, truth_raw: torch.Tensor, bridge,
                   lat: np.ndarray) -> float:
    truth_norm = bridge.normalize(truth_raw)
    delta = pred_norm[:, int(24//6)] - truth_norm[:, int(24//6)]
    weights = torch.as_tensor(np.cos(np.deg2rad(lat)),dtype=delta.dtype,device=delta.device)
    weights = weights[:,None] / weights.sum()
    # Mean over the 69 state channels and cosine-latitude weighted grid.
    per_issue = (delta.square() * weights).mean(dim=1).sum(dim=(-1,-2))
    return float(torch.sqrt(per_issue.mean()).detach().cpu())


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--spec",required=True); ap.add_argument("--run-dir",required=True)
    ap.add_argument("--norm-dir",required=True); ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--cache-dir",required=True); ap.add_argument("--job",default="local")
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args(); run=Path(args.run_dir).resolve(); run.mkdir(parents=True,exist_ok=True)
    spec=load_probe_spec(args.spec,expected_sha256="bf247b2abc86ef81be62a7aa5a26318532ca3e3b5ceaeb492c98387781518348")
    receipt={"schema":"earthdelta.probe.directions.v1","status":"BLOCKED","spec_sha256":spec.sha256,"job":args.job}
    try:
        qc=json.loads((run/"DATA_QC_RECEIPT.json").read_text())
        if qc.get("status")!="OBSERVED": raise RuntimeError("P1 QC is not OBSERVED")
        coord=qc.get("coordinate_cache",{}); cp=Path(coord.get("path",""))
        if not cp.exists() or digest(cp)!=coord.get("sha256"): raise RuntimeError("coordinate cache invalid")
        lat=np.asarray(np.load(cp,allow_pickle=False)["lat"],dtype=np.float64)
        device=torch.device(args.device)
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        loaded=load_bridge(args.checkpoint,args.norm_dir,device)
        if loaded.checkpoint_sha256!=spec.raw["bindings"]["checkpoint"]["sha256"]: raise RuntimeError("checkpoint hash mismatch")
        params=target_params(loaded.model); total=sum(p.numel() for _,p in params)
        # Direction gradients are only taken with respect to F1 target
        # projections; freezing all other parameters reduces graph memory while
        # leaving the forward map unchanged.
        for p in loaded.model.parameters(): p.requires_grad_(False)
        for _, p in params: p.requires_grad_(True)
        started=run/"P3_STARTED.marker"; started.write_text(datetime.now(timezone.utc).isoformat()+"\n")
        names=issue_sets(spec); d_issues=names["D_direction"]
        # Freeze the L6 denominators from pristine F0 on D_direction.
        f0_cells=[]; f0_data=[]
        t0=time.perf_counter()
        for issue in d_issues:
            data=load_issue(qc,issue,cache_dir=Path(args.cache_dir),ledger=run/"ACCESS_LEDGER.jsonl",run_id=run.name,job=args.job,stage="P3")
            x=loaded.bridge.normalize(torch.from_numpy(data[0:1]).float().to(device))
            truth=torch.from_numpy(np.ascontiguousarray(data[[1,4,12,20]])).float().unsqueeze(0).to(device)
            with torch.inference_mode(): f0=rollout_trajectory(loaded.bridge,x,steps=20)
            f0_cells.append(cell_mse(f0[:,[1,4,12,20]],truth,loaded.bridge,lat,spec.variables,spec.leads))
            f0_data.append((data,x,truth,f0.detach()))
        f0_cells=np.asarray(f0_cells); denom=np.sqrt(f0_cells.mean(axis=0)); denominator={f"{v['name']}@{lead}h":float(denom[vi,li]) for vi,v in enumerate(spec.variables) for li,lead in enumerate(spec.leads)}
        # One gradient per D issue, retained on CPU for the Gram construction.
        grads=[]; grad_seconds=[]
        for idx,(data,x,truth,_f0) in enumerate(f0_data):
            begin=time.perf_counter()
            loaded.model.zero_grad(set_to_none=True)
            pred=rollout_trajectory(loaded.bridge,x,steps=20,differentiable=True,checkpoint_steps=True)
            # Keep the full 0..20 trajectory: l6_per_issue indexes the
            # physical lead step directly (1, 4, 12, 20).
            loss=l6_per_issue(pred,truth,loaded.bridge,lat,spec.variables,spec.leads,denominator)
            g=torch.autograd.grad(loss,[p for _,p in params],retain_graph=False,allow_unused=False)
            grads.append(torch.cat([z.detach().reshape(-1).cpu() for z in g]).numpy().astype(np.float32))
            grad_seconds.append(time.perf_counter()-begin)
        G=np.stack(grads).astype(np.float64); gram=G@G.T
        evals,U=np.linalg.eigh(gram); order=np.argsort(evals)[::-1]; evals=np.maximum(evals[order],0); U=U[:,order]
        k=min(4,len(d_issues)); grad_dirs=[]
        for j in range(k):
            v=U[:,j]@G; v=v/np.linalg.norm(v); grad_dirs.append(v.astype(np.float32))
        grad_dirs=np.stack(grad_dirs)
        rng=np.random.default_rng(20260927); random_dirs=[]
        for _ in range(k):
            v=rng.standard_normal(total).astype(np.float64)
            for q in random_dirs: v-=np.dot(v,q)*q
            v/=np.linalg.norm(v); random_dirs.append(v.astype(np.float32))
        random_dirs=np.stack(random_dirs)
        # Calibration on CAL issues by the frozen log-bisection rule.
        cal=names["CAL_calibration"]
        cal_data=[]
        for issue in cal:
            data=load_issue(qc,issue,cache_dir=Path(args.cache_dir),ledger=run/"ACCESS_LEDGER.jsonl",run_id=run.name,job=args.job,stage="P3")
            x=loaded.bridge.normalize(torch.from_numpy(data[0:1]).float().to(device)); truth=torch.from_numpy(np.ascontiguousarray(data[[1,4,12,20]])).float().unsqueeze(0).to(device)
            with torch.inference_mode(): f0=rollout_trajectory(loaded.bridge,x,steps=4)
            cal_data.append((x,truth,f0.detach()))
        def effect(direction,a):
            vals=[]
            delta=flat_delta(direction*a,params,device)
            for x,truth,f0 in cal_data:
                with torch.inference_mode(): edited=rollout_trajectory(loaded.bridge,x,steps=4,deltas=delta)
                num=normalized_rms(edited,f0[:,[1,2,3,4]].detach() if False else loaded.bridge.denormalize(f0[:,[4]]),loaded.bridge,loaded.bridge,lat) if False else None
                # Both terms are measured in normalized state units at 24h.
                diff=edited[:,4]-f0[:,4]
                w=torch.as_tensor(np.cos(np.deg2rad(lat)),dtype=diff.dtype,device=diff.device); w=w[:,None]/w.sum()
                n=torch.sqrt((diff.square()*w).mean(dim=1).sum(dim=(-1,-2)))
                truthn=loaded.bridge.normalize(truth)[:,3]
                e=f0[:,4]-truthn; den=torch.sqrt((e.square()*w).mean(dim=1).sum(dim=(-1,-2)))
                vals.append(float((n/den).detach().cpu()))
            return float(np.sqrt(np.mean(np.square(vals))))
        def calibrate(direction):
            lo,hi=1e-4,1e3; target=.1
            elo,ehi=effect(direction,lo),effect(direction,hi)
            if not (elo<=target<=ehi): raise RuntimeError(f"calibration not bracketed: {elo}, {ehi}")
            for _ in range(12):
                mid=float(np.sqrt(lo*hi)); em=effect(direction,mid)
                if em<target: lo=mid
                else: hi=mid
            return float(np.sqrt(lo*hi))
        a_grad=[calibrate(v) for v in grad_dirs]; a_rand=[calibrate(v) for v in random_dirs]
        pristine=pristine_state_digest(loaded.model)
        out=run/"DIRECTIONS.npz"; np.savez_compressed(out,gradient_dirs=grad_dirs,random_dirs=random_dirs,a_gradient=np.asarray(a_grad),a_random=np.asarray(a_rand),denominator_keys=np.asarray(list(denominator)),denominator_values=np.asarray(list(denominator.values())),target_total_params=np.asarray([total]))
        receipt.update({"status":"OBSERVED","output":str(out),"output_sha256":digest(out),"n_direction_issues":len(d_issues),"target_blocks":list(TARGET_BLOCKS),"target_total_params":total,"singular_values":np.sqrt(evals[:k]).tolist(),"explained_fraction":(evals[:k]/max(float(evals.sum()),1e-30)).tolist(),"cosine_v1_mean_gradient":float(np.dot(grad_dirs[0],G.mean(axis=0))/(np.linalg.norm(G.mean(axis=0))+1e-30)),"a_gradient":a_grad,"a_random":a_rand,"denominator":denominator,"calibration_target":0.1,"calibration_iterations":12,"gradient_seconds_mean":float(np.mean(grad_seconds)),"f0_rollout_seconds":float(time.perf_counter()-t0),"pristine_state_sha256":pristine})
    except Exception as exc:
        receipt.update({"status":"BLOCKED","error":f"{type(exc).__name__}: {exc}"})
    (run/"DIRECTIONS_RECEIPT.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(receipt,indent=2,sort_keys=True)); return 0 if receipt["status"]=="OBSERVED" else 2


if __name__=="__main__": raise SystemExit(main())
