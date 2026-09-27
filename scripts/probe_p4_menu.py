#!/usr/bin/env python3
"""P4: evaluate the frozen direction menu, random control, and TTT diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from earthdelta.probe.contracts import issue_sets, load_probe_spec
from earthdelta.probe.edits import pristine_state_digest
from earthdelta.probe.rollout import l6_cells, l6_per_issue, load_bridge, rollout_trajectory
from scripts.probe_p3_directions import TARGET_BLOCKS, flat_delta, load_issue


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def target_params(model: torch.nn.Module):
    return [(f"blocks.{b}.attn.proj.weight", model.blocks[b].attn.proj.weight)
            for b in TARGET_BLOCKS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--norm-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--job", default="local")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir).resolve(); run.mkdir(parents=True, exist_ok=True)
    spec = load_probe_spec(args.spec, expected_sha256="bf247b2abc86ef81be62a7aa5a26318532ca3e3b5ceaeb492c98387781518348")
    receipt = {"schema": "earthdelta.probe.menu_eval.v1", "status": "BLOCKED",
               "spec_sha256": spec.sha256, "job": args.job, "shard_index": args.shard_index,
               "shards": args.shards, "dry_run": bool(args.dry_run)}
    try:
        qc = json.loads((run / "DATA_QC_RECEIPT.json").read_text())
        direction_receipt = json.loads((run / "DIRECTIONS_RECEIPT.json").read_text())
        if qc.get("status") != "OBSERVED": raise RuntimeError("P1 QC is not OBSERVED")
        if direction_receipt.get("status") != "OBSERVED": raise RuntimeError("P3 directions are not OBSERVED")
        coordinate = qc["coordinate_cache"]; cp = Path(coordinate["path"])
        if digest(cp) != coordinate["sha256"]: raise RuntimeError("coordinate cache hash mismatch")
        lat = np.asarray(np.load(cp, allow_pickle=False)["lat"], dtype=np.float64)
        directions = np.load(run / "DIRECTIONS.npz", allow_pickle=False)
        if digest(run / "DIRECTIONS.npz") != direction_receipt["output_sha256"]:
            raise RuntimeError("direction output hash mismatch")
        grad_dirs = np.asarray(directions["gradient_dirs"], dtype=np.float32)
        random_dirs = np.asarray(directions["random_dirs"], dtype=np.float32)
        a_grad = np.asarray(directions["a_gradient"], dtype=np.float64)
        a_rand = np.asarray(directions["a_random"], dtype=np.float64)
        denominator = direction_receipt["denominator"]
        device = torch.device(args.device)
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        loaded = load_bridge(args.checkpoint, args.norm_dir, device)
        if loaded.checkpoint_sha256 != spec.raw["bindings"]["checkpoint"]["sha256"]:
            raise RuntimeError("checkpoint hash mismatch")
        params = target_params(loaded.model)
        names = issue_sets(spec)
        issues = names["E_oracle_eval"]
        if args.shards <= 0 or not (0 <= args.shard_index < args.shards):
            raise ValueError("invalid shard index")
        issues = issues[args.shard_index::args.shards]
        if args.dry_run:
            issues = issues[:2]
            grad_dirs = grad_dirs[:1]; random_dirs = random_dirs[:1]
            a_grad = a_grad[:1]; a_rand = a_rand[:1]
            multipliers = [0.25, 0.5]
        else:
            multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
            (run / "P4_STARTED.marker").write_text(datetime.now(timezone.utc).isoformat() + "\n")

        grad_rows=[]; rand_rows=[]; ttt_rows=[]; ttt_grad_norms=[]
        grad_meta=[]; rand_meta=[]; ttt_meta=[]
        started=time.perf_counter()
        for issue in issues:
            data=load_issue(qc, issue, cache_dir=Path(args.cache_dir), ledger=run/"ACCESS_LEDGER.jsonl",
                            run_id=run.name, job=args.job, stage="P4")
            x=loaded.bridge.normalize(torch.from_numpy(data[0:1]).float().to(device))
            truth=torch.from_numpy(np.ascontiguousarray(data)).float().unsqueeze(0).to(device)
            with torch.inference_mode(): f0=rollout_trajectory(loaded.bridge,x,steps=20)
            row_grad=[l6_cells(f0,truth,loaded.bridge,lat,spec.variables,spec.leads,denominator)]
            row_rand=[row_grad[0].copy()]
            row_grad_meta=["F0"]; row_rand_meta=["F0"]
            for family, dirs, amps, row, meta in (("grad",grad_dirs,a_grad,row_grad,row_grad_meta),
                                                    ("rand",random_dirs,a_rand,row_rand,row_rand_meta)):
                for direction, amplitude in zip(dirs, amps):
                    for mult in multipliers:
                        for sign in (1.0, -1.0):
                            delta=flat_delta(direction*(float(amplitude)*mult*sign),params,device)
                            with torch.inference_mode(): pred=rollout_trajectory(loaded.bridge,x,steps=20,deltas=delta)
                            row.append(l6_cells(pred,truth,loaded.bridge,lat,spec.variables,spec.leads,denominator))
                            meta.append(f"{family}:dir={len(meta)}:mult={mult}:sign={int(sign)}")
            # Truth-gradient TTT direction is computed separately for this issue.
            for p in loaded.model.parameters(): p.requires_grad_(False)
            for _,p in params: p.requires_grad_(True)
            loaded.model.zero_grad(set_to_none=True)
            pred_grad=rollout_trajectory(loaded.bridge,x,steps=20,differentiable=True,checkpoint_steps=True)
            loss=l6_per_issue(pred_grad,truth,loaded.bridge,lat,spec.variables,spec.leads,denominator)
            grads=torch.autograd.grad(loss,[p for _,p in params],retain_graph=False)
            vector=torch.cat([g.detach().reshape(-1).cpu() for g in grads]).numpy().astype(np.float32)
            norm=float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 0: raise RuntimeError(f"TTT gradient invalid at {issue}")
            u=-vector/norm; a_ttt=float(np.median(a_grad)); row_ttt=[row_grad[0].copy()]; row_ttt_meta=["F0"]
            for mult in multipliers:
                delta=flat_delta(u*(a_ttt*mult),params,device)
                with torch.inference_mode(): pred=rollout_trajectory(loaded.bridge,x,steps=20,deltas=delta)
                row_ttt.append(l6_cells(pred,truth,loaded.bridge,lat,spec.variables,spec.leads,denominator))
                row_ttt_meta.append(f"ttt:mult={mult}")
            grad_rows.append(np.stack(row_grad))
            rand_rows.append(np.stack(row_rand))
            ttt_rows.append(np.stack(row_ttt))
            grad_meta=row_grad_meta; rand_meta=row_rand_meta; ttt_grad_norms.append(norm); ttt_meta=row_ttt_meta
        if not grad_rows: raise RuntimeError("shard has no issue rows")
        out=run/f"MENU_EVAL_SHARD_{args.shard_index:02d}.npz"
        np.savez_compressed(out, issue_times=np.asarray(issues), mse_grad=np.asarray(grad_rows),
                            mse_rand=np.asarray(rand_rows), mse_ttt=np.asarray(ttt_rows),
                            grad_arm_labels=np.asarray(grad_meta),rand_arm_labels=np.asarray(rand_meta),
                            ttt_arm_labels=np.asarray(ttt_meta),ttt_grad_norm=np.asarray(ttt_grad_norms))
        receipt.update({"status":"OBSERVED","output":str(out),"output_sha256":digest(out),
                        "n_issues":len(issues),"n_grad_arms":int(np.asarray(grad_rows).shape[1]),
                        "n_rand_arms":int(np.asarray(rand_rows).shape[1]),"n_ttt_arms":int(np.asarray(ttt_rows).shape[1]),
                        "elapsed_seconds":time.perf_counter()-started,
                        "pristine_state_sha256":pristine_state_digest(loaded.model),
                        "multipliers":multipliers})
    except Exception as exc:
        receipt.update({"status":"BLOCKED","error":f"{type(exc).__name__}: {exc}"})
    name="P4_DRY_RUN_RECEIPT.json" if args.dry_run else f"MENU_EVAL_RECEIPT_{args.shard_index:02d}.json"
    (run/name).write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(receipt,indent=2,sort_keys=True)); return 0 if receipt["status"]=="OBSERVED" else 2


if __name__=="__main__": raise SystemExit(main())
