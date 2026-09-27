#!/usr/bin/env python3
"""Step 2: fit the preregistered static Fs* adapter and OBC reference.

This worker intentionally keeps the probe objective separate from the older
native-loss training helpers.  The probe's frozen objective is L6 with the
denominator written by P3, and every rollout is a complete 21-state path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from earthdelta.bridge.stormer_bridge import DEFAULT_VARIABLES, controlled_rollout
from earthdelta.probe.contracts import issue_sets, load_probe_spec
from earthdelta.probe.rollout import l6_cells, l6_per_issue, load_bridge, rollout_trajectory
from earthdelta.static_adapter import build_fs_adapter, fs_edit_plan
from scripts.probe_p3_directions import load_issue


TARGET_BLOCKS = (18, 19, 20, 21, 22, 23)
SPEC_SHA = "bf247b2abc86ef81be62a7aa5a26318532ca3e3b5ceaeb492c98387781518348"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _lat_weights(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    w = w / w.sum()
    return w[:, None]


def metric_mse(pred: np.ndarray, truth: np.ndarray, lat: np.ndarray, ch: int) -> float:
    # This helper is used for one already-selected channel and accepts either
    # [B,H,W] or [H,W] arrays; keep the channel selection outside it.
    err = np.asarray(pred) - np.asarray(truth)
    return float((err * err * _lat_weights(lat)).sum(axis=(-2, -1)).mean())


def raw_trajectory(bridge, trajectory: torch.Tensor) -> np.ndarray:
    b, t, c, h, w = trajectory.shape
    return bridge.denormalize(trajectory.reshape(-1, c, h, w)).reshape_as(trajectory).detach().cpu().numpy()


def load_contract(run: Path, spec_path: str):
    spec = load_probe_spec(spec_path, expected_sha256=SPEC_SHA)
    qc = json.loads((run / "DATA_QC_RECEIPT.json").read_text())
    directions_receipt = json.loads((run / "DIRECTIONS_RECEIPT.json").read_text())
    if qc.get("status") != "OBSERVED":
        raise RuntimeError("P1 QC is not OBSERVED")
    if directions_receipt.get("status") != "OBSERVED":
        raise RuntimeError("P3 directions are not OBSERVED")
    d = np.load(run / "DIRECTIONS.npz", allow_pickle=False)
    if digest(run / "DIRECTIONS.npz") != directions_receipt["output_sha256"]:
        raise RuntimeError("DIRECTIONS.npz hash mismatch")
    denominator = directions_receipt["denominator"]
    coord = qc["coordinate_cache"]
    cp = Path(coord["path"])
    if digest(cp) != coord["sha256"]:
        raise RuntimeError("coordinate cache hash mismatch")
    lat = np.asarray(np.load(cp, allow_pickle=False)["lat"], dtype=np.float64)
    return spec, qc, denominator, lat, issue_sets(spec)


def issue_l6(loaded, data: np.ndarray, lat: np.ndarray, spec, denominator, device,
             adapters=None) -> tuple[float, np.ndarray, np.ndarray]:
    x = loaded.bridge.normalize(torch.from_numpy(np.ascontiguousarray(data[0:1])).float().to(device))
    truth = torch.from_numpy(np.ascontiguousarray(data)).float().unsqueeze(0).to(device)
    if adapters is None:
        with torch.inference_mode():
            trajectory = rollout_trajectory(loaded.bridge, x, steps=20)
    else:
        plan = fs_edit_plan(20)
        with torch.inference_mode():
            trajectory = controlled_rollout(
                loaded.bridge, x, list(DEFAULT_VARIABLES), interval=6, steps=20,
                plan=plan, expert_loras=adapters, target_blocks=TARGET_BLOCKS,
                return_trajectory=True,
            )
    value = float(l6_per_issue(trajectory, truth, loaded.bridge, lat, spec.variables,
                               spec.leads, denominator).detach().cpu())
    cells = l6_cells(trajectory, truth, loaded.bridge, lat, spec.variables, spec.leads, denominator)
    raw = raw_trajectory(loaded.bridge, trajectory)
    return value, cells, raw


def pooled(values: list[float]) -> float:
    if not values:
        raise RuntimeError("empty score list")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def make_adapter(loaded, seed: int):
    hidden = int(loaded.model.blocks[TARGET_BLOCKS[0]].attn.proj.in_features)
    adapters = build_fs_adapter(hidden, TARGET_BLOCKS, rank_per_expert=4, seed=seed)
    for lora in adapters.values():
        lora.to(next(loaded.model.parameters()).device)
        lora.requires_grad_(True)
    for p in loaded.model.parameters():
        p.requires_grad_(False)
    return adapters


def state_cpu(adapters):
    return {str(block): {k: v.detach().cpu().clone() for k, v in lora.state_dict().items()}
            for block, lora in adapters.items()}


def restore_state(adapters, state):
    for block, lora in adapters.items():
        lora.load_state_dict(state[str(block)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--norm-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--job", default="local")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--max-updates", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--do-obc", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir).resolve(); run.mkdir(parents=True, exist_ok=True)
    out_name = f"STEP2_LR_{args.learning_rate:.0e}.json" if not args.dry_run else "STEP2_DRY_RUN_RECEIPT.json"
    receipt = {"schema": "earthdelta.probe.step2.v1", "status": "BLOCKED", "job": args.job,
               "learning_rate": args.learning_rate, "dry_run": bool(args.dry_run),
               "spec_sha256": SPEC_SHA}
    started = time.perf_counter()
    try:
        spec, qc, denominator, lat, names = load_contract(run, args.spec)
        device = torch.device(args.device)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        loaded = load_bridge(args.checkpoint, args.norm_dir, device)
        if loaded.checkpoint_sha256 != spec.raw["bindings"]["checkpoint"]["sha256"]:
            raise RuntimeError("checkpoint hash mismatch")
        train = names["T_static_train"]
        holdout = names["H_static_holdout"]
        valid = names["V_static_eval"]
        if args.dry_run:
            train, holdout, valid = train[:2], holdout[:2], valid[:1]
            updates, eval_every = 2, 1
        else:
            updates, eval_every = int(args.max_updates), int(args.eval_every)
            if updates != 400 or eval_every != 50:
                raise ValueError("formal Step 2 is frozen at 400 updates/eval_every=50")
            (run / "STEP2_STARTED.marker").open("a").close()
        # F0 holdout baseline is fixed before looking at any adapter result.
        f0_hold = []
        for issue in holdout:
            data = load_issue(qc, issue, cache_dir=Path(args.cache_dir), ledger=run / "ACCESS_LEDGER.jsonl",
                              run_id=run.name, job=args.job, stage="STEP2")
            f0_hold.append(issue_l6(loaded, data, lat, spec, denominator, device)[0])
        f0_hold_score = pooled(f0_hold)
        adapters = make_adapter(loaded, args.seed)
        params = [p for lora in adapters.values() for p in lora.parameters()]
        optimizer = torch.optim.Adam(params, lr=float(args.learning_rate))
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(train)).tolist()
        losses = []
        eval_history = []
        best_score = float("inf"); best_update = 0; best_state = state_cpu(adapters)
        for update in range(1, updates + 1):
            issue = train[order[(update - 1) % len(order)]]
            data = load_issue(qc, issue, cache_dir=Path(args.cache_dir), ledger=run / "ACCESS_LEDGER.jsonl",
                              run_id=run.name, job=args.job, stage="STEP2")
            x = loaded.bridge.normalize(torch.from_numpy(np.ascontiguousarray(data[0:1])).float().to(device))
            truth = torch.from_numpy(np.ascontiguousarray(data)).float().unsqueeze(0).to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = controlled_rollout(
                loaded.bridge, x, list(DEFAULT_VARIABLES), interval=6, steps=20,
                plan=fs_edit_plan(20), expert_loras=adapters, target_blocks=TARGET_BLOCKS,
                return_trajectory=True,
            )
            loss = l6_per_issue(trajectory, truth, loaded.bridge, lat, spec.variables, spec.leads, denominator)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at update {update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if update % eval_every == 0 or update == updates:
                scores = []
                with torch.no_grad():
                    for h_issue in holdout:
                        hdata = load_issue(qc, h_issue, cache_dir=Path(args.cache_dir), ledger=run / "ACCESS_LEDGER.jsonl",
                                           run_id=run.name, job=args.job, stage="STEP2")
                        scores.append(issue_l6(loaded, hdata, lat, spec, denominator, device, adapters)[0])
                score = pooled(scores)
                eval_history.append({"update": update, "pooled_l6": score})
                if score < best_score:
                    best_score, best_update, best_state = score, update, state_cpu(adapters)
        restore_state(adapters, best_state)
        # Evaluate the selected Fs* on the V set.  This remains a fit-derived
        # diagnostic here; it is not the final 2021 confirmation.
        v_scores = []; v_cells = []; v_f0_scores = []; v_f0_cells = []
        with torch.no_grad():
            for issue in valid:
                data = load_issue(qc, issue, cache_dir=Path(args.cache_dir), ledger=run / "ACCESS_LEDGER.jsonl",
                                  run_id=run.name, job=args.job, stage="STEP2")
                f0s, f0c, _ = issue_l6(loaded, data, lat, spec, denominator, device)
                s, c, _ = issue_l6(loaded, data, lat, spec, denominator, device, adapters)
                v_scores.append(s); v_cells.append(c)
                v_f0_scores.append(f0s); v_f0_cells.append(f0c)
        receipt.update({"status": "OBSERVED", "f0_holdout_pooled_l6": f0_hold_score,
                        "selected_holdout_pooled_l6": best_score, "selected_update": best_update,
                        "qualified": bool(best_score < f0_hold_score), "loss_initial": losses[0],
                        "loss_final": losses[-1], "n_updates": updates, "n_train": len(train),
                        "n_holdout": len(holdout), "n_eval": len(valid),
                        "v_f0_comparison": {"selected_pooled_l6": pooled(v_scores),
                                            "f0_pooled_l6": pooled(v_f0_scores),
                                            "cells_mean": np.mean(np.asarray(v_cells), axis=0).tolist(),
                                            "f0_cells_mean": np.mean(np.asarray(v_f0_cells), axis=0).tolist(),
                                            "improvement_pct": (100.0 * (1.0 - np.sqrt(np.mean(np.asarray(v_cells), axis=0) /
                                                                            np.maximum(np.mean(np.asarray(v_f0_cells), axis=0), 1e-30)))).tolist()},
                        "eval_history": eval_history, "elapsed_seconds": time.perf_counter() - started})
        # Save only the selected adapter factors; no model/checkpoint is altered.
        adapter_path = run / f"FS_STAR_ADAPTER_{args.learning_rate:.0e}.pt"
        torch.save(best_state, adapter_path)
        receipt["adapter_path"] = str(adapter_path)
        receipt["adapter_sha256"] = digest(adapter_path)
        if args.do_obc:
            bias = np.zeros((len(spec.variables), len(spec.leads), len(lat), 256), dtype=np.float64)
            for issue in train:
                data = load_issue(qc, issue, cache_dir=Path(args.cache_dir), ledger=run / "ACCESS_LEDGER.jsonl",
                                  run_id=run.name, job=args.job, stage="STEP2_OBC")
                _, _, f0raw = issue_l6(loaded, data, lat, spec, denominator, device)
                for vi, var in enumerate(spec.variables):
                    ch = int(var["channel_index"])
                    for li, lead in enumerate(spec.leads):
                        step = int(lead // 6)
                        bias[vi, li] += f0raw[0, step, ch] - data[step, ch]
            bias /= len(train)
            bias_path = run / "OBC_BIAS.npy"; np.save(bias_path, bias)
            lambdas = [0., .25, .5, .75, 1.]
            h_scores = []
            for lam in lambdas:
                vals = []
                for issue in holdout:
                    data = load_issue(qc, issue, cache_dir=Path(args.cache_dir), ledger=run / "ACCESS_LEDGER.jsonl",
                                      run_id=run.name, job=args.job, stage="STEP2_OBC")
                    _, _, f0raw = issue_l6(loaded, data, lat, spec, denominator, device)
                    terms = []
                    for vi, var in enumerate(spec.variables):
                        ch = int(var["channel_index"])
                        for li, lead in enumerate(spec.leads):
                            step = int(lead // 6); pred = f0raw[0, step, ch] - lam * bias[vi, li]
                            terms.append(metric_mse(pred[None], data[step, ch][None], lat, 0) / denominator[f"{var['name']}@{lead}h"]**2)
                    vals.append(float(np.mean(terms)))
                h_scores.append(pooled(vals))
            best_li = int(np.argmin(h_scores)); selected_lambda = lambdas[best_li]
            receipt["obc"] = {"bias_path": str(bias_path), "bias_sha256": digest(bias_path),
                               "lambda_grid": lambdas, "holdout_pooled_l6": h_scores,
                               "selected_lambda": selected_lambda,
                               "useful_on_holdout": bool(h_scores[best_li] < f0_hold_score)}
            (run / "OBC_RECEIPT.json").write_text(json.dumps(receipt["obc"], indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        receipt.update({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.perf_counter() - started})
    (run / out_name).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "OBSERVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
