#!/usr/bin/env python3
"""Bounded follow-up decomposition on the sealed FP05b development cache.

This is deliberately a read-only analysis. It never loads model weights, reads a
new weather array, or changes any frozen FP05b file. It separates the attainable
expert gain from the loss caused by selecting the wrong expert and reports paired
whole-block bootstrap intervals at 6/24/72 hours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REQUIRED_CANDIDATES = ("reference", "expert_0", "expert_1", "expert_2", "expert_3")
REQUIRED_METHODS = ("M1", "M3")
HORIZONS = (6, 24, 72)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _ratio_ci(block_values: Mapping[int, tuple[float, float]], *, seed: int, draws: int) -> dict[str, float]:
    blocks = np.array(sorted(block_values), dtype=np.int64)
    if blocks.size < 2:
        raise ValueError("at least two non-empty blocks are required")
    num = np.array([block_values[int(b)][0] for b in blocks], dtype=np.float64)
    den = np.array([block_values[int(b)][1] for b in blocks], dtype=np.float64)
    if not np.isfinite(num).all() or not np.isfinite(den).all() or (den <= 0).any():
        raise ValueError("non-finite or non-positive block denominator")
    point = float(num.sum() / den.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, blocks.size, size=(draws, blocks.size))
    boot = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return {
        "point": point,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": "whole_7day_block_with_replacement",
        "n_blocks": int(blocks.size),
    }


def _mean_ci(block_values: Mapping[int, float], *, seed: int, draws: int) -> dict[str, float]:
    blocks = np.array(sorted(block_values), dtype=np.int64)
    vals = np.array([block_values[int(b)] for b in blocks], dtype=np.float64)
    if blocks.size < 2 or not np.isfinite(vals).all():
        raise ValueError("invalid block means")
    point = float(vals.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, blocks.size, size=(draws, blocks.size))
    boot = vals[idx].mean(axis=1)
    return {
        "point": point,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": "whole_7day_block_with_replacement",
        "n_blocks": int(blocks.size),
    }


def run(*, candidate_results: Path, predictions: Path, folds: Path, protocol: Path,
        out: Path, seed: int = 20260925, draws: int = 10000) -> dict[str, Any]:
    if draws < 1000:
        raise ValueError("draws must be >= 1000 for the preregistered interval")
    proto = load(protocol)
    if proto.get("human_decision") != "APPROVED_DEVELOPMENT_REPLAY_ONLY":
        raise ValueError("protocol is not approved for this development replay")
    if proto.get("new_stage4_authorized") is not False or proto.get("confirm_access_authorized") is not False:
        raise ValueError("development replay cannot authorize Stage4 or confirm access")
    cand_doc = load(candidate_results)
    pred_doc = load(predictions)
    fold_doc = load(folds)
    rows = cand_doc.get("rows")
    pred = pred_doc.get("predictions")
    if not isinstance(rows, list) or not isinstance(pred, dict):
        raise ValueError("invalid candidate or prediction schema")
    loss: dict[tuple[str, str, int], float] = {}
    issue_ids: set[str] = set()
    for row in rows:
        if row.get("status") != "PASS":
            raise ValueError("candidate results include non-PASS rows")
        iid, cid, lead = row.get("issue_id"), row.get("candidate_id"), row.get("lead_hours")
        if cid not in REQUIRED_CANDIDATES or lead not in HORIZONS or not isinstance(iid, str):
            raise ValueError("candidate result outside frozen contract")
        key = (iid, cid, int(lead))
        if key in loss:
            raise ValueError(f"duplicate candidate result {key}")
        val = float(row["cpu_loss"])
        if not math.isfinite(val):
            raise ValueError("non-finite CPU loss")
        loss[key] = val
        issue_ids.add(iid)
    if len(issue_ids) != 112:
        raise ValueError(f"expected frozen N=112, got {len(issue_ids)}")
    expected = len(issue_ids) * len(REQUIRED_CANDIDATES) * len(HORIZONS)
    if len(loss) != expected:
        raise ValueError(f"incomplete candidate matrix: {len(loss)} != {expected}")
    if set(pred) != issue_ids:
        raise ValueError("prediction issue IDs do not match candidate matrix")
    blocks = fold_doc.get("issue_blocks")
    if not isinstance(blocks, dict) or set(blocks) != issue_ids:
        raise ValueError("issue block map does not match candidate matrix")
    nonempty_blocks = sorted({int(blocks[i]) for i in issue_ids})
    if len(nonempty_blocks) != 23:
        raise ValueError(f"expected 23 represented blocks, got {len(nonempty_blocks)}")
    for iid, row in pred.items():
        for method in REQUIRED_METHODS:
            action = row.get(method)
            if action not in REQUIRED_CANDIDATES[1:]:
                raise ValueError(f"{iid} has invalid {method} action {action!r}")
    out.mkdir(parents=True, exist_ok=False)
    records: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {}
    for lead in HORIZONS:
        recs = []
        block_sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for iid in sorted(issue_ids):
            actions = pred[iid]
            fs = loss[(iid, "reference", lead)]
            m1 = loss[(iid, actions["M1"], lead)]
            m3 = loss[(iid, actions["M3"], lead)]
            expert_losses = [loss[(iid, c, lead)] for c in REQUIRED_CANDIDATES[1:]]
            best = min(expert_losses)
            block = int(blocks[iid])
            item = {
                "issue_id": iid,
                "block": block,
                "m1_action": actions["M1"],
                "m3_action": actions["M3"],
                "fs_loss": fs,
                "m1_loss": m1,
                "m3_loss": m3,
                "best_expert_loss": best,
                "direct_expert_gain_vs_m1": m1 - best,
                "chosen_gain_vs_m1": m1 - m3,
                "routing_selection_error": m3 - best,
                "m3_gain_vs_fs": fs - m3,
            }
            recs.append(item)
            for k, v in item.items():
                if k in {"direct_expert_gain_vs_m1", "chosen_gain_vs_m1", "routing_selection_error", "m3_gain_vs_fs"}:
                    block_sums[block][k] += float(v)
            block_sums[block]["fs_loss"] += fs
        records[str(lead)] = recs
        def ratio(metric: str, offset: int = 0) -> dict[str, float]:
            return _ratio_ci({b: (v[metric], v["fs_loss"]) for b, v in block_sums.items()}, seed=seed + lead + offset, draws=draws)
        direct = ratio("direct_expert_gain_vs_m1", 10)
        chosen = ratio("chosen_gain_vs_m1", 20)
        selection = ratio("routing_selection_error", 30)
        m3fs = ratio("m3_gain_vs_fs", 40)
        direct_mean = _mean_ci({b: v["direct_expert_gain_vs_m1"] for b, v in block_sums.items()}, seed=seed + lead + 50, draws=draws)
        summary[str(lead)] = {
            "n_issues": len(recs),
            "n_blocks": len(block_sums),
            "action_counts": {m: dict(Counter(pred[i][m] for i in issue_ids)) for m in REQUIRED_METHODS},
            "normalized_by_sum_fs": {
                "direct_expert_gain_vs_m1": direct,
                "chosen_gain_vs_m1_primary": chosen,
                "routing_selection_error": selection,
                "m3_gain_vs_fs": m3fs,
            },
            "unweighted_block_mean_direct_gain": direct_mean,
        }
    # The 24h estimand is the decision quantity; 72h guard uses the same chosen M3 action.
    primary = summary["24"]['normalized_by_sum_fs']['chosen_gain_vs_m1_primary']
    guard = summary["72"]['normalized_by_sum_fs']['m3_gain_vs_fs']
    summary["decision"] = {
        "primary_estimand": "chosen_action_gain_vs_M1_at_24h",
        "primary": primary,
        "direct_headroom": summary["24"]['normalized_by_sum_fs']['direct_expert_gain_vs_m1'],
        "routing_selection_error": summary["24"]['normalized_by_sum_fs']['routing_selection_error'],
        "harm72_guard_threshold_lower_ci": -0.0034,
        "harm72_guard": guard,
        "harm72_guard_pass": bool(guard["ci_low"] >= -0.0034),
        "interpretation": "development_replay_only; no confirm or generalization claim",
        "stop_rule_triggered": False,
    }
    (out / "per_issue.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps({"schema": "ed-r7-followup-decomposition/1", "seed": seed, "draws": draws, "summary": summary}, indent=2, sort_keys=True) + "\n")
    (out / "input_checks.json").write_text(json.dumps({
        "candidate_results_sha256": sha256(candidate_results),
        "predictions_sha256": sha256(predictions),
        "folds_sha256": sha256(folds),
        "protocol_sha256": sha256(protocol),
        "n_issues": len(issue_ids),
        "represented_blocks": nonempty_blocks,
        "new_weather_data_read": False,
        "new_model_inference": False,
        "confirm_data_read": False,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS_DEVELOPMENT_REPLAY", "n": len(issue_ids), "blocks": len(nonempty_blocks), "primary": primary, "harm72_guard_pass": summary["decision"]["harm72_guard_pass"]}, sort_keys=True))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-results", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--draws", type=int, default=10000)
    args = ap.parse_args()
    run(candidate_results=args.candidate_results, predictions=args.predictions, folds=args.folds, protocol=args.protocol, out=args.out, seed=args.seed, draws=args.draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
