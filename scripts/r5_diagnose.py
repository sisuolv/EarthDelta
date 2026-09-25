#!/usr/bin/env python3
"""CPU-only, read-only diagnostics for the sealed FP-05 policy results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from earthdelta import candidate_cache as cc
from earthdelta import policy_oof as po


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return cc.file_sha256(path)


def _inputs(path: Path) -> Dict[str, Any]:
    obj = _json(path)
    required = ("protocol", "cache_manifest", "prediction_freeze", "evaluation")
    for key in required:
        if key not in obj:
            raise ValueError(f"diagnostic input missing {key}")
    return obj


def freeze(args: argparse.Namespace) -> int:
    inp = _inputs(args.inputs)
    refs = {k: {"path": str(Path(inp[k]).resolve()), "sha256": _hash(Path(inp[k]))} for k in inp if k in {
        "protocol", "cache_manifest", "prediction_freeze", "evaluation"}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"schema": "ed-fp07-diagnostic-protocol/1", "inputs": refs,
                "read_only": True, "new_weather_reads": False,
                "labels_are_diagnostic_only": True}
    raw = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (out / "protocol.json").write_bytes(raw)
    (out / "protocol.sha256").write_text(digest + "\n")
    (out / "read_only_manifest.json").write_text(json.dumps({"schema": "ed-fp07-read-only-manifest/1", "before": refs}, indent=2, sort_keys=True) + "\n")
    (out / "strategy_framework.md").write_text(
        "# Diagnostic strategy framework\n\n"
        "All outputs are exploratory diagnostics on the sealed FP-05 development set. "
        "They cannot change the FP06 decision or authorize new data/GPU work.\n"
    )
    print(json.dumps({"protocol": str(out / "protocol.json"), "sha256": digest}))
    return 0


def diagnose(args: argparse.Namespace) -> int:
    inp = _inputs(args.inputs)
    protocol = _json(Path(args.protocol))
    refs = protocol.get("inputs") or {}
    for key, ref in refs.items():
        if _hash(Path(ref["path"])) != ref["sha256"]:
            raise ValueError(f"diagnostic input changed: {key}")
    manifest = _json(Path(inp["cache_manifest"]))
    freeze_rec = _json(Path(inp["prediction_freeze"]))
    rows = _json(Path(manifest["candidate_results"]["path"]))["rows"]
    preds = _json(Path(freeze_rec["oof_predictions"]["path"]))["predictions"]
    bg = _json(Path(manifest["background_f0"]["path"]))["issues"]
    f0 = {iid: rec["f0_cpu"] for iid, rec in bg.items()}
    real = po.realized_losses(rows, preds, f0)
    times = {r["issue_id"]: int(r["issue_time"]) for r in rows}
    blocks = {iid: po.block_of(times[iid]) for iid in preds}
    ids = sorted(preds)
    actions = real["actions"]
    candidates = list(po.CANDIDATES)
    candidate_loss = {c: {} for c in candidates}
    for row in rows:
        if int(row["lead_hours"]) == 24:
            candidate_loss[row["candidate_id"]][row["issue_id"]] = float(row["cpu_loss"])
    matrix = np.array([[candidate_loss[c][iid] for c in candidates] for iid in ids], dtype=np.float64)
    fs = matrix[:, 0]
    gains = fs[:, None] - matrix[:, 1:]
    selected = {m: np.array([matrix[n, candidates.index(actions[iid][m])] for n, iid in enumerate(ids)])
                for m in ("M1", "M2", "M3", "M4", "M5")}
    oracle = matrix.min(axis=1)
    m1 = selected["M1"]
    m3 = selected["M3"]
    switches = [iid for iid in ids if actions[iid]["M3"] != actions[iid]["M1"]]
    switch_deltas = {iid: float(m1[n] - m3[n]) for n, iid in enumerate(ids) if iid in switches}
    centered = gains - gains.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    hashes_before = {k: _hash(Path(v)) for k, v in inp.items() if k in ("protocol", "cache_manifest", "prediction_freeze", "evaluation")}
    (out / "headroom_decomposition.json").write_text(json.dumps({
        "schema": "ed-fp07-headroom/1", "n": len(ids), "mean_gain_vs_fs": {
            c: float(gains[:, j].mean()) for j, c in enumerate(candidates[1:])},
        "oracle_minus_m1_mean_loss": float(np.mean(m1 - oracle)),
        "m3_minus_m1_mean_gain": float(np.mean(m1 - m3)),
        "coverage": manifest.get("identity", {}),
    }, indent=2, sort_keys=True) + "\n")
    (out / "centered_expert_matrix.json").write_text(json.dumps({
        "schema": "ed-fp07-centered-expert-matrix/1", "candidate_order": candidates[1:],
        "n": len(ids), "mean": centered.mean(axis=0).tolist(),
        "singular_values": singular.tolist(), "effective_rank_tol_1e-8": int((singular > 1e-8).sum()),
    }, indent=2, sort_keys=True) + "\n")
    (out / "routing_regret.json").write_text(json.dumps({
        "schema": "ed-fp07-routing-regret/1", "m1_action_counts": _counts(actions, "M1"),
        "m3_action_counts": _counts(actions, "M3"), "switch_count": len(switches),
        "switch_ids": switches, "m3_minus_m1_gain_by_issue": switch_deltas,
        "all_issue_mean_m3_minus_m1_gain": float(np.mean(m1 - m3)),
    }, indent=2, sort_keys=True) + "\n")
    (out / "uncertainty_scope.json").write_text(json.dumps({
        "schema": "ed-fp07-uncertainty-scope/1", "bootstrap_unit": "paired whole 7-day block",
        "n_blocks": len(set(blocks.values())), "n_issues": len(ids),
        "coverage_caveat": _json(Path(inp["evaluation"])).get("coverage_caveat"),
        "interpretation": "descriptive conditional-on-frozen-OOF diagnostics; no causal or confirm claim",
    }, indent=2, sort_keys=True) + "\n")
    hashes_after = {k: _hash(Path(v)) for k, v in inp.items() if k in ("protocol", "cache_manifest", "prediction_freeze", "evaluation")}
    (out / "old_hashes_after.json").write_text(json.dumps({"before": hashes_before, "after": hashes_after, "unchanged": hashes_before == hashes_after}, indent=2, sort_keys=True) + "\n")
    (out / "REPORT.md").write_text(
        f"# FP-07 diagnostics\n\nN={len(ids)}, M3 switches relative to M1 on {len(switches)} issues. "
        "These are read-only diagnostics on the sealed development set.\n"
    )
    print(json.dumps({"n": len(ids), "switch_count": len(switches), "old_hashes_unchanged": hashes_before == hashes_after}))
    return 0


def _counts(actions: Dict[str, Dict[str, str]], method: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for act in actions.values():
        out[act[method]] = out.get(act[method], 0) + 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("freeze-diagnostics", "diagnose"):
        p = sub.add_parser(name)
        p.add_argument("--inputs", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "diagnose":
            p.add_argument("--protocol", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == "freeze-diagnostics":
        return freeze(args)
    return diagnose(args)


if __name__ == "__main__":
    raise SystemExit(main())
