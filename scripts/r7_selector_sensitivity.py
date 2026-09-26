#!/usr/bin/env python3
"""Read-only, predeclared margin-threshold sensitivity for the sealed DEV OOF.

The threshold grid is fixed in the protocol before labels are consumed. This
script reports a diagnostic curve and never chooses a threshold or writes back
frozen predictions.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.r7_followup_decomposition import _ratio_ci
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` entrypoint
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from r7_followup_decomposition import _ratio_ci

CANDIDATES = ("expert_0", "expert_1", "expert_2", "expert_3")
LEADS = (6, 24, 72)


def load(p: Path) -> Any:
    with p.open() as f:
        return json.load(f)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run(*, candidate_results: Path, predictions: Path, folds: Path, protocol: Path,
        out: Path) -> dict[str, Any]:
    spec = load(protocol)
    if spec.get("status") != "APPROVED_DEVELOPMENT_DIAGNOSTIC_ONLY":
        raise ValueError("selector diagnostic requires approved diagnostic protocol")
    if spec.get("human_decision") != "APPROVED_DEVELOPMENT_REPLAY_ONLY":
        raise ValueError("selector diagnostic requires explicit development approval")
    if spec.get("new_stage4_authorized") is not False or spec.get("confirm_access_authorized") is not False:
        raise ValueError("selector diagnostic cannot authorize Stage4 or confirm")
    if spec.get("candidate_set") != list(CANDIDATES):
        raise ValueError("candidate order differs from the frozen selector contract")
    if spec.get("baseline_method") != "M1":
        raise ValueError("selector diagnostic requires the frozen M1 baseline")
    thresholds = tuple(float(x) for x in spec.get("threshold_grid", []))
    expected = (0.0, 0.000001, 0.0000025, 0.000005, 0.00001, 0.00002, 0.00003, 0.00004)
    if thresholds != expected:
        raise ValueError("threshold grid differs from the predeclared protocol")
    rows = load(candidate_results).get("rows")
    pred = load(predictions).get("predictions")
    issue_blocks = load(folds).get("issue_blocks")
    if not isinstance(rows, list) or not isinstance(pred, dict) or not isinstance(issue_blocks, dict):
        raise ValueError("invalid frozen input schema")
    loss: dict[tuple[str, str, int], float] = {}
    ids: set[str] = set()
    for row in rows:
        iid, cid, lead = row.get("issue_id"), row.get("candidate_id"), row.get("lead_hours")
        if row.get("status") != "PASS" or cid not in ("reference",) + CANDIDATES or lead not in LEADS:
            raise ValueError("candidate matrix outside frozen contract")
        key = (iid, cid, int(lead))
        if key in loss:
            raise ValueError(f"duplicate candidate row {key}")
        value = float(row["cpu_loss"])
        if not math.isfinite(value):
            raise ValueError("non-finite candidate loss")
        loss[key] = value
        ids.add(iid)
    if len(ids) != 112 or set(pred) != ids or set(issue_blocks) != ids:
        raise ValueError("frozen DEV denominator or identity mismatch")
    if len({int(issue_blocks[i]) for i in ids}) != 23:
        raise ValueError("frozen DEV block coverage mismatch")
    expected_rows = 112 * 5 * 3
    if len(loss) != expected_rows:
        raise ValueError(f"incomplete candidate matrix {len(loss)} != {expected_rows}")
    for iid, row in pred.items():
        if row.get("M1") not in CANDIDATES or row.get("M3") not in CANDIDATES:
            raise ValueError("missing frozen M1/M3 action")
        # The frozen gain vector is ordered expert_0..expert_3 and its first
        # entry is the predicted gain for the registered M1 baseline.  The
        # current protocol freezes M1=expert_0; reject a future roster that
        # silently changes that identity instead of subtracting the wrong
        # component.
        if row["M1"] != CANDIDATES[0]:
            raise ValueError("frozen M1 baseline is not expert_0")
        gains = row.get("ridge_pred_gain24")
        if not isinstance(gains, list) or len(gains) != 4 or not all(math.isfinite(float(x)) for x in gains):
            raise ValueError("missing frozen ridge predictions")
    out.mkdir(parents=True, exist_ok=False)
    summaries: list[dict[str, Any]] = []
    records: dict[str, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        key = f"{threshold:.7g}"
        chosen: dict[str, str] = {}
        for iid in sorted(ids):
            row = pred[iid]
            gains = np.asarray(row["ridge_pred_gain24"], dtype=np.float64)
            best_idx = int(np.argmax(gains))
            margin = float(gains[best_idx] - gains[0])
            chosen[iid] = CANDIDATES[best_idx] if margin > threshold else row["M1"]
        recs: dict[str, list[dict[str, Any]]] = {}
        per_lead = {}
        for lead in LEADS:
            block_totals: dict[int, dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
            lead_rows = []
            for iid in sorted(ids):
                fs = loss[(iid, "reference", lead)]
                m1 = loss[(iid, pred[iid]["M1"], lead)]
                action = chosen[iid]
                selected = loss[(iid, action, lead)]
                block = int(issue_blocks[iid])
                item = {"issue_id": iid, "block": block, "threshold": threshold,
                        "action": action, "m1_action": pred[iid]["M1"],
                        "fs_loss": fs, "m1_loss": m1, "selected_loss": selected,
                        "selected_gain_vs_m1": m1 - selected, "selected_gain_vs_fs": fs - selected,
                        "switched": bool(action != pred[iid]["M1"])}
                lead_rows.append(item)
                block_totals[block]["gain_m1"] += m1 - selected
                block_totals[block]["gain_fs"] += fs - selected
                block_totals[block]["fs"] += fs
            recs[str(lead)] = lead_rows
            gain_m1 = _ratio_ci({b: (v["gain_m1"], v["fs"]) for b, v in block_totals.items()}, seed=20260925 + int(round(threshold * 1e8)) + lead * 10, draws=10000)
            gain_fs = _ratio_ci({b: (v["gain_fs"], v["fs"]) for b, v in block_totals.items()}, seed=20260925 + int(round(threshold * 1e8)) + lead * 20, draws=10000)
            per_lead[str(lead)] = {"n": len(lead_rows), "switch_count": sum(x["switched"] for x in lead_rows),
                                   "action_counts": dict(collections.Counter(x["action"] for x in lead_rows)),
                                   "gain_vs_m1": gain_m1, "gain_vs_fs": gain_fs}
        summaries.append({"threshold": threshold, "switch_count": sum(chosen[i] != pred[i]["M1"] for i in ids),
                          "action_counts": dict(collections.Counter(chosen.values())), "per_lead": per_lead})
        records[key] = [x for lead in LEADS for x in recs[str(lead)]]
    zero = next(x for x in summaries if x["threshold"] == 0.0)
    m3_switches = sum(pred[i]["M3"] != pred[i]["M1"] for i in ids)
    zero_actions = {}
    for iid in sorted(ids):
        gains = np.asarray(pred[iid]["ridge_pred_gain24"], dtype=np.float64)
        best_idx = int(np.argmax(gains))
        zero_actions[iid] = CANDIDATES[best_idx] if float(gains[best_idx] - gains[0]) > 0.0 else pred[iid]["M1"]
    if zero["switch_count"] != m3_switches or any(zero_actions[i] != pred[i]["M3"] for i in ids):
        raise ValueError("threshold zero does not reproduce M3 actions")
    (out / "sensitivity.json").write_text(json.dumps({"schema": "ed-r7-selector-sensitivity/1", "diagnostic_only": True, "threshold_grid": list(thresholds), "n_issues": 112, "n_blocks": 23, "summaries": summaries}, indent=2, sort_keys=True) + "\n")
    (out / "per_issue.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    (out / "input_checks.json").write_text(json.dumps({"candidate_results_sha256": sha256(candidate_results), "predictions_sha256": sha256(predictions), "folds_sha256": sha256(folds), "protocol_sha256": sha256(protocol), "m3_zero_threshold_switch_count": m3_switches, "zero_threshold_matches_m3": True, "zero_threshold_action_match": True, "new_weather_data_read": False, "new_model_inference": False, "confirm_data_read": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS_SELECTOR_DIAGNOSTIC", "n": 112, "thresholds": list(thresholds), "m3_zero_threshold_switch_count": m3_switches}, sort_keys=True))
    return {"summaries": summaries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-results", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(candidate_results=args.candidate_results, predictions=args.predictions, folds=args.folds, protocol=args.protocol, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
