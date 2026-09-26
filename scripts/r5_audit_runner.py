#!/usr/bin/env python3
"""Small CPU-only runners for the post-FP05 audit package."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from earthdelta.fp06_decision import decide_fp06


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_inputs(path: Path) -> dict[str, Any]:
    obj = _json(path)
    if "evaluation" in obj:
        ev = _json(Path(obj["evaluation"]))
    else:
        ev = obj
    root_caveat = obj.get("coverage_caveat") or ev.get("coverage_caveat")
    if "primary_7day_blocks" in ev:
        ev = ev["primary_7day_blocks"]
    comparisons = ev.get("comparisons", {})
    # This is a typed trust gate.  Missing values and values such as the
    # string "false" must never be coerced to a scientific PASS.
    raw_evidence_valid = obj.get("evidence_valid")
    verified_bundle = obj.get("verified_bundle") is True
    # A formal bundle must carry numeric CI endpoints so the categorical
    # labels cannot be supplied by a caller without recomputation.  The
    # historical unit-test fixture intentionally omits this field and remains
    # a pure rule test, never a scientific receipt.
    if verified_bundle:
        if not isinstance(obj.get("source_manifest"), dict) or not obj.get("input_manifest"):
            raw_evidence_valid = False
        delta = float(obj.get("delta_min", 0.0034))
        required = ("oracle_vs_Fs", "oracle_vs_M1", "M1_vs_Fs", "M3_vs_Fs", "M3_vs_M1", "M3_vs_Fs_72h")
        for key in required:
            row = comparisons.get(key)
            if not isinstance(row, dict) or not all(k in row for k in ("point", "ci_low", "ci_high", "status_vs_delta_min")):
                raw_evidence_valid = False
                continue
            try:
                point, lo, hi = float(row["point"]), float(row["ci_low"]), float(row["ci_high"])
                expected = "ABOVE" if lo >= delta else ("BELOW" if hi < delta else "STRADDLE")
                if not all(map(math.isfinite, (point, lo, hi))) or row["status_vs_delta_min"] != expected:
                    raw_evidence_valid = False
                if key == "M3_vs_Fs_72h" and row.get("harm72_guard_pass") != (lo >= -delta):
                    raw_evidence_valid = False
            except (TypeError, ValueError):
                raw_evidence_valid = False
    return {
        "evidence_valid": raw_evidence_valid is True,
        "evidence_valid_type": type(raw_evidence_valid).__name__,
        "verified_bundle": verified_bundle,
        "source_manifest": obj.get("source_manifest"),
        "input_manifest": obj.get("input_manifest"),
        "comparisons": comparisons,
        "coverage_caveat": root_caveat or ev.get("coverage_caveat"),
        "source": obj.get("evaluation", str(path)),
    }


def fp06_decision(args: argparse.Namespace) -> int:
    inputs = _load_inputs(args.inputs)
    decision = decide_fp06(inputs, delta_min=float(args.delta_min))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "ed-fp06-decision/1",
        "inputs": inputs,
        "decision": decision,
        # Keep the machine-facing verdict at the top level as well as in the
        # full decision object.  Older receipts only populated
        # ``decision.verdict``; downstream readers should not have to guess
        # which nesting convention a receipt used.
        "verdict": decision["verdict"],
        "rule_id": decision.get("reason"),
        "status": decision["verdict"],
        "coverage_caveat": inputs.get("coverage_caveat"),
    }
    (out / "rules_truth_table.json").write_text(json.dumps({
        "schema": "ed-fp06-rules-truth-table/1",
        "rule_order": [
            "INVALID_EVIDENCE", "STOP_CURRENT_BANK", "PIVOT_STATIC",
            "STOP_CURRENT_SELECTOR", "CONTINUE_NEXT_ITERATION", "INCONCLUSIVE",
        ],
        "current": decision,
    }, indent=2, sort_keys=True) + "\n")
    (out / "decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out / "claims.json").write_text(json.dumps({
        "fp06_verdict": decision["verdict"],
        "scientific_status": "PIVOT_STATIC" if decision["verdict"] == "PIVOT_STATIC" else decision["verdict"],
        "does_not_mean": ["static method passed delta_min", "new GPU is authorized", "novelty is established"],
        "coverage_caveat": inputs.get("coverage_caveat"),
    }, indent=2, sort_keys=True) + "\n")
    (out / "HANDOFF.md").write_text(
        "# FP06 decision\n\n"
        f"The frozen first-match rule returns **{decision['verdict']}**.\n\n"
        "This is a decision on the recorded FP-05 development evidence. It does not "
        "authorize new GPU work, confirm access, or upgrade the novelty claim.\n"
    )
    print(json.dumps(decision, sort_keys=True))
    # A valid negative science result is a successful computation.  Invalid
    # or uncovered evidence is an execution failure and must propagate a
    # nonzero status to shell/CI callers.
    if decision["verdict"] in {"INVALID", "INCONCLUSIVE_RULE_COVERAGE"}:
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fp06-decision")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--delta-min", type=float, default=0.0034)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fp06-decision":
        return fp06_decision(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
