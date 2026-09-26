#!/usr/bin/env python3
"""Read-only Fs/bank lineage and fail-closed admission audit.

This audit never loads model weights and never mutates historical evidence.  It
separates the legacy 500-update records from the qualified H32 assets and
reports whether the current working tree can be used for a new Stage-4 run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def ref(repo: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = repo / p
    return p.resolve()


def finite_trace(record: dict[str, Any]) -> bool:
    return all(math.isfinite(float(v)) for v in record.get("losses", [])) and all(
        math.isfinite(float(v)) for v in record.get("grad_norms", [])
    )


def formal_horizon_ok(record: dict[str, Any], horizon: int) -> bool:
    cfg = record.get("config") or {}
    return (
        record.get("mode") == "formal"
        and int(record.get("n_updates", -1)) == horizon
        and int(cfg.get("max_updates", -1)) == horizon
        and len(record.get("losses", [])) == horizon
        and len(record.get("grad_norms", [])) == horizon
        and finite_trace(record)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    inputs = load(args.inputs)
    repo = Path(inputs.get("repo", ".")).resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    horizon = 32

    legacy_root = repo / "artifacts/round2_cci/ed-r3-j4-bank-formal-v2-0922011307-2d8e12"
    legacy_rows = []
    for path in sorted(legacy_root.glob("expert*/fs_fit_record.json")):
        record = load(path)
        cfg = record.get("config") or {}
        reject_reasons = []
        if int(record.get("n_updates", -1)) != horizon:
            reject_reasons.append("LEGACY_HORIZON_NOT_CURRENT_FORMAL_H32")
        if not formal_horizon_ok(record, horizon):
            reject_reasons.append("FORMAL_TRACE_DOES_NOT_MATCH_H32_CONTRACT")
        if record.get("eligible") is True and reject_reasons:
            reject_reasons.append("RECORDED_ELIGIBLE_FLAG_CANNOT_OVERRIDE_LINEAGE_GATE")
        legacy_rows.append({
            "path": str(path),
            "sha256": sha(path),
            "mode": record.get("mode"),
            "n_updates": record.get("n_updates"),
            "config_max_updates": cfg.get("max_updates"),
            "recorded_eligible": record.get("eligible"),
            "quality_gate": record.get("quality_gate"),
            "accepted_by_current_h32_gate": not reject_reasons,
            "reject_reasons": reject_reasons,
        })
    (out / "assets.json").write_text(json.dumps({
        "schema": "ed-r7-fs-lineage-assets/1",
        "legacy500": {"root": str(legacy_root), "records": legacy_rows,
                       "status": "REJECTED_BY_CURRENT_H32_LINEAGE" if legacy_rows and all(not x["accepted_by_current_h32_gate"] for x in legacy_rows) else "BLOCKED"},
    }, indent=2, sort_keys=True) + "\n")

    fs_dir = repo / "plans/plan_v4_0923/run_20260924T013959Z_fp03_v2/certify/fs"
    rule = load(fs_dir / "quality_rule.json")
    training = load(fs_dir / "training_records.json")
    fs_checks = []
    for name, record_ref in (training.get("V2_P") or {}).items():
        if not isinstance(record_ref, dict) or not record_ref.get("path"):
            fs_checks.append({"name": name, "path": None, "declared_sha256": None,
                              "exists": False, "status": "MISSING_REFERENCE"})
            continue
        p = ref(repo, record_ref["path"])
        row = {"name": name, "path": str(p), "declared_sha256": record_ref.get("sha256"), "exists": p.is_file()}
        if p.is_file():
            row["actual_sha256"] = sha(p)
            if name == "fs_fit_record.json":
                record = load(p)
                row.update({"n_updates": record.get("n_updates"), "mode": record.get("mode"),
                            "eligible": record.get("eligible"), "quality_gate": record.get("quality_gate"),
                            "h32_trace_ok": formal_horizon_ok(record, int(rule.get("horizon", horizon)))})
            else:
                row["status"] = "HASHED_REFERENCE_ONLY"
        fs_checks.append(row)
    cert = load(fs_dir.parent / "V2_certify_decision.json")
    bank_dir = repo / "plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/certify/bank"
    bankq = load(bank_dir / "qualification.json")
    formal = bankq.get("formal") or {}
    bank_files = []
    for p in (bank_dir / "bank_manifest.json", bank_dir / "bank_verify.json", bank_dir / "bank_assembly.json", bank_dir / "registry.json"):
        bank_files.append({"path": str(p), "exists": p.is_file(), "sha256": sha(p) if p.is_file() else None})
    quality = {
        "schema": "ed-r7-fs-lineage-quality/1",
        "fs_rule_horizon": rule.get("horizon"),
        "fs_training_records": fs_checks,
        "fs_certify_verdict": cert.get("verdict"),
        "fs_replica_verdicts": cert.get("replica_verdicts"),
        "fs_substitution": cert.get("substitution"),
        "bank_formal_verdict": formal.get("verdict"),
        "bank_expert_verdicts": formal.get("expert_verdicts"),
        "bank_selected_for_assembly": formal.get("bank_selected_for_assembly"),
        "bank_substitution": formal.get("substitution"),
        "bank_partial_bank": formal.get("partial_bank"),
        "bank_files": bank_files,
        "current_source_is_dirty": True,
        "interpretation": "Historical H32 qualification is separated from legacy 500-update records; it is not a fresh current-HEAD qualification.",
    }
    (out / "quality_replay.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")

    cases = []
    for n in (500, 32):
        fake = {"mode": "formal", "n_updates": n, "config": {"max_updates": n},
                "losses": [0.1] * n, "grad_norms": [0.1] * n}
        cases.append({"n_updates": n, "accepted": formal_horizon_ok(fake, horizon),
                      "expected": n == horizon})
    negative_pass = all(x["accepted"] == x["expected"] for x in cases)
    (out / "negative_entry_tests.json").write_text(json.dumps({
        "schema": "ed-r7-fs-negative-entry-tests/1", "passed": negative_pass,
        "cases": cases, "new_stage4_authorized": False,
    }, indent=2, sort_keys=True) + "\n")

    readiness = {
        "schema": "ed-r7-stage4-readiness/1",
        "ready": False,
        "new_stage4_authorized": False,
        "reasons": [
            "CURRENT_WORKTREE_SOURCE_DIFFERS_FROM_HISTORICAL_GPU_RECEIPTS",
            "NO_NEW_STAGE4_AUTHORIZATION_IN_THIS_PLAN",
            "FP06_PIVOT_STATIC_AND_DYNAMIC_VALUE_NOT_ESTABLISHED",
        ],
        "legacy500_rejected": bool(legacy_rows) and all(not x["accepted_by_current_h32_gate"] for x in legacy_rows),
        "negative_entry_tests_passed": negative_pass,
        "historical_h32_assets_observed": formal.get("verdict") in {"BANK_QUAL_PASS_PENDING_ASSEMBLY", "BANK_QUAL_PASS"},
    }
    (out / "stage4_readiness.json").write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "OBSERVED_READ_ONLY", "legacy_records": len(legacy_rows), "negative_entry_tests": negative_pass, "stage4_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
