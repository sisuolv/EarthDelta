#!/usr/bin/env python3
"""Audit existing WBX Tier-A/Tier-B receipts without rerunning a benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    inp = json.loads(args.inputs.read_text())
    repo = Path(inp.get("repo", ".")).resolve()
    tier = repo / "artifacts/wbx_smoke/20260924T135007Z_tierB/tierb_summary.json"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not tier.is_file():
        status = {"status": "MISSING", "path": str(tier), "new_stage4_authorized": False}
        (out / "decision.json").write_text(json.dumps(status, indent=2) + "\n")
        return 0
    doc = json.loads(tier.read_text())
    table = doc.get("table", [])
    keys = [(r.get("family"), r.get("issue_id"), int(r.get("lead_h"))) for r in table]
    unique = len(keys) == len(set(keys))
    expected = sum((doc.get("expected_anchored_rows") or {}).values())
    anchored = [r for r in table if r.get("anchor") != "none"]
    informational = [r for r in table if r.get("anchor") == "none"]
    counts = {}
    for r in anchored:
        counts[r.get("family")] = counts.get(r.get("family"), 0) + 1
    missing = [k for k, n in (doc.get("expected_anchored_rows") or {}).items() if counts.get(k, 0) != n]
    manual_pairs = [r for r in anchored if r.get("export_bitwise") and r.get("in_job_equals_recorded_bitwise")][:2]
    prov = doc.get("official_code") or {}
    provenance = {
        "actual_head_weatherbench2": ((prov.get("vendored") or {}).get("weatherbench2") or {}).get("actual_head"),
        "actual_head_weatherbenchX": ((prov.get("vendored") or {}).get("weatherbenchX") or {}).get("actual_head"),
        "jax_version": prov.get("jax_version"),
        "numpy_version": prov.get("numpy_version"),
        "source_summary_sha256": sha(tier),
    }
    comparison = {
        "schema": "ed-r7-wbx-receipt-audit/1",
        "source": str(tier),
        "source_sha256": sha(tier),
        "job_id": doc.get("job_id"),
        "n_rows": len(table),
        "n_anchored_rows": len(anchored),
        "n_informational_rows": len(informational),
        "expected_rows": expected,
        "n_anchored_issues": doc.get("n_anchored_issues"),
        "unique_issue_family_lead_keys": unique,
        "family_counts": counts,
        "missing_or_mismatched_families": missing,
        "manual_raw_pairs": manual_pairs,
        "provenance": provenance,
        "informational_rows_retained": bool(len(informational) == len(table) - expected),
        "tierb_claim_supported": bool(len(anchored) == expected and unique and not missing and len(manual_pairs) >= 2),
        "interpretation": "Existing Tier-B receipt is audited read-only; this does not create a new benchmark run or confirm-set result.",
    }
    (out / "full312_comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    (out / "upstream_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (out / "decision.json").write_text(json.dumps({
        "schema": "ed-r7-wbx-decision/1",
        "status": "OBSERVED_EXISTING_TIERB" if comparison["tierb_claim_supported"] else "BLOCKED_OR_INCOMPLETE",
        "tier_a_current_round": "NOT_RUN_AS_CLEAN_PINNED_RECEIPT" ,
        "tier_b": comparison["tierb_claim_supported"],
        "new_stage4_authorized": False,
        "source_sha256": sha(tier),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "OBSERVED_EXISTING_TIERB", "rows": len(table), "unique": unique, "missing": missing}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
