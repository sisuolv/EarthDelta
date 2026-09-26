#!/usr/bin/env python3
"""Classify the existing 2022 pull log without opening forecast data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--metadata-only", action="store_true", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    text = args.log.read_text(errors="replace") if args.log.is_file() else ""
    h = hashlib.sha256(args.log.read_bytes()).hexdigest() if args.log.is_file() else None
    patterns = {
        "network": [r"ClientConnectorError", r"ConnectionResetError", r"Cannot connect to host"],
        "object_or_manifest": [r"KeyError: '.zmetadata'", r"\.zmetadata"],
        "schema_or_time": [r"KeyError: '2022'", r"not all values found in index 'time'"],
        "verification": [r'"status": "verification_failed"', r"verification_failed"],
        "decoder": [r"decode", r"decoder"],
    }
    matches = {k: sorted({m.group(0) for p in ps for m in re.finditer(p, text, flags=re.I)}) for k, ps in patterns.items()}
    categories = [k for k, vals in matches.items() if vals]
    findings = {
        "schema": "ed-r7-data-metadata-findings/1",
        "log": str(args.log.resolve()),
        "log_exists": args.log.is_file(),
        "log_sha256": h,
        "metadata_only": bool(args.metadata_only),
        "failure_categories_observed": categories,
        "matches": matches,
        "latest_finalize_verification_failed": bool(matches["verification"]),
        "network_failures_observed": bool(matches["network"]),
        "no_forecast_skill_read": True,
        "no_new_download_started": True,
        "interpretation": "The log shows network/object metadata failures followed by a local finalize verification failure; it does not prove a complete 2022 store or any model skill result.",
    }
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out / "findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
    (out / "required_integrity_reads.json").write_text(json.dumps({
        "schema": "ed-r7-data-required-integrity-reads/1",
        "required": ["expected_timestep_count", "all_channel_finite_scan", "zarr_metadata", "manifest_hash", "access_ledger"],
        "status": "MISSING_NO_NUMERIC_INTEGRITY_READ_PERFORMED",
        "new_data_read": False,
    }, indent=2, sort_keys=True) + "\n")
    (out / "access_ledger.json").write_text(json.dumps({
        "schema": "ed-r7-data-access-ledger/1", "year": 2022,
        "audit_reads": [{"path": str(args.log.resolve()), "kind": "metadata_log", "sha256": h}],
        "forecast_or_truth_array_reads": [], "new_downloads": [],
        "status": "LOG_ONLY_NO_MODEL_READS",
    }, indent=2, sort_keys=True) + "\n")
    (out / "root_cause_status.json").write_text(json.dumps({
        "schema": "ed-r7-data-root-cause/1",
        "status": "OBSERVED_METADATA_FAILURES_AND_FINALIZE_VERIFICATION_FAILURE",
        "categories": categories,
        "complete_2022_data_status": "BLOCKED",
        "model_effect_status": "NOT_ASSESSED",
        "new_stage4_authorized": False,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "OBSERVED_METADATA_ONLY", "categories": categories, "log_sha256": h}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
