#!/usr/bin/env python3
"""FP-03 Phase 2: manifests for the certified bank_fit admission and holdout panel.

Train rows: the SAME 8 issue times as pt-e8y7ib01, in its round-robin order.
Holdout rows: each training issue time + 288h (48 steps; a 6h-grid midpoint
between consecutive training issues, 97 steps apart), never trained on.
Rows are the library's own `build_manifest(years=[2020], lead_hours=[72])`
rows (72h lead so the admission certificate spans t-12h .. t+72h), filtered
and ordered; nothing is hand-constructed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from earthdelta.data.make_splits import SplitManifest, build_manifest

REPO = Path("/mnt/afs/260010168/EarthDelta")
HERE = Path(__file__).resolve().parent
SOURCE_RECORD = REPO / ("artifacts/round2_cci/ed-r4fsformal0923-0923180756-7b1776/"
                        "experts/expert0/fs_fit_record.json")
NORMALIZATION_HASH = "3e0b216bfbf34ab0"  # ed-norm-identity/1 digest in the S0 gate config
HOLDOUT_OFFSET_SECONDS = 288 * 3600


def main() -> None:
    source_sha = hashlib.sha256(SOURCE_RECORD.read_bytes()).hexdigest()
    order = json.loads(SOURCE_RECORD.read_text())["training_issue_ids"][:8]
    full = build_manifest(years=[2020], lead_hours=[72],
                          normalization_hash=NORMALIZATION_HASH, formal=True)
    by_issue = {r.resolved_issue_id(): r for r in full.rows}
    by_time = {r.issue_time: r for r in full.rows}
    train = [by_issue[iid] for iid in order]
    holdout = [by_time[r.issue_time + HOLDOUT_OFFSET_SECONDS] for r in train]
    assert not {r.resolved_issue_id() for r in train} & {r.resolved_issue_id() for r in holdout}
    common = {"built_by": str(Path(__file__).relative_to(REPO)),
              "build_manifest": {"years": [2020], "lead_hours": [72],
                                 "normalization_hash": NORMALIZATION_HASH, "formal": True},
              "issue_order_source": {"path": str(SOURCE_RECORD.relative_to(REPO)),
                                     "sha256": source_sha}}
    SplitManifest(rows=train, metadata={**common, "purpose": "FP-03 Fs bank_fit training rows",
                                        "order": "pt-e8y7ib01 round-robin order"}
                  ).to_json(HERE / "bank_fit_manifest.json")
    SplitManifest(rows=holdout, metadata={**common, "purpose": "FP-03 Fs holdout panel (never trained)",
                                          "rule": "each training issue time + 288h"}
                  ).to_json(HERE / "holdout_panel_manifest.json")
    print(json.dumps({"train": [r.resolved_issue_id() for r in train],
                      "holdout": [r.resolved_issue_id() for r in holdout]}, indent=1))


if __name__ == "__main__":
    main()
