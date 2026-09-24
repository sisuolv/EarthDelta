#!/usr/bin/env python3
"""FP-04 admission consumer check + the pinned diversity grouping (CPU only).

1. `static_adapter.certify_admission_for_fs` on the final admission record, the
   same fail-closed consumer certification every real FP-04 worker runs
   (lead steps 1/4/12, history 2 steps, normalization 3e0b216bfbf34ab0,
   canonical grid / variable-order hashes, FRESH content re-hash per row).
2. `bank_training.assign_diversity_groups(rows, 4, hold_steps=4)` on the
   admitted rows (the grouping reads issue_time / history_time / valid_time /
   split_id / data_role only -- no data value, no model output).

Writes admission_consumer_check.json and bank_fit_grouping.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from earthdelta import bank_training as bt
from earthdelta import static_adapter as sa
from earthdelta.data.pull_wb2 import grid_hash, variable_order_hash

HERE = Path(__file__).resolve().parent
RECORD = HERE / "bank_fit_admission.json"
NORMALIZATION_DIGEST = "3e0b216bfbf34ab0"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    record = json.loads(RECORD.read_text())
    report = sa.certify_admission_for_fs(
        record, lead_steps=(1, 4, 12), required_history_steps=2,
        expected_normalization_digest=NORMALIZATION_DIGEST,
        expected_grid_hash=grid_hash(), expected_variable_order_hash=variable_order_hash(),
        reverify_content=True)
    fresh_equal = all(r.get("fresh_content_sha256") == r.get("content_sha256") for r in report["rows"])
    (HERE / "admission_consumer_check.json").write_text(json.dumps({
        "record": {"path": str(RECORD), "sha256": sha(RECORD)},
        "passed": report["passed"], "n_rows": report["n_rows"],
        "fresh_content_sha256_equals_stored_for_every_row": fresh_equal,
        "report": report}, indent=1) + "\n")

    rows = record["admission"]["admitted"]
    grouping = bt.assign_diversity_groups(rows, 4, hold_steps=4)
    payload = grouping.to_dict()
    payload.pop("created_at", None)
    out = {
        "schema_version": "ed-fp04-bank-fit-grouping/1",
        "admission": {"path": str(RECORD), "sha256": sha(RECORD)},
        "rule": grouping.rule,
        "num_experts": 4,
        "hold_steps": 4,
        "group_sizes": [g.n_samples for g in grouping.groups],
        "groups": [g.to_dict() for g in grouping.groups],
        "purged": grouping.purged,
        "n_input": grouping.n_input,
        "n_assigned": grouping.n_assigned,
    }
    (HERE / "bank_fit_grouping.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"consumer_passed": report["passed"], "fresh_equal": fresh_equal,
                      "group_sizes": out["group_sizes"],
                      "months": [g["months"] for g in out["groups"]],
                      "purged": [(p["issue_id"], p.get("issue_time_utc"), p["reason"])
                                 for p in grouping.purged]}, indent=1))


if __name__ == "__main__":
    main()
