#!/usr/bin/env python3
"""X1 (EXPLORATORY, not FS-QUAL-v1): nested training manifests for N=32 and N=64.

Rows are the library's own `build_manifest(years=[2020], lead_hours=[72])` rows.
  * The original 8 FP-03 training issues come first, in the protocol-v1 order,
    so the first 8 updates of every X1 run replay J3's first 8 updates exactly.
  * New issues come only from 2020 slots at least 288h (the N=8 design's own
    minimum train-holdout separation) from EVERY one of the 8 fixed holdout
    issues. That leaves the block 2020-07-13T18Z .. 2020-12-28T18Z (plus
    single slots 6h before each original training issue, excluded as
    near-duplicates). So each holdout issue's nearest training issue stays
    288h away at every N, and a holdout change cannot come from proximity.
  * G56 = 56 slots evenly spaced over that block; N=64 adds G56; N=32 adds the
    24-point evenly spaced subset of G56, so N=8 < N=32 < N=64 are nested.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from earthdelta.data.make_splits import SplitManifest, build_manifest

REPO = Path("/mnt/afs/260010168/EarthDelta")
HERE = Path(__file__).resolve().parent
V1 = REPO / "plans/plan_v4_0923/run_20260923T192759Z_fp03_fs/protocol/fs_protocol_v1.json"
NORMALIZATION_HASH = "3e0b216bfbf34ab0"
EPOCH_2020 = 1577836800  # 2020-01-01T00:00:00Z
STEP = 21600
BLOCK = (779, 1451)       # first/last eligible step of the purge-respecting block
PURGE_STEPS = 48          # 288h, the N=8 design's minimum train-holdout separation


def main() -> None:
    protocol = json.loads(V1.read_text())
    train_ids = list(protocol["inputs"]["train_issue_ids"])
    holdout_ids = list(protocol["inputs"]["holdout_issue_ids"])
    full = build_manifest(years=[2020], lead_hours=[72],
                          normalization_hash=NORMALIZATION_HASH, formal=True)
    by_issue = {r.resolved_issue_id(): r for r in full.rows}
    by_step = {(r.issue_time - EPOCH_2020) // STEP: r for r in full.rows}
    original = [by_issue[i] for i in train_ids]
    holdout_steps = [(by_issue[i].issue_time - EPOCH_2020) // STEP for i in holdout_ids]

    lo, hi = BLOCK
    g56 = [lo + round(i * (hi - lo) / 55) for i in range(56)]
    n32_new = [g56[round(j * 55 / 23)] for j in range(24)]
    assert len(set(g56)) == 56 and len(set(n32_new)) == 24 and set(n32_new) <= set(g56)
    for s in g56:
        assert s in by_step, s
        assert all(abs(s - h) >= PURGE_STEPS for h in holdout_steps), s
    new64 = [by_step[s] for s in g56]
    new32 = [by_step[s] for s in n32_new]
    ids = lambda rows: [r.resolved_issue_id() for r in rows]
    assert not set(ids(new64)) & set(holdout_ids) and not set(ids(new64)) & set(train_ids)

    common = {
        "status": "EXPLORATORY_X1 - not FS-QUAL-v1, not pre-registered, cannot select/freeze/certify an Fs",
        "built_by": str(Path(__file__).relative_to(REPO)),
        "build_manifest": {"years": [2020], "lead_hours": [72],
                           "normalization_hash": NORMALIZATION_HASH, "formal": True},
        "original_8_from": {"path": str(V1.relative_to(REPO)),
                            "sha256": hashlib.sha256(V1.read_bytes()).hexdigest()},
        "purge_rule": "every new issue >= 288h from every one of the 8 fixed holdout issues",
        "new_issue_block_steps": list(BLOCK),
        "g56_steps": g56,
    }
    SplitManifest(rows=original + new32, metadata={**common, "N": 32, "new_steps": n32_new}
                  ).to_json(HERE / "x1_n32_manifest.json")
    SplitManifest(rows=original + new64, metadata={**common, "N": 64, "new_steps": g56}
                  ).to_json(HERE / "x1_n64_manifest.json")
    print(json.dumps({"n32_new_steps": n32_new, "n64_new_steps": g56,
                      "n32_first_last_new": [str(new32[0].event_id), str(new32[-1].event_id)],
                      "min_new_to_holdout_hours": min(abs(s - h) for s in g56 for h in holdout_steps) * 6}))


if __name__ == "__main__":
    main()
