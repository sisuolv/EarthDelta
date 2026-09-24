#!/usr/bin/env python3
"""FP-05b policy_dev manifest: POLICY-DEV-SELECT-v1 exactly as declared.

Inputs (all hash-checked): the declaration, the 2019H2 finiteness scan, the
split freeze, and the coordinator resolution of the >8-drop STOP. Calendar
arithmetic + the finiteness scan only; no store value is read. Rows are the
library's own build_manifest(years=[2019], lead_hours=[72], ...) rows; every row
must pass split_freeze.assert_rows_clear(role='policy_dev', pre_admission=True)
before anything is written.

Writes policy_dev_manifest.json (SplitManifest) and policy_dev_selection.json
(per-target record, drops, nested subsets, strata coverage).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from earthdelta import split_freeze as sf
from earthdelta.data.make_splits import SplitManifest, build_manifest

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
DECLARATION = HERE / "policy_dev_selection_declaration.json"
STOP = HERE / "policy_dev_selection_STOP.json"
RESOLUTION = HERE / "policy_dev_selection_coordinator_resolution.json"
SCAN = RUN / "splits/finiteness_scan_2019_H2.json"
FREEZE = RUN / "splits/split_freeze_v1.json"
OUT = HERE / "policy_dev_manifest.json"
SELECTION = HERE / "policy_dev_selection.json"
NORMALIZATION_HASH = "3e0b216bfbf34ab0"
FIRST, LAST, N_TARGETS, BLOCK_SLOTS, N_BLOCKS, DROP_CAP = 746, 1443, 128, 28, 25, 8


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pinned(p: Path) -> str:
    return (p.parent / (p.stem + ".sha256")).read_text().split()[0]


def main() -> None:
    for p in (OUT, SELECTION):
        if p.exists():
            raise SystemExit(f"{p} exists (write-once)")
    assert sha(DECLARATION) == pinned(DECLARATION)
    assert sha(SCAN) == pinned(SCAN)
    freeze = sf.load_freeze(FREEZE, pinned(FREEZE))
    decl = json.loads(DECLARATION.read_text())
    assert decl["split_freeze"]["sha256"] == freeze.sha256
    resolution = json.loads(RESOLUTION.read_text())
    assert resolution["resolves"]["sha256"] == sha(STOP)
    assert resolution["declaration"]["sha256"] == sha(DECLARATION)
    scan = json.loads(SCAN.read_text())
    assert scan["steps_scanned"] == [736, 1459] and scan["declaration"]["sha256"] == sha(DECLARATION)
    pool = freeze.role("policy_dev")["store_index_window"]
    assert pool == [FIRST, LAST]

    bad = {int(k) for k in scan["steps_with_nonfinite"]}
    candidates = [t for t in range(FIRST, LAST + 1)
                  if all(s not in bad and 736 <= s <= 1459 for s in range(t - 2, t + 13))]
    block = lambda t: (t - FIRST) // BLOCK_SLOTS  # 7-day blocks from 2019-07-06T12Z (store 746)
    targets = [FIRST + int(i * (LAST - FIRST) / (N_TARGETS - 1) + 0.5) for i in range(N_TARGETS)]
    used, prev, record = set(), -1, []
    for i, t in enumerate(targets):
        pool_i = [c for c in candidates if block(c) == block(t) and c not in used and c > prev]
        if not pool_i:
            record.append({"i": i, "target_step": t, "target_utc": sf.iso(sf.parse_utc("2019-01-01T00") + t * 21600),
                           "stratum": block(t), "action": "DROPPED"})
            continue
        pick = min(pool_i, key=lambda c: (abs(c - t), c))
        used.add(pick)
        prev = pick
        record.append({"i": i, "target_step": t, "stratum": block(t), "pick_step": pick,
                       "shift_hours": 6 * (pick - t), "action": "target" if pick == t else "replaced"})
    drops = [r for r in record if r["action"] == "DROPPED"]
    stop = json.loads(STOP.read_text())
    assert len(drops) == stop["declared_rule_outcome"]["n_drops"] == 16 > DROP_CAP
    assert [d["i"] for d in drops] == [d["i"] for d in stop["declared_rule_outcome"]["drops"]]
    assert resolution["consequences"]["N_target"] == N_TARGETS - len(drops)

    t0 = sf.parse_utc("2019-01-01T00")
    full = build_manifest(years=[2019], lead_hours=[72], normalization_hash=NORMALIZATION_HASH,
                          formal=True)
    by_time = {r.issue_time: r for r in full.rows}
    kept = [r for r in record if r["action"] != "DROPPED"]
    rows = []
    for r in kept:
        r["issue_utc"] = sf.iso(t0 + r["pick_step"] * 21600)
        rows.append(by_time[t0 + r["pick_step"] * 21600])
    assert [x.issue_time for x in rows] == sorted(x.issue_time for x in rows)
    clear = sf.assert_rows_clear(freeze, [x.to_dict() for x in rows], "policy_dev", pre_admission=True)
    strata = {b: sum(1 for r in kept if r["stratum"] == b) for b in range(N_BLOCKS)}
    empty = [b for b, n in strata.items() if n == 0]
    assert empty == [16, 19], empty
    block_utc = lambda b: [sf.iso(t0 + (FIRST + BLOCK_SLOTS * b) * 21600),
                           sf.iso(t0 + (FIRST + BLOCK_SLOTS * b + BLOCK_SLOTS - 1) * 21600)]
    gap = [{"stratum": b, "utc": block_utc(b), "n_policy_dev_issues": 0,
            "reason": "REAL_DATA_GAP: no clean t-12h..t+72h window (non-finite runs in 2019.zarr)"}
           for b in empty]
    nested = {"N": [x.issue_id for x in rows],
              "N/2": [x.issue_id for x, r in zip(rows, kept) if r["i"] % 2 == 0],
              "N/4": [x.issue_id for x, r in zip(rows, kept) if r["i"] % 4 == 0]}
    common = {"built_by": Path(__file__).name,
              "declaration": {"path": str(DECLARATION), "sha256": sha(DECLARATION)},
              "finiteness_scan": {"path": str(SCAN), "sha256": sha(SCAN)},
              "split_freeze": {"path": str(FREEZE), "sha256": freeze.sha256},
              "stop_record": {"path": str(STOP), "sha256": sha(STOP)},
              "coordinator_resolution": {"path": str(RESOLUTION), "sha256": sha(RESOLUTION)},
              "build_manifest": {"years": [2019], "lead_hours": [72],
                                 "normalization_hash": NORMALIZATION_HASH, "formal": True},
              "N_target": len(rows), "n_dropped": len(drops),
              "strata_without_policy_dev_representation": gap}
    SplitManifest(rows=rows, metadata={**common, "purpose": "FP-05b policy_dev rows (POLICY-DEV-SELECT-v1)"}
                  ).to_json(OUT)
    selection = {**common, "schema_version": "ed-fp05b-policy-dev-selection/1",
                 "targets": record, "dropped": drops, "strata_counts": strata,
                 "n_strata_represented": N_BLOCKS - len(empty),
                 "nested_subsets": {k: {"n": len(v), "issue_ids": v} for k, v in nested.items()},
                 "max_abs_shift_hours": max(abs(r["shift_hours"]) for r in kept),
                 "n_replaced": sum(r["action"] == "replaced" for r in kept),
                 "pre_admission_clearance": {"passed": clear["passed"], "n_clear": clear["n_clear"],
                                             "freeze_sha256": clear["freeze_sha256"]},
                 "manifest": {"path": str(OUT), "sha256": sha(OUT)}}
    with SELECTION.open("x") as f:
        json.dump(selection, f, indent=1)
        f.write("\n")
    print(json.dumps({"N_target": len(rows), "dropped": len(drops), "strata_empty": empty,
                      "nested": {k: len(v) for k, v in nested.items()},
                      "first": rows[0].event_id, "last": rows[-1].event_id,
                      "max_abs_shift_hours": selection["max_abs_shift_hours"],
                      "manifest_sha256": sha(OUT)}, indent=1))


if __name__ == "__main__":
    main()
