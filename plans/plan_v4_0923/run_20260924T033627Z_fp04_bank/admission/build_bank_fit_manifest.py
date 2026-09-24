#!/usr/bin/env python3
"""FP-04 bank_fit manifest: the 49-slot 90h calendar grid of the frozen declaration.

Implements bank_fit_selection_declaration.json exactly: t_i = 2020-01-02T00Z +
90h * i (i = 0..48), the library's own `build_manifest(years=[2020],
lead_hours=[72], ...)` rows at those issue times, in time order. Uses calendar
arithmetic only -- no data value is read here.

    python3 build_bank_fit_manifest.py                      # targets (attempt 1)
    python3 build_bank_fit_manifest.py --replace i:+h ...   # final manifest after
                                                            # the declared replacement rule
    python3 build_bank_fit_manifest.py --candidates i ...   # replacement candidates for slots
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from earthdelta.data.make_splits import SplitManifest, build_manifest

HERE = Path(__file__).resolve().parent
DECLARATION = HERE / "bank_fit_selection_declaration.json"
NORMALIZATION_HASH = "3e0b216bfbf34ab0"
T0 = datetime(2020, 1, 2, 0, tzinfo=timezone.utc)
STRIDE_H = 90
N_SLOTS = 49
WINDOW = (datetime(2020, 1, 1, 12, tzinfo=timezone.utc), datetime(2020, 7, 4, 18, tzinfo=timezone.utc))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target(i: int) -> datetime:
    return T0 + timedelta(hours=STRIDE_H * i)


def in_window(t: datetime) -> bool:
    return t - timedelta(hours=12) >= WINDOW[0] and t + timedelta(hours=72) <= WINDOW[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", nargs="*", default=[], help="slot:+hours (declared rule)")
    parser.add_argument("--drop", nargs="*", type=int, default=[], help="slots with no replacement")
    parser.add_argument("--candidates", nargs="*", type=int, default=None,
                        help="write the replacement-candidate manifest for these slots")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    declaration_sha = sha(DECLARATION)
    full = build_manifest(years=[2020], lead_hours=[72], normalization_hash=NORMALIZATION_HASH,
                          formal=True)
    by_time = {r.issue_time: r for r in full.rows}
    targets = [target(i) for i in range(N_SLOTS)]
    assert all(in_window(t) for t in targets) and not in_window(target(N_SLOTS))

    common = {"built_by": str(Path(__file__).name),
              "declaration": {"path": str(DECLARATION), "sha256": declaration_sha},
              "build_manifest": {"years": [2020], "lead_hours": [72],
                                 "normalization_hash": NORMALIZATION_HASH, "formal": True}}
    if args.candidates is not None:
        rows, meta = [], []
        for i in args.candidates:
            for h in (6, 12, 18, 24):
                t = targets[i] + timedelta(hours=h)
                if t.month != targets[i].month or not in_window(t):
                    continue
                rows.append(by_time[int(t.timestamp())])
                meta.append({"slot": i, "shift_hours": h, "issue_utc": t.isoformat()})
        SplitManifest(rows=rows, metadata={**common, "purpose": "FP-04 replacement candidates",
                                           "candidates": meta}).to_json(args.out)
        print(json.dumps({"n_candidates": len(rows), "candidates": meta}))
        return

    shifts = {}
    for item in args.replace:
        slot, hours = item.split(":")
        shifts[int(slot)] = int(hours)
    chosen, record = [], []
    for i, t in enumerate(targets):
        if i in args.drop:
            record.append({"slot": i, "target_utc": t.isoformat(), "action": "dropped"})
            continue
        h = shifts.get(i, 0)
        assert h in (0, 6, 12, 18, 24)
        pick = t + timedelta(hours=h)
        assert pick.month == t.month and in_window(pick)
        chosen.append(by_time[int(pick.timestamp())])
        record.append({"slot": i, "target_utc": t.isoformat(), "pick_utc": pick.isoformat(),
                       "shift_hours": h, "action": "target" if h == 0 else "replaced"})
    assert [r.issue_time for r in chosen] == sorted(r.issue_time for r in chosen)
    SplitManifest(rows=chosen, metadata={**common, "purpose": "FP-04 bank_fit training rows",
                                         "slots": record}).to_json(args.out)
    print(json.dumps({"n_rows": len(chosen), "first": chosen[0].event_id,
                      "last": chosen[-1].event_id,
                      "replaced": [r for r in record if r["action"] == "replaced"],
                      "dropped": [r for r in record if r["action"] == "dropped"]}))


if __name__ == "__main__":
    main()
