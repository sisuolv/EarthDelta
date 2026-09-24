#!/usr/bin/env python3
"""v2 fresh holdout manifest under amendment A1 (integrity-aware nearest clean slot).

Implements holdout_selection_amendment_A1.json exactly: candidates are 2019
slots whose t-12h..t+72h window is finite in finiteness_scan_2019.json; for
each target T_i = 4 + i*723/15 take the nearest candidate (ties earlier).
Uses no model output and no data statistic beyond finiteness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from earthdelta.data.make_splits import SplitManifest, build_manifest

HERE = Path(__file__).resolve().parent
AMENDMENT = HERE / "holdout_selection_amendment_A1.json"
SCAN = HERE / "finiteness_scan_2019.json"
EPOCH_2019 = 1546300800
STEP = 21600


def main() -> None:
    bad = {int(k) for k in json.loads(SCAN.read_text())["steps_with_nonfinite"]}
    clean = [t for t in range(4, 728) if all(u not in bad for u in range(t - 2, t + 13))]
    targets = [4 + i * 723 / 15 for i in range(16)]
    picks = [min(clean, key=lambda c: (abs(c - T), c)) for T in targets]
    assert len(set(picks)) == 16 and picks == sorted(picks)
    assert all((b - a) * 6 > 84 for a, b in zip(picks, picks[1:]))
    full = build_manifest(years=[2019], lead_hours=[72],
                          normalization_hash="3e0b216bfbf34ab0", formal=True)
    by_step = {(r.issue_time - EPOCH_2019) // STEP: r for r in full.rows}
    rows = [by_step[s] for s in picks]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    SplitManifest(rows=rows, metadata={
        "status": "PROTOCOL_V2 fresh holdout under amendment A1 (never trained on; no model output before the confirmatory job)",
        "amendment": {"path": str(AMENDMENT), "sha256": sha(AMENDMENT)},
        "integrity_scan": {"path": str(SCAN), "sha256": sha(SCAN)},
        "targets": targets, "picks": picks,
    }).to_json(HERE / "v2_holdout_manifest_A1.json")
    print(json.dumps({"picks": picks, "event_ids": [r.event_id for r in rows]}))


if __name__ == "__main__":
    main()
