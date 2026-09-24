#!/usr/bin/env python3
"""v2 fresh holdout manifest: 16 issues, 2019-01-02T00Z .. 2019-07-01T18Z, evenly spaced.

Implements holdout_selection_declaration.json exactly (calendar-only; reads no
data values). Rows are the library's own build_manifest rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from earthdelta.data.make_splits import SplitManifest, build_manifest

HERE = Path(__file__).resolve().parent
DECLARATION = HERE / "holdout_selection_declaration.json"
EPOCH_2019 = 1546300800  # 2019-01-01T00:00:00Z
STEP = 21600


def main() -> None:
    declaration_sha = hashlib.sha256(DECLARATION.read_bytes()).hexdigest()
    steps = [4 + round(i * 723 / 15) for i in range(16)]
    assert len(set(steps)) == 16 and steps[0] == 4 and steps[-1] == 727
    full = build_manifest(years=[2019], lead_hours=[72],
                          normalization_hash="3e0b216bfbf34ab0", formal=True)
    by_step = {(r.issue_time - EPOCH_2019) // STEP: r for r in full.rows}
    rows = [by_step[s] for s in steps]
    SplitManifest(rows=rows, metadata={
        "status": "PROTOCOL_V2 fresh holdout (never trained on; no model output before the confirmatory job)",
        "declaration": {"path": str(DECLARATION), "sha256": declaration_sha},
        "steps": steps,
    }).to_json(HERE / "v2_holdout_manifest.json")
    print(json.dumps({"steps": steps, "event_ids": [r.event_id for r in rows]}))


if __name__ == "__main__":
    main()
