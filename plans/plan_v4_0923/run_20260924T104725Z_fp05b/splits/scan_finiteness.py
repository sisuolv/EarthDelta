#!/usr/bin/env python3
"""FP-05b integrity-only scan of 2019 H2: which timesteps contain non-finite values.

Copied from FP-03 v2's holdout/scan_finiteness.py (finiteness only; records no
value, statistic or model output) and given a first-step argument. Must run
AFTER admission/policy_dev_selection_declaration.json is hashed (checked below)
and BEFORE the policy_dev manifest exists.

    PYTHONPATH=.pydeps python3 scan_finiteness.py --year 2019 --first-step 736 \
        --last-step 1459 --out finiteness_scan_2019_H2.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
DECLARATION = HERE.parent / "admission/policy_dev_selection_declaration.json"
MANIFEST = HERE.parent / "admission/policy_dev_manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--first-step", type=int, required=True)
    ap.add_argument("--last-step", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    decl_sha = hashlib.sha256(DECLARATION.read_bytes()).hexdigest()
    pinned = (DECLARATION.parent / "policy_dev_selection_declaration.sha256").read_text().split()[0]
    assert decl_sha == pinned, "declaration changed after it was hashed"
    assert not MANIFEST.exists(), "the scan must precede the manifest"
    if args.out.exists():
        raise SystemExit(f"{args.out} exists (write-once)")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    g = zarr.open(f"/mnt/afs/260010168/EarthDelta/data/era5_1p40625/{args.year}.zarr", "r")
    data = g["data"]
    channels = [str(c) for c in g["channel"][:]]
    bad = {}
    for t in range(args.first_step, args.last_step + 1):
        x = data[t]
        finite = np.isfinite(x)
        if not finite.all():
            per_channel = (~finite).reshape(x.shape[0], -1).any(axis=1)
            bad[t] = [channels[i] for i in np.nonzero(per_channel)[0]]
    t0 = dt.datetime(args.year, 1, 1, tzinfo=dt.timezone.utc)
    utc = lambda i: (t0 + dt.timedelta(hours=6 * i)).strftime("%Y-%m-%dT%HZ")
    out = {"year": args.year, "steps_scanned": [args.first_step, args.last_step],
           "utc_range": [utc(args.first_step), utc(args.last_step)],
           "n_steps_scanned": args.last_step - args.first_step + 1,
           "n_steps_with_nonfinite": len(bad),
           "steps_with_nonfinite": {str(k): v for k, v in bad.items()},
           "declaration": {"path": str(DECLARATION), "sha256": decl_sha},
           "started_utc": started, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "exposure_kind": "INTEGRITY_READ_ONLY",
           "note": "finiteness only; no values, statistics or model outputs recorded"}
    with args.out.open("x") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(args.year, "steps with non-finite values:", len(bad), "of", out["n_steps_scanned"])


if __name__ == "__main__":
    main()
