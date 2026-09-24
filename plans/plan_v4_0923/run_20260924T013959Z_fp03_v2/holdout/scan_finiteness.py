#!/usr/bin/env python3
"""Integrity-only scan: which timesteps of a year's store contain non-finite values.

Reads raw values ONLY to test finiteness (ARTIFACT_CONTRACT section 3 allows
raw integrity checks). Records no statistics and runs no model. Steps 0..735
(2019/2018/2015: Jan 1 00Z .. Jul 3 18Z) cover the holdout window plus history
and 72h margins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent


def main(year: int, last_step: int = 735) -> None:
    g = zarr.open(f"/mnt/afs/260010168/EarthDelta/data/era5_1p40625/{year}.zarr", "r")
    data = g["data"]
    channels = [str(c) for c in g["channel"][:]]
    bad = {}
    for t in range(0, last_step + 1):
        x = data[t]
        finite = np.isfinite(x)
        if not finite.all():
            per_channel = (~finite).reshape(x.shape[0], -1).any(axis=1)
            bad[t] = [channels[i] for i in np.nonzero(per_channel)[0]]
    out = {"year": year, "steps_scanned": [0, last_step], "n_steps_with_nonfinite": len(bad),
           "steps_with_nonfinite": {str(k): v for k, v in bad.items()},
           "note": "finiteness only; no values, statistics or model outputs recorded"}
    (HERE / f"finiteness_scan_{year}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(year, "steps with non-finite values:", len(bad), "of", last_step + 1)


if __name__ == "__main__":
    main(int(sys.argv[1]))
