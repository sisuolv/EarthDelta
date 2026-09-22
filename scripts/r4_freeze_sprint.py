#!/usr/bin/env python3
"""Freeze UTC roles and pilot contracts before observing any forecast outcomes."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_once(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    args = p.parse_args()
    run, assets = args.run.resolve(), args.assets.resolve()
    spec = json.loads((run / "planned_spec.json").read_text())
    rng = np.random.default_rng(spec["data"]["sampling_seed"])
    axes, stores = {}, {}
    expected_channels = json.loads((assets / "scripts/s0_gate_inputs/channels.json").read_text())
    if isinstance(expected_channels, dict):
        expected_channels = expected_channels["channels"]
    for year in (2018, 2019, 2020):
        path = assets / f"data/era5_1p40625/{year}.zarr"
        z = zarr.open_group(str(path), mode="r")
        assert list(z["channel"][:]) == expected_channels
        t = z["time"][:]
        units = z["time"].attrs["units"]
        assert units == f"hours since {year}-01-01 00:00:00"
        assert z["time"].attrs["calendar"] == "proleptic_gregorian"
        assert np.array_equal(t, np.arange(len(t)) * 6)
        times = np.datetime64(f"{year}-01-01T00", "h") + t.astype("timedelta64[h]")
        axes[year] = times
        grid = {"lat": z["lat"][:].tolist(), "lon": z["lon"][:].tolist(), "channels": expected_channels}
        assert np.array_equal(z["lat"][:], np.load(assets / "scripts/s0_gate_inputs/lat.npy"))
        assert np.array_equal(z["lon"][:], np.load(assets / "scripts/s0_gate_inputs/lon.npy"))
        stores[str(year)] = {"path": str(path), "shape": list(z["data"].shape),
                             "time_hours_utc": times.astype(np.int64).tolist(),
                             "time_sha256": hashlib.sha256(times.astype(np.int64).tobytes()).hexdigest(),
                             "grid_channels": grid, "grid_channels_sha256": digest(grid),
                             "metadata_sha256": {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
                                                 for f in sorted(path.rglob(".z*")) if f.is_file()}}
    assert len({v["grid_channels_sha256"] for v in stores.values()}) == 1

    rows = []

    def add(role, year, start, end, n, expert=None, exact=None):
        times = axes[year]
        valid = np.flatnonzero((times - np.timedelta64(6, "h") >= np.datetime64(start)) &
                               (times + np.timedelta64(72, "h") < np.datetime64(end)))
        if exact is None:
            selected = rng.permutation(valid)
        else:
            selected = np.array([np.flatnonzero(times == np.datetime64(t))[0] for t in exact])
            assert set(selected).issubset(valid)
        assert len(selected) >= n
        for order, i in enumerate(selected):
            rows.append({"issue_id": f"{role}_{str(times[i]).replace('-', '').replace('T', '_')}",
                         "role": role, "year": year, "store": stores[str(year)]["path"],
                         "index": int(i), "issue_utc": str(times[i]) + ":00:00Z",
                         "history_index": int(i - 1), "target_indices": [int(i + s) for s in (1, 4, 12)],
                         "support_start_utc": str(times[i] - np.timedelta64(6, "h")) + ":00:00Z",
                         "support_end_utc": str(times[i] + np.timedelta64(72, "h")) + ":00:00Z",
                         "selection_order": order, "primary": order < n,
                         "expert_index": expert, "role_window": [start, end]})

    for k, (start, boundary, end) in enumerate([
        ("2018-01-01", "2018-03-10", "2018-04-01"),
        ("2018-04-01", "2018-06-10", "2018-07-01"),
        ("2018-07-01", "2018-09-10", "2018-10-01"),
        ("2018-10-01", "2018-12-10", "2019-01-01"),
    ]):
        add("bank_fit", 2018, start, boundary, 32, k)
        add("bank_qualification", 2018, boundary, end, 8, k)
    add("policy_fit", 2019, "2019-01-01", "2019-09-01", 256)
    add("calibration", 2019, "2019-09-01", "2020-01-01", 64)
    add("sprint_holdout", 2020, "2020-02-01", "2021-01-01", 128)
    add("debug", 2020, "2020-01-01", "2020-02-01", 8,
        exact=[f"2020-01-{d:02d}T00" for d in (2, 5, 8, 11, 14, 17, 20, 23)])
    actions = [{"candidate_id": "R", "expert": None, "alpha": 0.0, "known": True}]
    for a in (0.125, 0.25, 0.1875):
        for k in range(4):
            actions.append({"candidate_id": f"e{k}_a{int(a * 10000):04d}",
                            "expert": k, "alpha": a, "known": a != 0.1875})
    frozen = {**spec, "status": "FROZEN_BEFORE_OUTCOMES", "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "role_sampling": "seeded ordered candidates; replace invalid content with next candidate in same role/expert only",
              "bank_grouping": "four contiguous calendar quarters, disjoint complete support",
              "data_manifest_digest": digest(stores), "issue_candidates_digest": digest(rows), "actions": actions,
              "objective_exact": {"space": "raw", "scale": "official input std, per variable",
                                  "q": "cos(latitude) broadcast to all variables and longitudes; uniform variable weights",
                                  "component_leads": [6, 24, 72], "weights_joint": [0.2, 0.5, 0.3]},
              "fixed_expert_selection": "min calibration joint loss averaged equally over known amplitudes; include no-edit",
              "head_selection": "min calibration 24h native selected loss over known actions; no holdout labels",
              "failed_candidate_policy": "fail entire issue/shard; no successful-subset scoring",
              "reduced_lists": "first 128/32/64 admitted issues in frozen order; curves first 64/128/256"}
    write_once(run / "manifests/data_stores.json", stores)
    write_once(run / "manifests/issue_candidates.json", rows)
    write_once(run / "frozen_spec.json", frozen)
    write_once(run / "manifests/frozen_spec_identity.json", {"sha256": hashlib.sha256((run / "frozen_spec.json").read_bytes()).hexdigest()})
    for role in sorted({r["role"] for r in rows}):
        chosen = [r for r in rows if r["role"] == role and r["primary"]]
        print(role, len(chosen), min(r["issue_utc"] for r in chosen), max(r["issue_utc"] for r in chosen))


if __name__ == "__main__":
    main()
