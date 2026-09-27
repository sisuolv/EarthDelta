#!/usr/bin/env python3
"""P1: validate the enrolled 2020 store and create the data QC receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from earthdelta.bridge.stormer_bridge import DEFAULT_VARIABLES
from earthdelta.probe import ProbeAccessController, load_probe_spec, issue_sets, issue_index


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--norm-dir", default="reference/stormer/normalization_constants")
    ap.add_argument("--job", default="local")
    args = ap.parse_args()
    run = Path(args.run_dir).resolve(); run.mkdir(parents=True, exist_ok=True)
    spec = load_probe_spec(args.spec, expected_sha256="bf247b2abc86ef81be62a7aa5a26318532ca3e3b5ceaeb492c98387781518348")
    store = Path(spec.store_path).resolve()
    ledger = run / "ACCESS_LEDGER.jsonl"
    guard = ProbeAccessController(spec, ledger, run_id=run.name)
    receipt = {"schema": "earthdelta.probe.data_qc.v1", "status": "BLOCKED", "spec_sha256": spec.sha256,
               "store": str(store), "job": args.job, "failure_list": [], "excluded_issues": {},
               "per_index": {}, "metadata": {}}
    try:
        complete = store / "COMPLETE.json"
        expected_complete = "56a6f69aa99e76d9ca3dc977860205bdc05b04fbac85a8343ad53b39798a2d9c"
        if sha256_file(complete) != expected_complete:
            raise RuntimeError("2020 COMPLETE.json hash mismatch")
        receipt["metadata"]["COMPLETE_sha256"] = sha256_file(complete)
        group = guard.open_zarr(store, stage="P1", job=args.job)
        # Coordinate reads are explicit ledger entries; they do not authorize
        # payload access from any other store.
        channels = [decode_name(x) for x in guard.read_coordinate(store, "channel", purpose="channel_identity", stage="P1", job=args.job)]
        lat = np.asarray(guard.read_coordinate(store, "lat", purpose="coordinate_identity", stage="P1", job=args.job), dtype=np.float64)
        lon = np.asarray(guard.read_coordinate(store, "lon", purpose="coordinate_identity", stage="P1", job=args.job), dtype=np.float64)
        time_values = np.asarray(guard.read_coordinate(store, "time", purpose="time_identity", stage="P1", job=args.job))
        if channels != list(DEFAULT_VARIABLES):
            raise RuntimeError("channel identity/order differs from bridge DEFAULT_VARIABLES")
        if tuple(group["data"].shape) != (1464, 69, 128, 256):
            raise RuntimeError(f"unexpected data shape: {group['data'].shape}")
        time_attrs = dict(group["time"].attrs)
        if time_attrs.get("units") != "hours since 2020-01-01 00:00:00" or time_attrs.get("calendar") != "proleptic_gregorian":
            raise RuntimeError(f"unexpected time coordinate attrs: {time_attrs}")
        expected_time = np.arange(1464, dtype=np.int64) * 6
        if not np.array_equal(time_values.astype(np.int64), expected_time):
            raise RuntimeError("time coordinate is not the signed 6h 2020 grid")
        receipt["metadata"].update({"shape": list(group["data"].shape), "channels": channels,
                                     "lat_sha256": sha256_bytes(np.ascontiguousarray(lat).tobytes()),
                                     "lon_sha256": sha256_bytes(np.ascontiguousarray(lon).tobytes()),
                                     "time_sha256": sha256_bytes(np.ascontiguousarray(time_values).tobytes())})
        means = np.load(Path(args.norm_dir) / "normalize_mean.npz")
        stds = np.load(Path(args.norm_dir) / "normalize_std.npz")
        mean = np.asarray([means[name].reshape(-1)[0] for name in channels], dtype=np.float64)
        std = np.asarray([stds[name].reshape(-1)[0] for name in channels], dtype=np.float64)
        if not np.isfinite(std).all() or np.any(std <= 0):
            raise RuntimeError("normalization std invalid")
        names = issue_sets(spec)
        indices = sorted({i for name in names for issue in names[name]
                          for i in range(issue_index(issue), issue_index(issue) + max(spec.rollout_steps) + 1)})
        failures = set()
        # Read each registered index exactly once through the guard.  Keeping a
        # per-index digest makes retry/resume and provenance auditable.
        for index in indices:
            array = guard.read(store, [index], purpose="data_qc", stage="P1", job=args.job)[0]
            finite = bool(np.isfinite(array).all())
            z = np.abs((array.astype(np.float64) - mean[:, None, None]) / std[:, None, None])
            max_z = float(np.nanmax(z)) if np.isfinite(z).any() else float("inf")
            argmax = int(np.nanargmax(z)) if np.isfinite(z).any() else -1
            digest = sha256_bytes(np.ascontiguousarray(array).tobytes())
            expected_dt = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() + index * 6 * 3600
            row = {"index": index, "finite": finite, "max_abs_z": max_z, "argmax_flat": argmax,
                   "decoded_sha256": digest, "expected_epoch": expected_dt}
            receipt["per_index"][str(index)] = row
            if (not finite) or (not np.isfinite(max_z)) or max_z > 40.0:
                failures.add(index)
        receipt["failure_list"] = sorted(failures)
        for name, issues in names.items():
            excluded = [issue for issue in issues if failures.intersection(
                range(issue_index(issue), issue_index(issue) + max(spec.rollout_steps) + 1))]
            receipt["excluded_issues"][name] = excluded
        total_excluded = sum(len(v) for v in receipt["excluded_issues"].values())
        unexpected_eval = sum(len(receipt["excluded_issues"].get(name, [])) for name in ("E_oracle_eval", "EQ_engine_qual", "V_static_eval"))
        expected = set(spec.raw["expected_exclusions"]["issues"])
        actual = set(receipt["excluded_issues"].get("D_direction", [])) | set(receipt["excluded_issues"].get("T_static_train", []))
        receipt["expected_exclusions_match"] = expected.issubset(actual | set(receipt["excluded_issues"].get("E_oracle_eval", [])))
        receipt["total_excluded_issues"] = total_excluded
        receipt["unexpected_eval_excluded"] = unexpected_eval
        if total_excluded > 8 or unexpected_eval > 4:
            raise RuntimeError(f"data QC exclusion gate exceeded: total={total_excluded}, eval={unexpected_eval}")
        receipt["status"] = "OBSERVED"
    except Exception as exc:
        receipt["status"] = "BLOCKED"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        (run / "DATA_QC_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2
    (run / "DATA_QC_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
