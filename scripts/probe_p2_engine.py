#!/usr/bin/env python3
"""P2: H100/RTX5090 engine qualification and F0 20-step receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from earthdelta.probe import load_probe_spec, issue_sets, issue_index
from earthdelta.probe.rollout import load_bridge, rollout_trajectory, TARGET_BLOCKS


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache_row(qc: dict, index: int, *, cache_dir: Path, ledger: Path,
                   run_id: str, job: str) -> np.ndarray:
    """Load one hash-bound P1 payload; GPU stages never open the Zarr store."""
    row = qc.get("cache_files", {}).get(str(index))
    if not row:
        raise RuntimeError(f"P1 cache is missing index {index}")
    path = Path(row["path"])
    if not path.is_absolute():
        path = cache_dir / path
    if digest(path) != row.get("sha256"):
        raise RuntimeError(f"P1 cache hash mismatch at index {index}")
    array = np.asarray(np.load(path, allow_pickle=False))
    if list(array.shape) != list(row.get("shape", [])):
        raise RuntimeError(f"P1 cache shape mismatch at index {index}: {array.shape}")
    decoded = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    expected = qc.get("per_index", {}).get(str(index), {}).get("decoded_sha256")
    if decoded != expected:
        raise RuntimeError(f"P1 decoded cache hash mismatch at index {index}")
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"schema": "earthdelta.probe.access.v1", "run_id": run_id,
                            "path": str(path), "indices": [index],
                            "purpose": "engine_qualification_cache", "stage": "P2",
                            "job": job, "array_payload": True,
                            "decoded_sha256": decoded, "shape": list(array.shape)},
                           sort_keys=True, separators=(",", ":")) + "\n")
    return array


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--norm-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--engine", choices=("H100", "RTX5090"), required=True)
    ap.add_argument("--cache-dir", required=True,
                    help="P1 per-index NPY cache; the worker must not open Zarr")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--job", default="local")
    args = ap.parse_args()
    run = Path(args.run_dir).resolve(); run.mkdir(parents=True, exist_ok=True)
    spec = load_probe_spec(args.spec, expected_sha256="bf247b2abc86ef81be62a7aa5a26318532ca3e3b5ceaeb492c98387781518348")
    receipt = {"schema": "earthdelta.probe.engine_qual.v1", "status": "BLOCKED", "engine": args.engine,
               "job": args.job, "spec_sha256": spec.sha256, "issues": [], "failure_list": []}
    try:
        qc_path = run / "DATA_QC_RECEIPT.json"
        qc = json.loads(qc_path.read_text())
        if qc.get("status") != "OBSERVED":
            raise RuntimeError("P1 DATA_QC_RECEIPT is not OBSERVED")
        excluded = set(qc.get("excluded_issues", {}).get("EQ_engine_qual", []))
        issue_times = [x for x in issue_sets(spec)["EQ_engine_qual"] if x not in excluded]
        if not issue_times:
            raise RuntimeError("all engine qualification issues excluded")
        cache_dir = Path(args.cache_dir).resolve()
        coordinate = json.loads(json.dumps(qc.get("coordinate_cache", {})))
        coord_path = Path(coordinate.get("path", ""))
        if not coord_path.exists() or digest(coord_path) != coordinate.get("sha256"):
            raise RuntimeError("P1 coordinate cache is missing or hash-invalid")
        coords = np.load(coord_path, allow_pickle=False)
        lat = np.asarray(coords["lat"], dtype=np.float64)
        if lat.shape != (128,):
            raise RuntimeError(f"unexpected latitude cache shape: {lat.shape}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device = torch.device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        loaded = load_bridge(args.checkpoint, args.norm_dir, device)
        if loaded.checkpoint_sha256 != spec.raw["bindings"]["checkpoint"]["sha256"]:
            raise RuntimeError("checkpoint hash mismatch")
        rows = []; mse_rows = []; six_norm = []; zero_edit_max = []
        start = time.perf_counter()
        for issue in issue_times:
            i0 = issue_index(issue)
            values = list(range(i0, i0 + 21))
            data = np.stack([load_cache_row(qc, idx, cache_dir=cache_dir,
                                            ledger=run / "ACCESS_LEDGER.jsonl",
                                            run_id=run.name, job=args.job)
                             for idx in values], axis=0)
            x_raw = torch.from_numpy(np.ascontiguousarray(data[0:1])).float().to(device)
            truth = torch.from_numpy(np.ascontiguousarray(data[[1, 4, 12, 20]])).float().unsqueeze(0).to(device)
            x_norm = loaded.bridge.normalize(x_raw)
            with torch.inference_mode():
                f0 = rollout_trajectory(loaded.bridge, x_norm, steps=20, deltas=None)
                zero = {b: torch.zeros_like(loaded.model.blocks[b].attn.proj.weight) for b in TARGET_BLOCKS}
                edited_zero = rollout_trajectory(loaded.bridge, x_norm, steps=20, deltas=zero)
            diff = float((f0 - edited_zero).abs().max().detach().cpu())
            zero_edit_max.append(diff)
            if not torch.equal(f0, edited_zero):
                receipt["failure_list"].append({"issue": issue, "reason": "zero_edit_not_bitwise", "max_abs": diff})
            f0_raw = loaded.bridge.denormalize(f0[:, [1, 4, 12, 20]].reshape(-1, 69, 128, 256)).reshape(1, 4, 69, 128, 256)
            # Save six named cells in physical units for the cross-engine check.
            cell = []
            for var in spec.variables:
                ch = int(var["channel_index"])
                for step in range(4):
                    field = f0_raw[0, step, ch].detach().cpu().numpy()
                    tfield = truth[0, step, ch].detach().cpu().numpy()
                    w = np.cos(np.deg2rad(lat))[:, None]; w = w / w.sum()
                    cell.append(float(np.sum((field - tfield) ** 2 * w)))
            mse_rows.append(cell)
            six_norm.append(f0[:, 1].detach().cpu().numpy().astype(np.float32)[0])
            rows.append({"issue_time": issue, "index": i0, "zero_edit_bitwise": bool(diff == 0.0),
                         "zero_edit_max_abs": diff})
        elapsed = time.perf_counter() - start
        out_path = run / f"ENGINE_QUAL_{args.engine}.npz"
        np.savez_compressed(out_path, issue_times=np.asarray(issue_times), mse=np.asarray(mse_rows, dtype=np.float64),
                            f0_norm_6h=np.asarray(six_norm, dtype=np.float32))
        receipt.update({"status": "OBSERVED" if not receipt["failure_list"] else "BLOCKED",
                        "checkpoint_sha256": loaded.checkpoint_sha256, "n_issues": len(issue_times),
                        "issues": rows, "output": str(out_path), "output_sha256": digest(out_path),
                        "zero_edit_max_abs": max(zero_edit_max) if zero_edit_max else None,
                        "elapsed_seconds": elapsed, "torch_version": torch.__version__,
                        "cuda_device": torch.cuda.get_device_name(0)})
    except Exception as exc:
        receipt["status"] = "BLOCKED"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    path = run / f"ENGINE_QUAL_{args.engine}_RECEIPT.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "OBSERVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
