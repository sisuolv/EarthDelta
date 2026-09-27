#!/usr/bin/env python3
"""P5: merge P4, compute preregistered estimands and paired block CIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from earthdelta.probe.contracts import issue_sets, load_probe_spec


SPEC_SHA = "bf247b2abc86ef81be62a7aa5a26318532ca3e3b5ceaeb492c98387781518348"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def improvement(mse: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """Pooled-MSE-then-RMSE improvement, per cell."""
    return 100.0 * (1.0 - np.sqrt(np.mean(mse, axis=0) / np.maximum(np.mean(f0, axis=0), 1e-30)))


def menu(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # One arm is chosen using the mean over all 24 cells.  The returned selected
    # rows retain all cells; no per-cell oracle is used for the decision.
    chosen = np.argmin(values.mean(axis=2), axis=1)
    selected = values[np.arange(values.shape[0]), chosen]
    return chosen, selected, improvement(selected, values[:, 0])


def static(values: np.ndarray) -> tuple[int, np.ndarray]:
    arm_scores = values.mean(axis=(0, 2))
    arm = int(np.argmin(arm_scores))
    return arm, improvement(values[:, arm], values[:, 0])


def block_indices(issue_times: np.ndarray) -> list[np.ndarray]:
    import datetime as dt
    origin = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    by = {}
    for i, value in enumerate(issue_times.astype(str)):
        t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        key = int((t - origin).total_seconds() // (7 * 86400))
        by.setdefault(key, []).append(i)
    return [np.asarray(v, dtype=np.int64) for _, v in sorted(by.items())]


def bootstrap(g: np.ndarray, r: np.ndarray, t: np.ndarray, times: np.ndarray,
              draws: int = 10000, seed: int = 20260927) -> dict:
    blocks = block_indices(times)
    rng = np.random.default_rng(seed)
    n_cells = g.shape[-1]
    out_g = np.empty((draws, n_cells), dtype=np.float32)
    out_r = np.empty((draws, n_cells), dtype=np.float32)
    out_s = np.empty((draws, n_cells), dtype=np.float32)
    out_t = np.empty((draws, n_cells), dtype=np.float32)
    for b in range(draws):
        selected_blocks = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[j] for j in selected_blocks])
        gg = g[idx]; rr = r[idx]; tt = t[idx]
        _, sg, _ = menu(gg)
        _, sr, _ = menu(rr)
        _, ss = static(gg)
        _, st, _ = menu(tt)
        out_g[b] = improvement(sg, gg[:, 0])
        out_r[b] = improvement(sr, rr[:, 0])
        out_s[b] = ss
        out_t[b] = improvement(st, tt[:, 0])
    def ci(x):
        return {"lo": np.percentile(x, 2.5, axis=0).tolist(),
                "hi": np.percentile(x, 97.5, axis=0).tolist()}
    return {"method": "paired_block_bootstrap", "block_days": 7, "draws": draws,
            "seed": seed, "n_blocks": len(blocks), "menu_grad_ci": ci(out_g),
            "menu_rand_ci": ci(out_r), "static_grad_ci": ci(out_s), "ttt_ci": ci(out_t)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--run-dir", required=True)
    ap.add_argument("--draws", type=int, default=10000); ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()
    run = Path(args.run_dir).resolve(); run.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "earthdelta.probe.p5.v1", "status": "BLOCKED", "spec_sha256": SPEC_SHA}
    started = time.perf_counter()
    try:
        spec = load_probe_spec(args.spec, expected_sha256=SPEC_SHA)
        names = issue_sets(spec)["E_oracle_eval"]
        expected = {str(x) for x in names}
        shards = sorted(run.glob("MENU_EVAL_SHARD_*.npz"))
        if not shards:
            raise RuntimeError("no P4 shards")
        rows = []
        for path in shards:
            z = np.load(path, allow_pickle=False)
            if digest(path) == "":
                raise RuntimeError("unreachable hash failure")
            issues = [str(x) for x in z["issue_times"]]
            rows.append((issues, np.asarray(z["mse_grad"], dtype=np.float64),
                         np.asarray(z["mse_rand"], dtype=np.float64),
                         np.asarray(z["mse_ttt"], dtype=np.float64)))
        all_issues = [x for part, *_ in rows for x in part]
        if set(all_issues) != expected or len(all_issues) != len(set(all_issues)):
            raise RuntimeError(f"P4 coverage mismatch: got {len(all_issues)} unique expected {len(expected)}")
        order = np.argsort(np.asarray(all_issues, dtype=str))
        times = np.asarray(all_issues, dtype=str)[order]
        grad = np.concatenate([x[1] for x in rows], axis=0)[order]
        rand = np.concatenate([x[2] for x in rows], axis=0)[order]
        ttt = np.concatenate([x[3] for x in rows], axis=0)[order]
        if grad.shape[1:] != (41, 6, 4) or rand.shape[1:] != (41, 6, 4):
            raise RuntimeError(f"unexpected P4 shape grad={grad.shape} rand={rand.shape}")
        if ttt.shape[1:] != (6, 6, 4):
            raise RuntimeError(f"unexpected TTT shape {ttt.shape}")
        # The saved P4 arrays are normalized per-cell MSE; arm 0 is always F0.
        chosen_g, selected_g, imp_g = menu(grad)
        chosen_r, selected_r, imp_r = menu(rand)
        static_arm, imp_s = static(grad)
        chosen_t, selected_t, imp_t = menu(ttt)
        static_point = imp_s
        primary = [0 * 4 + 2, 1 * 4 + 2]
        # `imp_*` has [variable, lead], preserving the 24-cell contract.
        go = bool((imp_g[0, 2] >= 2.0 and imp_g[1, 2] >= 2.0 and
                   (imp_g[0, 2] - static_point[0, 2] >= 1.0) and
                   (imp_g[1, 2] - static_point[1, 2] >= 1.0) and
                   (imp_g[0, 2] - imp_r[0, 2] >= 1.0) and
                   (imp_g[1, 2] - imp_r[1, 2] >= 1.0) and
                   np.all(imp_g >= -0.5)))
        relax = bool((not go) and (max(float(imp_g[0, 2]), float(imp_g[1, 2])) >= 0.5 or
                                   (imp_t[0, 2] >= 2.0 and imp_t[1, 2] >= 2.0)))
        if go:
            decision = "GO_V8"
        elif relax:
            decision = "RELAX_ONCE"
        else:
            decision = "STOP_PARAM_EDIT"
        # The extension trigger is checked before interpreting the menu result.
        # A multiplier-4 arm has the fixed arm order F0, then direction-major,
        # multiplier-major, sign-minor. Its indices are 9, 19, 29, 39.
        mult4 = np.asarray([1 + (direction * 5 + 4) * 2 + sign
                            for direction in range(4) for sign in range(2)])
        extension = {"gradient": bool(np.mean(np.isin(chosen_g, mult4)) >= 0.15),
                     "random": bool(np.mean(np.isin(chosen_r, mult4)) >= 0.15),
                     "multiplier4_arm_indices": mult4.tolist()}
        boot = bootstrap(grad, rand, ttt, times, draws=int(args.draws), seed=int(args.seed))
        results = {
            "schema": "earthdelta.probe.oracle_results.v1", "status": "OBSERVED",
            "issue_times": times.tolist(), "n_issues": len(times),
            "gradient_menu": {"selected_arm": chosen_g.tolist(), "f0_fraction": float(np.mean(chosen_g == 0)),
                              "improvement_pct": imp_g.tolist()},
            "random_menu": {"selected_arm": chosen_r.tolist(), "f0_fraction": float(np.mean(chosen_r == 0)),
                             "improvement_pct": imp_r.tolist()},
            "static_gradient": {"arm": static_arm, "improvement_pct": static_point.tolist()},
            "ttt": {"selected_arm": chosen_t.tolist(), "f0_fraction": float(np.mean(chosen_t == 0)),
                    "improvement_pct": imp_t.tolist(), "label": "diagnostic_truth_informed_upper_bound"},
            "primary_cells": {"Z500@72h": {"menu_grad": float(imp_g[0, 2]), "static_grad": float(static_point[0, 2]),
                                             "menu_rand": float(imp_r[0, 2]), "ttt": float(imp_t[0, 2])},
                              "T850@72h": {"menu_grad": float(imp_g[1, 2]), "static_grad": float(static_point[1, 2]),
                                             "menu_rand": float(imp_r[1, 2]), "ttt": float(imp_t[1, 2])}},
            "extension_required": extension, "bootstrap": boot,
        }
        (run / "ORACLE_RESULTS.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        receipt.update({"status": "OBSERVED", "decision": decision, "extension_required": extension,
                        "oracle_results": str(run / "ORACLE_RESULTS.json"),
                        "elapsed_seconds": time.perf_counter() - started})
        (run / "P5_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (run / "DECISION.json").write_text(json.dumps({"schema": "earthdelta.probe.decision.v1",
                                                         "status": "OBSERVED", "decision": decision,
                                                         "route": "A" if decision == "GO_V8" else ("B" if decision == "STOP_PARAM_EDIT" else "R1_REQUIRED"),
                                                         "extension_required": extension,
                                                         "primary_cells": results["primary_cells"]},
                                                        indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        receipt.update({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.perf_counter() - started})
        (run / "P5_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "OBSERVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
