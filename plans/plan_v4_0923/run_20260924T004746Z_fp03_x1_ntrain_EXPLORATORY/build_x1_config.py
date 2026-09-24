#!/usr/bin/env python3
"""Write x1_exploratory_config.json: the EXPLORATORY X1 run declarations.

This is NOT a pre-registration and NOT protocol v1. It exists because
r2_fs_bank_train.py refuses undeclared real runs; declaring the X1 runs here
keeps S0-certificate consumption, the certified-backend guard and admission
certification switched on. Every hash is computed from the actual bytes.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

REPO = Path("/mnt/afs/260010168/EarthDelta")
X = REPO / "plans/plan_v4_0923/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY"
OUT = X / "x1_exploratory_config.json"
V1RUN = REPO / "plans/plan_v4_0923/run_20260923T192759Z_fp03_fs"
GATE_CONFIG = REPO / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/s0/gate_config.json"
S0_CERT = REPO / ("plans/plan_v4_0923/run_20260923T_fp01_trace/gate_output/"
                  "s0-gate-20260923t174135587010z/s0_gate_result.json")
HOLDOUT = V1RUN / "admission/holdout_panel_admission.json"
OVERLAY = REPO / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/cuda_site"
BASELINE_DECISION = V1RUN / "decisions/J3_formal_decision.json"
LABEL = "EXPLORATORY_X1 - not FS-QUAL-v1, not pre-registered, cannot select/freeze/certify an Fs"


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    common = {
        "_status": LABEL, "stage": "fs_fit", "mode": "stability_screen", "max_updates": 256,
        "learning_rate": 0.0025, "seed": 20260921, "rank_per_expert": 4,
        "target_blocks": [18, 19, 20, 21, 22, 23], "train_steps": 4, "lead_steps": [4],
        "panel_lead_steps": [1, 4, 12], "hold_steps": 4, "data_role": "bank_fit",
        "device": "cuda:0", "limit": None,
        "config": ref(GATE_CONFIG), "panel_admission": ref(HOLDOUT), "s0_certificate": ref(S0_CERT),
    }
    configs = {
        f"X1_N{n}": {**common, "admission": ref(X / f"admission/x1_n{n}_admission.json"),
                     "_output_subdir": f"X1_N{n}", "_n_train": n}
        for n in (32, 64)
    }

    def argv(cid: str) -> list:
        n = configs[cid]
        body = (
            f'set -u; export PYTHONPATH="{REPO}/.pydeps:{OVERLAY}:${{PYTHONPATH:-}}"; '
            "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; "
            f'echo "== {LABEL}"; sha256sum "{OUT}" "{S0_CERT}" "{n["admission"]["path"]}" "{HOLDOUT}"; '
            "python3 -c \"import torch, xformers; print('env torch', torch.__version__, 'xformers', "
            "xformers.__version__, 'gpus', torch.cuda.device_count())\"; "
            f"D=@OUTPUT@/{cid}; mkdir -p $D; "
            "python3 -u @SNAPSHOT@/scripts/r2_fs_bank_train.py --stage fs_fit --mode stability_screen "
            f"--max-updates 256 --learning-rate 0.0025 --config-id {cid} --device cuda:0 "
            "--seed 20260921 --rank-per-expert 4 --target-blocks 18 19 20 21 22 23 --train-steps 4 "
            "--lead-steps 4 --hold-steps 4 --data-role bank_fit --panel-lead-steps 1 4 12 "
            f"--config {GATE_CONFIG} --admission {n['admission']['path']} --panel-admission {HOLDOUT} "
            f"--s0-certificate {S0_CERT} --protocol {OUT} --protocol-sha256 <CONFIG_SHA256> "
            f'--output-dir $D > $D/stdout.log 2>&1; rc=$?; echo "{cid} rc=$rc"; exit $rc')
        return ["bash", "-lc", body]

    h8 = 1.038925  # J3 (N=8) holdout 24h mean ratio, all 4 replicas bitwise identical
    config = {
        "schema_version": "ed-fs-protocol/1",
        "schema_note": ("schema name required by the CLI loader only; this file is NOT a "
                        "protocol/pre-registration"),
        "protocol_id": "FP03-X1-EXPLORATORY-NTRAIN",
        "status": LABEL,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "why_this_file_exists": (
            "r2_fs_bank_train.py refuses undeclared real runs by design. Declaring the X1 runs "
            "here keeps S0 consumption, the certified-backend guard (xformers 0.0.27 / official "
            "Stormer) and fail-closed admission certification on. It is not bound to "
            "fs_protocol_v1 and no go-ahead was requested on its hash."),
        "question": ("Does the 24h train/holdout gap of the lr=2.5e-3, 256-update static adapter "
                     "shrink as the number of training issues grows (N = 8 -> 32 [-> 64])?"),
        "baseline_N8": {"job": "J3 pt-sz6iv529", "decision": ref(BASELINE_DECISION),
                        "train_24h_mean_ratio": 0.910518, "holdout_24h_mean_ratio": h8},
        "design": {
            "fixed": "lr 2.5e-3, 256 updates, Adam defaults, clip 1.0, seed 20260921, rank 4, "
                     "blocks 18-23, 4x6h rollout, 24h objective, same 8-issue holdout panel, "
                     "mode stability_screen (diagnostic; never freezable), certified backend",
            "varied": "training issue count N (nested: N=8 subset of N=32 subset of N=64)",
            "training_order": "original 8 (protocol-v1 order) first, then new issues in time order; "
                              "the first 8 updates replay J3 exactly",
            "purge_rule": ("every new training issue >= 288h from every holdout issue: the same "
                           "minimum separation the N=8 design has, so each holdout issue's nearest "
                           "training issue is 288h away at every N (no proximity gain)"),
            "consequence": ("new issues come only from 2020-07-13T18Z .. 2020-12-28T18Z (evenly "
                            "spaced); the holdout is Jan-Jul, so N and season change together"),
        },
        "measurements": [
            "train 24h mean ratio L1/L0 over the run's full training set",
            "train 24h mean ratio over the original 8 issues (same issues as the baseline)",
            "holdout 24h mean ratio (fixed 8 issues) - the primary comparison; 6h/72h reported",
            "per-issue holdout 24h ratios; descriptive stability (clip events, per-visit ratio, "
            "epoch-mean onset); continuity: first 8 losses bitwise equal to J3",
        ],
        "decision_rule_for_N64": {
            "written_before_any_X1_job": True,
            "definitions": {"h32": "X1_N32 holdout 24h mean ratio",
                            "d8": round(h8 - 1.0, 6), "no_help_bound": round(1 + 0.75 * (h8 - 1.0), 6)},
            "stop_unstable": "X1_N32 not STABLE (FS-SCREEN-v1 Q1+Q2 criteria, descriptive) -> stop, report, no N=64",
            "stop_resolved": "h32 <= 1.000 (holdout no longer degraded at 24h) -> no N=64",
            "stop_no_help": "h32 >= no_help_bound (less than 25% reduction of the N=8 holdout degradation) -> no N=64",
            "run_N64": "otherwise (1.000 < h32 < no_help_bound): shrinking but not closed -> run X1_N64",
        },
        "interpretation_guide": [
            "A holdout improvement with N is robust: it cannot come from proximity (purge held at 288h).",
            "No improvement is only weakly informative about N itself: the new issues are Jul-Dec while "
            "the holdout is Jan-Jul; a same-season multi-year test would be the clean follow-up.",
            "holdout <= 1.00 with train clearly < 1 would support a v2 with expanded training data; "
            "holdout <= 1.00 with train ~ 1 would mean the adapter simply learns less.",
        ],
        "qualification_rule": {
            "_note": ("Only q0 (environment pins) and q2 (descriptive stability thresholds) are read "
                      "by the CLI. Nothing here qualifies, selects or certifies an Fs."),
            "q0": {"torch_version": "2.3.1+cu121", "xformers_version": "0.0.27"},
            "q2": {"clip_events_max": 0, "per_visit_ratio_max": 1.25, "epoch_onset_factor": 1.02,
                   "epoch_size": 8},
        },
        "inputs": {
            "s0_certificate": ref(S0_CERT),
            "gate_config": ref(GATE_CONFIG),
            "holdout_admission": ref(HOLDOUT),
            "initial_adapter_digest": "2d0b92da4782f3e708e7ca1fb0b406b26ba070352cc52cb4501b2009c92beffa",
            "manifests": {f"N{n}": ref(X / f"admission/x1_n{n}_manifest.json") for n in (32, 64)},
            "manifest_builder": ref(X / "admission/build_manifests.py"),
        },
        "budget": {"max_jobs": 3, "planned": ["X1_N32", "X1_N64 only per decision_rule_for_N64"],
                   "spare": "1 infra retry"},
        "configs": configs,
        "jobs": {cid: {"label": f"r4fs03x1n{configs[cid]['_n_train']}", "timeout_seconds": 5400,
                       "argv_template": argv(cid)} for cid in configs},
    }
    OUT.write_text(json.dumps(config, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
