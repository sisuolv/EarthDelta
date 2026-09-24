#!/usr/bin/env python3
"""Assemble protocol/fs_protocol_v2.json (FP-03 protocol v2, frozen before any v2 job)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

REPO = Path("/mnt/afs/260010168/EarthDelta")
V = REPO / "plans/plan_v4_0923/run_20260924T013959Z_fp03_v2"
V1 = REPO / "plans/plan_v4_0923/run_20260923T192759Z_fp03_fs"
OUT = V / "protocol" / "fs_protocol_v2.json"
V1_PROTOCOL = V1 / "protocol/fs_protocol_v1.json"
GATE_CONFIG = REPO / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/s0/gate_config.json"
S0_CERT = REPO / ("plans/plan_v4_0923/run_20260923T_fp01_trace/gate_output/"
                  "s0-gate-20260923t174135587010z/s0_gate_result.json")
TRAIN_ADMISSION = V1 / "admission/bank_fit_admission.json"
HOLDOUT_ADMISSION = V / "holdout/v2_holdout_admission_A1.json"
F0_REFERENCE = V1 / "decisions/J1_diag_decision_r2.json"
OVERLAY = REPO / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/cuda_site"
INIT_DIGEST = "2d0b92da4782f3e708e7ca1fb0b406b26ba070352cc52cb4501b2009c92beffa"
SOURCE_FILES = [
    "earthdelta/static_adapter.py", "earthdelta/fs_protocol.py", "scripts/r2_fs_bank_train.py",
    "scripts/r4_fs_decide.py", "tests/test_fs_quality_gate.py", "tests/test_fs_v2_chain.py",
    "tests/test_loaded_admission_binding.py", "tests/test_plan_gate_chain.py",
    "tests/test_fs_decide_cli.py", "tests/test_fs_static_adapter.py",
    "earthdelta/contracts.py", "earthdelta/bridge/stormer_bridge.py",
    "earthdelta/bridge/stormer_arch.py", "earthdelta/lowrank.py", "earthdelta/metrics_contract.py",
]

FS_QUAL_V2 = (V / "protocol/PREREGISTRATION_v2.md").read_text().split("```")[1].strip()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": sha(path)}


def main() -> None:
    v1 = json.loads(V1_PROTOCOL.read_text())
    holdout = json.loads(HOLDOUT_ADMISSION.read_text())
    train = json.loads(TRAIN_ADMISSION.read_text())
    hold_rows = holdout["admission"]["admitted"]
    train_rows = train["admission"]["admitted"]
    iso = lambda t: dt.datetime.fromtimestamp(int(t), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    htimes = [int(r["issue_time"]) for r in hold_rows]

    common = {
        "seed": 20260921, "rank_per_expert": 4, "target_blocks": [18, 19, 20, 21, 22, 23],
        "train_steps": 4, "lead_steps": [4], "panel_lead_steps": [1, 4, 12], "hold_steps": 4,
        "data_role": "bank_fit", "limit": None,
        "config": ref(GATE_CONFIG), "admission": ref(TRAIN_ADMISSION),
        "panel_admission": ref(HOLDOUT_ADMISSION), "s0_certificate": ref(S0_CERT),
    }
    devices = {"one_of": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]}
    configs = {
        "V2_P": {"job": "V2-J3", "stage": "fs_fit", "mode": "formal", "max_updates": 32,
                 "learning_rate": 0.0025, "device": devices,
                 "_output_subdir_template": "replica{index}", **common},
        "V2_F": {"job": "V2-J3b", "stage": "fs_fit", "mode": "formal", "max_updates": 32,
                 "learning_rate": 0.00125, "device": devices,
                 "_output_subdir_template": "replica{index}", **common},
    }
    j4_common = {k: common[k] for k in ("seed", "rank_per_expert", "target_blocks", "hold_steps",
                                         "data_role", "limit", "config", "admission",
                                         "s0_certificate")}
    configs["V2_J4_FREEZE"] = {"job": "V2-J4", "stage": "fs_freeze", "device": "cuda:0",
                               "num_experts": 4, "_output_subdir": "freeze", **j4_common}
    configs["V2_J4_VERIFY"] = {"job": "V2-J4", "stage": "fs_verify", "device": "cuda:1",
                               "num_experts": 4, "_output_subdir": "verify", **j4_common}

    env = (f'set -u; export PYTHONPATH="{REPO}/.pydeps:{OVERLAY}:${{PYTHONPATH:-}}"; '
           "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; "
           'echo "== FP-03 PROTOCOL V2"; '
           f'sha256sum "{OUT}" "{S0_CERT}" "{TRAIN_ADMISSION}" "{HOLDOUT_ADMISSION}"; '
           "python3 -c \"import torch, xformers; print('env torch', torch.__version__, 'xformers', "
           "xformers.__version__, 'gpus', torch.cuda.device_count())\"; ")
    script = "python3 -u @SNAPSHOT@/scripts/r2_fs_bank_train.py"
    fit_common = (
        "--seed 20260921 --rank-per-expert 4 --target-blocks 18 19 20 21 22 23 --train-steps 4 "
        "--lead-steps 4 --hold-steps 4 --data-role bank_fit --panel-lead-steps 1 4 12 "
        f"--config {GATE_CONFIG} --admission {TRAIN_ADMISSION} --panel-admission {HOLDOUT_ADMISSION} "
        f"--s0-certificate {S0_CERT} --protocol {OUT} --protocol-sha256 <PROTOCOL_SHA256>")

    def fit_job(cid: str, lr: str) -> str:
        return (env + "status=0; pids=(); for i in 0 1 2 3; do D=@OUTPUT@/replica$i; mkdir -p $D; "
                f"{script} --stage fs_fit --mode formal --max-updates 32 --learning-rate {lr} "
                f"--config-id {cid} --device cuda:$i {fit_common} --output-dir $D > $D/stdout.log 2>&1 "
                "& pids+=($!); done; for i in 0 1 2 3; do if wait ${pids[$i]}; then echo \""
                + cid + " replica $i rc=0\"; else rc=$?; echo \"" + cid + " replica $i rc=$rc\"; "
                "status=1; fi; done; exit $status")

    j4_flags = (
        "--seed 20260921 --rank-per-expert 4 --num-experts 4 --target-blocks 18 19 20 21 22 23 "
        f"--hold-steps 4 --data-role bank_fit --config {GATE_CONFIG} --admission {TRAIN_ADMISSION} "
        f"--s0-certificate {S0_CERT} --protocol {OUT} --protocol-sha256 <PROTOCOL_SHA256> "
        "--fs-adapter <DESIGNATED_ADAPTER> --fs-adapter-sha256 <DESIGNATED_ADAPTER_SHA256>")
    j4 = (env + "mkdir -p @OUTPUT@/freeze @OUTPUT@/verify; "
          f"{script} --stage fs_freeze --config-id V2_J4_FREEZE --device cuda:0 {j4_flags} "
          "--fs-decision <FORMAL_DECISION> --output-dir @OUTPUT@/freeze > @OUTPUT@/freeze/stdout.log 2>&1; "
          'rc=$?; echo "V2_J4_FREEZE rc=$rc"; if [ $rc -ne 0 ]; then exit $rc; fi; '
          f"{script} --stage fs_verify --config-id V2_J4_VERIFY --device cuda:1 {j4_flags} "
          "--fs-merged-backbone @OUTPUT@/freeze/fs_merged_backbone.pt --output-dir @OUTPUT@/verify "
          '> @OUTPUT@/verify/stdout.log 2>&1; rc=$?; echo "V2_J4_VERIFY rc=$rc"; exit $rc')

    decider = "python3 <JOB_RUN_DIR>/source/scripts/r4_fs_decide.py"
    protocol = {
        "schema_version": "ed-fs-protocol/1",
        "protocol_id": "FP03-FS-v2",
        "status": "PREREGISTERED_BEFORE_V2_JOBS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(V),
        "supersedes_nothing": ("Protocol v1 and its final STOP_FITTED_FS_QUALITY result stand; v2 is a "
                               "new, separately budgeted qualification attempt with H=32."),
        "self_reference": ("This file's SHA-256 is every v2 job's --protocol-sha256 (placeholder "
                           "<PROTOCOL_SHA256> in jobs.*.argv_template); it cannot be stored inside."),
        "rules_verbatim": {
            "FS-SELECT-v1": v1["rules_verbatim"]["FS-SELECT-v1"],
            "FS-QUAL-v2_and_FS-BUDGET-v2": FS_QUAL_V2,
        },
        "changes_from_v1": [
            "H_formal 256 -> 32 (Q1 n_updates == 32; Q2 epoch range k = 1..3)",
            "Q6 holdout: the exposed 8-issue 2020 holdout -> a fresh 16-issue 2019 holdout",
            "Q0 F0 reference: train-panel F0 vs the pinned v1 J1 reference (unchanged panel issues); "
            "fresh-holdout F0 bitwise identical across the 4 replicas (no earlier reference can exist)",
            "Budget: FS-BUDGET-v2 (max 4 GPU jobs)",
        ],
        "qualification_rule": {
            "rule": "FS-QUAL-v2", "horizon": 32,
            "q0": {"torch_version": "2.3.1+cu121", "xformers_version": "0.0.27",
                   "official_model": "stormer.models.hub.stormer.Stormer",
                   "tf32": "off (effective getters, DEV-001)",
                   "f0_reference_groups": ["train"],
                   "f0_reference": "v1 J1 arm A0 train F0 panel, carried by the pinned inputs.f0_reference_decision",
                   "holdout_f0_cross_replica_bitwise": True,
                   "f0_panel_rel_tol": 1e-6},
            "q2": {"clip_events_max": 0, "per_visit_ratio_max": 1.25, "epoch_onset_factor": 1.02,
                   "epoch_size": 8},
            "q3": {"required_issues": 8},
            "q4": {"pass_max_ratio_lt": 1.0, "pass_mean_ratio_le": 0.99,
                   "fail_mean_ratio_gt": 1.00, "fail_max_ratio_gt": 1.02},
            "q5": {"mean_ratio_72h_le": 1.01, "report_only": False},
            "q6": {"holdout_mean_ratio_24h_le": 1.02, "report_only": False,
                   "panel": "fresh 2019 holdout (16 issues)"},
            "report_only_leads_hours": [6],
            "implementation": "earthdelta.static_adapter.evaluate_fs_qualification",
        },
        "selection_rule": {"rule": "FS-SELECT-v1", "n_replicas": 4, "designated_device_index": 0,
                           "designated_dir": "replica0",
                           "implementation": "earthdelta.static_adapter.decide_formal_fs"},
        "budget": {"rule": "FS-BUDGET-v2", "jobs": ["V2-J3", "V2-J3b", "V2-J4"], "max_gpu_jobs": 4,
                   "infra_retries_total": 1,
                   "v2_j3b_only_if": "V2-J3 verdict STOP_FITTED_FS_QUALITY",
                   "v2_j4_only_if": "a formal decision QUALITY_PASS_PENDING_DESIGNATED_GATES",
                   "both_fail": "stop and report; no further candidate without the coordinator",
                   "formal_horizon": 32},
        "merge_equivalence_tolerance": v1["merge_equivalence_tolerance"],
        "inputs": {
            "gate_config": ref(GATE_CONFIG),
            "s0_certificate": ref(S0_CERT),
            "admission": ref(TRAIN_ADMISSION),
            "panel_admission": ref(HOLDOUT_ADMISSION),
            "f0_reference_decision": ref(F0_REFERENCE),
            "checkpoint_sha256": v1["inputs"]["checkpoint_sha256"],
            "normalization_identity": v1["inputs"]["normalization_identity"],
            "certified_overlay": str(OVERLAY),
            "initial_adapter_digest": INIT_DIGEST,
            "train_issue_ids": [r["issue_id"] for r in train_rows],
            "train_issue_times_utc": [iso(r["issue_time"]) for r in train_rows],
            "holdout_issue_ids": [r["issue_id"] for r in hold_rows],
            "holdout_issue_times_utc": [iso(t) for t in htimes],
            "holdout_exposure_support_interval_utc": [iso(min(htimes) - 12 * 3600), iso(max(htimes) + 72 * 3600)],
            "holdout_declaration": ref(V / "holdout/holdout_selection_declaration.json"),
            "holdout_attempt1_failed": ref(V / "holdout/holdout_attempt1_failed.json"),
            "holdout_amendment_A1": ref(V / "holdout/holdout_selection_amendment_A1.json"),
            "integrity_scan_2019": ref(V / "holdout/finiteness_scan_2019.json"),
            "old_data_analysis": ref(V / "analysis/h_and_risk_from_old_data.json"),
            "v1_protocol": ref(V1_PROTOCOL),
        },
        "expected_from_old_data_not_decision_inputs": {
            "V2-J3_train_side": "bitwise replay of v1 J1 arm A2: Q1-Q5 PASS (Q4 0.9741 / max 0.9800, Q5 0.9924)",
            "V2-J3b_train_side": "bitwise replay of v1 J1 arm A3: Q1-Q5 PASS (Q4 0.9839 / max 0.9877, Q5 0.9886)",
            "open": "Q0 validity and Q6 on the fresh holdout",
            "V2-J4_risk": "merge gate margin ~2x expected at lr 2.5e-3 (analysis file); not guaranteed",
        },
        "configs": configs,
        "jobs": {
            "V2-J3": {"label": "r4fs03v2p", "timeout_seconds": 3600,
                      "argv_template": ["bash", "-lc", fit_job("V2_P", "0.0025")],
                      "decision": (f"{decider} --kind formal --job V2_P --protocol {OUT} --protocol-sha256 "
                                   f"<PROTOCOL_SHA256> --job-dir <JOB_RUN_DIR> --diag-decision {F0_REFERENCE} "
                                   f"--out {V}/decisions/V2_J3_formal_decision.json")},
            "V2-J3b": {"label": "r4fs03v2f", "timeout_seconds": 3600,
                       "argv_template": ["bash", "-lc", fit_job("V2_F", "0.00125")],
                       "decision": (f"{decider} --kind formal --job V2_F --protocol {OUT} --protocol-sha256 "
                                    f"<PROTOCOL_SHA256> --job-dir <JOB_RUN_DIR> --diag-decision {F0_REFERENCE} "
                                    f"--out {V}/decisions/V2_J3b_formal_decision.json")},
            "V2-J4": {"label": "r4fs03v2j4", "timeout_seconds": 3600,
                      "argv_template": ["bash", "-lc", j4],
                      "placeholders": {
                          "<FORMAL_DECISION>": "the QUALITY_PASS_PENDING_DESIGNATED_GATES formal decision file",
                          "<DESIGNATED_ADAPTER>": "that decision's replica0 fs_adapter.pt path",
                          "<DESIGNATED_ADAPTER_SHA256>": "that decision's designated_adapter_sha256"},
                      "certify": (f"{decider} --kind certify --protocol {OUT} --protocol-sha256 <PROTOCOL_SHA256> "
                                  "--formal-decision <FORMAL_DECISION> --freeze-dir <JOB_RUN_DIR>/freeze "
                                  f"--verify-dir <JOB_RUN_DIR>/verify --out {V}/certify/V2_certify_decision.json "
                                  f"(publishes {V}/certify/fs/ only on FS_SELECTED)")},
        },
        "limitations": [
            "2019 is Stormer's checkpoint-selection year and this project's 'val' split; the 16 holdout "
            "issues are now Fs-exposed and must be purged from any later policy_dev/confirm use.",
            "2015/2018/2019 stores have multi-day holes in whole variable groups (integrity scan); the "
            "holdout was reselected on integrity only (amendment A1).",
            "S0 certifies the forward pass only; the training backward is exercised, not certified.",
            "2020 training rows carry split_id 'test' while admitted as bank_fit (as in v1).",
            "delta_min is not pre-registered: v2 can qualify 'no harm', not 'useful gain'.",
        ],
        "source_at_preregistration": {p: sha(REPO / p) for p in SOURCE_FILES},
        "preregistration_md": ref(V / "protocol/PREREGISTRATION_v2.md"),
    }
    OUT.write_text(json.dumps(protocol, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
