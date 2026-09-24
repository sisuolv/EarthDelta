#!/usr/bin/env python3
"""Assemble protocol/bank_protocol_v1.json (FP-04, frozen before any FP-04 GPU job).

Every pin is read from the file it names (hashes recomputed here); the bank and
Fs digests come from ../analysis/cpu_pins_and_fs_dryrun.json AND are cross-
checked against the certified bundle's own independent_reload.json record.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

REPO = Path("/mnt/afs/260010168/EarthDelta")
RUN = REPO / "plans/plan_v4_0923/run_20260924T033627Z_fp04_bank"
OUT = RUN / "protocol" / "bank_protocol_v1.json"
FP03 = REPO / "plans/plan_v4_0923/run_20260924T013959Z_fp03_v2"
FS_BUNDLE = FP03 / "certify/fs"
FS_CERTIFY = FP03 / "certify/V2_certify_decision.json"
FS_PROTOCOL = FP03 / "protocol/fs_protocol_v2.json"
GATE_CONFIG = REPO / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/s0/gate_config.json"
S0_CERT = REPO / ("plans/plan_v4_0923/run_20260923T_fp01_trace/gate_output/"
                  "s0-gate-20260923t174135587010z/s0_gate_result.json")
OVERLAY = REPO / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/cuda_site"
ADMISSION = RUN / "admission/bank_fit_admission.json"
GROUPING = RUN / "admission/bank_fit_grouping.json"
PINS = RUN / "analysis/cpu_pins_and_fs_dryrun.json"
BLOCKS = [18, 19, 20, 21, 22, 23]
SOURCE_FILES = [
    "earthdelta/bank_training.py", "earthdelta/registry.py", "earthdelta/static_adapter.py",
    "earthdelta/fs_protocol.py", "earthdelta/metrics_contract.py", "earthdelta/pilot_contract.py",
    "earthdelta/data/make_splits.py",
    "scripts/r2_fs_bank_train.py", "scripts/r4_bank_decide.py",
    "tests/test_fitted_fs_bank_roundtrip.py", "tests/test_bank_quality_gate.py",
    "tests/test_profile_immutability.py", "tests/test_plan_gate_chain.py",
    "earthdelta/contracts.py", "earthdelta/bridge/stormer_bridge.py",
    "earthdelta/bridge/stormer_arch.py", "earthdelta/lowrank.py",
]
RULES = (RUN / "protocol/PREREGISTRATION.md").read_text().split("```")[1].strip()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": sha(path)}


def main() -> None:
    pins = json.loads(PINS.read_text())
    dry = pins["certified_fs_cpu_dryrun"]
    assert dry["identity_pass"] is True and dry["recomputed_equals_bundle_record"] is True
    manifest = json.loads((FS_BUNDLE / "reference_manifest.json").read_text())
    certify = json.loads(FS_CERTIFY.read_text())
    assert certify["verdict"] == "FS_SELECTED" and manifest["protocol_sha256"] == sha(FS_PROTOCOL)
    grouping = json.loads(GROUPING.read_text())
    assert grouping["admission"]["sha256"] == sha(ADMISSION)
    assert grouping["group_sizes"] == [15, 16, 7, 6]
    admission = json.loads(ADMISSION.read_text())
    rows = admission["admission"]["admitted"]
    assert admission["passed"] is True and len(rows) == 48
    iso = lambda t: dt.datetime.fromtimestamp(int(t), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    fs_reference = {
        "reference_manifest": ref(FS_BUNDLE / "reference_manifest.json"),
        "certify_decision": ref(FS_CERTIFY),
        "fs_adapter": ref(FS_BUNDLE / "fs_adapter.pt"),
        "fs_merged_backbone": ref(FS_BUNDLE / "fs_merged_backbone.pt"),
        "fs_protocol_sha256": manifest["protocol_sha256"],
        "fs_protocol": ref(FS_PROTOCOL),
        "digests": dict(dry["recomputed_digests"]),
    }
    assert fs_reference["fs_adapter"]["sha256"] == manifest["fs_adapter_sha256"]
    assert fs_reference["fs_merged_backbone"]["sha256"] == manifest["fs_merged_backbone_sha256"]
    bank_pins = {
        "initial_bank_digest": pins["initial_bank"]["bank_digest"],
        "initial_expert_digests": pins["initial_bank"]["expert_digests"],
        "initial_bank_call": pins["initial_bank"]["call"],
        "grouping": {str(g["expert_index"]): list(g["issue_ids"]) for g in grouping["groups"]},
        "probe_issue_id": rows[0]["issue_id"],
    }

    common = {"seed": 20260921, "rank_per_expert": 4, "num_experts": 4, "target_blocks": BLOCKS,
              "hold_steps": 4, "train_steps": 4, "lead_steps": [4], "panel_lead_steps": [1, 4, 12],
              "a0": 0.25, "rho": 0.25, "data_role": "bank_fit", "limit": None,
              "config": ref(GATE_CONFIG), "admission": ref(ADMISSION), "s0_certificate": ref(S0_CERT),
              "fs_reference_manifest": fs_reference["reference_manifest"],
              "fs_certify_decision": fs_reference["certify_decision"],
              "fs_adapter": fs_reference["fs_adapter"],
              "fs_merged_backbone": fs_reference["fs_merged_backbone"]}
    measure = {"profile_steps": 3, "profile_warmup": 1, "memory_budget_gib": 80.0,
               "memory_safety_fraction": 0.8, "max_seconds_per_update": 30.0}
    configs = {}
    for k in range(4):
        configs[f"BJ1_E{k}"] = {"job": "B-J1", "stage": "bank_train", "mode": "gradient_check",
                                "max_updates": 16, "learning_rate": 0.0025, "expert_index": k,
                                "device": f"cuda:{k}", "_output_subdir": f"expert{k}", **common}
        configs[f"BJ2_E{k}"] = {"job": "B-J2", "stage": "bank_train", "mode": "formal",
                                "max_updates": 32, "learning_rate": 0.0025, "expert_index": k,
                                "device": f"cuda:{k}", "_output_subdir": f"expert{k}", **common}
        configs[f"BJ2F_E{k}"] = {"job": "B-J2F", "stage": "bank_train", "mode": "formal",
                                 "max_updates": 32, "learning_rate": 0.00125, "expert_index": k,
                                 "device": f"cuda:{k}", "_output_subdir": f"expert{k}", **common}
    configs["BJ1_PROFILE"] = {"job": "B-J1", "stage": "profile", "mode": "gradient_check",
                              "max_updates": 16, "learning_rate": 0.0025, "expert_index": 0,
                              "device": "cuda:0", "_output_subdir": "profile", **common, **measure}
    configs["BJ1_HORIZON"] = {"job": "B-J1", "stage": "horizon_check", "mode": "gradient_check",
                              "max_updates": 16, "learning_rate": 0.0025, "expert_index": 0,
                              "max_differentiable_steps": None, "device": "cuda:1",
                              "_output_subdir": "horizon", **common, **measure}
    configs["BJ3_ASSEMBLE"] = {"job": "B-J3", "stage": "bank_assemble", "device": "cuda:0",
                               "_output_subdir": "assemble", **common}
    configs["BJ3_VERIFY"] = {"job": "B-J3", "stage": "bank_verify", "device": "cuda:1",
                             "_output_subdir": "verify", **common}

    env = (f'set -u; export PYTHONPATH="{REPO}/.pydeps:{OVERLAY}:${{PYTHONPATH:-}}"; '
           "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; echo \"== FP-04 BANK PROTOCOL V1\"; "
           f'sha256sum "{OUT}" "{S0_CERT}" "{ADMISSION}" "{FS_BUNDLE}/reference_manifest.json" '
           f'"{FS_BUNDLE}/fs_adapter.pt"; '
           "python3 -c \"import torch, xformers; print('env torch', torch.__version__, 'xformers', "
           "xformers.__version__, 'gpus', torch.cuda.device_count())\"; ")
    script = "python3 -u @SNAPSHOT@/scripts/r2_fs_bank_train.py"
    base = (
        "--seed 20260921 --rank-per-expert 4 --num-experts 4 --target-blocks 18 19 20 21 22 23 "
        "--hold-steps 4 --train-steps 4 --lead-steps 4 --panel-lead-steps 1 4 12 --a0 0.25 --rho 0.25 "
        f"--data-role bank_fit --config {GATE_CONFIG} --admission {ADMISSION} "
        f"--s0-certificate {S0_CERT} --protocol {OUT} --protocol-sha256 <PROTOCOL_SHA256> "
        f"--fs-reference-manifest {FS_BUNDLE}/reference_manifest.json "
        f"--fs-reference-manifest-sha256 {fs_reference['reference_manifest']['sha256']} "
        f"--fs-certify-decision {FS_CERTIFY} --fs-adapter {FS_BUNDLE}/fs_adapter.pt "
        f"--fs-merged-backbone {FS_BUNDLE}/fs_merged_backbone.pt")
    meas = ("--profile-steps 3 --profile-warmup 1 --memory-budget-gib 80.0 "
            "--memory-safety-fraction 0.8 --max-seconds-per-update 30.0")

    def workers(prefix: str, mode: str, updates: int, lr: str) -> str:
        return ("status=0; pids=(); for i in 0 1 2 3; do D=@OUTPUT@/expert$i; mkdir -p $D; "
                f"{script} --stage bank_train --expert-index $i --mode {mode} --max-updates {updates} "
                f"--learning-rate {lr} --config-id {prefix}_E$i --device cuda:$i {base} --output-dir $D "
                "> $D/stdout.log 2>&1 & pids+=($!); done; for i in 0 1 2 3; do if wait ${pids[$i]}; "
                f"then echo \"{prefix}_E$i rc=0\"; else rc=$?; echo \"{prefix}_E$i rc=$rc\"; status=1; fi; done; ")

    bj1 = (env + workers("BJ1", "gradient_check", 16, "0.0025")
           + "mkdir -p @OUTPUT@/profile @OUTPUT@/horizon; "
           f"{script} --stage profile --expert-index 0 --mode gradient_check --max-updates 16 "
           f"--learning-rate 0.0025 --config-id BJ1_PROFILE --device cuda:0 {base} {meas} "
           "--output-dir @OUTPUT@/profile > @OUTPUT@/profile/stdout.log 2>&1 & p1=$!; "
           f"{script} --stage horizon_check --expert-index 0 --mode gradient_check --max-updates 16 "
           f"--learning-rate 0.0025 --config-id BJ1_HORIZON --device cuda:1 {base} {meas} "
           "--output-dir @OUTPUT@/horizon > @OUTPUT@/horizon/stdout.log 2>&1 & p2=$!; "
           'if wait $p1; then echo "BJ1_PROFILE rc=0"; else echo "BJ1_PROFILE rc=$?"; status=1; fi; '
           'if wait $p2; then echo "BJ1_HORIZON rc=0"; else echo "BJ1_HORIZON rc=$?"; status=1; fi; '
           "exit $status")
    bj2 = env + workers("BJ2", "formal", 32, "0.0025") + "exit $status"
    bj2f = env + workers("BJ2F", "formal", 32, "0.00125") + "exit $status"
    bj3 = (env + "mkdir -p @OUTPUT@/assemble @OUTPUT@/verify; "
           f"{script} --stage bank_assemble --config-id BJ3_ASSEMBLE --device cuda:0 {base} "
           "--bank-decision <FORMAL_DECISION> --output-dir @OUTPUT@/assemble "
           '> @OUTPUT@/assemble/stdout.log 2>&1; rc=$?; echo "BJ3_ASSEMBLE rc=$rc"; '
           "if [ $rc -ne 0 ]; then exit $rc; fi; "
           f"{script} --stage bank_verify --config-id BJ3_VERIFY --device cuda:1 {base} "
           "--bank-decision <FORMAL_DECISION> --bank-assembly @OUTPUT@/assemble/bank_assembly.json "
           '--output-dir @OUTPUT@/verify > @OUTPUT@/verify/stdout.log 2>&1; rc=$?; '
           'echo "BJ3_VERIFY rc=$rc"; exit $rc')
    decider = "python3 <JOB_RUN_DIR>/source/scripts/r4_bank_decide.py"
    dec = RUN / "decisions"

    protocol = {
        "schema_version": "ed-fs-protocol/1",
        "protocol_kind": "bank",
        "protocol_id": "FP04-BANK-v1",
        "status": "PREREGISTERED_BEFORE_FP04_JOBS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(RUN),
        "self_reference": ("This file's SHA-256 is every FP-04 job's --protocol-sha256 (placeholder "
                           "<PROTOCOL_SHA256> in jobs.*.argv_template); it cannot be stored inside."),
        "rules_verbatim": {"BANK-QUAL-v1_BANK-SELECT-v1_BANK-SHORTSTEP-v1_BANK-CERTIFY-v1_BANK-BUDGET-v1": RULES},
        "design": {"num_experts": 4, "rank_per_expert": 4, "a0": 0.25, "rho": 0.25, "max_active": 1,
                   "hold_steps": 4, "target_blocks": BLOCKS, "train_steps": 4, "objective_lead_steps": [4],
                   "optimizer": "Adam defaults, grad clip 1.0, constant lr, seed 20260921 + expert_index",
                   "diversity_rule": grouping["rule"]},
        "qualification_rule": {
            "rule": "BANK-QUAL-v1", "horizon": 32,
            "e0": {"torch_version": "2.3.1+cu121", "xformers_version": "0.0.27",
                   "require_official_backend": True, "tf32": "off (effective getters)"},
            "e2": {"clip_events_max": 0, "per_visit_ratio_max": 1.25, "epoch_onset_factor": 1.02,
                   "epoch_size": "group_size"},
            "e3": {"own_group_mean_ratio_24h_lt": 1.0},
            "report_only": ["own_group_6h", "own_group_72h", "own_group_24h_per_issue",
                            "per_visit_direction", "prefix_equals_shortstep"],
            "implementation": "earthdelta.bank_training.evaluate_bank_expert_qualification"},
        "shortstep_rule": {"rule": "BANK-SHORTSTEP-v1", "horizon": 16, "n_workers": 4,
                           "require_real_cuda_profile": True,
                           "implementation": "scripts/r4_bank_decide.py decide_shortstep"},
        "selection_rule": {"rule": "BANK-SELECT-v1", "num_experts": 4,
                           "fallback": {"job": "B-J2F", "after_job": "B-J2",
                                        "requires_failed_criterion": "E2",
                                        "scope": "every failed expert has E2 == FAIL (other "
                                                 "criteria may fail alongside); else no fallback"},
                           "implementation": "earthdelta.bank_training.decide_bank_formal"},
        "assembly_rule": {"rule": "BANK-CERTIFY-v1", "tolerance": 0.0, "probe_steps": [1, 4, 12],
                          "continuation_total_steps": 12,
                          "implementation": "earthdelta.bank_training.evaluate_bank_assembly + "
                                            "earthdelta.registry.verify_bank_bundle"},
        "budget": {"rule": "BANK-BUDGET-v1", "jobs": ["B-J1", "B-J2", "B-J2F", "B-J3"],
                   "max_gpu_jobs": 7, "expected_gpu_jobs": 3, "infra_retries_total": 1,
                   "b_j2_only_if": "B-J1 shortstep decision PASS",
                   "b_j2f_only_if": ("B-J2 formal decision STOP_CURRENT_BANK with fallback_authorized "
                                     "(every failed expert failed E2); never after INVALID; a STOP with "
                                     "any non-E2 failed expert (e.g. E3 only) is final, no fallback"),
                   "fallback_refinement": ("E2-only fallback trigger approved by the coordinator before "
                                           "this freeze, while no FP-04 job had been submitted"),
                   "b_j3_only_if": "a formal decision BANK_QUAL_PASS_PENDING_ASSEMBLY",
                   "stop": "no further job without the coordinator"},
        "inputs": {
            "gate_config": ref(GATE_CONFIG),
            "s0_certificate": ref(S0_CERT),
            "admission": ref(ADMISSION),
            "admission_declaration": ref(RUN / "admission/bank_fit_selection_declaration.json"),
            "admission_attempt1": ref(RUN / "admission/bank_fit_admission_attempt1.json"),
            "admission_slot47_candidates": ref(RUN / "admission/replacement_candidates_slot47_admission.json"),
            "admission_consumer_check": ref(RUN / "admission/admission_consumer_check.json"),
            "grouping": ref(GROUPING),
            "cpu_pins_and_fs_dryrun": ref(PINS),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "normalization_identity": manifest["normalization_identity"],
            "certified_overlay": str(OVERLAY),
            "fs_reference": fs_reference,
            "bank": bank_pins,
            "train_issue_ids": [r["issue_id"] for r in rows],
            "train_issue_times_utc": [iso(r["issue_time"]) for r in rows],
            "exposure_window_utc": ["2020-01-01T12:00:00Z", "2020-07-04T18:00:00Z"],
        },
        "configs": configs,
        "jobs": {
            "B-J1": {"label": "r4bankbj1", "timeout_seconds": 3600, "argv_template": ["bash", "-lc", bj1],
                     "decision": (f"{decider} --kind shortstep --job B-J1 --protocol {OUT} --protocol-sha256 "
                                  f"<PROTOCOL_SHA256> --job-dir <JOB_RUN_DIR> --out {dec}/BJ1_shortstep_decision.json")},
            "B-J2": {"label": "r4bankbj2", "timeout_seconds": 3600, "argv_template": ["bash", "-lc", bj2],
                     "decision": (f"{decider} --kind formal --job B-J2 --protocol {OUT} --protocol-sha256 "
                                  f"<PROTOCOL_SHA256> --job-dir <JOB_RUN_DIR> --shortstep-decision "
                                  f"{dec}/BJ1_shortstep_decision.json --out {dec}/BJ2_formal_decision.json")},
            "B-J2F": {"label": "r4bankbj2f", "timeout_seconds": 3600, "argv_template": ["bash", "-lc", bj2f],
                      "decision": (f"{decider} --kind formal --job B-J2F --protocol {OUT} --protocol-sha256 "
                                   f"<PROTOCOL_SHA256> --job-dir <JOB_RUN_DIR> --prior-formal-decision "
                                   f"{dec}/BJ2_formal_decision.json --out {dec}/BJ2F_formal_decision.json")},
            "B-J3": {"label": "r4bankbj3", "timeout_seconds": 3600, "argv_template": ["bash", "-lc", bj3],
                     "placeholders": {"<FORMAL_DECISION>": "the BANK_QUAL_PASS_PENDING_ASSEMBLY formal decision"},
                     "certify": (f"{decider} --kind certify --protocol {OUT} --protocol-sha256 <PROTOCOL_SHA256> "
                                 "--formal-decision <FORMAL_DECISION> --assemble-dir <JOB_RUN_DIR>/assemble "
                                 f"--verify-dir <JOB_RUN_DIR>/verify --out {RUN}/certify/BJ3_certify_decision.json "
                                 f"(publishes {RUN}/certify/bank/ only on BANK_CERTIFIED)")},
        },
        "historical_bank_jobs_ledger_only": {
            "pt-cdj1s2le": ("artifacts/round2_cci/ed-r3-j4-bank-formal-v2-0922011307-2d8e12: 4 workers each "
                            "re-fit their own Fs (fs_backbone_digest fa7c2871/701387a5/0c7fbcd3/b64a26d9), "
                            "groups 3/2/2/1 of the old 8 issues, lr 1e-2, 500 updates, expert 2 clipped 17x. "
                            "Protocol violation; ledger only; never evidence for FP-04."),
        },
        "limitations": [
            "The 48 bank_fit issues lie inside the already Fs-exposed 2020 window; they and the 2019 FP-03 "
            "holdout must be purged from any policy_dev/confirm use.",
            "2020 rows carry split_id 'test' while admitted as bank_fit (as in FP-03).",
            "2020 store index 712 (2020-06-27T00Z) has a specific_humidity_100 slab at 102 sigma; slot 47 was "
            "dropped by the declared rule; any later window containing that step must re-certify content.",
            "S0 certifies the forward pass only; the training backward is exercised, not certified.",
            "FP-04 PASS = a qualified, exactly-assembled, registry-certified K=4 bank exists; not usefulness "
            "(delta_min not pre-registered; out-of-sample evaluation is FP-05).",
        ],
        "source_at_preregistration": {p: sha(REPO / p) for p in SOURCE_FILES},
        "preregistration_md": ref(RUN / "protocol/PREREGISTRATION.md"),
    }
    OUT.write_text(json.dumps(protocol, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
