#!/usr/bin/env python3
"""Assemble protocol/fs_protocol_v1.json (FP-03 pre-registration, frozen before J1).

Every hash is computed here from the actual bytes. The written file is made
read-only by the caller; its SHA-256 is what every job argv carries.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

REPO = Path("/mnt/afs/260010168/EarthDelta")
RUN = REPO / "plans/plan_v4_0923/run_20260923T192759Z_fp03_fs"
OUT = RUN / "protocol" / "fs_protocol_v1.json"
GATE_CONFIG = REPO / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/s0/gate_config.json"
S0_CERT = REPO / ("plans/plan_v4_0923/run_20260923T_fp01_trace/gate_output/"
                  "s0-gate-20260923t174135587010z/s0_gate_result.json")
ADMISSION = RUN / "admission/bank_fit_admission.json"
PANEL_ADMISSION = RUN / "admission/holdout_panel_admission.json"
OVERLAY = REPO / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/cuda_site"
PROTOCOL_PATH_STR = str(OUT)
INIT_DIGEST = "2d0b92da4782f3e708e7ca1fb0b406b26ba070352cc52cb4501b2009c92beffa"
ARM_LR = {"A0": 1e-2, "A1": 5e-3, "A2": 2.5e-3, "A3": 1.25e-3}
NEGATIVE_CONTROLS = [  # known-bad historical adapters; must be REJECTED
    ("pt-3x63g0c6", "artifacts/round2_cci/ed-r3-j4-bank-formal500-0922004700-0a3c68/expert1"),
    ("pt-cdj1s2le", "artifacts/round2_cci/ed-r3-j4-bank-formal-v2-0922011307-2d8e12/expert2"),
    ("pt-e8y7ib01", "artifacts/round2_cci/ed-r4fsformal0923-0923180756-7b1776/experts/expert1"),
    ("pt-e8y7ib01", "artifacts/round2_cci/ed-r4fsformal0923-0923180756-7b1776/experts/expert0"),
]
SOURCE_FILES = [
    "earthdelta/static_adapter.py", "earthdelta/fs_protocol.py", "scripts/r2_fs_bank_train.py",
    "scripts/r4_fs_decide.py", "tests/test_fs_quality_gate.py",
    "tests/test_loaded_admission_binding.py", "tests/test_plan_gate_chain.py",
    "tests/test_fs_decide_cli.py", "tests/test_fs_static_adapter.py",
    "earthdelta/contracts.py", "earthdelta/bridge/stormer_bridge.py",
    "earthdelta/bridge/stormer_arch.py", "earthdelta/lowrank.py", "earthdelta/metrics_contract.py",
]

RULES_VERBATIM = {
    "FS-SELECT-v1": """FS-SELECT-v1. (1) The formal configuration is fitted once, in one ACP job, as 4 independent
processes on cuda:0..3 with byte-identical argv except --device/--output-dir. (2) The cuda:0
process (replica0/) is the DESIGNATED CANDIDATE, fixed by device index before submission;
cuda:1..3 are REPRODUCIBILITY WITNESSES and can never become Fs. (3) Fs is selected iff all 4
processes satisfy FS-QUAL-v1 AND the designated candidate then passes numerical_merge, reload
and identity gates in independent processes. (4) Otherwise no Fs is selected from that job; no
witness, intermediate checkpoint, screen arm, or artifact of pt-3x63g0c6/pt-cdj1s2le/pt-e8y7ib01
may be substituted. (5) No loss, gain, panel ratio, weight norm, or any policy_dev/confirm
quantity is consulted to choose among processes. FP-03 reads no policy_dev/confirm data.""",
    "FS-QUAL-v1": """FS-QUAL-v1 (per formal replica; L0 = common initial checkpoint = zero-init Fs, bitwise equal to
F0; L1 = checkpoint after exactly H_formal=256 updates; same 8 train issues, same Q/scale,
float64 reduction, no-grad 12-step rollout; E_k = mean train loss over updates 8k..8k+7)
 Q0 validity (else INVALID): official backend + torch 2.3.1+cu121 + xformers 0.0.27, TF32 off,
    S0 certificate consumed, admission consumer PASS, protocol/config-id match, init digest
    match, F0 panel identical to J1's (bitwise; pre-registered fallback bound 1e-6 relative).
 Q1 all losses/grad norms finite; n_updates == 256.
 Q2 stability: clip_events == 0; max_t L_t / L0_24h(issue_t) <= 1.25;
    for k=1..31: E_k <= 1.02 * min(E_0..E_{k-1}).
 Q3 per-visit last < first on 8/8 issues (listed separately).
 Q4 train panel 24h, r_i = L1/L0, m = mean L1 / mean L0:
    PASS iff max r_i < 1 and m <= 0.99; FAIL iff m > 1.00 or max r_i > 1.02;
    else INCONCLUSIVE_QUALIFICATION.
 Q5 72h guard on train panel: m72 <= 1.01          [or null -> report-only + limitation]
 Q6 holdout panel 24h mean ratio <= 1.02
 6h reported only. quality_pass = Q0..Q3 and Q4==PASS and Q5 and Q6.""",
    "FS-SCREEN-v1": """FS-SCREEN-v1. Arms A0..A3 on cuda:0..3: lr = 1e-2 (positive control), 5e-3, 2.5e-3, 1.25e-3;
all else identical (Adam defaults, clip 1.0, seed 20260921, same init digest, rank 4, blocks
18-23, 4x6h rollout, 24h objective, round-robin in the pt-e8y7ib01 issue order, no warmup/decay),
H=256, mode stability_screen. STABLE = Q1 and Q2 criteria. PROGRESSING = final train-panel 24h
mean ratio <= 0.99. If A0 is STABLE -> NOT_TESTABLE, STOP. S = {A1..A3: STABLE and PROGRESSING}.
S empty -> LR hypothesis REFUTED, STOP. Otherwise LR hypothesis CONFIRMED for H=256;
selected = highest-lr arm in S; fallback = next-lower-lr arm in S (or none). Report only:
onset update and lr x onset for each unstable arm.""",
    "FS-BUDGET-v1": """FS-BUDGET-v1. Scientific GPU jobs: J1, J2, J3, J3b (only if J3 is FAIL, never if INCONCLUSIVE),
J4: max 5. Infra retries <= 1 per job and <= 2 in total, all recorded; the same blocker 3 times
-> BLOCKED. Formal horizon = screened horizon (no extrapolation). Anything beyond this needs
protocol v2 and user approval.""",
}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": sha(path)}


def main() -> None:
    admission = json.loads(ADMISSION.read_text())
    panel = json.loads(PANEL_ADMISSION.read_text())
    train_rows = admission["admission"]["admitted"]
    hold_rows = panel["admission"]["admitted"]
    iso = lambda t: dt.datetime.fromtimestamp(int(t), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    times = [int(r["issue_time"]) for r in train_rows + hold_rows]
    support = [iso(min(times) - 12 * 3600), iso(max(times) + 72 * 3600)]

    common = {
        "seed": 20260921, "rank_per_expert": 4, "target_blocks": [18, 19, 20, 21, 22, 23],
        "train_steps": 4, "lead_steps": [4], "panel_lead_steps": [1, 4, 12], "hold_steps": 4,
        "data_role": "bank_fit", "limit": None,
        "config": ref(GATE_CONFIG), "admission": ref(ADMISSION),
        "panel_admission": ref(PANEL_ADMISSION), "s0_certificate": ref(S0_CERT),
    }
    configs = {}
    for i, (arm, lr) in enumerate(ARM_LR.items()):
        configs[f"J1_{arm}"] = {"job": "J1", "stage": "fs_fit", "mode": "gradient_check",
                                "max_updates": 32, "learning_rate": lr, "device": f"cuda:{i}",
                                "_output_subdir": f"arms/{arm}", **common}
        configs[f"J2_{arm}"] = {"job": "J2", "stage": "fs_fit", "mode": "stability_screen",
                                "max_updates": 256, "learning_rate": lr, "device": f"cuda:{i}",
                                "_output_subdir": f"arms/{arm}", **common}
    for i, (job_id, rel) in enumerate(NEGATIVE_CONTROLS):
        adapter = REPO / rel / "fs_adapter.pt"
        configs[f"J1_NC{i}"] = {
            "job": "J1", "stage": "fs_panel", "device": f"cuda:{i}",
            "fs_adapter": ref(adapter), "fs_adapter_sha256": sha(adapter),
            "_historical_fit_record": ref(REPO / rel / "fs_fit_record.json"),
            "_historical_job_id": job_id, "_output_subdir": f"negctl/NC{i}",
            "expected": "REJECTED", **{k: v for k, v in common.items()},
        }
    devices = {"one_of": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]}
    for job_id, field in (("J3", "selected_lr"), ("J3b", "fallback_lr")):
        configs[job_id] = {"job": job_id, "stage": "fs_fit", "mode": "formal", "max_updates": 256,
                           "learning_rate": {"from_screen_decision": field}, "device": devices,
                           "_output_subdir_template": "replica{index}", **common}
    j4_common = {k: common[k] for k in ("seed", "rank_per_expert", "target_blocks", "hold_steps",
                                         "data_role", "limit", "config", "admission",
                                         "s0_certificate")}
    configs["J4_FREEZE"] = {"job": "J4", "stage": "fs_freeze", "device": "cuda:0",
                            "num_experts": 4, "_output_subdir": "freeze", **j4_common}
    configs["J4_VERIFY"] = {"job": "J4", "stage": "fs_verify", "device": "cuda:1",
                            "num_experts": 4, "_output_subdir": "verify", **j4_common}

    env = (f'set -u; export PYTHONPATH="{REPO}/.pydeps:{OVERLAY}:${{PYTHONPATH:-}}"; '
           "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; "
           f'echo "== pre-registration binding"; sha256sum "{OUT}" "{S0_CERT}" "{ADMISSION}" '
           f'"{PANEL_ADMISSION}"; python3 -c "import torch, xformers; print(\'env torch\', '
           "torch.__version__, 'xformers', xformers.__version__, 'gpus', torch.cuda.device_count())\"; ")
    common_flags = (
        "--seed 20260921 --rank-per-expert 4 --target-blocks 18 19 20 21 22 23 --train-steps 4 "
        "--lead-steps 4 --hold-steps 4 --data-role bank_fit --panel-lead-steps 1 4 12 "
        f"--config {GATE_CONFIG} --admission {ADMISSION} --panel-admission {PANEL_ADMISSION} "
        f"--s0-certificate {S0_CERT} --protocol {OUT} --protocol-sha256 <PROTOCOL_SHA256>")
    script = "python3 -u @SNAPSHOT@/scripts/r2_fs_bank_train.py"

    def wave(label, subdir, stage_flags, lrs=None, extra=None):
        body = (f"pids=(); for i in 0 1 2 3; do D=@OUTPUT@/{subdir}; mkdir -p $D; "
                f"{script} {stage_flags} --device cuda:$i {common_flags} --output-dir $D "
                "> $D/stdout.log 2>&1 & pids+=($!); done; "
                "for i in 0 1 2 3; do if wait ${pids[$i]}; then echo \"" + label +
                " $i rc=0\"; else rc=$?; echo \"" + label + " $i rc=$rc\"; status=1; fi; done; ")
        return body

    lr_array = "LR=(0.01 0.005 0.0025 0.00125); "
    j1 = (env + "status=0; " + lr_array
          + wave("J1 arm A", "arms/A$i", "--stage fs_fit --mode gradient_check --max-updates 32 "
                 "--learning-rate ${LR[$i]} --config-id J1_A$i")
          + "NCA=(" + " ".join(str(REPO / rel / "fs_adapter.pt") for _, rel in NEGATIVE_CONTROLS)
          + "); NCS=(" + " ".join(sha(REPO / rel / "fs_adapter.pt") for _, rel in NEGATIVE_CONTROLS)
          + "); "
          + wave("J1 negctl NC", "negctl/NC$i", "--stage fs_panel --config-id J1_NC$i "
                 "--fs-adapter ${NCA[$i]} --fs-adapter-sha256 ${NCS[$i]}")
          + "exit $status")
    j2 = (env + "status=0; " + lr_array
          + wave("J2 arm A", "arms/A$i", "--stage fs_fit --mode stability_screen --max-updates 256 "
                 "--learning-rate ${LR[$i]} --config-id J2_A$i")
          + "exit $status")
    j3_template = (env + "status=0; "
                   + wave("J3 replica", "replica$i", "--stage fs_fit --mode formal --max-updates 256 "
                          "--learning-rate <SELECTED_LR_FROM_J2_DECISION> --screen-decision "
                          f"{RUN}/decisions/J2_screen_decision.json --config-id J3")
                   + "exit $status")

    protocol = {
        "schema_version": "ed-fs-protocol/1",
        "protocol_id": "FP03-FS-v1",
        "status": "PREREGISTERED_BEFORE_J1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(RUN),
        "self_reference": ("This file's SHA-256 is the value every job passes as "
                           "--protocol-sha256 (the only placeholder, <PROTOCOL_SHA256>, in "
                           "jobs.*.argv_template); it cannot be stored inside the file itself."),
        "rules_verbatim": RULES_VERBATIM,
        "qualification_rule": {
            "rule": "FS-QUAL-v1", "horizon": 256,
            "q0": {"torch_version": "2.3.1+cu121", "xformers_version": "0.0.27",
                   "official_model": "stormer.models.hub.stormer.Stormer", "tf32": "off",
                   "f0_panel_reference": "J1 arm A0 (cuda:0) F0 train+holdout panels, carried "
                                         "in decisions/J1_diag_decision.json",
                   "f0_panel_rel_tol": 1e-6},
            "q2": {"clip_events_max": 0, "per_visit_ratio_max": 1.25, "epoch_onset_factor": 1.02,
                   "epoch_size": 8},
            "q3": {"required_issues": 8},
            "q4": {"pass_max_ratio_lt": 1.0, "pass_mean_ratio_le": 0.99,
                   "fail_mean_ratio_gt": 1.00, "fail_max_ratio_gt": 1.02},
            "q5": {"mean_ratio_72h_le": 1.01, "report_only": False},
            "q6": {"holdout_mean_ratio_24h_le": 1.02, "report_only": False},
            "report_only_leads_hours": [6],
            "implementation": "earthdelta.static_adapter.evaluate_fs_qualification",
        },
        "screen_rule": {"rule": "FS-SCREEN-v1", "horizon": 256, "control_arm": "A0",
                        "candidate_arms": ["A1", "A2", "A3"], "arm_lr": ARM_LR,
                        "progressing_mean_ratio_24h_le": 0.99,
                        "implementation": "earthdelta.static_adapter.decide_lr_screen"},
        "selection_rule": {"rule": "FS-SELECT-v1", "n_replicas": 4, "designated_device_index": 0,
                           "designated_dir": "replica0",
                           "implementation": "earthdelta.static_adapter.decide_formal_fs"},
        "budget": {"rule": "FS-BUDGET-v1", "scientific_jobs": ["J1", "J2", "J3", "J3b", "J4"],
                   "max_scientific_jobs": 5, "infra_retries_per_job": 1, "infra_retries_total": 2,
                   "same_blocker_limit": 3, "j3b_only_if": "J3 verdict STOP_FITTED_FS_QUALITY (FAIL)",
                   "formal_horizon_equals_screen_horizon": True,
                   "this_dispatch": ["J1", "J2"],
                   "requires_separate_authorization": ["J3", "J3b", "J4"]},
        "merge_equivalence_tolerance": {
            "atol_floor": 1e-5, "atol_relative": 1.5e-3, "rtol": 1e-5,
            "atol_rule": "max(atol_floor, atol_relative * delta_max_abs)", "steps": 4,
            "zero_edit_atol": 0.0, "continuation_after_hold": "exact (0.0), and must differ from F0",
            "scope": ("Fs merged-vs-always-on NUMERICAL MERGE check only. Frozen here as the "
                      "existing record_post_freeze_reference defaults. NEVER an S0 tolerance; "
                      "S0 stays max_abs_diff <= 1e-5 on the certified path."),
        },
        "inputs": {
            "gate_config": ref(GATE_CONFIG),
            "s0_certificate": ref(S0_CERT),
            "admission": ref(ADMISSION),
            "panel_admission": ref(PANEL_ADMISSION),
            "checkpoint_sha256": "7fde884ec85f4999d4bcb02d55ed72c142912071956083ebee08d3d86311b695",
            "normalization_identity": "ed-norm-identity/1:3e0b216bfbf34ab0",
            "certified_overlay": str(OVERLAY),
            "initial_adapter_digest": INIT_DIGEST,
            "initial_adapter_digest_recipe": ("static_adapter_digest(build_fs_adapter(1024, "
                                              "(18,...,23), rank_per_expert=4, seed=20260921))"),
            "train_issue_ids": [r["issue_id"] for r in train_rows],
            "train_issue_times_utc": [iso(r["issue_time"]) for r in train_rows],
            "holdout_issue_ids": [r["issue_id"] for r in hold_rows],
            "holdout_issue_times_utc": [iso(r["issue_time"]) for r in hold_rows],
            "fs_exposure_support_interval_utc": support,
            "forensics": ref(RUN / "forensics/historical_fs_forensics.json"),
            "ledger": ref(RUN / "ledger/fs_attempts_ledger.json"),
        },
        "configs": configs,
        "jobs": {
            "J1": {"label": "r4fs03diag", "timeout_seconds": 1800, "argv_template": ["bash", "-lc", j1],
                   "decision": ("python3 <J1 run dir>/source/scripts/r4_fs_decide.py --kind diag "
                                f"--protocol {OUT} --protocol-sha256 <PROTOCOL_SHA256> --job-dir "
                                f"<J1 run dir> --out {RUN}/decisions/J1_diag_decision.json"),
                   "pass_condition": ("decision verdict PASS: every arm Q0 facts true and code-path "
                                      "checks true; every negative control REJECTED; else STOP "
                                      "entirely (no J2)")},
            "J2": {"label": "r4fs03scrn", "timeout_seconds": 2700, "argv_template": ["bash", "-lc", j2],
                   "decision": ("python3 <J2 run dir>/source/scripts/r4_fs_decide.py --kind screen "
                                f"--protocol {OUT} --protocol-sha256 <PROTOCOL_SHA256> --job-dir "
                                f"<J2 run dir> --diag-decision {RUN}/decisions/J1_diag_decision.json "
                                f"--out {RUN}/decisions/J2_screen_decision.json")},
            "J3": {"label": "r4fs03formal", "timeout_seconds": 5400,
                   "argv_template": ["bash", "-lc", j3_template],
                   "note": ("<SELECTED_LR_FROM_J2_DECISION> is the decision's selected_lr; the CLI "
                            "refuses any other value. J3b: identical with --config-id J3b and "
                            "fallback_lr. Not submitted in this dispatch.")},
        },
        "decision_notes": {
            "screen_arm_validity": ("A screen arm whose Q0 facts fail makes the whole screen "
                                    "INVALID (not a scientific verdict); arms are never compared "
                                    "on quality numbers beyond FS-SCREEN-v1."),
            "negative_controls": ("NC evaluation = FS-QUAL-v1 Q1..Q6 on the historical training "
                                  "record + the new panels, with the Q1 horizon set to the "
                                  "record's own n_updates so Q1 cannot reject trivially; an NC is "
                                  "REJECTED iff the substantive verdict is FAIL with >=1 of Q2..Q6 "
                                  "FAIL. Q0 facts are required of the PANEL process only."),
            "training_vs_panel_losses": ("Q2's L_t is the grad-enabled training loss; L0_24h is "
                                         "the no-grad panel loss of the same issue at L0. At "
                                         "update 0 their ratio is ~1 up to kernel differences."),
            "formal_argv": ("J3/J3b use ONE config id for all 4 replicas so the argv is "
                            "byte-identical except --device/--output-dir; the decider re-checks "
                            "this from each process's recorded argv."),
        },
        "limitations": [
            ("DELAY_NOT_FIX: Adam moves each Fs parameter ~lr per update regardless of gradient "
             "reliability, so a lower lr may only DELAY the lr=1e-2 instability (onset epoch 10-11, "
             "i.e. updates 80-88, in all 12 historical formal replicas) beyond 256 updates rather "
             "than remove it. Because the J2 screen and the J3 formal fit share H=256 exactly, "
             "'no instability within 256 updates' is the correct bar for THIS Fs, and no claim is "
             "made about any longer horizon (formal horizon = screened horizon; no extrapolation)."),
            ("SPLIT_LABEL: build_manifest stamps 2020 rows split_id='test' (YEAR_SPLITS) while "
             "FP-03 admits them with data_role bank_fit, as the historical Fs runs did. The Fs "
             f"exposure support interval {support[0]} .. {support[1]} (train + holdout, t-12h..t+72h) "
             "is recorded here; any later policy_dev/confirm selection must exclude it with purge. "
             "The full bank_fit/policy_dev freeze remains FP-02/FP-05 work."),
            ("S0_SCOPE: S0 certifies the backbone FORWARD pass; the xformers backward used in "
             "training is exercised (J1 checks finite gradients) but is not separately certified."),
            ("HISTORICAL_RUNS: pt-3x63g0c6 / pt-cdj1s2le / pt-e8y7ib01 ran without the certified "
             "overlay (probable SDPA fallback, unproven) and on the uncertified admission; all 29 "
             "historical Fs records are INELIGIBLE_AS_FS (ledger) and serve only as negative "
             "controls / forensics."),
        ],
        "source_at_preregistration": {p: sha(REPO / p) for p in SOURCE_FILES},
        "preregistration_md": ref(RUN / "protocol/PREREGISTRATION.md"),
    }
    OUT.write_text(json.dumps(protocol, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
