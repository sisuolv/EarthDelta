#!/usr/bin/env python3
"""Assemble protocol/cache_protocol_v1.json (FP-05b, frozen before any FP-05b GPU job).

Every pin is re-hashed here from the file it names. FP-04's inputs.fs_reference
block is copied verbatim (digests included). The 17 FP-04-pinned source files
must still equal FP-04's source_at_preregistration; the FP-05b source/test
files are added. Write-once: refuses to overwrite an existing protocol.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

REPO = Path("/mnt/afs/260010168/EarthDelta")
RUN = REPO / "plans/plan_v4_0923/run_20260924T104725Z_fp05b"
OUT = RUN / "protocol/cache_protocol_v1.json"
PREREG = RUN / "protocol/PREREGISTRATION.md"
FP04 = REPO / "plans/plan_v4_0923/run_20260924T033627Z_fp04_bank"
FP04_PROTOCOL = FP04 / "protocol/bank_protocol_v1.json"
FP04_PROTOCOL_SHA = "394562dde7cae5b3b805e6f5344f680ec9f24c71d9ed90c3893d96e8867da0d1"
BANK = FP04 / "certify/bank"
FP05A = REPO / "plans/plan_v4_0923/run_20260924T093810Z_fp05a"
DRAFT = FP05A / "protocol/fp05_protocol.DRAFT.json"
ANCHOR_JOB = REPO / "artifacts/round2_cci/ed-r4bankbj2-0924052624-dde08d"
OVERLAY = REPO / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/cuda_site"
SUBMIT = REPO / "plans/plans_v2_0921/cci/submit_job.py"
NEW_SOURCES = [
    "earthdelta/split_freeze.py", "earthdelta/candidate_cache.py", "earthdelta/policy_oof.py",
    "scripts/r4_candidate_cache.py", "scripts/r4_cache_decide.py", "scripts/r4_policy_oof.py",
    "tests/test_split_freeze.py", "tests/test_fp05_consumption_chain.py",
    "tests/test_plan_cache_evaluation.py", "tests/test_policy_oof_isolation.py",
]
RULES = PREREG.read_text().split("```")[1].strip()


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(p: Path) -> dict:
    return {"path": str(p), "sha256": sha(p)}


def pinned(p: Path) -> str:
    return (p.parent / (p.stem + ".sha256")).read_text().split()[0]


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists (write-once)")
    assert sha(FP04_PROTOCOL) == FP04_PROTOCOL_SHA
    fp04 = json.loads(FP04_PROTOCOL.read_text())
    core = fp04["source_at_preregistration"]
    drift = [p for p, h in core.items() if sha(REPO / p) != h]
    assert not drift, drift
    assert len(core) == 17

    freeze_path = RUN / "splits/split_freeze_v1.json"
    assert sha(freeze_path) == pinned(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    adm = RUN / "admission"
    pd_adm = adm / "policy_dev_admission.json"
    pd_rec = json.loads(pd_adm.read_text())
    assert pd_rec["passed"] and pd_rec["admission"]["n_admitted"] == 112
    pd_check = json.loads((adm / "policy_dev_consumer_check.json").read_text())
    dbg_check = json.loads((adm / "debug_consumer_check.json").read_text())
    assert pd_check["passed"] and pd_check["record"]["sha256"] == sha(pd_adm)
    assert dbg_check["passed"] and dbg_check["issue_ids"] == freeze["roles"]["debug"]["allowed_issue_ids"]
    selection = json.loads((adm / "policy_dev_selection.json").read_text())
    assert selection["manifest"]["sha256"] == sha(adm / "policy_dev_manifest.json")
    pd_ids = [r["issue_id"] for r in pd_rec["admission"]["admitted"]]
    assert pd_ids == selection["nested_subsets"]["N"]["issue_ids"]
    gap = selection["strata_without_policy_dev_representation"]

    bank_manifest = json.loads((BANK / "bank_manifest.json").read_text())
    binding = bank_manifest["binding"]
    registry = json.loads((BANK / "registry.json").read_text())
    import sys
    sys.path.insert(0, str(REPO))
    from earthdelta import candidate_cache as cc

    registry_pin = cc.registry_entries_pin(registry)
    anchors = {str(k): ref(ANCHOR_JOB / f"expert{k}" / f"bank_panels_expert{k}.json") for k in range(4)}
    for k in range(4):
        src = binding["experts"][k]["source"]["expert_file"]["path"]
        assert src.startswith(str(ANCHOR_JOB)), src

    debug_ids = freeze["roles"]["debug"]["allowed_issue_ids"]
    configs = {}
    for w in range(4):
        own = debug_ids[2 * w:2 * w + 2]
        dup = debug_ids[(2 * w + 2) % 8:(2 * w + 2) % 8 + 2]
        configs[f"CJ1_W{w}"] = {"job": "C-J1", "stage": "cache", "kind": "debug", "role": "debug",
                                "worker": w, "device": f"cuda:{w}", "output_subdir": f"w{w}",
                                "own_issue_ids": own, "duplicate_issue_ids": dup,
                                "issue_ids": own + dup}
        configs[f"CJ2_S{w}"] = {"job": "C-J2", "stage": "cache", "kind": "dev", "role": "policy_dev",
                                "shard": w, "n_shards": 4, "device": f"cuda:{w}",
                                "output_subdir": f"s{w}",
                                "issue_ids": "dev_protocol.json shard_map == shard (i mod 4 over time order)"}

    env = (f'set -u; export PYTHONPATH="{REPO}/.pydeps:{OVERLAY}:${{PYTHONPATH:-}}"; '
           "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; echo \"== FP-05B CACHE PROTOCOL V1\"; "
           f'sha256sum "{OUT}" "{freeze_path}" "{pd_adm}" "{BANK}/bank.pt" '
           f'"{binding["fs"]["reference_manifest"]["path"]}"; '
           "python3 -c \"import torch, xformers; print('env torch', torch.__version__, 'xformers', "
           "xformers.__version__, 'gpus', torch.cuda.device_count())\"; ")
    script = "python3 -u @SNAPSHOT@/scripts/r4_candidate_cache.py cache"

    def workers(prefix: str, sub: str, extra: str = "") -> str:
        return ("status=0; pids=(); for i in 0 1 2 3; do D=@OUTPUT@/" + sub + "$i; mkdir -p $D; "
                f"{script} --protocol {OUT} --protocol-sha256 <PROTOCOL_SHA256> --config-id {prefix}$i "
                f"--device cuda:$i --output-dir $D {extra}> $D/stdout.log 2>&1 & pids+=($!); done; "
                "for i in 0 1 2 3; do if wait ${pids[$i]}; then echo \"" + prefix + "$i rc=0\"; else rc=$?; "
                "echo \"" + prefix + "$i rc=$rc\"; status=1; fi; done; exit $status")

    cj1 = env + workers("CJ1_W", "w")
    cj2 = env + workers("CJ2_S", "s", "--dev-protocol <DEV_PROTOCOL> --dev-protocol-sha256 <DEV_PROTOCOL_SHA256> ")
    decide = (f"PYTHONPATH=.pydeps python3 scripts/r4_cache_decide.py --kind debug --protocol {OUT} "
              f"--protocol-sha256 <PROTOCOL_SHA256> --job-dir <JOB_RUN_DIR> "
              f"--out {RUN}/decisions/CJ1_debug_decision.json --dev-protocol-out {RUN}/cache/dev_protocol.json")
    merge = (f"PYTHONPATH=.pydeps python3 scripts/r4_candidate_cache.py merge --protocol {OUT} "
             f"--protocol-sha256 <PROTOCOL_SHA256> --dev-protocol {RUN}/cache/dev_protocol.json "
             "--dev-protocol-sha256 <DEV_PROTOCOL_SHA256> --shard-dirs <JOB_RUN_DIR>/s0 <JOB_RUN_DIR>/s1 "
             f"<JOB_RUN_DIR>/s2 <JOB_RUN_DIR>/s3 --out {RUN}/cache/dev")
    oof = f"PYTHONPATH=.pydeps python3 scripts/r4_policy_oof.py"
    common = f"--protocol {OUT} --protocol-sha256 <PROTOCOL_SHA256> --cache-manifest {RUN}/cache/dev/cache_manifest.json"
    caveat = {
        "short": "strata 16 and 19 (2019-10-26..11-02, 2019-11-16..11-23) have no policy_dev issue: real data gap",
        "text": ("policy_dev covers 23 of 25 weekly strata: strata 16 (2019-10-26T12Z..11-02T06Z) and 19 "
                 "(2019-11-16T12Z..11-23T06Z) have zero issues because 2019.zarr holds non-finite values there "
                 "(no clean t-12h..t+72h window). N_target = 112 of 128 (16 targets dropped by the declared "
                 "POLICY-DEV-SELECT-v1 rule; outcome accepted by the coordinator). A data-availability gap, not a "
                 "selection choice; every DEV result and every FP-06 statement must carry this caveat."),
        "strata": gap, "N_target": 112, "n_strata_represented": 23,
        "coordinator_resolution": ref(adm / "policy_dev_selection_coordinator_resolution.json")}

    draft = json.loads(DRAFT.read_text())
    protocol = {
        "schema_version": "ed-fs-protocol/1",
        "protocol_kind": "cache",
        "protocol_id": "FP05B-CACHE-v1",
        "status": "PREREGISTERED_BEFORE_FP05_JOBS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(RUN),
        "self_reference": ("This file's SHA-256 is every FP-05b job's and CPU stage's --protocol-sha256 "
                           "(placeholder <PROTOCOL_SHA256>); it cannot be stored inside."),
        "supersedes_draft": {**ref(DRAFT), "draft_status": draft["status"],
                             "undecided_fields_index_now_empty": True},
        "authorization": {"policy_dev_admission_authorized": True, "gpu_jobs_authorized": True,
                          "gpu_jobs_require_coordinator_go_ahead_per_job": True,
                          "training_authorized": False, "confirm_access_authorized": False},
        "rules_verbatim": {"FP05B_RULES": RULES},
        "design": {"candidates": list(cc.CANDIDATE_IDS), "lead_steps": list(cc.LEAD_STEPS),
                   "lead_hours": list(cc.LEAD_HOURS), "target_blocks": [18, 19, 20, 21, 22, 23],
                   "objective": "build_objective_spec(bridge, lat, lead_steps=(4,), space='raw'); per lead via dataclasses.replace",
                   "endpoint_space": "raw (denormalized), float32",
                   "F0_background": "report-only, never a candidate",
                   "features": {"grid": list(cc.FEATURE_GRID), "n_features": cc.N_FEATURES,
                                "inputs": "x_t, x_{t-6h}, x_{t-12h}, issue time"}},
        "decided": {
            "boundary_rule": "conservative_full_24h_buffer_incl_S0_aggregate_result_2020-01-01T00Z",
            "confirm": None, "confirm_status": "UNASSIGNED_NO_ACCESS",
            "F0_background_trajectory": "included_report_only", "primary_lead_hours": 24,
            "report_leads_hours": [6, 24, 72], "gain_normalizer": "sum L_Fs, recomputed per bootstrap draw",
            "delta_min_fs_relative": 0.0034,
            "delta_min_anchor": {"fs_denominator": 0.0033810878144209864, "f0_denominator": 0.0033925583521250413},
            "max_harm72_vs_Fs": "CI_low(G72(M vs Fs)) >= -delta_min", "dynamic_over_static_required": True,
            "N_final": "DEV-SCALE-v1 choice from {112, 56, 30}, chosen once", "N_target": 112,
            "debug_issues": debug_ids, "replacement_rule": "POLICY-DEV-SELECT-v1 as declared (outcome accepted)",
            "outer_mode": "bidirectional blocked cross-fit", "outer_folds": 4, "inner_folds": 3,
            "base_block_days": 7, "ridge_grid": [0.1, 1.0, 10.0, 100.0, 1e3, 1e4], "k_grid": [2, 3, 4],
            "pca_dim": 8, "confidence_level": 0.95, "bootstrap_unit": "paired whole 7-day UTC block",
            "bootstrap_draws": 10000, "block_days": 7, "sensitivity_block_days": [14], "seed": 20260924,
            "MDE": "report-only, exposed data only", "max_gpu_jobs_total": 4, "infra_retries_total": 1,
            "gpu_budget_seconds": 4 * 3600,
            "profile": "FP-04 measured 1.43 s per 4-step fwd+bwd and 14.8 GiB; the C-J1 measurement is binding",
            "coordinator_decisions": ["GPU cap 4", "outer folds 4",
                                      "accept POLICY-DEV-SELECT-v1 outcome N_target=112"]},
        "cache_run_rule": {"rule": "CACHE-RUN-v1",
                           "e0": {"torch_version": "2.3.1+cu121", "xformers_version": "0.0.27",
                                  "require_official_backend": True, "tf32": "off (effective getters)"}},
        "cache_valid_rule": {"rule": "CACHE-VALID-v1", "v4_max_nonfinite_fraction": 0.02,
                             "v5_relative_tol": 1e-12, "v6_atol": 1e-11, "v6_rtol": 1e-8,
                             "implementation": "earthdelta.candidate_cache.validate_cache_matrix"},
        "debug_rule": {"rule": "DEBUG-DECIDE-v1", "anchor_tolerance": 0.0, "inexact_band_rel": 1e-6,
                       "implementation": "earthdelta.candidate_cache.decide_debug via scripts/r4_cache_decide.py"},
        "dev_scale_rule": {"rule": "DEV-SCALE-v1", "sizes": {"N": 112, "N/2": 56, "N/4": 30},
                           "time_budget_s": 2700, "bytes_cap": 60e9, "memory_cap_gib": 64.0,
                           "artifact_root": str(REPO / "artifacts/round2_cci"),
                           "implementation": "earthdelta.candidate_cache.decide_dev_scale"},
        "oof_rule": {"rule": "OOF-v1", "implementation": "earthdelta.policy_oof",
                     "outer_block_groups": [[0, 5], [6, 11], [12, 17], [18, 24]]},
        "eval_rule": {"rule": "EVAL-v1 + DELTA-MIN-v1", "implementation": "earthdelta.policy_oof.paired_block_bootstrap"},
        "budget": {"rule": "CACHE-BUDGET-v1", "jobs": ["C-J1", "C-J1b", "C-J2"], "max_gpu_jobs": 4,
                   "expected_gpu_jobs": 2, "infra_retries_total": 1, "timeout_seconds": 3600,
                   "c_j2_only_if": "DEBUG-DECIDE-v1 PASS and dev_protocol.json with DEV-SCALE CHOSEN",
                   "stop": "no job without the coordinator's go-ahead"},
        "split": {
            "freeze": ref(freeze_path),
            "stores": {"policy_dev": "2019.zarr", "debug": "2020.zarr"},
            "allow_lists": {"policy_dev": {"admission": ref(pd_adm), "issue_ids": pd_ids}},
            "debug_allow_list": "split_freeze_v1.json roles.debug (8 ids + FP-04 admission sha256)",
            "coverage_caveat": caveat,
            "labels": freeze["labels"],
        },
        "inputs": {
            "gate_config": fp04["inputs"]["gate_config"],
            "s0_certificate": fp04["inputs"]["s0_certificate"],
            "checkpoint_sha256": fp04["inputs"]["checkpoint_sha256"],
            "normalization_identity": fp04["inputs"]["normalization_identity"],
            "certified_overlay": str(OVERLAY),
            "fs_reference": fp04["inputs"]["fs_reference"],
            "fp04_protocol": ref(FP04_PROTOCOL),
            "bank_bundle": {"dir": str(BANK), "manifest": ref(BANK / "bank_manifest.json"),
                            "registry_sha256": sha(BANK / "registry.json"),
                            "bank_sha256": binding["bank"]["sha256"], "bank_digest": binding["bank"]["bank_digest"],
                            "expert_digests": binding["bank"]["expert_digests"],
                            "fp04_protocol_sha256": FP04_PROTOCOL_SHA,
                            "certify_decision": ref(FP04 / "certify/BJ3_certify_decision.json")},
            "registry_entries": registry_pin,
            "exposure_ledger": ref(FP05A / "entry/exposure_ledger.json"),
            "exposure_reconstruction_report": ref(FP05A / "entry/exposure_reconstruction_report.json"),
            "fp05a_task_result": ref(FP05A / "task_result_FP-05a.json"),
            "split_freeze": ref(freeze_path),
            "exposure_addendum_cp0": ref(RUN / "splits/exposure_ledger_addendum_fp05b_cp0.json"),
            "policy_dev": {
                "admission": ref(pd_adm),
                "consumer_check": ref(adm / "policy_dev_consumer_check.json"),
                "declaration": ref(adm / "policy_dev_selection_declaration.json"),
                "finiteness_scan": ref(RUN / "splits/finiteness_scan_2019_H2.json"),
                "stop_record": ref(adm / "policy_dev_selection_STOP.json"),
                "coordinator_resolution": ref(adm / "policy_dev_selection_coordinator_resolution.json"),
                "manifest": ref(adm / "policy_dev_manifest.json"),
                "selection": ref(adm / "policy_dev_selection.json")},
            "debug": {
                "admission": ref(FP04 / "admission/bank_fit_admission.json"),
                "grouping": ref(FP04 / "admission/bank_fit_grouping.json"),
                "consumer_check": ref(adm / "debug_consumer_check.json"),
                "selection": ref(adm / "debug_selection.json"),
                "anchors": anchors},
        },
        "configs": configs,
        "jobs": {
            "C-J1": {"label": "r4cachecj1", "timeout_seconds": 3600, "argv_template": ["bash", "-lc", cj1],
                     "submit": f"python3 {SUBMIT} --label r4cachecj1 --timeout-seconds 3600 --argv-file {RUN}/jobs/C-J1.argv.json",
                     "decision": decide},
            "C-J1b": {"label": "r4cachecj1b", "timeout_seconds": 3600, "argv_template": ["bash", "-lc", cj1],
                      "only_if": "C-J1 STOP on a defect in NEW FP-05 code, with a deviation record; at most once"},
            "C-J2": {"label": "r4cachecj2", "timeout_seconds": 3600, "argv_template": ["bash", "-lc", cj2],
                     "placeholders": {"<DEV_PROTOCOL>": f"{RUN}/cache/dev_protocol.json",
                                      "<DEV_PROTOCOL_SHA256>": "its sha256"},
                     "merge": merge},
            "CPU-policies": {"folds": f"{oof} folds {common} --out {RUN}/policies/folds.json",
                             "fit_predict": f"{oof} fit-predict {common} --folds {RUN}/policies/folds.json --out-dir {RUN}/policies",
                             "score": f"{oof} score {common} --prediction-freeze {RUN}/policies/prediction_freeze.json --out-dir {RUN}/evaluation"},
        },
        "limitations": [
            "policy_dev is a retrospective held period (2019H2), not a deployment; 147 pool-period times were integrity-read by the stopped shared-F0 sprint (disclosed).",
            "Coverage caveat: strata 16 and 19 unrepresented (real data gap); N_target 112; 23 bootstrap blocks.",
            "Seasonal mismatch: experts are Jan-Jun 2020 month-block regimes, policy_dev is Jul-Dec 2019.",
            "The certified Fs is slightly worse than F0 out of sample; H_Fs and G_F0 are reported.",
            "No confirm set exists; FP-05b neither names nor reads one.",
            "The C-J1 debug rows are FP-04 training rows (already exposed); they validate machinery only.",
        ],
        "source_at_preregistration": {**core, **{p: sha(REPO / p) for p in NEW_SOURCES}},
        "fp04_core_pins_unchanged": True,
        "preregistration_md": ref(PREREG),
    }
    OUT.write_text(json.dumps(protocol, indent=1) + "\n")
    digest = sha(OUT)
    (OUT.parent / "cache_protocol_v1.sha256").write_text(
        f"{digest}  cache_protocol_v1.json\nfrozen_utc {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
    argv = ["bash", "-lc", cj1.replace("<PROTOCOL_SHA256>", digest)]
    (RUN / "jobs/C-J1.argv.json").write_text(json.dumps(argv, indent=1) + "\n")
    print(json.dumps({"protocol": str(OUT), "sha256": digest, "n_sources": len(protocol["source_at_preregistration"]),
                      "c_j1_argv": str(RUN / "jobs/C-J1.argv.json")}))


if __name__ == "__main__":
    main()
