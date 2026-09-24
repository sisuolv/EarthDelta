#!/usr/bin/env python3
"""FP-05a: build the UNFROZEN FP-05 protocol revision draft from this run's own evidence.

READ-ONLY w.r.t. the repository (writes only into this run dir), stdlib only.
Every data-role fact is copied mechanically from entry/exposure_reconstruction_report.json
and entry/certificate_recompute.json (this run). Every statistical/resource parameter that
has not been decided is null with status PROPOSED_NOT_FROZEN; earlier proposals (FP05_PLAN.md,
review package v1, review package v2) are listed only under `proposals_for_reference`.

Output (refuses to overwrite): protocol/fp05_protocol.DRAFT.json
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
P4 = "plans/plan_v4_0923"
V1 = "plans/plans_v5_0924/extracted_v1/EarthDelta_Codex_Followup_Plan_20260924"
V2 = "plans/plans_v5_0924/extracted_v2/EarthDelta_Codex_Followup_Plan_20260924"
PNF = "PROPOSED_NOT_FROZEN"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ref(rel: str) -> dict:
    return {"path": rel, "sha256": sha256(REPO / rel)}


def jl(p):
    return json.loads(Path(p).read_text())


def undecided(proposals: dict, note: str = "") -> dict:
    return {"value": None, "status": PNF, "proposals_for_reference": proposals, "note": note}


def main() -> int:
    out = RUN / "protocol/fp05_protocol.DRAFT.json"
    if out.exists():
        print("refusing to overwrite", out)
        return 2
    runrel = str(RUN.relative_to(REPO))
    rep = jl(RUN / "entry/exposure_reconstruction_report.json")
    cert = jl(RUN / "entry/certificate_recompute.json")
    fs, bank = cert["fs"], cert["bank"]
    pool = rep["2019H2_policy_dev_pool"]
    h2 = rep["2020H2"]
    bm = jl(REPO / f"{P4}/run_20260924T033627Z_fp04_bank/certify/bank/bank_manifest.json")["binding"]
    s0 = jl(REPO / f"{P4}/run_20260923T_fp01_trace/gate_output/s0-gate-20260923t174135587010z/s0_gate_result.json")

    draft = {
        "schema": "ed-fp05-protocol-draft/1",
        "status": "DRAFT_NOT_FROZEN",
        "execution_ready": False,
        "final_protocol_sha256": None,
        "frozen_utc": None,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "created_by": "FP-05a (read-only, CPU-only)",
        "authorization": {"gpu_jobs_authorized": False, "training_authorized": False,
                          "policy_dev_admission_authorized": False, "confirm_access_authorized": False},
        "supersedes": {
            "document": ref(f"{P4}/FP05_PLAN.md"),
            "withdrawn_claims": [
                {"id": "FP05_PLAN.D1.confirm", "claim": "2020H2 is an untouched confirm reservation",
                 "status": "WITHDRAWN",
                 "reason": "X1_N64 (pt-t3bfl3e9) trained on 56 issues 2020-07-13T18Z..2020-12-28T18Z; their t-12h..t+72h supports form one contiguous run 2020-07-13T06Z..2020-12-31T18Z (this run's exposure ledger). The v1 holdout issue 2020-07-01T18Z also exposes 2020-07-01T06Z..07-04T18Z to model results.",
                 "evidence": f"{runrel}/entry/exposure_reconstruction_report.json#2020H2"},
                {"id": "FP05_PLAN.D1.policy_dev_window", "claim": "policy_dev = 2019H2 (whole half-year)",
                 "status": "REPLACED_BY_LEDGER_DERIVED_CANDIDATE_POOL"},
                {"id": "FP05_PLAN.D2.delta_min", "claim": "delta_min = 0.34% decided",
                 "status": "DOWNGRADED_TO_PROPOSED_NOT_FROZEN"},
            ],
        },
        "baseline_bindings": {
            "s0": {"job_id": "pt-g3344e9z",
                   "certificate": ref(f"{P4}/run_20260923T_fp01_trace/gate_output/s0-gate-20260923t174135587010z/s0_gate_result.json"),
                   "s0_gate_pass": s0["s0_gate_pass"], "verdict_committed": s0["verdict_committed"],
                   "config_digest": s0["config_digest"], "checkpoint_sha256": s0["computed_checkpoint_sha256"],
                   "normalization_identity": s0["normalization_identity"],
                   "raw_platform_receipt": f"{runrel}/entry/job_receipts_index.json#FP01-gate-e"},
            "fs": {"reference_manifest": ref(f"{P4}/run_20260924T013959Z_fp03_v2/certify/fs/reference_manifest.json"),
                   "certify_decision": ref(f"{P4}/run_20260924T013959Z_fp03_v2/certify/V2_certify_decision.json"),
                   "protocol": ref(f"{P4}/run_20260924T013959Z_fp03_v2/protocol/fs_protocol_v2.json"),
                   "adapter_sha256": "7b39226b550a2795b1ed85020057bec46064f132b1c36d77a4db9b5682323089",
                   "merged_backbone_sha256": "1e48438ee98dc72552786011d25d9db3cc208672ceb8a7fc725c7f8fde77a79b",
                   "recomputed_in_fp05a": {k: fs["recomputed"][k]["mean_ratio"] for k in ("q4_train_24h", "q5_train_72h", "q6_holdout_24h")},
                   "holdout24_worse_count": fs["recomputed"]["q6_holdout_24h"]["n_worse_than_initial"],
                   "scientific_scope": "limited-harm static reference; no established out-of-sample benefit"},
            "bank": {"bank_manifest": ref(f"{P4}/run_20260924T033627Z_fp04_bank/certify/bank/bank_manifest.json"),
                     "registry": ref(f"{P4}/run_20260924T033627Z_fp04_bank/certify/bank/registry.json"),
                     "certify_decision": ref(f"{P4}/run_20260924T033627Z_fp04_bank/certify/BJ3_certify_decision.json"),
                     "protocol": ref(f"{P4}/run_20260924T033627Z_fp04_bank/protocol/bank_protocol_v1.json"),
                     "bank_file_sha256": bm["bank"]["sha256"], "bank_digest": bm["bank"]["bank_digest"],
                     "own_group24_ratios_recomputed_in_fp05a": bank["own_group24_mean_ratios_recomputed"],
                     "group_sizes": bank["group_sizes_recomputed"],
                     "A1_A5_exact_zero_recomputed": bank["assembly_all_exact_zero"],
                     "registry_verify_deep_passed": cert["bank_bundle_registry_verify"].get("passed"),
                     "out_of_sample_gate": "NOT_PERFORMED"},
            "protected_source": {"source": ref(f"{P4}/run_20260924T033627Z_fp04_bank/protocol/bank_protocol_v1.json"),
                                 "field": "source_at_preregistration", "n_files": 17,
                                 "fp05a_check_before": f"{runrel}/entry/core_source_pin_check_before.json"},
        },
        "data_roles": {
            "authority": {"exposure_ledger": ref(f"{runrel}/entry/exposure_ledger.json"),
                          "exposure_reconstruction_report": ref(f"{runrel}/entry/exposure_reconstruction_report.json")},
            "support_convention": "issue t on 6h grid; support t-12h..t+72h; exposure buffer 24h each side",
            "exposed_gradient_or_result_unbuffered": rep["runs_by_year_gradient_or_result_unbuffered"],
            "exposed_gradient_or_result_24h_buffered": rep["runs_by_year_gradient_or_result_24h_buffered"],
            "2020H2": {"label": "EXPOSED_NOT_CONFIRM",
                       "residual_free_runs_after_24h_buffer": h2["free_runs_in_H2_after_24h_buffer"],
                       "max_support_disjoint_candidates_after_buffer": h2["max_candidates_support_disjoint_buffered_(spacing>=90h)"],
                       "usable_for_confirm": False},
            "policy_dev_candidate_pool": {
                "status": "CANDIDATE_POOL_INTEGRITY_NOT_SCANNED",
                "store": "2019.zarr",
                "bounds_by_boundary_rule": {
                    "conservative_full_24h_buffer_incl_S0_aggregate_result_2020-01-01T00Z": pool["with_S0_aggregate_result_and_full_cross_year_buffer"],
                    "full_24h_buffer_certified_rows_only": pool["without_S0_aggregate_result_(only_certified_row_exposure)_full_buffer"],
                    "no_cross_year_buffer": pool["ignoring_cross_year_buffer_(2019_exposure_only)"],
                },
                "boundary_rule": undecided({"fp05a_recommendation": "conservative_full_24h_buffer_incl_S0_aggregate_result_2020-01-01T00Z",
                                            "FP05A_PLAN.md_stated": "[2019-07-06T12Z, 2019-12-28T18Z], 702 points (equals the no_cross_year_buffer variant)"}),
                "integrity_scan": {"done": False, "existing_scan_ends": pool["finiteness_scan_2019_last_step_utc"],
                                   "owner": "FP-05b"},
                "disclosure_sharedF0_sprint": pool["sharedF0_sprint_disclosure"],
                "label": "RETROSPECTIVE_HELD_PERIOD_DEV_NOT_HISTORICAL_DEPLOYMENT",
            },
            "confirm": None,
            "confirm_status": "UNASSIGNED_NO_ACCESS",
            "confirm_rule": "Any consumer that finds confirm_status UNASSIGNED_NO_ACCESS must refuse to read. No automatic fetch of 2021 data. Naming a confirm window is a separate, explicitly authorised decision outside FP-05a.",
            "other_years": {
                "2015": "backbone pretraining year - not usable (CARRIED_FROM_FP05A_PLAN, not re-verified in FP-05a)",
                "2018": "backbone pretraining year - not usable (CARRIED_FROM_FP05A_PLAN, not re-verified in FP-05a)",
                "2016": "store reported empty - not usable (CARRIED_FROM_FP05A_PLAN; verifying would require opening the .zarr store, out of FP-05a scope)",
                "2017": "store reported empty - not usable (CARRIED_FROM_FP05A_PLAN; same caveat)",
                "2021": "no 2021.zarr locally; a 2021_intermediate.zarr directory name exists under data/era5_1p40625/ (not opened; presumably the separate download task) - NOT usable/authorised",
                "2022": "not present locally",
            },
        },
        "split": {
            "policy_dev_pool": "see data_roles.policy_dev_candidate_pool",
            "confirm": {"status": "UNASSIGNED_NO_ACCESS", "window": None,
                        "rejected_reservation": "2020H2_EXPOSED_BY_X1 (EXPOSED_NOT_CONFIRM)", "new_window": None},
            "confirm_window": None,
            "exposure_buffer_hours": 24,
            "all_readers_must_verify_before_io": True,
        },
        "candidates": {
            "ids": ["reference", "expert_0", "expert_1", "expert_2", "expert_3"], "count": 5,
            "reference": "certified Fs (no-edit)", "background_F0_not_a_candidate": True,
            "F0_background_trajectory": undecided({"FP05_PLAN.md": "report-only F0 trajectory (+~20% GPU)"}),
            "alpha_a0": bm["a0"], "rho": bm["rho"], "hold_steps": bm["hold_steps"],
            "max_active": 1, "continuation": "Fs_from_edited_state",
            "source": "bank_manifest binding (certified); not a new decision",
        },
        "objective": {
            "primary_lead_hours": undecided({"v2": 24, "v1": 24, "FP05_PLAN.md": "24h (implicit)"}),
            "report_leads_hours": undecided({"v2": [6, 24, 72], "v1": {"diagnostic": 6, "harm": 72}}),
            "scoring": "reuse certified float64 full-objective evaluator (exact identities to be pinned at freeze)",
            "gain_normalizer": undecided({"v2": "mean Fs loss over same issues, recomputed inside each bootstrap draw"}),
        },
        "decision": {
            "delta_min_fs_relative": undecided(
                {"FP05_PLAN.md": 0.0034, "v1": 0.0034, "v2": 0.0034},
                "Scale anchors recomputed in FP-05a from the 2019H1 Fs-certification holdout: harm/F0 = %r, harm/Fs = %r. Engineering screen only, not a break-even proof; must not be re-derived after DEV outcomes are seen."
                % (fs["delta_min_recompute"]["holdout24_harm_F0_denominator"], fs["delta_min_recompute"]["holdout24_harm_Fs_denominator"])),
            "report_H_Fs_and_G_F0": "G_F0 = G_Fs - H_Fs must be reported alongside any gain vs Fs",
            "max_harm72_vs_Fs": None,
            "max_harm72_status": PNF,
            "dynamic_over_static_required": undecided({"v2": True}),
            "CI_crosses_threshold": "INCONCLUSIVE_STOP_AT_FIXED_CAP",
        },
        "sampling": {
            "N_final": None, "N_status": PNF,
            "proposals_for_reference": {"FP05_PLAN.md": 128, "v2_nominal_N": 128, "v1_max_dev_issues": 128},
            "debug_issues": undecided({"v1": 8, "FP05_PLAN.md": "8 from FP-04 trained data"}),
            "choose_N_once_before_dev_outcomes": True,
            "outcome_based_expansion_allowed": False,
            "replacement_rule": undecided({"v2": "nearest eligible unused slot in same 7-day stratum, tie earlier"}),
            "declaration_before_integrity_scan": True,
        },
        "validation": {
            "outer_mode": undecided({"v2": "bidirectional blocked cross-fit", "FP05_PLAN.md": "bidirectional blocked cross-fit"}),
            "outer_folds": undecided({"v2": 5, "v1": None}),
            "inner_folds": undecided({"v2": 3, "v1": None}),
            "base_block_days": undecided({"v2": 7, "v1": None}),
            "ridge_grid": undecided({"v2": "<= 4 configs, grid TO_BE_FROZEN"}),
            "poisoned_validation_label_test_scope": "must be confined to that fold's own OOF predictions (v1 FULL_REVIEW caution, accepted by FP05A)",
            "all_fitted_transforms_inner_fold_local": True,
        },
        "inference": {
            "confidence_level": undecided({"v2": 0.95, "v1": 0.95}),
            "bootstrap_unit": undecided({"v2": "paired whole 7-day UTC block"}),
            "bootstrap_draws": undecided({"v2": 10000, "v1": None}),
            "block_days": undecided({"v2": 7, "v1": None}),
            "sensitivity_block_days": undecided({"v2": [14], "v1": None},
                                                "v1 caution: only ~12-13 blocks of 14 days in a half-year; independence not guaranteed"),
            "seed": undecided({"v2": 20260924}),
            "MDE": None, "MDE_status": PNF,
        },
        "resource": {
            "max_gpu_jobs_total": None, "max_gpu_jobs_status": PNF,
            "proposals_for_reference": {"v1_total_job_ceiling": 4, "v2_max_total_submissions": 3,
                                        "FP05_PLAN.md": "2 planned, cap 4", "FP05A_PLAN.md": "decide at FP-05b freeze, leaning 3"},
            "infra_retries_total": undecided({"v1": 1, "v2": 1, "FP05_PLAN.md": 1}),
            "gpu_budget_authorized": False, "gpu_budget_seconds": None,
            "profile": "MISSING_NEEDS_MEASURED_PROFILE",
        },
        "new_source_manifest_sha256": None,
        "forbidden": ["new Fs fit", "bank retraining or altered alpha", "F0 in selector candidate set", "read confirm",
                      "read validation labels during fit/predict", "unregistered extra runs",
                      "modify any of the 17 FP-04-pinned source files", "claim dual-head novelty"],
        "undecided_fields_index": [],
    }

    # index every PROPOSED_NOT_FROZEN / null leaf for the reviewer
    idx = []

    def walk(o, path):
        if isinstance(o, dict):
            if o.get("status") == PNF and "value" in o:
                idx.append(path)
                return
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k)
        elif o is None:
            idx.append(path + " = null")

    walk(draft, "")
    draft["undecided_fields_index"] = sorted(idx)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(draft, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"out": str(out.relative_to(REPO)), "sha256": sha256(out),
                      "n_undecided": len(idx)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
