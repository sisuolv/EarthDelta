#!/usr/bin/env python3
"""FP-05a closing step: evaluate FP05A_PLAN.md acceptance criteria from this run's own
entry/ files, record every disagreement with FP05A_PLAN.md, write entry/decision.json and
task_result_FP-05a.json (with a sha256 manifest of every file in the run dir).
Stdlib only; refuses to overwrite.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
E = RUN / "entry"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(name):
    return json.loads((E / name).read_text())


def git(*a):
    p = subprocess.run(["git", "--git-dir", str(REPO / ".git"), "--work-tree", str(REPO), *a],
                       capture_output=True, text=True)
    return p.stdout.strip()


def main() -> int:
    dec_p, tr_p = E / "decision.json", RUN / "task_result_FP-05a.json"
    for p in (dec_p, tr_p):
        if p.exists():
            print("refusing to overwrite", p)
            return 2
    pin_b, pin_a = jl("core_source_pin_check_before.json"), jl("core_source_pin_check_after.json")
    inv = jl("evidence_inventory.json")
    rc = jl("job_receipts_index.json")
    cert = jl("certificate_recompute.json")
    led = jl("exposure_ledger.json")
    rep = jl("exposure_reconstruction_report.json")
    fp02 = jl("fp02_remaining.json")
    pchk = jl("protocol_draft_structural_check.json")
    v2val = jl("package_v2_validation.json")
    v1log = (RUN / "logs/package_v1_validate.stdout.log").read_text()
    draft = json.loads((RUN / "protocol/fp05_protocol.DRAFT.json").read_text())

    tracked = [r for r in inv["inputs"] if r["kind"] == "tracked_json"]
    large = [r for r in inv["inputs"] if r["kind"] == "large_asset"]
    receipts_in_inv = [r for r in inv["inputs"] if r["kind"] == "raw_job_receipt"]
    x1 = rep["x1"]
    sp = led["sharedF0_sprint"]

    crit = {
        "C1_17_pins_match_before_and_after": pin_b["all_match"] and pin_a["all_match"]
        and pin_b["n_pinned"] == pin_a["n_pinned"] == 17,
        "C2_8_tracked_input_hashes_match": len(tracked) == 8 and all(r["state"] == "HASH_VERIFIED_NOT_SEMANTIC_GATE" for r in tracked),
        "C2b_3_large_assets_rehashed_match": len(large) == 3 and all(r["state"] == "HASH_VERIFIED_NOT_SEMANTIC_GATE" for r in large),
        "C3_every_job_receipt_matches": rc["overall"] == "ALL_RECEIPTS_VERIFIED" and rc["n_missing"] == 0 and rc["n_mismatch"] == 0,
        "C4_certificates_recomputed_within_1e-12": cert["fs"]["all_replicas_within_1e-12"] and cert["bank"]["all_experts_within_1e-12"],
        "C4b_assembly_equivalence_exactly_zero": cert["bank"]["assembly_all_exact_zero"],
        "C4c_verify_bank_bundle_deep_passed_no_zarr": bool(cert["bank_bundle_registry_verify"].get("passed"))
        and cert["bank_bundle_registry_verify"].get("zarr_imported") is False,
        "C5_ledger_has_all_56_X1_H2_issues": x1["x1_n64_extra_in_2020H2_count"] == 56 and x1["support_union_contiguous"],
        "C5b_ledger_has_sharedF0_616_reads": sp["n_rows_listed"] == 616 and len(led["sharedF0_sprint_rows"]) == 616,
        "C6_2020H2_EXPOSED_NOT_CONFIRM": rep["2020H2"]["verdict"] == "EXPOSED_NOT_CONFIRM"
        and draft["data_roles"]["2020H2"]["label"] == "EXPOSED_NOT_CONFIRM",
        "C7_confirm_null": draft["data_roles"]["confirm"] is None and draft["split"]["confirm_window"] is None,
        "C8_protocol_draft_v2_structural_check": pchk["structural_pass"] and pchk["literal_only_expected_failures"],
        "C9_package_validators_pass": v2val["passed"] is True and '"passed": false' not in v1log and "PASS_PACKAGE_ONLY" in v1log,
        "C10_fp02_gaps_recorded": fp02["all_required_still_missing"] is True,
    }

    h2 = rep["2020H2"]
    pool = rep["2019H2_policy_dev_pool"]
    unbuf20 = rep["runs_by_year_gradient_or_result_unbuffered"]["2020"]
    delta = cert["fs"]["delta_min_recompute"]
    disagreements = [
        {"id": "D-1", "severity": "REAL_DISAGREEMENT (conclusion unchanged)",
         "FP05A_PLAN_states": "2020H2 residual gap holds at most 2 non-overlapping candidates",
         "fp05a_recomputed": f"after the 24h buffer the only free H2 timesteps are {h2['free_runs_in_H2_after_24h_buffer']}; issues with full t-12h..t+72h support there: {h2['candidate_issue_times_full_support_in_buffered_gap']}; max support-disjoint candidates = {h2['max_candidates_support_disjoint_buffered_(spacing>=90h)']} (even allowing touching supports: {h2['max_candidates_support_touching_allowed_buffered_(spacing>=84h)']}). 2 is reached only with NO buffer ({h2['max_candidates_support_disjoint_NO_buffer']}).",
         "impact": "none on the verdict: 2020H2 cannot host a confirm set either way"},
        {"id": "D-2", "severity": "CONVENTION_ONLY",
         "FP05A_PLAN_states": "gap 2020-07-05T18Z..2020-07-12T06Z (156 h)",
         "fp05a_recomputed": "those are the last/first BUFFERED timesteps (exclusive bounds); the free timesteps themselves are 2020-07-06T00Z..2020-07-12T00Z inclusive (144 h span, 25 grid points)",
         "impact": "none"},
        {"id": "D-3", "severity": "REAL_DISAGREEMENT (small)",
         "FP05A_PLAN_states": "policy_dev candidate pool [2019-07-06T12Z, 2019-12-28T18Z], 702 grid points",
         "fp05a_recomputed": f"702/[..12-28T18Z] only if no 24h buffer is applied across the 2019/2020 year boundary. Honouring the 24h buffer against the 2020-01-01T12Z certified-row exposure: {pool['without_S0_aggregate_result_(only_certified_row_exposure)_full_buffer']}; additionally honouring the S0 gate's aggregate-result exposure from 2020-01-01T00Z: {pool['with_S0_aggregate_result_and_full_cross_year_buffer']}",
         "impact": "pool end moves 12-18 h earlier (2-4 grid points); boundary rule left PROPOSED_NOT_FROZEN in the draft, conservative variant recommended"},
        {"id": "D-4", "severity": "ADDITIONAL_EXPOSURE_NOT_IN_FP05A",
         "FP05A_PLAN_states": "exposed 2020 starts 2020-01-01T12Z",
         "fp05a_recomputed": f"S0 gate compute_rmse_sanity scores model vs truth on jan2020_full.npy indices 0..8 (2020-01-01T00Z..01-03T00Z, time axis assumed from file name/shape, not byte-verified) and loads the whole file (2020-01-01T00Z..01-31T18Z). Unbuffered 2020 gradient/result runs: {unbuf20}",
         "impact": "2020H1 was already exposed; only affects the 2019H2 pool end (D-3)"},
        {"id": "D-5", "severity": "PRECISION_NOTE",
         "FP05A_PLAN_states": "exposed = 2020-01-01T12Z~07-04T18Z and 2019-01-01T12Z~07-04T18Z",
         "fp05a_recomputed": "these are envelopes, not unions: 2019H1 is 16 disjoint 84 h runs (A1 holdout); 2020H1 has a hole at the dropped FP-04 slot 47 (free 2020-06-25T18Z..06-29T06Z unbuffered, 06-26T18Z..06-28T06Z buffered, integrity-read by FP-04 attempt1/replacements). The v1 holdout issue 2020-07-01T18Z puts 2020-07-01T06Z..07-04T18Z (inside H2) under result exposure.",
         "impact": "none on roles (all envelope-internal gaps are too short or already integrity-read)"},
        {"id": "D-6", "severity": "ADDITIONAL_DISCLOSURE",
         "FP05A_PLAN_states": "shared-F0 sprint read 147 time points in 2019H2",
         "fp05a_recomputed": f"147 admitted rows confirmed ({pool['sharedF0_sprint_disclosure']['roles_of_2019H2_admitted_rows']}); in addition {pool['sharedF0_sprint_disclosure']['excluded_rows_with_issue_in_2019H2_(read_attempted,_failed_nonfinite)']} EXCLUDED sprint rows with 2019H2 issue times were read and rejected as non-finite (sprint total: 616 admitted + {sp['n_excluded_listed']} excluded). Integrity-only; no gradient/result.",
         "impact": "disclosure only (FP05A decision 3: disclosed, not excluded)"},
        {"id": "D-7", "severity": "MINOR_FACT",
         "FP05A_PLAN_states": "2021/2022 not present locally",
         "fp05a_recomputed": "no 2021.zarr, no 2022 store; but a directory named 2021_intermediate.zarr exists under data/era5_1p40625/ (not opened)",
         "impact": "none (not usable/authorised; download is a separate task)"},
        {"id": "AGREE", "severity": "INDEPENDENTLY_CONFIRMED",
         "items": [
             "17/17 pins unchanged (before and after)", "8/8 tracked input hashes (+3 large assets) match",
             f"X1_N64 adds exactly 56 issues 2020-07-13T18Z..2020-12-28T18Z (store indices 779..1451, spacing 72-78 h); contiguous support run 2020-07-13T06Z..2020-12-31T18Z",
             f"delta scale anchors: harm/F0 = {delta['holdout24_harm_F0_denominator']:.10f}, harm/Fs = {delta['holdout24_harm_Fs_denominator']:.10f}",
             "raw platform receipts exist locally for every r4-session job (20 jobs; 14 named in the dispatch + 6 failed/aux attempts)"]},
    ]

    passed = all(crit.values())
    decision = {
        "schema": "ed-fp05a-decision/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task": "FP-05a",
        "verdict": "FP05A_COMPLETE" if passed else "BLOCKED",
        "acceptance_criteria": crit,
        "acceptance_notes": {
            "C8": "the unmodified v2 validator, run on a copy with the draft substituted, fails only manifest_hashes_match (by construction) and protocol_not_falsely_frozen (it hard-codes status 'TO_BE_RUN'; the dispatch mandates 'DRAFT_NOT_FROZEN'). All other protocol predicates pass literally.",
            "C3": "receipts verified = raw job_result.json present, run_id/hostname/timestamps consistent, rc/status/elapsed equal to every tracked claim within its printed rounding, and (where a submission record exists) job_id/run_dir/invocation/source_manifest/argv hashes joined.",
        },
        "withdrawn_claim": {"claim": "FP05_PLAN.md D1: 2020H2 = untouched confirm reservation", "status": "WITHDRAWN",
                            "basis": "entry/exposure_reconstruction_report.json#2020H2 (independently recomputed from raw admission certificates)"},
        "data_role_summary": {
            "2020H2": "EXPOSED_NOT_CONFIRM",
            "policy_dev": "CANDIDATE_POOL_INTEGRITY_NOT_SCANNED (2019H2; boundary rule PROPOSED_NOT_FROZEN)",
            "confirm": None, "confirm_status": "UNASSIGNED_NO_ACCESS"},
        "disagreements_with_FP05A_PLAN": disagreements,
        "fp02": "PARTIAL; scripts/plan_followup_runner.py and earthdelta/split_freeze.py still absent",
        "next": {"FP-05b": "ELIGIBLE_NOT_AUTHORIZED (requires explicit user go-ahead; first steps: 2019H2 integrity scan + selection declaration, split_freeze/runner, protocol freeze)",
                 "gpu": "NOT_AUTHORIZED", "confirm": "NO_ACCESS"},
        "forbidden_actions_observed": {"gpu_job_submitted": False, "zarr_opened": False, "model_forward": False,
                                       "pinned_file_modified": not (pin_b["all_match"] and pin_a["all_match"]),
                                       "policy_dev_admission_built": False, "2019H2_integrity_scan": False,
                                       "confirm_window_named": False, "git_commit_or_push": False},
    }
    with dec_p.open("x") as f:
        json.dump(decision, f, indent=1, ensure_ascii=False)
        f.write("\n")

    manifest = []
    for p in sorted(RUN.rglob("*")):
        if p.is_file() and p != tr_p:
            manifest.append({"path": str(p.relative_to(REPO)), "sha256": sha256(p), "bytes": p.stat().st_size})
    tr = {
        "schema": "ed-fp05a-task-result/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task": "FP-05a", "run_dir": str(RUN.relative_to(REPO)),
        "verdict": decision["verdict"],
        "repo_head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain=v1"),
        "scope": "read-only, CPU-only; no GPU, no .zarr read, no model forward, no pinned-file change, no commit",
        "deliverables": {
            "1_package_validation": ["entry/package_v2_validation.json", "logs/package_v2_validate.stdout.log",
                                     "logs/package_v1_validate.stdout.log", "entry/evidence_inventory.json",
                                     "entry/package_v2_arithmetic.json"],
            "2_job_receipts": ["entry/job_receipts_index.json"],
            "3_certificate_recompute": ["entry/certificate_recompute.json", "logs/verify_bank_bundle.child.log"],
            "4_exposure": ["entry/exposure_ledger.json", "entry/exposure_reconstruction_report.json"],
            "5_fp02": ["entry/fp02_remaining.json"],
            "6_protocol_draft": ["protocol/fp05_protocol.DRAFT.json", "entry/protocol_draft_structural_check.json"],
            "7_close": ["entry/decision.json", "task_result_FP-05a.json"],
            "pins": ["entry/core_source_pin_check_before.json", "entry/core_source_pin_check_after.json"],
        },
        "headline_numbers": {
            "jobs_receipt_verified": f"{rc['n_receipt_verified']}/{rc['n_jobs']}",
            "fs_q4_q5_q6": [cert["fs"]["recomputed"][k]["mean_ratio"] for k in ("q4_train_24h", "q5_train_72h", "q6_holdout_24h")],
            "bank_own_group24": cert["bank"]["own_group24_mean_ratios_recomputed"],
            "x1_h2_issues": x1["x1_n64_extra_in_2020H2_count"],
            "h2_free_after_buffer": h2["free_runs_in_H2_after_24h_buffer"],
            "h2_max_disjoint_candidates_after_buffer": h2["max_candidates_support_disjoint_buffered_(spacing>=90h)"],
            "policy_dev_pool_variants": {k: v for k, v in pool.items() if isinstance(v, dict) and "count" in v},
            "sharedF0_2019H2": pool["sharedF0_sprint_disclosure"],
        },
        "hash_manifest": manifest,
    }
    with tr_p.open("x") as f:
        json.dump(tr, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"verdict": decision["verdict"], "criteria": crit, "n_manifest": len(manifest)}, indent=1))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
