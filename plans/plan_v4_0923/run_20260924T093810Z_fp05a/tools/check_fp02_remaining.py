#!/usr/bin/env python3
"""FP-05a: FP-02 remaining-gap check. Read-only filesystem/git inspection, stdlib only.

Output (refuses to overwrite): entry/fp02_remaining.json
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
P4 = "plans/plan_v4_0923"

REQUIRED_MISSING = ["scripts/plan_followup_runner.py", "earthdelta/split_freeze.py"]
# other NEW entry points the reviewed FP-05b plan (v2 FOLLOWUP_PLAN.md FP-05b inputs) names
FP05B_NEW = ["earthdelta/candidate_cache.py", "earthdelta/policy_oof.py", "scripts/r4_candidate_cache.py",
             "scripts/r4_cache_decide.py", "scripts/r4_policy_oof.py", "tests/test_split_freeze.py",
             "tests/test_plan_cache_evaluation.py", "tests/test_fp05_consumption_chain.py",
             "tests/test_policy_oof_isolation.py"]
DONE_EVIDENCE = {
    "stage failure blocks successors": "tests/test_plan_gate_chain.py",
    "loaded-admission binding (Fs subset)": "tests/test_loaded_admission_binding.py",
    "profile immutability (added in FP-04)": "tests/test_profile_immutability.py",
    "certified-Fs authorization / single Fs identity": f"{P4}/run_20260924T033627Z_fp04_bank/task_result_fp04.json",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def find(name: str) -> list[str]:
    p = subprocess.run(["find", str(REPO), "-path", str(REPO / ".git"), "-prune", "-o",
                        "-path", str(REPO / "artifacts"), "-prune", "-o", "-name", name, "-print"],
                       capture_output=True, text=True, timeout=600)
    return sorted(x for x in p.stdout.splitlines() if x)


def tracked(rel: str) -> bool:
    p = subprocess.run(["git", "--git-dir", str(REPO / ".git"), "--work-tree", str(REPO), "ls-files",
                        "--error-unmatch", rel], capture_output=True, text=True)
    return p.returncode == 0


def main() -> int:
    out = RUN / "entry/fp02_remaining.json"
    if out.exists():
        print("refusing to overwrite", out)
        return 2
    req = []
    for rel in REQUIRED_MISSING:
        req.append({"path": rel, "exists": (REPO / rel).exists(), "git_tracked": tracked(rel),
                    "find_by_basename_outside_artifacts": find(Path(rel).name)})
    other = [{"path": rel, "exists": (REPO / rel).exists()} for rel in FP05B_NEW]
    done = [{"subpart": k, "evidence": v, "exists": (REPO / v).is_file(),
             "sha256": sha256(REPO / v) if (REPO / v).is_file() else None} for k, v in DONE_EVIDENCE.items()]
    # role-tag observation: every admission used so far is tagged bank_fit, incl. holdout panels
    roles = {}
    for rel in [f"{P4}/run_20260923T192759Z_fp03_fs/admission/bank_fit_admission.json",
                f"{P4}/run_20260923T192759Z_fp03_fs/admission/holdout_panel_admission.json",
                f"{P4}/run_20260924T013959Z_fp03_v2/holdout/v2_holdout_admission_A1.json",
                f"{P4}/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/admission/x1_n64_admission.json",
                f"{P4}/run_20260924T033627Z_fp04_bank/admission/bank_fit_admission.json"]:
        d = json.loads((REPO / rel).read_text())
        roles[rel] = {"data_role": d["data_role"],
                      "split_ids": sorted({r.get("split_id") for r in d["admission"]["admitted"]})}
    res = {
        "schema": "ed-fp05a-fp02-remaining/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder": str(Path(__file__).relative_to(REPO)), "builder_sha256": sha256(Path(__file__)),
        "fp02_status": "PARTIAL (not upgraded to DONE by FP-03/FP-04 progress)",
        "required_still_missing": req,
        "all_required_still_missing": all(not r["exists"] and not r["git_tracked"] and not r["find_by_basename_outside_artifacts"] for r in req),
        "fp05b_named_new_entry_points": other,
        "done_subparts_with_evidence": done,
        "missing_subparts": [
            "thin common runner (scripts/plan_followup_runner.py)",
            "project-wide frozen split / exposure enforcement (earthdelta/split_freeze.py) consuming entry/exposure_ledger.json",
            "complete FP-05 fit/predict/cache/evaluate consumption checks (tests named above)",
        ],
        "observation_role_tags": {
            "finding": "Every admission consumed so far carries data_role='bank_fit', including the v1 holdout panel and the FP-03 v2 2019 holdout (A1); split_id is 'test' on 2020 rows. Role tags therefore do not currently distinguish training from evaluation exposure; the exposure ledger (not the role tag) is the authority until split_freeze exists.",
            "admissions": roles,
        },
        "fp05a_action": "RECORD_ONLY (creating these entry points is FP-05b scope)",
    }
    with out.open("x") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps({"all_required_still_missing": res["all_required_still_missing"],
                      "required": [(r["path"], r["exists"]) for r in req],
                      "other": [(o["path"], o["exists"]) for o in other]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
