#!/usr/bin/env python3
"""Validate only this delivery: documents, task DAG, initial locks and SHA256 integrity."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

REQUIRED = [
 "START_HERE_FOR_CODEX.md", "EarthDelta_Codex_Execution_Plan.md", "FINAL_RESEARCH_DECISION.md",
 "CODEX_TASKS.md", "codex_tasks.json", "STOP_CONDITIONS.md", "IMPLEMENTATION_AUDIT_ACTIONS.md",
 "experiments/P0_SURVIVAL_EXPERIMENTS.md", "experiments/BASELINES.md",
 "research/ROUND2_REVIEW_SUMMARY.md", "EXECUTION_DAG.md", "configs/pilot_config.template.json",
 "evidence/REPOSITORY_FREEZE.json", "evidence/pytest_cpu_prior.log", "tools/preflight.py",
 "tools/validate_package.py",
]
FIELDS = ["task_id","title","priority","scientific_purpose","depends_on","files",
 "files_to_inspect","files_to_modify","files_to_create","implementation_steps","tests",
 "commands","artifacts","success_criteria","failure_criteria","failure_action",
 "decision_after_completion","suggested_commit_message","gate"]
TOKEN = "TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE"

def validate(root: Path) -> dict:
    checks = []
    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name":name, "passed":bool(ok), "detail":detail})
    check("required_files_present", all((root/p).is_file() for p in REQUIRED))
    if not checks[-1]["passed"]:
        return {"status":"FAIL", "checks":checks, "scope":"PACKAGE_ONLY"}
    state = json.loads((root/"codex_tasks.json").read_text(encoding="utf-8"))
    tasks = state.get("tasks", [])
    ids = [t.get("task_id") for t in tasks]
    check("five_unique_tasks", len(tasks)==5 and len(set(ids))==5)
    check("uniform_task_fields", all(all(f in t for f in FIELDS) for t in tasks))
    check("only_first_task_authorized", state.get("current_task")=="P0-01" and
          state.get("allowed_tasks")==["P0-01"] and state.get("do_not_proceed_beyond")=="P0-01" and
          not state.get("allow_auto_unlock") and
          [t["task_id"] for t in tasks if t.get("authorized_now")] == ["P0-01"])
    check("unexecuted_tasks", all(t.get("executed_in_this_delivery") is False for t in tasks))
    done = set()
    acyclic = True
    for t in tasks:
        deps = t.get("depends_on", [])
        if any(d not in done for d in deps):
            acyclic = False
        done.add(t["task_id"])
    check("dag_topologically_ordered", acyclic)
    check("all_gates_have_scope_and_ci", all(all(t["gate"].get(k) for k in
           ["why","pass_implies","fail_implies","ci_crossing","failure_action"]) for t in tasks))
    check("no_automatic_success_unlock", all(t["gate"].get("requires_explicit_next_authorization")
                                              is True for t in tasks))
    main = (root/"EarthDelta_Codex_Execution_Plan.md").read_text(encoding="utf-8")
    check("main_plan_has_same_tasks", re.findall(r"^## TASK (P0-\d+) —", main, re.M)==ids)
    check("all_commands_in_main_plan", all(c in main for t in tasks for c in t["commands"]))
    md_files = list(root.rglob("*.md"))
    check("markdown_fences_balanced", all(sum(line.lstrip().startswith("```") for line in
                p.read_text(encoding="utf-8").splitlines()) % 2 == 0 for p in md_files))
    config = json.loads((root/"configs/pilot_config.template.json").read_text(encoding="utf-8"))
    check("template_training_gpu_confirm_disabled", all(config["execution"][k] is False for k in
                ["allow_training","allow_gpu_run","allow_confirm_read","allow_network_download","allow_dependency_install"]))
    check("thresholds_not_invented", all(config["statistics"][k]==TOKEN for k in
                ["delta_min_vs_reference","delta_min_dynamic","delta_min_legal_policy",
                 "delta_min_factorization","noninferiority_margin_long_lead"]))
    check("resource_caps_unfilled", all(v is None for k,v in config["resources"].items() if k!="status"))
    actions = state.get("audit_action_map", [])
    check("all_B01_B15_mapped", [a.get("id") for a in actions]==[f"B{i:02d}" for i in range(1,16)])
    classes = {"BLOCKS_NEXT_PILOT","FIX_BEFORE_TRAINING","FIX_LATER","NOT_A_BLOCKER","AUDIT_FINDING_REJECTED"}
    check("valid_audit_priorities", all(a.get("classification") in classes for a in actions))
    freeze = json.loads((root/"evidence/REPOSITORY_FREEZE.json").read_text(encoding="utf-8"))
    check("head_vs_audit_distinguished", freeze["observed_branch_head"] != freeze["audited_source_head"] and
          freeze["comparison"]["common_entries_equal"] is True and
          state["repository"]["observed_branch_head"]==freeze["observed_branch_head"])
    check("historical_cpu_result_not_claimed_as_rerun", freeze["prior_cpu_log"]["passed"]==283 and
          freeze["prior_cpu_log"]["skipped"]==7 and freeze["prior_cpu_log"]["failed"]==0 and
          freeze["prior_cpu_log"]["rerun_in_package_creation"] is False and
          state["runtime"]["repository_tests_this_delivery"]=="NOT_RUN")
    manifest_path = root/"MANIFEST_SHA256.json"
    integrity = manifest_path.is_file()
    failures = []
    if integrity:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("files", {})
        actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                        and p.name!="MANIFEST_SHA256.json" and "__pycache__" not in p.parts}
        if set(entries)!=actual_files:
            failures.append("manifest inventory mismatch")
        for relative, expected in entries.items():
            p = root/relative
            if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
                failures.append(relative)
        integrity = not failures
    check("sha256_inventory_and_content", integrity, "; ".join(failures))
    return {"status":"PASS" if all(c["passed"] for c in checks) else "FAIL",
            "scope":"PACKAGE_ONLY_NOT_REPOSITORY_OR_WEATHER_VALIDATION",
            "check_count":len(checks),"checks":checks,
            "repository_tests_executed":False,"gpu_s0_executed":False,
            "training_executed":False,"remote_repository_modified":False}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = validate(args.root.expanduser().resolve())
    except (OSError, KeyError, ValueError, TypeError) as exc:
        result = {"status":"FAIL","scope":"PACKAGE_ONLY","error":type(exc).__name__+": "+str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"]=="PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
