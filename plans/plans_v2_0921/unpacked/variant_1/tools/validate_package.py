#!/usr/bin/env python3
"""Validate this delivery only. Does not execute any EarthDelta task or model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

CORE = ["START_HERE_FOR_CODEX.md", "EarthDelta_Codex_Execution_Plan.md",
        "FINAL_RESEARCH_DECISION.md", "CODEX_TASKS.md", "codex_tasks.json",
        "STOP_CONDITIONS.md", "IMPLEMENTATION_AUDIT_ACTIONS.md",
        "experiments/P0_SURVIVAL_EXPERIMENTS.md", "experiments/BASELINES.md",
        "research/ROUND2_REVIEW_SUMMARY.md", "configs/iteration1.template.json",
        "provenance.json", "tools/preflight.py", "tools/validate_package.py"]
SECTIONS = ["Scientific purpose", "Depends on", "Files to inspect", "Files to modify/create",
            "Implementation steps", "Tests", "Run command", "Expected artifacts",
            "Success criteria", "Failure criteria", "Decision after completion",
            "Suggested commit message"]
SENTINEL = "TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE"
CATS = ["BLOCKS_NEXT_PILOT", "FIX_BEFORE_TRAINING", "FIX_LATER",
        "NOT_A_BLOCKER", "AUDIT_FINDING_REJECTED"]

def validate(root: Path, manifest: bool = False) -> dict:
    checks = []
    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(ok), "detail": detail})
    for p in CORE:
        check("file:"+p, (root/p).is_file() and (root/p).stat().st_size > 0)
    if not all(c["passed"] for c in checks):
        return {"status": "FAIL", "checks": checks, "scope": "PACKAGE_ONLY"}
    for p in root.rglob("*.json"):
        try:
            json.loads(p.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            check("json:"+str(p.relative_to(root)), True)
        except (ValueError, OSError) as e:
            check("json:"+str(p.relative_to(root)), False, str(e))
    data = json.loads((root/"codex_tasks.json").read_text(encoding="utf-8"))
    cfg = json.loads((root/"configs/iteration1.template.json").read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    ids = [t.get("task_id") for t in tasks]
    check("task_count_3_to_6", 3 <= len(tasks) <= 6)
    check("unique_task_ids", len(ids) == len(set(ids)))
    check("initial_stop", data.get("current_task") == "R2I-01" and data.get("do_not_proceed_beyond") == "R2I-01")
    check("one_task_unlocked", data.get("unlocked_task_ids") == ["R2I-01"])
    check("no_auto_advance", data.get("auto_advance") is False and cfg["authorization"]["auto_advance"] is False)
    check("gpu_training_confirm_locked", all(cfg["authorization"][k] is False for k in
          ["allow_gpu_s0", "allow_model_training", "allow_weather_cache", "allow_confirmation",
           "allow_download", "allow_dependency_install", "allow_repository_push"]))
    check("audit_vs_delivery_distinct", data["repository"]["observed_branch_head"] != data["repository"]["audited_source_head"])
    check("historical_cpu_not_new_test", data["existing_cpu_evidence"]["evidence_type"] == "PREVIOUS_REVIEW_READ_LOG_NOT_RERUN_IN_PACKAGE_CREATION")
    check("no_scientific_runs_claimed", all(v == "NOT_RUN" for k,v in data["package_creation_execution"].items()
         if k in ["repository_tests", "gpu_s0", "weather_training", "weather_data_download"]))
    plan = (root/"EarthDelta_Codex_Execution_Plan.md").read_text(encoding="utf-8")
    stops = (root/"STOP_CONDITIONS.md").read_text(encoding="utf-8")
    actions = (root/"IMPLEMENTATION_AUDIT_ACTIONS.md").read_text(encoding="utf-8")
    boxes = (root/"CODEX_TASKS.md").read_text(encoding="utf-8")
    seen = set()
    required = ["task_id", "title", "priority", "depends_on", "files", "commands", "artifacts",
                "success_criteria", "failure_action", "implementation_steps", "tests", "command_gates"]
    for t in tasks:
        tid=t["task_id"]
        check(tid+":schema", all(k in t for k in required))
        check(tid+":nonempty", all(bool(t.get(k)) for k in ["title","files","commands","artifacts","tests","success_criteria"]))
        check(tid+":DAG_order", all(d in seen for d in t["depends_on"]))
        seen.add(tid)
        check(tid+":manual_next_grant", t.get("requires_explicit_next_task_authorization") is True)
        check(tid+":command_gates_aligned", len(t["commands"]) == len(t["command_gates"]) and
              all(c == g["command"] for c,g in zip(t["commands"], t["command_gates"])))
        heading = "## TASK ID: "+tid+" — "
        parts = plan.split(heading)
        check(tid+":one_main_task_block", len(parts)==2)
        if len(parts)==2:
            block=parts[1].split("## TASK ID:",1)[0]
            check(tid+":uniform_sections", all("### "+s in block for s in SECTIONS))
            check(tid+":commands_in_main", all(c in block for c in t["commands"]))
        check(tid+":checkbox", "- [ ] **"+tid+"**" in boxes)
        check(tid+":locked_state", t["status"] == ("AUTHORIZED_FOR_SCOPE_ONLY" if tid=="R2I-01" else "LOCKED"))
        for c in t["commands"]:
            check(tid+":no_destructive_command:"+str(t["commands"].index(c)), not any(x in c for x in
                  ["git reset", "git push", "rm -rf", "pip install", "curl ", "wget "]))
    check("all_B_findings", all(re.search(r"\| B"+f"{i:02d}"+r" \|", actions) for i in range(1,16)))
    check("all_action_categories_explained", all(c in actions for c in CATS))
    check("placeholder_not_number", SENTINEL in stops and cfg["statistics"]["delta_min_legal_vs_static"] == SENTINEL)
    check("mde_distinct_from_value", "value_justification_independent_of_mde" in cfg["statistics"] and cfg["statistics"]["mde"] is None)
    check("oracle_pass_not_deployment", "HEADROOM_PASS_ONLY" in plan and "INCONCLUSIVE" in plan)
    for p in root.rglob("*.py"):
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
            check("syntax:"+str(p.relative_to(root)), True)
        except (SyntaxError,OSError) as e:
            check("syntax:"+str(p.relative_to(root)), False, str(e))
    # Validate only package-local Markdown links, not source URLs or proposed repository files.
    for p in root.rglob("*.md"):
        text=p.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)",text):
            if "://" in target or target.startswith("#"):
                continue
            target=target.split("#",1)[0]
            check("local_link:"+str(p.relative_to(root))+":"+target, (p.parent/target).exists())
    if manifest:
        manifest_path=root/"MANIFEST.sha256"
        check("manifest_exists",manifest_path.is_file())
        listed=set()
        if manifest_path.is_file():
            for line in manifest_path.read_text().splitlines():
                digest,name=line.split("  ",1)
                p=(root/name).resolve()
                safe=p.is_relative_to(root.resolve())
                check("manifest:"+name, safe and p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest)
                listed.add(name)
            actual={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name!="MANIFEST.sha256"}
            check("manifest_file_set",listed==actual)
    return {"status":"PASS" if all(c["passed"] for c in checks) else "FAIL",
            "scope":"PACKAGE_STRUCTURE_JSON_DAG_SYNTAX_LINKS"+("_MANIFEST" if manifest else ""),
            "check_count":len(checks),"failed_count":sum(not c["passed"] for c in checks),
            "repository_tests":"NOT_RUN","weather_experiments":"NOT_RUN","checks":checks}

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-manifest",action="store_true")
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    result=validate(args.root.resolve(),args.verify_manifest)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open("x",encoding="utf-8") as f:
            json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False); f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="checks"},ensure_ascii=False))
    if result["status"]!="PASS":
        for c in result["checks"]:
            if not c["passed"]: print(c,file=sys.stderr)
    return 0 if result["status"]=="PASS" else 1

if __name__=="__main__":
    raise SystemExit(main())
