#!/usr/bin/env python3
"""Validate package files and task DAG; does not validate scientific results."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

REQUIRED = [
 'START_HERE_FOR_CODEX.md','EarthDelta_Codex_Execution_Plan.md',
 'CODEX_TASKS.md','codex_tasks.json','STOP_CONDITIONS.md',
 'research/CURRENT_METHOD_AUDIT.md','research/NOVELTY_AUDIT.md',
 'research/RELATED_WORK.md','research/FINAL_RESEARCH_DECISION.md',
 'research/EARTHDELTA_V2.md','research/MATHEMATICAL_AUDIT.md',
 'experiments/P0_SURVIVAL_EXPERIMENTS.md','experiments/P1_CORE_EXPERIMENTS.md',
 'experiments/P2_NOVELTY_EXPERIMENTS.md','experiments/BASELINES.md',
 'implementation/FILE_BY_FILE_PLAN.md','implementation/TEST_PLAN.md',
 'implementation/ARTIFACT_CONTRACT.md','implementation/REPRODUCIBILITY.md',
 'implementation/EVALUATION_CONTRACT.md','implementation/LITERATURE_TO_CODE.md',
 'configs/survival_template.json','tools/repo_preflight.py','tools/check_math.py']
FIELDS = ['id','title','priority','depends_on','files_to_inspect','files_to_modify',
 'files_to_create','scientific_purpose','implementation_requirements','must_not_change',
 'tests_to_add','commands','expected_artifacts','success_criteria','failure_criteria',
 'decision_after_completion','commit_suggestion']

def strict_json(p):
    def bad(value): raise ValueError(f'Non-standard JSON numeric constant: {value}')
    return json.loads(p.read_text(encoding='utf-8'),parse_constant=bad)

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.out.exists(): parser.error('Output exists; choose a new path.')
    errors=[]
    for name in REQUIRED:
        p=root/name
        if not p.is_file() or p.stat().st_size==0: errors.append('Missing/empty: '+name)
    for p in root.rglob('*.json'):
        try: strict_json(p)
        except Exception as exc: errors.append(f'Invalid JSON {p.relative_to(root)}: {exc}')
    graph={}; task_count=0
    try:
        data=strict_json(root/'codex_tasks.json');tasks=data['tasks'];task_count=len(tasks)
        ids=[t['id'] for t in tasks]
        if len(ids)!=len(set(ids)):errors.append('Duplicate task IDs')
        if data['initial_allowed_tasks']!=['P0-01']:errors.append('Initial unlock must be only P0-01')
        if data['automatic_phase_advancement'] is not False:errors.append('Automatic phase advancement forbidden')
        for t in tasks:
            for key in FIELDS:
                if key not in t:errors.append(f'{t.get("id")}: missing {key}')
            graph[t['id']]=t['depends_on']
            for dep in t['depends_on']:
                if dep not in ids:errors.append(f'{t["id"]}: missing dependency {dep}')
        seen=set();active=set()
        def visit(node):
            if node in active:raise ValueError('Cyclic dependency: '+node)
            if node in seen:return
            active.add(node)
            for d in graph.get(node,[]):visit(d)
            active.remove(node);seen.add(node)
        for node in graph:visit(node)
        plan=(root/'EarthDelta_Codex_Execution_Plan.md').read_text()
        for t in tasks:
            if plan.count('### TASK '+t['id']+' —')!=1:errors.append('Task not exactly once in plan: '+t['id'])
        cfg=strict_json(root/'configs/survival_template.json')
        if cfg['status']!='TEMPLATE_NOT_READY':errors.append('Cannot pretend uncalibrated config is ready')
        if cfg['resource_caps']['authorized'] is not False:errors.append('No GPU authorization can be assumed')
    except Exception as exc: errors.append(type(exc).__name__+': '+str(exc))
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS' if not errors else 'FAIL',
            'task_count':task_count,'required_files':len(REQUIRED),'errors':errors,
            'scope':'package integrity/schema/DAG only; not real-repository tests or scientific validation'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if errors else 0
if __name__=='__main__':raise SystemExit(main())
