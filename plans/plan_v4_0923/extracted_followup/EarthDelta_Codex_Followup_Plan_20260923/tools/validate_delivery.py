#!/usr/bin/env python3
"""Validate this delivery, NOT the EarthDelta model or a GPU certificate."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
REQ=['START_HERE_FOR_CODEX.md','FOLLOWUP_PLAN.md','EXECUTION_DAG.md','STOP_CONDITIONS.md','CODEX_TASKS.json','ARTIFACT_CONTRACT.md','TEST_PLAN.md','REPRODUCIBILITY.md','CURRENT_GAP_ANALYSIS.md','evidence/PLAN_AUDIT_RESULT.json','evidence/EVIDENCE_MANIFEST.json','evidence/REQUIRED_INPUTS.md','prompts/CODEX_EXECUTION_PROMPT.md']
TOP={'overall_verdict','baseline','plan_items','special_verdicts','blocking_findings','required_next_actions','claims_that_must_not_be_made'}
OVER={'PLAN_EXECUTED','PARTIALLY_EXECUTED','BLOCKED','FAILED_PLAN_COMPLIANCE','INSUFFICIENT_EVIDENCE'}
IMPLEMENTED={'COMPLETE','PARTIAL','BLOCKED','NOT_STARTED','NOT_APPLICABLE'}
GOAL={'ACHIEVED','PARTIALLY_ACHIEVED','NOT_ESTABLISHED'}
EVIDENCE={'STATIC_CONFIRMED','EXECUTION_CONFIRMED','PLAUSIBLE_UNVERIFIED','MISSING'}
TASKFIELDS={'id','priority','depends_on','scope','commands','acceptance_criteria','evidence_outputs','stop_if','estimated_minutes'}

def validate(root:Path)->dict:
    errors=[];checks=[]
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
        if not condition:errors.append(name)
    check('required_paths',all((root/p).is_file() for p in REQ))
    try:
        audit=json.loads((root/'evidence/PLAN_AUDIT_RESULT.json').read_text())
        check('audit_top_fields',set(audit)==TOP)
        check('overall_enum',audit['overall_verdict'] in OVER)
        check('baseline_fields',set(audit['baseline'])=={'planned_source','implementation_commit','comparison_base'})
        valid=True
        fields={'id','planned_requirement','implementation_status','code_evidence','test_evidence','execution_evidence','goal_status','evidence_status','gap_or_risk'}
        for row in audit['plan_items']:
            valid &= set(row)==fields and row['implementation_status'] in IMPLEMENTED and row['goal_status'] in GOAL and row['evidence_status'] in EVIDENCE
            for c in row['code_evidence']:
                valid &= isinstance(c['line'],int) and c['line']>0 and isinstance(c['file'],str) and bool(c['file'])
            valid &= isinstance(row['test_evidence'],list) and isinstance(row['execution_evidence'],list)
        check('plan_item_template',valid and len(audit['plan_items'])>0)
        check('special_verdict_fields',set(audit['special_verdicts'])=={'s0_parity_verdict','fs_fit_divergence_verdict','stage4_readiness','novelty_claim_status','weather_utility_claim_status'})
        tasks=json.loads((root/'CODEX_TASKS.json').read_text())
        check('task_list_fields',isinstance(tasks,list) and all(TASKFIELDS<=set(t) for t in tasks))
        ids=[t['id'] for t in tasks];byid={t['id']:t for t in tasks}
        check('unique_tasks_and_dependencies',len(ids)==len(set(ids)) and all(d in byid for t in tasks for d in t['depends_on']))
        grey=set();black=set()
        def visit(i):
            if i in grey:raise ValueError('cycle')
            if i in black:return
            grey.add(i)
            for d in byid[i]['depends_on']:visit(d)
            grey.remove(i);black.add(i)
        try:
            for i in ids:visit(i)
            dag=True
        except (KeyError,ValueError):dag=False
        check('dag_acyclic',dag)
        check('future_status_no_fake_results',all(t['status']=='TO_BE_RUN' for t in tasks))
        check('initial_authorization',byid['FP-00']['authorization']=='OPEN_READ_ONLY' and byid['FP-01']['authorization']=='S0_ONLY_AFTER_FP00_AND_RESOURCE_APPROVAL' and all(byid[i]['authorization']=='LOCKED' for i in ids if i not in ('FP-00','FP-01')))
        check('fs_precedes_bank_and_science',byid['FP-04']['depends_on']==['FP-03'] and byid['FP-05']['depends_on']==['FP-04'] and byid['FP-03']['depends_on']==['FP-02'])
        cfg=json.loads((root/'evidence/FOLLOWUP_CONFIG.template.json').read_text())
        check('frozen_protocol',cfg['reference_kind']=='fitted_fs' and cfg['s0']['max_absolute_tolerance']==1e-5 and cfg['s0']['registered_steps']==[1,4,12] and cfg['bank']['K']==4 and cfg['bank']['rank']==4 and cfg['bank']['a0']==0.25 and cfg['data']['confirmation_access'] is False)
        sums=json.loads((root/'SHA256SUMS.json').read_text())
        actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json' and '__pycache__' not in p.parts}
        check('manifest_coverage',set(sums)==actual)
        check('sha256_integrity',all((root/p).is_file() and hashlib.sha256((root/p).read_bytes()).hexdigest()==s for p,s in sums.items()))
    except Exception as ex:
        errors.append(f'{type(ex).__name__}: {ex}')
    return {'status':'OBSERVED','scope':'DELIVERY_STRUCTURE_ONLY','passed':not errors,'checks':checks,'errors':errors,'repository_tests_run':False,'gpu_runs':0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args()
    result=validate(a.package.resolve());print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
