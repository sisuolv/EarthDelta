#!/usr/bin/env python3
"""Validate this delivery package only; never execute EarthDelta jobs/tests."""
import argparse
import hashlib
import json
from pathlib import Path

REQUIRED=['START_HERE_FOR_CODEX.md','FOLLOWUP_PLAN.md','EXECUTION_DAG.md','STOP_CONDITIONS.md','CODEX_TASKS.json',
          'ARTIFACT_CONTRACT.md','TEST_PLAN.md','REPRODUCIBILITY.md','CURRENT_GAP_ANALYSIS.md',
          'evidence/PLAN_AUDIT_RESULT.json','evidence/EVIDENCE_MANIFEST.json','evidence/REQUIRED_INPUTS.md','prompts/CODEX_EXECUTION_PROMPT.md']
FIELDS=['id','priority','depends_on','scope','commands','acceptance_criteria','evidence_outputs','stop_if','status','estimated_minutes']

def validate(root:Path):
    checks={}
    checks['all_required_files_exist']=all((root/p).is_file() and (root/p).stat().st_size>0 for p in REQUIRED)
    if not checks['all_required_files_exist']:return checks
    tasks=json.loads((root/'CODEX_TASKS.json').read_text());byid={t['id']:t for t in tasks}
    checks['task_list']=isinstance(tasks,list)
    checks['required_task_fields']=all(all(k in t for k in FIELDS) for t in tasks)
    checks['unique_task_ids']=len(byid)==len(tasks)
    checks['all_dependencies_exist']=all(d in byid for t in tasks for d in t['depends_on'])
    grey=set();black=set()
    def visit(x):
        if x in grey:raise ValueError('cycle')
        if x in black:return
        grey.add(x)
        for d in byid[x]['depends_on']:visit(d)
        grey.remove(x);black.add(x)
    try:
        for x in byid:visit(x)
        checks['acyclic']=True
    except (ValueError,KeyError):checks['acyclic']=False
    checks['historical_done_retained']=all(byid[x]['status']=='DONE' for x in ['FP-00','FP-01','FP-03','FP-04'])
    checks['fp02_not_falsely_closed']=byid['FP-02']['status']=='PARTIAL'
    checks['no_historical_reexecution']=all(not byid[x]['commands'] for x in ['FP-00','FP-01','FP-02','FP-03','FP-04'])
    checks['only_fp05a_ready']=[t['id'] for t in tasks if t['status']=='READY']==['FP-05a']
    checks['future_states_to_be_run']=all(t['evidence_state']=='TO_BE_RUN' for t in tasks if t['id'].startswith('FP-05') or t['id']=='FP-06')
    checks['estimates_not_invented']=all(t['estimated_minutes'] is None for t in tasks)
    auth=json.loads((root/'evidence/INITIAL_AUTHORIZATION.json').read_text())
    checks['no_initial_gpu_or_training']=auth['gpu_jobs_authorized'] is False and auth['training_authorized'] is False
    proto=json.loads((root/'evidence/FP05_PROTOCOL_TEMPLATE.json').read_text())
    checks['confirm_not_h2_clean']=proto['split']['confirm']['status']=='UNASSIGNED_NO_ACCESS' and 'EXPOSED' in proto['split']['confirm']['rejected_reservation']
    checks['five_candidates']=proto['candidates']['count']==5 and proto['candidates']['background_F0_not_a_candidate']
    checks['protocol_not_falsely_frozen']=proto['status']=='TO_BE_RUN' and proto['execution_ready'] is False and proto['final_protocol_sha256'] is None
    checks['safety_margin_unbound_visible']=proto['decision']['max_harm72_vs_Fs'] is None
    audit=json.loads((root/'evidence/PLAN_AUDIT_RESULT.json').read_text())
    checks['audit_verdict_allowed']=audit['overall_verdict'] in ['PLAN_EXECUTED','PARTIALLY_EXECUTED','BLOCKED','FAILED_PLAN_COMPLIANCE','INSUFFICIENT_EVIDENCE']
    checks['science_not_claimed']='INCONCLUSIVE' in audit['novelty_claim_status'] and 'INCONCLUSIVE' in audit['weather_utility_claim_status']
    checks['S0_threshold_not_relaxed']=audit['key_observations']['s0']['tolerance']==1e-5
    checks['X1_exposure_documented']='X1' in (root/'STOP_CONDITIONS.md').read_text()
    checks['FP06_depends_on_real_evaluation']=byid['FP-06']['depends_on']==['FP-05f']
    manifest_path=root/'PACKAGE_MANIFEST.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        checks['manifest_hashes_match']=all((root/f['path']).is_file() and hashlib.sha256((root/f['path']).read_bytes()).hexdigest()==f['sha256'] for f in manifest['files'])
    else:
        checks['manifest_hashes_match']=False
    return checks

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args()
    checks=validate(a.package);result={'scope':'DELIVERY_ONLY_NOT_REPOSITORY_TESTS_OR_GPU','passed':all(checks.values()),'checks':checks,'check_count':len(checks)}
    if a.out:
        if a.out.exists():p.error('validation output already exists')
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
