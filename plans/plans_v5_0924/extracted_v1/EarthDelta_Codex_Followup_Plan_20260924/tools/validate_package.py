#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
REQ=['START_HERE_FOR_CODEX.md','FOLLOWUP_PLAN.md','EXECUTION_DAG.md','STOP_CONDITIONS.md','CODEX_TASKS.json','ARTIFACT_CONTRACT.md','TEST_PLAN.md','REPRODUCIBILITY.md','CURRENT_GAP_ANALYSIS.md','evidence/PLAN_AUDIT_RESULT.json','evidence/EVIDENCE_MANIFEST.json','evidence/REQUIRED_INPUTS.md','prompts/CODEX_EXECUTION_PROMPT.md']
FIELDS=['id','priority','depends_on','scope','commands','acceptance_criteria','evidence_outputs','stop_if','status','estimated_minutes']
def check(root,check_hashes=True):
    checks=[]
    def need(name,condition):
        checks.append({'check':name,'passed':bool(condition)})
        if not condition: raise ValueError(name)
    need('required_files',all((root/n).is_file() for n in REQ))
    ts=json.loads((root/'CODEX_TASKS.json').read_text()); by={t['id']:t for t in ts}
    need('unique_task_ids',len(ts)==len(by))
    need('required_task_fields',all(all(k in t for k in FIELDS) for t in ts))
    need('existing_dependencies',all(set(t['depends_on'])<=by.keys() for t in ts))
    visiting=set(); done=set()
    def dfs(k):
        if k in visiting: raise ValueError('cycle')
        if k in done:return
        visiting.add(k)
        for d in by[k]['depends_on']:dfs(d)
        visiting.remove(k);done.add(k)
    for k in by:dfs(k)
    need('acyclic_DAG',True)
    need('only_FP05a_authorized',[t['id'] for t in ts if t.get('authorized')] == ['FP-05a'])
    need('historical_done_scoped',all(by[k]['status']=='DONE' and by[k]['evidence'] for k in ('FP-00','FP-01','FP-03','FP-04')))
    need('FP02_partial_preserved',by['FP-02']['status']=='PARTIALLY_DONE')
    p=json.loads((root/'evidence/PROPOSED_PROTOCOL.json').read_text())
    need('no_gpu_authorization',p['authorization']['gpu_authorized'] is False)
    need('confirm_not_laundered',p['split']['confirm_window'] is None and p['split']['confirm_access_authorized'] is False and p['split']['2020_H2_status']=='EXPOSED_NOT_CLEAN_CONFIRM')
    need('five_candidates_F0_report_only',p['actions']['num_candidates']==5 and p['actions']['F0_report_only'] is True)
    o=json.loads((root/'evidence/OBSERVED_STAGE_EVIDENCE.json').read_text())
    need('S0_frozen_tolerance',o['s0']['tolerance']==1e-5)
    need('no_fake_repo_test_execution',all(x['new_audit_rerun'] is False for x in o['cpu_logs']))
    ar=json.loads((root/'evidence/PLAN_AUDIT_RESULT.json').read_text())
    need('audit_required_fields',all(k in ar for k in ['overall_verdict','fp00_04_disposition','s0_parity_verdict','fs_fit_divergence_verdict','bank_certification_verdict','fp05_plan_soundness_verdict','novelty_claim_status','weather_utility_claim_status','blocking_items']))
    need('no_invented_minutes',all(t['estimated_minutes'] is None for t in ts))
    if check_hashes:
        m=json.loads((root/'SHA256SUMS.json').read_text())
        actual={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file() and f.name!='SHA256SUMS.json'}
        need('manifest_complete_and_hashes_equal',m==actual)
    return checks

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args()
    checks=check(a.package.resolve())
    print(json.dumps({'status':'PASS_PACKAGE_ONLY','checks':checks,'repository_tests_run':False,'gpu_experiments_run':False},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
