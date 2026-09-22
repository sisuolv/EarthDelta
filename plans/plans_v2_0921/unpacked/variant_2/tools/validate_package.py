#!/usr/bin/env python3
"""Validate this delivery's documents, DAG, locks, templates and SHA-256 manifest.

Standard library only. Does not import EarthDelta, train models, grant permissions,
or establish that any research experiment passed. Run on the original package;
maintain mutable execution state in a separate working copy.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

REQUIRED = [
    'START_HERE_FOR_CODEX.md', 'EarthDelta_Codex_Execution_Plan.md',
    'FINAL_RESEARCH_DECISION.md', 'CODEX_TASKS.md', 'codex_tasks.json',
    'STOP_CONDITIONS.md', 'IMPLEMENTATION_AUDIT_ACTIONS.md',
    'experiments/P0_SURVIVAL_EXPERIMENTS.md', 'experiments/BASELINES.md',
    'research/ROUND2_REVIEW_SUMMARY.md', 'evidence/REPOSITORY_SNAPSHOT.json',
    'configs/pilot_config.template.json', 'configs/authorization.template.json',
    'tools/repo_preflight.py', 'tools/validate_package.py',
]
HEADERS = ['Scientific purpose', 'Depends on', 'Files to inspect',
           'Files to modify/create', 'Implementation steps', 'Tests',
           'Run command', 'Expected artifacts', 'Success criteria',
           'Failure criteria', 'Decision after completion', 'Suggested commit message']
FIELDS = ['task_id', 'title', 'priority', 'depends_on', 'files', 'commands',
          'artifacts', 'success_criteria', 'failure_action']


def validate(root: Path, skip_integrity: bool = False) -> dict:
    checks = []
    def check(name: str, condition: bool, details: str = '') -> None:
        checks.append({'check': name, 'passed': bool(condition), 'details': details})
    check('required_files_exist', all((root / p).is_file() for p in REQUIRED))
    if not checks[-1]['passed']:
        return {'scope': 'delivery_only', 'passed': False, 'checks': checks}
    texts = {p: (root/p).read_text(encoding='utf-8') for p in REQUIRED if p.endswith('.md')}
    check('required_texts_nonempty', all(text.strip() for text in texts.values()))
    state = json.loads((root/'codex_tasks.json').read_text(encoding='utf-8'))
    tasks = state['tasks']
    ids = [t['task_id'] for t in tasks]
    check('task_count_3_to_6', 3 <= len(tasks) <= 6)
    check('unique_task_ids', len(ids) == len(set(ids)))
    check('machine_required_fields', all(all(k in t for k in FIELDS) for t in tasks))
    check('only_first_task_authorized', state['current_task'] == 'R2-P0-01'
          and state['do_not_proceed_beyond'] == 'R2-P0-01'
          and state['approved_task_ids'] == ['R2-P0-01'])
    check('no_automatic_unlock', state['automatic_unlock'] is False
          and state['evidence_pass_is_not_authorization'] is True)
    check('initial_task_statuses', all(t['status'] == ('READY' if t['task_id'] == 'R2-P0-01' else 'LOCKED') for t in tasks))
    dependencies = {t['task_id']: t['depends_on'] for t in tasks}
    valid_deps = all(dep in ids and dep != t for t, deps in dependencies.items() for dep in deps)
    check('dependencies_exist', valid_deps)
    visiting, visited = set(), set()
    def visit(node: str) -> bool:
        if node in visiting: return False
        if node in visited: return True
        visiting.add(node)
        for dep in dependencies.get(node, []):
            if not visit(dep): return False
        visiting.remove(node)
        visited.add(node)
        return True
    check('dag_acyclic', valid_deps and all(visit(node) for node in ids))
    edges = {(d,t) for t,deps in dependencies.items() for d in deps}
    check('declared_edges_match_dependencies', edges == {tuple(e) for e in state['dag_edges']})
    plan = texts['EarthDelta_Codex_Execution_Plan.md']
    complete = True
    for t in tasks:
        marker = '## TASK ' + t['task_id'] + ' — '
        if marker not in plan:
            complete = False
            continue
        part = plan.split(marker, 1)[1].split('\n## TASK ',1)[0]
        complete &= all('### ' + h in part for h in HEADERS)
        complete &= all(cmd in part for cmd in t['commands'])
    check('main_plan_has_uniform_tasks_and_exact_commands', complete)
    check('main_plan_each_task_has_pass_fail_ci_semantics', all(set(t['gate']) == {'pass_means','fail_means','ci_crossing'} for t in tasks))
    audit = texts['IMPLEMENTATION_AUDIT_ACTIONS.md']
    check('all_15_findings_mapped', all('B%02d' % i in audit for i in range(1,16)))
    check('all_audit_categories_explained', all(x in audit for x in ['BLOCKS_NEXT_PILOT','FIX_BEFORE_TRAINING','FIX_LATER','NOT_A_BLOCKER','AUDIT_FINDING_REJECTED']))
    config = json.loads((root/'configs/pilot_config.template.json').read_text(encoding='utf-8'))
    pending = 'TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE'
    needed = ['delta_gain_min','delta_dynamic_min','delta_policy_min','delta_factorization_min',
              'correction_noninferiority_margin','guard_harm_margin','alpha','target_power',
              'confirm_n_processes_max','mde_vs_zero','planned_alternative']
    check('no_fabricated_science_thresholds', all(config['statistics'][k] == pending for k in needed))
    check('memory_jepa_calibration_disabled', all(config['policy'][k] is False for k in ['memory_enabled','jepa_enabled','gain_calibration_enabled']))
    auth = json.loads((root/'configs/authorization.template.json').read_text(encoding='utf-8'))
    ops = auth['approved_operations']
    check('heavy_execution_locked', auth['approved_task_ids'] == ['R2-P0-01']
          and all(not v for k,v in ops.items() if k not in ['repo_readonly','minimal_code_patch','synthetic_cpu_tests']))
    snapshot = json.loads((root/'evidence/REPOSITORY_SNAPSHOT.json').read_text(encoding='utf-8'))
    check('historical_tests_not_claimed_as_rerun', snapshot['cpu_record']['evidence_type'] == 'REPOSITORY_LOG_READ_NOT_RERUN'
          and snapshot['cpu_record']['passed'] == 283 and snapshot['cpu_record']['skipped'] == 7)
    check('heads_separated', snapshot['observed_branch_head'] != snapshot['audited_source_head']
          and snapshot['root_comparison']['all_preexisting_root_entries_equal'] is True)
    for path in sorted((root/'tools').glob('*.py')):
        try:
            ast.parse(path.read_text(encoding='utf-8'))
            check('python_syntax_' + path.name, True)
        except SyntaxError as exc:
            check('python_syntax_' + path.name, False, str(exc))
    if not skip_integrity:
        mf = root/'MANIFEST.sha256.json'
        check('manifest_exists', mf.is_file())
        if mf.is_file():
            manifest = json.loads(mf.read_text(encoding='utf-8'))
            entries = manifest['files']
            actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != mf}
            check('manifest_coverage', actual == set(entries))
            ok = True
            for rel, expected in entries.items():
                path = (root/rel).resolve()
                if root not in path.parents or not path.is_file():
                    ok = False
                    continue
                ok &= hashlib.sha256(path.read_bytes()).hexdigest() == expected
            check('manifest_all_sha256_match', ok)
    return {'scope':'delivery_integrity_and_contract_consistency_only',
            'passed': all(c['passed'] for c in checks), 'checks_count':len(checks),
            'checks':checks, 'repository_tests_run':False, 'weather_experiments_run':False,
            'grants_execution_authority':False}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--skip-integrity', action='store_true', help='Builder-only: validate before the final checksum manifest is created.')
    parser.add_argument('--report', type=Path, help='Optional new report path outside the immutable package.')
    args = parser.parse_args()
    root = args.package.expanduser().resolve()
    try:
        result = validate(root, args.skip_integrity)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {'scope':'delivery_only','passed':False,'error':str(exc)}
    if args.report:
        report = args.report.expanduser().resolve()
        if report == root or root in report.parents:
            print('Report must be outside the immutable package.', file=sys.stderr)
            return 2
        report.parent.mkdir(parents=True, exist_ok=True)
        try:
            with report.open('x', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                f.write('\n')
        except FileExistsError:
            print('Refusing to overwrite report: ' + str(report), file=sys.stderr)
            return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
