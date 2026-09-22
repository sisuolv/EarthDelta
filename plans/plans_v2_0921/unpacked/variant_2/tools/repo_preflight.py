#!/usr/bin/env python3
"""Read-only Git/source preflight. No model, data, dependency or network access.

Writes a new JSON report, never overwrites. This verifies source identity only;
it does NOT certify runtime correctness, checkpoint availability or science.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

OBSERVED_HEAD = '403b55db65f4c35c1a85d0794ad0de2765b07d96'
AUDITED_HEAD = 'fb767f7f6efbc428be39c9ad84f5905331d6e40f'
EXPECTED_BRANCH = 'audit/round2-review-20260921'
EXPECTED = {
    'earthdelta': '867ea7c5ad43ffd7850cda3b6627287032a4db8f',
    'scripts': '17893b917ce06ffd7fed2655e15bb0a129268320',
    'tests': 'e3c5d9d8bd207b492554487e7d9b0bd669ab4193',
    'pyproject.toml': 'adcad73a0ae57fa8fe3bdaf830f200c20969fa55',
    'reference': 'f127c7c902a02525645bffd09a1acff0dc4b51a4',
}

def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(['git', '-C', str(repo), *args], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False, timeout=30)
    if proc.returncode != 0:
        # Do not dump remotes or environment variables; git command here is local.
        raise RuntimeError('git ' + ' '.join(args[:2]) + ' failed: ' + proc.stderr.strip()[:500])
    return proc.stdout.strip()

def inspect(repo: Path) -> dict:
    root = Path(git(repo, 'rev-parse', '--show-toplevel')).resolve()
    if root != repo:
        raise ValueError('--repo must be the actual repository root: ' + str(root))
    head = git(repo, 'rev-parse', 'HEAD')
    branch = git(repo, 'rev-parse', '--abbrev-ref', 'HEAD')
    objects = {p: git(repo, 'rev-parse', 'HEAD:' + p) for p in EXPECTED}
    drift = {p: {'expected': EXPECTED[p], 'actual': objects[p]}
             for p in EXPECTED if EXPECTED[p] != objects[p]}
    tracked = git(repo, 'status', '--porcelain=v1', '--untracked-files=no')
    # Only names/status, not file contents or secrets.
    status = git(repo, 'status', '--porcelain=v1', '--untracked-files=normal')
    if drift:
        verdict = 'BLOCKED_SOURCE_DRIFT_REVIEW_REQUIRED'
    elif tracked:
        verdict = 'DIRTY_WORKTREE_REVIEW_REQUIRED'
    elif head != OBSERVED_HEAD or branch != EXPECTED_BRANCH:
        verdict = 'SOURCE_EQUIVALENT_BASELINE_RECONCILIATION_REQUIRED'
    else:
        verdict = 'READONLY_BASELINE_MATCH'
    return {
        'observed_package_head': OBSERVED_HEAD, 'audited_source_head': AUDITED_HEAD,
        'local_head': head, 'local_branch': branch, 'repo_root': str(root),
        'source_objects': objects, 'source_object_drift': drift,
        'tracked_status_lines': tracked.splitlines(), 'worktree_status_lines': status.splitlines(),
        'verdict': verdict, 'science_validated': False,
        'allowed_next_operation': 'R2-P0-01 minimal patch only after any reconciliation above',
        'notes': ['No network, checkout/reset, model load, data scan or repository tests performed.',
                  'Untracked package/artifact directories are recorded, not deleted.',
                  'Future commits created by the task need fresh run manifests, not a reset to this baseline.'],
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        print('Refusing to overwrite: ' + str(out), file=sys.stderr)
        return 2
    result = {'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        result.update(inspect(args.repo.expanduser().resolve()))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result.update(verdict='BLOCKED_PREFLIGHT', error=str(exc), science_validated=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
    except FileExistsError:
        print('Refusing to overwrite: ' + str(out), file=sys.stderr)
        return 2
    print(json.dumps({'verdict': result['verdict'], 'report': str(out)}, ensure_ascii=False))
    return 0 if result['verdict'] == 'READONLY_BASELINE_MATCH' else 2

if __name__ == '__main__':
    raise SystemExit(main())
