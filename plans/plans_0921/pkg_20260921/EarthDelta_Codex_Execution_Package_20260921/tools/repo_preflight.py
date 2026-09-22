#!/usr/bin/env python3
"""Read-only repository preflight. Does not import research modules or load weights."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

EXPECTED = '4fe55a7af90ea92f62a3232a571af92bfbd6114d'

def git(repo: Path, *args: str) -> str:
    result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                            text=True, check=True, timeout=20)
    return result.stdout.strip()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if not repo.is_dir():
        parser.error('Repository directory does not exist.')
    if output.exists():
        parser.error('Output already exists; choose a new run directory.')
    try:
        root = Path(git(repo, 'rev-parse', '--show-toplevel')).resolve()
        if root != repo:
            parser.error('Use the repository root, not a subdirectory.')
        head = git(repo, 'rev-parse', 'HEAD')
        branch = git(repo, 'branch', '--show-current') or 'DETACHED'
        status = git(repo, 'status', '--porcelain=v1')
        tracked_count = len(git(repo, 'ls-files').splitlines())
        # Store only a digest, not potentially sensitive diff content.
        diff = git(repo, 'diff', 'HEAD', '--binary')
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f'Git preflight failed ({type(exc).__name__}); no research was run.', file=sys.stderr)
        return 2
    versions = {}
    for package in ['torch', 'numpy', 'scipy', 'pytest', 'timm', 'xformers',
                    'lightning', 'xarray', 'zarr', 'weatherbenchX']:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    assets = {}
    # Names and byte counts only; no traversal of arbitrary user folders.
    for name in ['earthdelta', 'tests', 'scripts', 'plans', 'reference', 'checkpoints', 'data']:
        p = repo / name
        assets[name] = {'exists': p.exists(), 'is_directory': p.is_dir()}
        if name == 'checkpoints' and p.is_dir() and not p.is_symlink():
            assets[name]['files'] = [
                {'name': f.name, 'size_bytes': f.stat().st_size, 'sha256': 'NOT_COMPUTED'}
                for f in sorted(p.iterdir()) if f.is_file() and not f.is_symlink()
            ][:50]
    report = {
        'schema_version': 'earthdelta_read_only_preflight_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'repo_root': str(repo), 'branch': branch, 'head': head,
        'expected_audit_commit': EXPECTED, 'head_matches_audit': head == EXPECTED,
        'dirty': bool(status), 'working_tree_status': status.splitlines(),
        'working_tree_diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
        'tracked_file_count': tracked_count,
        'python': platform.python_version(), 'packages': versions, 'assets': assets,
        'gpu_probe': 'NOT_RUN', 'repository_tests': 'NOT_RUN',
        'checkpoint_load': 'NOT_RUN', 'weather_experiments': 'NOT_RUN',
        'status': 'READ_ONLY_INVENTORY_COMPLETE',
        'next_action': 'Reconcile HEAD/dirty state and asset roles; stop after P0-01.'
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(f'Read-only inventory saved: {output}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
