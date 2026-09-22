#!/usr/bin/env python3
"""Submit a frozen EarthDelta source snapshot to one 4-H100 ACP worker."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import uuid


SCO = '/mnt/afs/260010168/bin/sco'
IMAGE = ('registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/'
         'nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-'
         'transformerengine1.5:v1.0.0-20241130-nvdia-base-image')
STORAGE = '01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs'
WORKSPACE = 'share-space'


def write_json(path: Path, value: object) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ['git', '--git-dir=.git', '--work-tree=.', *args], cwd=repo, text=True
    ).strip()


def snapshot(repo: Path, dest: Path) -> dict[str, str]:
    suffixes = {'.py', '.json', '.yaml', '.yml', '.toml', '.md', '.sh', '.txt', '.npz'}
    paths = set()
    for rel in ('earthdelta', 'scripts', 'tests', 'reference/stormer'):
        root = repo / rel
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in
                       {'.git', '__pycache__', '.pytest_cache', 'checkpoints', 'logs'}
                       and not (Path(current) / d).is_symlink()]
            for name in files:
                path = Path(current) / name
                if path.is_symlink():
                    continue
                if path.suffix in suffixes or name in {'LICENSE', 'LICENSE.txt'}:
                    if path.stat().st_size > 10 * 1024 * 1024:
                        raise ValueError(f'Unexpected large source file: {path}')
                    paths.add(path)
    for rel in ('pyproject.toml', 'research_spec_v6.yaml', 'README.md'):
        if (repo / rel).is_file():
            paths.add(repo / rel)
    hashes = {}
    for path in sorted(paths):
        rel = path.relative_to(repo)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(rel)] = hashlib.sha256(target.read_bytes()).hexdigest()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        target = dest / 'cci' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(target.relative_to(dest))] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--label', default='probe')
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--argv-file', type=Path,
                        help='JSON array; supports @SNAPSHOT@, @OUTPUT@, @ASSETS@ placeholders.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,27}', args.label):
        parser.error('label must be 1-28 lowercase letters, digits or hyphens')
    if args.timeout_seconds <= 0:
        parser.error('timeout-seconds must be positive')
    command = None
    if args.argv_file:
        command = json.loads(args.argv_file.read_text())
        if not isinstance(command, list) or not command or not all(
            isinstance(x, str) and '\x00' not in x for x in command
        ):
            parser.error('argv-file must contain a nonempty array of strings')
    repo = args.repo.resolve()
    run_id = 'ed-' + args.label + '-' + dt.datetime.now(dt.timezone.utc).strftime('%m%d%H%M%S') + '-' + uuid.uuid4().hex[:6]
    root = repo / 'artifacts' / 'round2_cci' / run_id
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source'
    source.mkdir()
    hashes = snapshot(repo, source)
    write_json(root / 'source_manifest.json', {
        'repo': str(repo), 'head': git(repo, 'rev-parse', 'HEAD'),
        'tracked_diff': git(repo, 'diff', '--stat'), 'files': hashes,
        'assets_note': 'Data and checkpoints stay on AFS; source snapshot excludes weights and NPY/Zarr.'
    })
    if command is None:
        command = ['python3', '-u', '@SNAPSHOT@/cci/probe.py', '--run-dir', '@OUTPUT@', '--assets', '@ASSETS@']
    substitutions = {'@SNAPSHOT@': str(source), '@OUTPUT@': str(root), '@ASSETS@': str(repo)}
    for key, value in substitutions.items():
        command = [part.replace(key, value) for part in command]
    write_json(root / 'invocation.json', {
        'run_id': run_id, 'argv': command, 'timeout_seconds': args.timeout_seconds,
        'cwd': str(source), 'source_cci_hostname': socket.gethostname(),
        'claim': 'Execution evidence only; no implied S0 or research PASS.'
    })
    worker_command = shlex.join(['python3', '-u', str(source / 'cci' / 'run_job.py'), str(root)])
    argv = [SCO, 'acp', 'jobs', 'create', f'--workspace-name={WORKSPACE}',
            '--aec2-name=share-cluster', f'--job-name={run_id}',
            f'--container-image-url={IMAGE}', '--training-framework=pytorch',
            '--worker-nodes=1', '--worker-spec=n6ls.iu.i40.4.32c512g',
            f'--storage-mount={STORAGE}', '--quota-type=reserved', '--priority=NORMAL',
            '--retry-times=0', '--wait', f'--command={worker_command}']
    write_json(root / 'create_argv.json', argv)
    print(json.dumps({'run_dir': str(root), 'dry_run': args.dry_run}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    # This installed launcher contains the working ACP client; its regional bootstrap URL is stale.
    env['SCO_LAUNCHED_BY'] = 'launcher'
    try:
        result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=120)
    except subprocess.TimeoutExpired:
        write_json(root / 'submission.json', {'status': 'SUBMISSION_UNKNOWN', 'reason': 'CLI timeout; query by unique display name before retrying.'})
        print(f'Submission state unknown. Query display name {run_id}; do not submit a duplicate.')
        return 2
    write_json(root / 'submission.json', {
        'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
        'sco_dispatch': 'SCO_LAUNCHED_BY=launcher',
    })
    print(result.stdout, end='')
    print(result.stderr, end='')
    if result.returncode:
        return result.returncode
    match = re.search(r'\bjob\s+(pt-[a-z0-9]+)\s+submitted', result.stdout, re.I)
    if not match:
        print('Submission returned success without a parsed job ID; query by unique display name before retrying.')
        return 2
    (root / 'job-id.txt').write_text(match.group(1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
