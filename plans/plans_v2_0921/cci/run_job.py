#!/usr/bin/env python3
"""Verify source snapshot, execute one bounded command and persist its exit status."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    invocation = json.loads((root / 'invocation.json').read_text())
    result_path = root / 'job_result.json'
    if result_path.exists():
        raise RuntimeError('Use a new run directory; an execution result already exists.')
    result = {'run_id': invocation['run_id'], 'hostname': socket.gethostname(),
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'FAILED'}
    started = time.monotonic()
    returncode = 1
    try:
        if socket.gethostname() == invocation['source_cci_hostname']:
            raise RuntimeError('This command must run in an ACP worker, not the submitting CCI.')
        source = root / 'source'
        manifest = json.loads((root / 'source_manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Source snapshot changed: {name}')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(source) + ':' + str(source / 'reference/stormer')
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        env['OMP_NUM_THREADS'] = '4'
        with (root / 'worker.log').open('x') as log:
            process = subprocess.Popen(invocation['argv'], cwd=source, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=invocation['timeout_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = 124
                result['status'] = 'TIMEOUT'
        if returncode == 0:
            result['status'] = 'SUCCEEDED'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result.update(returncode=returncode, elapsed_seconds=round(time.monotonic() - started, 3),
                  finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    with result_path.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)
    return returncode


if __name__ == '__main__':
    raise SystemExit(main())
