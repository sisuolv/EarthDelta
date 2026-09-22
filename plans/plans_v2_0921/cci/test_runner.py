"""Bounded runner failure-path checks; these do not submit cloud jobs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RunnerTests(unittest.TestCase):
    def execute(self, body, *, timeout=5, tamper=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            program = source / 'task.py'
            program.write_text(body)
            digest = hashlib.sha256(program.read_bytes()).hexdigest()
            (root / 'source_manifest.json').write_text(json.dumps({'files': {'task.py': digest}}))
            (root / 'invocation.json').write_text(json.dumps({
                'run_id': 'synthetic-runner-test', 'source_cci_hostname': 'synthetic-other-host',
                'argv': [sys.executable, str(program)], 'timeout_seconds': timeout,
            }))
            if tamper:
                program.write_text("raise RuntimeError('must not execute')\n")
            p = subprocess.run([sys.executable, str(Path(__file__).with_name('run_job.py')), str(root)],
                               capture_output=True, text=True, timeout=25)
            result = json.loads((root / 'job_result.json').read_text())
            return p.returncode, result, (root / 'worker.log').exists()

    def test_success(self):
        rc, result, log_exists = self.execute("print('finished')\n")
        self.assertEqual((rc, result['status'], log_exists), (0, 'SUCCEEDED', True))

    def test_failure_propagates(self):
        rc, result, _ = self.execute('raise SystemExit(7)\n')
        self.assertEqual((rc, result['returncode'], result['status']), (7, 7, 'FAILED'))

    def test_snapshot_tamper_prevents_execution(self):
        rc, result, log_exists = self.execute('pass\n', tamper=True)
        self.assertEqual((rc, result['status'], log_exists), (1, 'FAILED', False))
        self.assertIn('Source snapshot changed', result['error'])

    def test_timeout(self):
        rc, result, _ = self.execute('import time\ntime.sleep(30)\n', timeout=1)
        self.assertEqual((rc, result['returncode'], result['status']), (124, 124, 'TIMEOUT'))


if __name__ == '__main__':
    unittest.main()
