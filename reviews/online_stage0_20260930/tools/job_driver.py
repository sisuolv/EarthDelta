import datetime
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

request = json.loads(Path(sys.argv[1]).read_text())
out = Path(request['job_dir'])
context = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'hostname': socket.gethostname(), 'python': platform.python_version(),
           'cwd': request['cwd']}
(out / 'worker_environment.json').write_text(json.dumps(context, indent=2)+'\n')
results = []
code = 0
with (out / 'worker.log').open('x') as log:
    for action in request['actions']:
        log.write('ACTION '+action['name']+'\n'); log.flush()
        started = time.monotonic()
        try:
            proc = subprocess.run(action['argv'], cwd=request['cwd'],
                                  env=dict(os.environ, **action['env']),
                                  stdout=log, stderr=subprocess.STDOUT,
                                  timeout=action['timeout'])
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
        results.append({'action':action['name'], 'exit_code':code,
                        'wall_seconds':time.monotonic()-started})
        if code:
            break
(out/'actions.json').write_text(json.dumps(results,indent=2)+'\n')
(out/'exit_code.txt').write_text(str(code)+'\n')
(out/'finished_utc.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
sys.exit(code)
