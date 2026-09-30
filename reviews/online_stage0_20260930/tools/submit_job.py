import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

RUN = Path(__file__).resolve().parents[1]
stage, resource = sys.argv[1:3]
launcher = '/mnt/afs/260010168/.sco/state/regions/cnsh01/quarantine/20260920T105828/sco' if '--fallback' in sys.argv[3:] else '/mnt/afs/260010168/bin/sco'
out=RUN/'jobs'/stage
req=json.loads((out/'request.json').read_text())
configs={
 'spot_h100':('computing-cluster-01g-02','N6lS.Iu.I10.1.8c128g','spot'),
 'reserved_h100':('share-cluster','n6ls.iu.i40.1.4c64g','reserved'),
}
cluster,spec,quota=configs[resource]
image='registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image'
name='ed-v92-'+stage.replace('_','-')+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')
command=shlex.join(['timeout',str(req['timeout_seconds'])+'s','python3','-B',str(RUN/'tools/job_driver.py'),str(out/'request.json')])
argv=[launcher,'acp','jobs','create','--workspace-name=share-space','--aec2-name='+cluster,'--job-name='+name,'--container-image-url='+image,'--training-framework=pytorch','--worker-nodes=1','--worker-spec='+spec,'--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--quota-type='+quota,'--priority=NORMAL','--command='+command]
assert not (out/'submission.json').exists(), 'do not retry uncertain submission'
(out/'create-argv.json').write_text(json.dumps(argv,indent=2)+'\n')
(out/'command.txt').write_text(command+'\n')
record={'stage':stage,'job_name':name,'resource':resource,'gpus':1,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'request_sha256':hashlib.sha256((out/'request.json').read_bytes()).hexdigest(),
        'driver_sha256':hashlib.sha256((RUN/'tools/job_driver.py').read_bytes()).hexdigest()}
try:
 p=subprocess.run(argv,capture_output=True,text=True,timeout=60,env=dict(os.environ,SCO_HOME='/mnt/afs/260010168/.sco',SCO_DATA_HOME='/mnt/afs/260010168/.local/share/sco',SCO_CONFIG='/mnt/afs/260010168/.config/sco'))
 record.update(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
except subprocess.TimeoutExpired:
 record.update(exit_code=None,state='UNKNOWN_SUBMISSION')
(out/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
if record.get('exit_code')==0:
 match=re.search(r'\bpt-[a-z0-9]+\b',record['stdout'])
 if match:
  record['job_id']=match.group();(out/'job-id.txt').write_text(match.group()+'\n')
with (RUN/'BUDGET_LEDGER.jsonl').open('a') as f:
 f.write(json.dumps({'event':'submission','stage':stage,'job_id':record.get('job_id'),'gpus':1,'resource':resource,'utc':record['created_utc'],'card_hours':'PENDING_FINAL_ALLOCATION_TIMESTAMPS'})+'\n')
print(json.dumps(record))
