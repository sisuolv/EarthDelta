import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

run=Path(__file__).resolve().parents[1]
p=run/'jobs'/sys.argv[1];job=(p/'job-id.txt').read_text().strip()
env=dict(os.environ,SCO_HOME='/mnt/afs/260010168/.sco',SCO_DATA_HOME='/mnt/afs/260010168/.local/share/sco',SCO_CONFIG='/mnt/afs/260010168/.config/sco')
q=subprocess.run(['/mnt/afs/260010168/bin/sco','acp','jobs','describe',job,'--workspace-name=share-space','-o','json'],env=env,capture_output=True,text=True,timeout=40)
if q.returncode:print(q.stderr);raise SystemExit(q.returncode)
raw=json.loads(q.stdout);(p/'describe_latest.json').write_text(json.dumps(raw,indent=2)+'\n')
def parse(s):
    dt,tz=s.replace('Z','+00:00').rsplit('+',1)
    if '.' in dt:
        a,b=dt.split('.');dt=a+'.'+b[:6].ljust(6,'0')
    return datetime.datetime.fromisoformat(dt+'+'+tz)
row={'event':'reconciled_charge','stage':p.name,'job_id':job,'state':raw['state'],'gpus':1}
if raw['state'] in ('SUCCEEDED','FAILED','SUSPENDED','CANCELLED'):
    row.update(conservative_card_hours=(parse(raw['update_time'])-parse(raw['create_time'])).total_seconds()/3600,basis='creation to terminal update upper bound',describe_path=str(p/'describe_latest.json'))
    entries=[json.loads(x) for x in (run/'BUDGET_LEDGER.jsonl').read_text().splitlines()]
    if not any(x.get('event')=='reconciled_charge' and x.get('job_id')==job for x in entries):
        with (run/'BUDGET_LEDGER.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row))
