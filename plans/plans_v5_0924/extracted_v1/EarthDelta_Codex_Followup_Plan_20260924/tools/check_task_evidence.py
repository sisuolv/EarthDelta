#!/usr/bin/env python3
import argparse, hashlib, json, subprocess
from pathlib import Path
from datetime import datetime, timezone

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description='Read-only metadata collection; never certifies a scientific run.')
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); repo=a.repo.resolve(); out=a.out.resolve()
    if out.exists(): raise FileExistsError(out)
    root=Path(__file__).resolve().parents[1]
    obs=json.loads((root/'evidence/OBSERVED_STAGE_EVIDENCE.json').read_text())
    result={'status':'OBSERVED','verdict':'METADATA_COLLECTION_ONLY','generated_at_utc':datetime.now(timezone.utc).isoformat(),'repo':str(repo),'model_or_gpu_execution':False,'commands':[],'files':[],'missing':[]}
    for cmd in [['git','rev-parse','HEAD'],['git','status','--porcelain=v1']]:
        r=subprocess.run(cmd,cwd=repo,capture_output=True,text=True,check=False)
        result['commands'].append({'argv':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    paths=[obs['s0']['artifact']]
    jobs=[obs['s0']]+obs['fs_v2']['jobs']+obs['bank']['jobs']
    for j in jobs: paths.append(j['run_dir']+'/job_result.json')
    for rel in paths:
        f=repo/rel
        rec={'path':rel,'exists':f.is_file()}
        if f.is_file():
            rec.update(sha256=sha(f),bytes=f.stat().st_size)
            try:
                v=json.loads(f.read_text()); rec['selected_fields']={k:v.get(k) for k in ('run_id','job_id','status','returncode','elapsed_seconds','started_utc','finished_utc','s0_gate_pass','verdict_committed')}
            except Exception as exc: rec['parse_error']=f'{type(exc).__name__}: {exc}'
        else: result['missing'].append(rel)
        result['files'].append(rec)
    result['completeness']='MISSING' if result['missing'] else 'PRESENT_NOT_YET_IDENTITY_AUDITED'
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'out':str(out),'missing_count':len(result['missing']),'scope':result['verdict']}))
    return 0
if __name__=='__main__': raise SystemExit(main())
