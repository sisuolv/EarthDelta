#!/usr/bin/env python3
"""Read-only file/receipt inventory. Never runs a model or certifies a scientific gate."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def receipt_errors(record: dict, expected_rc: int, expected_job: str) -> list[str]:
    errors=[]
    if not isinstance(record,dict):
        return ['job_result is not an object']
    if type(record.get('returncode')) is not int or record.get('returncode') != expected_rc:
        errors.append('returncode missing, invalid or inconsistent with expected attempt')
    elapsed=record.get('elapsed_seconds')
    if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<0:
        errors.append('elapsed_seconds missing or invalid')
    for key in ('run_id','hostname','started_utc','finished_utc'):
        if not isinstance(record.get(key),str) or not record[key].strip():
            errors.append(f'{key} missing')
    if record.get('job_id') is not None and record['job_id'] != expected_job:
        errors.append('job_id mismatch')
    for key in ('started_utc','finished_utc'):
        try:
            t=dt.datetime.fromisoformat(record.get(key,'').replace('Z','+00:00'))
            if t.tzinfo is None: errors.append(key+' has no timezone')
        except (TypeError,ValueError):
            errors.append(key+' is not ISO time')
    # A platform receipt can link its job id through its matching submission record.
    # Its absence here is NOT silently promoted to complete identity binding.
    return errors


def git_read(repo: Path,args: list[str]) -> dict:
    p=subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,timeout=30)
    return {'argv':['git','-C',str(repo),*args],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}


def inventory(repo: Path, specs: list[dict], mapping: dict[str,str], hash_large: bool) -> list[dict]:
    results=[]
    for entry in specs:
        override=mapping.get(entry['id'])
        path=Path(override) if override else repo/entry['path']
        record={'id':entry['id'],'path':str(path),'kind':entry['kind'],'required_for':entry['required_for'],
                'expected_sha256':entry.get('expected_sha256'),'state':'MISSING','semantic_gate_certified':False}
        if not path.is_file():
            record['reason']='file not available; supply an exact read-only path, not a fabricated summary'
            results.append(record); continue
        record['size_bytes']=path.stat().st_size
        if entry['kind']=='large_asset' and not hash_large:
            record.update(state='NOT_HASHED',reason='explicit --hash-large-assets required; existence is not verification')
            results.append(record);continue
        actual=sha256(path);record['computed_sha256']=actual
        expected=entry.get('expected_sha256')
        if expected and actual!=expected:
            record.update(state='HASH_MISMATCH');results.append(record);continue
        if entry['kind']=='raw_job_receipt':
            try:
                payload=json.loads(path.read_text())
                raw=payload.get('job_result',payload) if isinstance(payload,dict) else payload
                errors=receipt_errors(raw,entry.get('expected_returncode',0),entry['job_id'])
                record.update(state='INVALID_RECEIPT' if errors else 'OBSERVED_RECEIPT_FIELDS',errors=errors,
                              job_identity_join='TO_VERIFY_WITH_ORIGINAL_SUBMISSION_AND_INVOCATION',
                              raw_job_id=raw.get('job_id') if isinstance(raw,dict) else None,
                              run_id=raw.get('run_id') if isinstance(raw,dict) else None)
            except (ValueError,TypeError,OSError) as exc:
                record.update(state='INVALID_RECEIPT',errors=[str(exc)])
        else:
            record['state']='HASH_VERIFIED_NOT_SEMANTIC_GATE' if expected else 'UNBOUND_COMPUTED_HASH_ONLY'
        results.append(record)
    return results


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--asset-map',type=Path)
    p.add_argument('--hash-large-assets',action='store_true')
    a=p.parse_args()
    pkg=Path(__file__).resolve().parents[1]
    specs=json.loads((pkg/'evidence/EXPECTED_INPUTS.json').read_text())
    mapping=json.loads(a.asset_map.read_text()) if a.asset_map else {}
    if not isinstance(mapping,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in mapping.items()):
        p.error('--asset-map must map known input ids to path strings')
    if set(mapping)-{s['id'] for s in specs}:p.error('--asset-map has unknown ids')
    repo=a.repo.resolve()
    out=a.out.resolve()
    if out.exists():p.error('output exists; use a new evidence path')
    if not repo.is_dir():p.error('--repo must exist')
    result={'schema':'ed-read-only-inventory/1','state':'OBSERVED','scope':'INVENTORY_NOT_GATE_NOT_GPU_NOT_SCIENCE',
            'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'repo':str(repo),
            'commands':{},'inputs':[],'scientific_gate_pass':False}
    for name,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain=v1']),('diff',['diff','--stat'])]:
        try:result['commands'][name]=git_read(repo,args)
        except (OSError,subprocess.TimeoutExpired) as exc:result['commands'][name]={'returncode':None,'error':str(exc)}
    result['inputs']=inventory(repo,specs,mapping,a.hash_large_assets)
    accepted={'HASH_VERIFIED_NOT_SEMANTIC_GATE','OBSERVED_RECEIPT_FIELDS'}
    result['missing_or_unverified_now']=[r['id'] for r in result['inputs']
        if r['required_for'].startswith('FP-05a') and r['state'] not in accepted]
    result['later_gpu_inputs_pending']=[r['id'] for r in result['inputs']
        if not r['required_for'].startswith('FP-05a') and r['state'] not in accepted]
    result['status']='BLOCKED_INPUTS' if result['missing_or_unverified_now'] else 'INVENTORY_COLLECTED_REQUIRES_SEMANTIC_REVIEW'
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'output':str(out),'scientific_gate_pass':False},ensure_ascii=False))
    return 2 if result['missing_or_unverified_now'] else 0

if __name__=='__main__':raise SystemExit(main())
