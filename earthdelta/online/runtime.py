"""Bound execution inputs; authorization records are not fabricated signatures."""
import json
from pathlib import Path
from .io import checked_json


def load_runtime(path):
    config=json.loads(Path(path).read_text())
    read=lambda key:checked_json(config[key]['path'],config[key]['sha256'])
    contract=read('contract');auth=read('session_authorization');roster=read('roster')
    if auth.get('contract_sha256')!=config['contract']['sha256']:
        raise PermissionError('session instruction does not bind this contract')
    if contract['data']['allowed_year']!=2020 or roster['year']!=2020:
        raise PermissionError('only the frozen 2020 role is authorized')
    if contract['data']['download_authorized']:
        raise PermissionError('stage 0 does not authorize downloads')
    receipt=read('t2_receipt')
    if not receipt.get('engineering_pass') or receipt.get('required_skips')!=0:
        raise PermissionError('T2 receipt does not release real-data QC')
    return config,contract,roster


def ledger(path,event):
    with Path(path).open('a') as f:f.write(json.dumps(event,sort_keys=True,allow_nan=False)+'\n')


def require_fit_index(index):
    from .timeindex import checked_index,index_of
    i=checked_index(index)
    if i>index_of('2020-03-31T00:00:00Z'):
        raise PermissionError('fit worker cannot open analysis-period truth')
    return i


def fit_cache_row(manifest,index,access_ledger):
    from .qc_reuse import load_bound_cache
    i=require_fit_index(index)
    row=manifest['indices'].get(str(i))
    ledger(access_ledger,{'stage':'fit','index':i,'source':'bound_cache','purpose':'fit_input_or_truth'})
    return None if row is None else load_bound_cache(row)
