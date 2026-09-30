"""Mechanical receipt aggregation only; does not open scientific arrays."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--phase',choices=('calibration','validation'),required=True)
ap.add_argument('--shards',nargs='+',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
cfg=json.loads(Path(args.config).read_text())
def checked(info):
    raw=Path(info['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==info['sha256'],'receipt hash mismatch'
    return json.loads(raw)
c=checked(cfg['contract']);inf=c['inference'];sources=[];seen={};factors=inf['width_factors']
if args.phase=='validation':
    cal=checked(cfg['coverage_calibration']);assert cal['status']=='OBSERVED'
    factors=[cal['selected_factor']]
for directory in args.shards:
    path=Path(directory)/'SHARD_RECEIPT.json';raw=path.read_bytes();r=json.loads(raw)
    assert r['status']=='OBSERVED' and r['phase']==args.phase
    meta=Path(directory)/'INPUTS.json'
    assert hashlib.sha256(meta.read_bytes()).hexdigest()==r['inputs_sha256']
    m=json.loads(meta.read_text());assert m['contract_sha256']==cfg['contract']['sha256']
    assert m['config_sha256']==hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    expected_reps=inf['coverage_calibration']['mc_validation_reps' if args.phase=='validation' else 'mc_calibration_reps']
    expected_draws=inf['coverage_calibration']['bootstrap_validation_draws' if args.phase=='validation' else 'bootstrap_screen_draws']
    assert r['mc_per_scenario']==expected_reps and r['draws']==expected_draws
    for scenario in r['scenarios']:
        i=scenario['scenario']['id'];assert i not in seen
        rows=scenario['families'];assert len(rows)==len(inf['families'])*len(factors)
        assert {(x['family'],x['factor']) for x in rows}=={(f,k) for f in inf['families'] for k in factors}
        for row in rows:
            n=row['mc_n'];k=row['successes'];assert n==expected_reps and 0<=k<=n
            p=k/n;z=1.959963984540054
            lower=(p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
            assert abs(lower-row['wilson95'][0])<1e-12
            assert row['pass']==(p>=.95 and lower>=.93)
        seen[i]=scenario
    sources.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
assert set(seen)==set(range(32)),'all 32 preregistered scenarios required'
passing=[k for k in factors if all(row['pass'] for s in seen.values() for row in s['families'] if row['factor']==k)]
selected=min(passing) if passing else None
qualified=args.phase=='validation' and selected is not None
result={'status':'OBSERVED' if selected is not None else 'STATISTICAL_INCONCLUSIVE','phase':args.phase,
    'selected_factor':selected,'qualified':qualified,'contract_sha256':cfg['contract']['sha256'],
    'scenarios':[seen[i] for i in sorted(seen)],'sources':sources,'scientific_support':False,
    'interpretation':'synthetic simulation qualification only; no guarantee for actual weather errors',
    'on_failure':'no factor escalation, no formal analysis, no scientific rejection'}
with Path(args.out).open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ('scenarios','sources')},indent=2))
