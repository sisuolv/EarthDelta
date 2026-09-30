import argparse
import hashlib
import json
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--field',choices=['f0_bank','output_fit','gradients','parameter_fit'],required=True)
ap.add_argument('--receipt',required=True);a=ap.parse_args()
p=Path(a.receipt);raw=p.read_bytes();receipt=json.loads(raw)
if receipt.get('status')!='OBSERVED':raise RuntimeError('only completed observed receipts may release dependent stages')
cfg=json.loads(Path(a.input).read_text());cfg[a.field]={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()}
with Path(a.out).open('x') as f:f.write(json.dumps(cfg,indent=2)+'\n')
