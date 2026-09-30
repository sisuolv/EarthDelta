#!/usr/bin/env python3
"""Synthetic tests with an explicit pre-open data boundary; run on ACP CPUs."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--existing',action='store_true')
    args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'tmp').mkdir()
    repo=Path(__file__).resolve().parents[1]
    import pytest
    events=[]
    afs=Path('/mnt/afs/260010168')
    roots=[(repo/x).resolve() for x in ('data','checkpoints')]
    def audit(event,values):
        if event!='open' or not isinstance(values[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(values[0])).resolve()
        if path.is_relative_to(out):return
        forbidden=any(path.is_relative_to(p) for p in roots)
        forbidden|=path.is_relative_to(afs) and (path.suffix in {'.npy','.npz','.ckpt','.pt','.pth','.nc'} or '.zarr' in str(path))
        if forbidden:
            events.append({'path':str(path),'action':'REFUSED_BEFORE_OPEN'})
            if args.existing:pytest.skip('synthetic CPU run refuses real arrays/checkpoints')
            raise PermissionError('new online tests cannot open real arrays/checkpoints')
    sys.addaudithook(audit)
    files=['tests'] if args.existing else [str(p.relative_to(repo)) for p in sorted((repo/'tests').glob('test_online_*.py'))]
    if not files:raise RuntimeError('no online tests collected')
    os.chdir(repo)
    code=pytest.main(files+['-q','-m','not slow','-rs','-p','no:cacheprovider',
                          '--basetemp='+str(out/'tmp/pytest'),'--junitxml='+str(out/'JUNIT.xml')])
    (out/'DATA_GUARD.json').write_text(json.dumps(events,indent=2)+'\n')
    return code


if __name__=='__main__':raise SystemExit(main())
