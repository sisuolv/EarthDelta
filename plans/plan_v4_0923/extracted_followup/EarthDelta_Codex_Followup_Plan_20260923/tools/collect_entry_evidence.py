#!/usr/bin/env python3
"""Read-only code inventory. Does not load weights, run tests or submit jobs."""
from __future__ import annotations
import argparse,datetime,hashlib,json,subprocess,sys
from pathlib import Path

IMPL="a3596e6b804e9d23b72d1247b08c47129d6b50b1"
BASE="d749a1c62521226df857587e08f7d067b0f15355"

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); repo=a.repo.resolve();out=a.out.resolve()
    if not repo.is_dir():p.error('repo directory does not exist')
    if out.exists():p.error('output must be new; existing evidence is never overwritten')
    out.mkdir(parents=True)
    def git(*args):
        r=subprocess.run(['git','-C',str(repo),*args],text=True,capture_output=True,timeout=60,check=False)
        return {'argv':['git',*args],'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    records={'head':git('rev-parse','HEAD'),'worktree':git('status','--porcelain=v1'),
             'diff_names':git('diff','--name-status',BASE,IMPL),
             'diff_stat':git('diff','--stat',BASE,IMPL),
             'current_tree':git('ls-tree','HEAD'),'implementation_tree':git('ls-tree',IMPL),
             'worktree_diff':git('diff','HEAD','--','earthdelta','scripts','tests','pyproject.toml')}
    (out/'worktree.diff').write_text(records['worktree_diff']['stdout'],encoding='utf-8')
    files=[]
    for folder in ('earthdelta','scripts','tests'):
        for f in sorted((repo/folder).rglob('*.py')):
            if f.is_file() and not f.is_symlink():
                files.append({'path':str(f.relative_to(repo)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    (out/'source_files.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
    ok=all(v['returncode']==0 for v in records.values())
    result={'status':'OBSERVED','verdict':'PASS_INVENTORY_ONLY' if ok else 'BLOCKED',
            'scope':'READ_ONLY_CODE_METADATA_NOT_S0_NOT_TESTS_NOT_SCIENCE',
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'repo':str(repo),'implementation_expected':IMPL,'comparison_base':BASE,
            'commands':records,'source_file_count':len(files),'python':sys.version,
            'worktree_diff_sha256':hashlib.sha256((out/'worktree.diff').read_bytes()).hexdigest(),
            'source_files_sha256':hashlib.sha256((out/'source_files.json').read_bytes()).hexdigest(),
            'tests_run':False,'models_loaded':False,'gpu_jobs_submitted':False,
            'next':'Resolve required_inputs; differing source or dirty worktree needs review, never reset automatically.'}
    (out/'entry_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'verdict':result['verdict'],'path':str(out/'entry_evidence.json')},ensure_ascii=False))
    return 0 if ok else 2
if __name__=='__main__':raise SystemExit(main())
