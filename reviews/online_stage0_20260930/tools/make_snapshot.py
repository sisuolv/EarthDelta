import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

run=Path(__file__).resolve().parents[1];repo=run.parents[2]
name=sys.argv[1];snap=run/'code'/name;snap.mkdir()
archive=run/'code/t2_001_base.tar'
with tarfile.open(archive) as f:f.extractall(snap)
added=sorted((repo/'earthdelta/online').glob('*.py'))+sorted((repo/'tests').glob('test_online_*.py'))+sorted((repo/'scripts').glob('online_*.py'))
for p in added:
    d=snap/p.relative_to(repo);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
for name in ('.pydeps','.pydeps_wbx'):(snap/name).symlink_to(repo/name,target_is_directory=True)
for p in (repo/'reference').iterdir():
    if p.is_dir() and not (snap/'reference'/p.name).exists():(snap/'reference'/p.name).symlink_to(p,target_is_directory=True)
manifest={str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for sub in ('earthdelta','scripts','tests') for p in (snap/sub).rglob('*.py')}
out={'head':'d55ad70854bcf535b86c761b8d9cf7a7981374a0','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':manifest}
(run/'code'/f'{snap.name.upper()}_SNAPSHOT.json').write_text(json.dumps(out,indent=2)+'\n')
print(snap)
