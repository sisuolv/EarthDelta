import sys
sys.dont_write_bytecode = True
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('delivery_validator',ROOT/'tools/validate_package.py')
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)

class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'pkg'
        shutil.copytree(ROOT,self.root)
    def read(self,p): return json.loads((self.root/p).read_text())
    def write(self,p,o): (self.root/p).write_text(json.dumps(o,ensure_ascii=False),encoding='utf-8')
    def result(self): return v.validate(self.root,True)
    def test_valid_delivery(self): self.assertTrue(self.result()['passed'])
    def test_unauthorized_task_unlock_rejected(self):
        s=self.read('codex_tasks.json');s['approved_task_ids'].append('R2-P0-02');self.write('codex_tasks.json',s)
        self.assertFalse(self.result()['passed'])
    def test_dependency_cycle_rejected(self):
        s=self.read('codex_tasks.json');s['tasks'][0]['depends_on']=['R2-P0-04'];self.write('codex_tasks.json',s)
        r=self.result();self.assertFalse(r['passed']);self.assertFalse(next(c['passed'] for c in r['checks'] if c['check']=='dag_acyclic'))
    def test_fabricated_threshold_rejected(self):
        s=self.read('configs/pilot_config.template.json');s['statistics']['delta_gain_min']=.02;self.write('configs/pilot_config.template.json',s)
        self.assertFalse(self.result()['passed'])
    def test_gpu_authorization_tamper_rejected(self):
        s=self.read('configs/authorization.template.json');s['approved_operations']['gpu_execution']=True;self.write('configs/authorization.template.json',s)
        self.assertFalse(self.result()['passed'])
    def test_missing_required_file_rejected(self):
        (self.root/'STOP_CONDITIONS.md').unlink();self.assertFalse(self.result()['passed'])
    def test_non_git_preflight_fails_without_fake_head(self):
        empty=Path(self.tmp.name)/'not_a_repo';empty.mkdir()
        out=Path(self.tmp.name)/'preflight.json'
        r=subprocess.run([sys.executable,str(ROOT/'tools/repo_preflight.py'),'--repo',str(empty),'--out',str(out)],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,2)
        s=json.loads(out.read_text());self.assertEqual(s['verdict'],'BLOCKED_PREFLIGHT');self.assertNotIn('local_head',s)
    def test_preflight_refuses_overwrite(self):
        out=Path(self.tmp.name)/'preflight.json';out.write_text('ORIGINAL')
        r=subprocess.run([sys.executable,str(ROOT/'tools/repo_preflight.py'),'--repo',self.tmp.name,'--out',str(out)],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,2);self.assertEqual(out.read_text(),'ORIGINAL')
    def test_main_command_drift_rejected(self):
        p=self.root/'EarthDelta_Codex_Execution_Plan.md';p.write_text(p.read_text().replace('python3 "$PACKAGE/tools/repo_preflight.py" --repo "$REPO" --out "$RUN/preflight.json"','REMOVED_COMMAND'))
        self.assertFalse(self.result()['passed'])
    def test_integrity_detects_tamper(self):
        import hashlib
        mapping={p.relative_to(self.root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256.json'}
        self.write('MANIFEST.sha256.json',{'files':mapping})
        self.assertTrue(v.validate(self.root,False)['passed'])
        p=self.root/'FINAL_RESEARCH_DECISION.md';p.write_text(p.read_text()+'\nTAMPER\n')
        self.assertFalse(v.validate(self.root,False)['passed'])

if __name__=='__main__':unittest.main(verbosity=2)
