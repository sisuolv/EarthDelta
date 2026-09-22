#!/usr/bin/env python3
"""Real four-GPU smoke plus environment/data checks, explicitly not scientific S0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    invocation = json.loads((root / 'invocation.json').read_text())
    smoke = root / 'gpu_smoke'
    smoke.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__).with_name('gpu_smoke.py'), smoke / 'smoke.py')
    (smoke / 'request.json').write_text(json.dumps({
        'run_id': invocation['run_id'], 'expected_gpu': 'H100', 'expected_gpu_count': 4,
        'afs_marker': str(uuid.uuid4()), 'source_cci_hostname': invocation['source_cci_hostname'],
    }))
    gpu = subprocess.run([sys.executable, '-u', str(smoke / 'smoke.py')], timeout=300)
    report = {'scope': 'INFRASTRUCTURE_AND_ASSET_SAMPLE_ONLY', 'gpu_smoke_returncode': gpu.returncode,
              'scientific_s0_executed': False, 'weather_skill_measured': False, 'imports': {}, 'assets': {}}
    snippet = ('import importlib,json,sys; m=importlib.import_module(sys.argv[1]); '
               'print(json.dumps({"version":str(getattr(m,"__version__","unknown"))}))')
    for module in ('torch', 'torchvision', 'numpy', 'scipy', 'pytest', 'timm',
                   'xformers.ops', 'pytorch_lightning', 'zarr', 'xarray'):
        try:
            p = subprocess.run([sys.executable, '-c', snippet, module], capture_output=True,
                               text=True, timeout=40)
            report['imports'][module] = {'returncode': p.returncode, 'stdout': p.stdout[-1500:],
                                         'stderr': p.stderr[-1500:]}
        except subprocess.TimeoutExpired:
            report['imports'][module] = {'status': 'TIMEOUT'}
    checkpoint = args.assets / 'checkpoints/stormer_1.40625_patch_size_4.ckpt'
    report['assets']['checkpoint_ps4'] = {'exists': checkpoint.is_file(),
                                         'bytes': checkpoint.stat().st_size if checkpoint.is_file() else None,
                                         'deserialized': False}
    import numpy as np
    pilot = args.assets / 'scripts/s0_gate_inputs/jan2020_full.npy'
    if pilot.is_file():
        array = np.load(pilot, mmap_mode='r', allow_pickle=False)
        sample = np.asarray(array[0])
        report['assets']['pilot_npy'] = {'path': str(pilot), 'shape': list(array.shape),
                                       'dtype': str(array.dtype), 'first_sample_finite': bool(np.isfinite(sample).all()),
                                       'validation_scope': 'First sample only; no full content/time/target certification.'}
    else:
        report['assets']['pilot_npy'] = {'exists': False}
    with (root / 'environment_probe.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
    return gpu.returncode


if __name__ == '__main__':
    raise SystemExit(main())
