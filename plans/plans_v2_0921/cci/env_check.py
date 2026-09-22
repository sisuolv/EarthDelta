#!/usr/bin/env python3
"""J0: real dependency install attempt + real xformers CUDA op + official import chain + asset readability.

Explicitly NOT a scientific S0: no model weights are loaded, no forward pass through the
real backbone is run. Writes plans_v2_0921/cci/env_check.py's own report incrementally so a
wall-clock timeout still leaves partial evidence on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def now_flush(*parts: str) -> None:
    print(*parts, flush=True)


def pip_install(pkg: str, timeout: int) -> dict:
    cmd = [sys.executable, '-m', 'pip', 'install', '--no-input', pkg]
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {'cmd': cmd, 'returncode': p.returncode, 'stdout': p.stdout[-4000:],
                'stderr': p.stderr[-4000:], 'elapsed_seconds': round(time.monotonic() - t0, 1)}
    except subprocess.TimeoutExpired:
        return {'cmd': cmd, 'status': 'TIMEOUT', 'timeout_seconds': timeout,
                'elapsed_seconds': round(time.monotonic() - t0, 1)}


def import_check(module: str, timeout: int = 60, env: dict | None = None) -> dict:
    snippet = ("import importlib,json,sys\n"
               "m=importlib.import_module(sys.argv[1])\n"
               "print(json.dumps({'version':str(getattr(m,'__version__','unknown'))}))")
    try:
        p = subprocess.run([sys.executable, '-c', snippet, module], capture_output=True,
                            text=True, timeout=timeout, env=env)
        return {'returncode': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {'status': 'TIMEOUT', 'timeout_seconds': timeout}


_XFORMERS_OP_SCRIPT = r"""
import json, sys
import torch
report = {"cuda_available": torch.cuda.is_available()}
try:
    from xformers.ops import memory_efficient_attention
    report["xformers_ops_import"] = True
except Exception as exc:
    report["xformers_ops_import"] = False
    report["xformers_ops_import_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report)); sys.exit(0)
if not report["cuda_available"]:
    report["forward_backward_ran"] = False
    report["reason"] = "no CUDA device visible in this process"
    print(json.dumps(report)); sys.exit(0)
try:
    device = torch.device("cuda")
    torch.manual_seed(0)
    b, h, s, d = 2, 4, 64, 32
    q = torch.randn(b, s, h, d, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(b, s, h, d, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(b, s, h, d, device=device, dtype=torch.float16, requires_grad=True)
    out = memory_efficient_attention(q, k, v)
    loss = out.float().pow(2).sum()
    loss.backward()
    report["forward_backward_ran"] = True
    report["output_shape"] = list(out.shape)
    report["output_finite"] = bool(torch.isfinite(out).all().item())
    report["grad_q_finite"] = bool(torch.isfinite(q.grad).all().item())
    report["grad_q_abs_sum"] = float(q.grad.abs().sum().item())
except Exception as exc:
    report["forward_backward_ran"] = False
    report["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report))
"""


def real_xformers_op_check(timeout: int = 180) -> dict:
    try:
        p = subprocess.run([sys.executable, '-c', _XFORMERS_OP_SCRIPT], capture_output=True,
                            text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'status': 'TIMEOUT', 'timeout_seconds': timeout}
    out = {'returncode': p.returncode, 'stderr': p.stderr[-3000:]}
    try:
        out['report'] = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        out['stdout_raw'] = p.stdout[-3000:]
    return out


_OFFICIAL_CHAIN_SCRIPT = r"""
import json
report = {}
try:
    from lightning import LightningModule
    report["lightning_LightningModule_import"] = True
except Exception as exc:
    report["lightning_LightningModule_import"] = False
    report["lightning_error"] = f"{type(exc).__name__}: {exc}"
try:
    from stormer.models.iterative_module import GlobalForecastIterativeModule
    report["GlobalForecastIterativeModule_import"] = True
except Exception as exc:
    report["GlobalForecastIterativeModule_import"] = False
    report["GlobalForecastIterativeModule_error"] = f"{type(exc).__name__}: {exc}"
try:
    from stormer.models.hub.stormer import Stormer
    report["Stormer_hub_import"] = True
except Exception as exc:
    report["Stormer_hub_import"] = False
    report["Stormer_hub_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report))
"""


def official_import_chain_check(timeout: int = 120, env: dict | None = None) -> dict:
    # Relies on PYTHONPATH already set by run_job.py to include the snapshot's
    # reference/stormer directory; does not repoint imports at any mutable path.
    try:
        p = subprocess.run([sys.executable, '-c', _OFFICIAL_CHAIN_SCRIPT], capture_output=True,
                            text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {'status': 'TIMEOUT', 'timeout_seconds': timeout}
    out = {'returncode': p.returncode, 'stderr': p.stderr[-3000:]}
    try:
        out['report'] = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        out['stdout_raw'] = p.stdout[-3000:]
    return out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--skip-install', action='store_true',
                         help='Skip pip install attempts; only probe what is already present.')
    args = parser.parse_args()
    root = args.run_dir
    report_path = root / 'j0_environment_report.json'
    report: dict = {'scope': 'J0_ENVIRONMENT_PREP_ONLY', 'scientific_s0_executed': False,
                     'weather_skill_measured': False}

    def save() -> None:
        with report_path.open('w') as f:
            json.dump(report, f, indent=2)
            f.write('\n')

    # J0 (pt-j66hv5ph) showed all 4 quick pip installs (lightning/timm/zarr/xarray,
    # 240s timeout each) burned their full timeout with zero completed output --
    # evidence the worker has no usable PyPI egress at all, not merely slow.
    # Rather than repeat that, extend PYTHONPATH to the repo's vendored .pydeps
    # (which now includes lightning/torchmetrics/lightning-utilities/timm/zarr/xarray,
    # added this window) for ONLY the specific checks below that need those packages.
    # baseline_versions and real_xformers_op_check deliberately keep the unmodified
    # image environment: .pydeps vendors its own numpy (1.26.4) that differs from the
    # image's numpy (1.24.4), and shadowing it during real torch CUDA tensor math or
    # the baseline version read would contaminate that evidence with an environment
    # the actual training/S0 jobs won't use. xformers is NOT vendored (211.8MB,
    # CUDA-11.8-targeted wheel vs the image's CUDA 12.4 torch build -- a real
    # ABI-compatibility risk on top of the network problem) and is intentionally left
    # unresolved: its absence is reported honestly below, not papered over.
    pydeps = args.assets / '.pydeps'
    pydeps_env: dict | None = None
    if pydeps.is_dir():
        pydeps_env = dict(os.environ)
        existing = pydeps_env.get('PYTHONPATH', '')
        pydeps_env['PYTHONPATH'] = str(pydeps) + (':' + existing if existing else '')
        report['pydeps_pythonpath_extended'] = str(pydeps)
    else:
        report['pydeps_pythonpath_extended'] = None

    now_flush('[env_check] baseline versions')
    report['baseline_versions'] = {m: import_check(m) for m in
                                    ('torch', 'torchvision', 'numpy', 'scipy', 'pytest')}
    save()

    if not args.skip_install:
        report['pip_install'] = {}
        for pkg, timeout in (('lightning', 240), ('timm', 240), ('zarr', 240),
                              ('xarray', 240), ('xformers', 2400)):
            now_flush(f'[env_check] pip install {pkg} (timeout={timeout}s)')
            report['pip_install'][pkg] = pip_install(pkg, timeout)
            save()

    now_flush('[env_check] post-install import versions (via .pydeps PYTHONPATH extension)')
    report['post_install_versions'] = {m: import_check(m, env=pydeps_env) for m in
                                        ('lightning', 'timm', 'zarr', 'xarray', 'xformers', 'xformers.ops')}
    save()

    now_flush('[env_check] real xformers CUDA forward/backward op (unmodified image environment)')
    report['xformers_real_cuda_op'] = real_xformers_op_check()
    save()

    now_flush('[env_check] official import chain (lightning.LightningModule / GlobalForecastIterativeModule / Stormer), via .pydeps PYTHONPATH extension')
    report['official_import_chain'] = official_import_chain_check(env=pydeps_env)
    save()

    now_flush('[env_check] asset readability')
    ckpt = args.assets / 'checkpoints/stormer_1.40625_patch_size_4.ckpt'
    npy = args.assets / 'scripts/s0_gate_inputs/jan2020_full.npy'
    norm_dir = args.assets / 'reference/stormer/normalization_constants'
    ckpt_info: dict = {'exists': ckpt.is_file()}
    if ckpt.is_file():
        ckpt_info['bytes'] = ckpt.stat().st_size
        now_flush('[env_check] hashing checkpoint (may take a few minutes for ~5.5GB)')
        ckpt_info['sha256'] = sha256_file(ckpt)
    report['assets'] = {
        'checkpoint_ps4': ckpt_info,
        'pilot_npy': {'exists': npy.is_file(),
                      'bytes': npy.stat().st_size if npy.is_file() else None},
        'normalization_constants_dir': {
            'exists': norm_dir.is_dir(),
            'entries': sorted(p.name for p in norm_dir.iterdir()) if norm_dir.is_dir() else None,
        },
    }
    save()

    import torch
    report['gpu'] = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_names': ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                          if torch.cuda.is_available() else []),
    }
    save()
    print(json.dumps(report, indent=2))

    op_ok = report.get('xformers_real_cuda_op', {}).get('report', {}).get('forward_backward_ran') is True
    chain_ok = report.get('official_import_chain', {}).get('report', {}).get(
        'lightning_LightningModule_import') is True
    return 0 if (op_ok and chain_ok) else 1


if __name__ == '__main__':
    raise SystemExit(main())
