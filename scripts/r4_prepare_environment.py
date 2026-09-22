#!/usr/bin/env python3
"""Install a pinned, private CUDA package overlay and execute official S0."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    source = Path(__file__).resolve().parents[1]
    site = a.run / "cuda_site"
    records = {"status": "RUNNING", "stages": [], "site": str(site)}
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(map(str, [source, source / "reference/stormer", site, a.assets / ".pydeps"]))
    env["OMP_NUM_THREADS"] = "4"

    def execute(name, argv, timeout):
        start = time.monotonic()
        with (a.out / (name + ".log")).open("x") as log:
            result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        records["stages"].append({"name": name, "returncode": result.returncode,
                                  "seconds": time.monotonic() - start})
        (a.out / "environment_stages.json").write_text(json.dumps(records, indent=2))
        if result.returncode:
            raise RuntimeError(f"{name} failed with {result.returncode}")

    try:
        execute("install", [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(a.run / "wheels"),
                            "--target", str(site), "torch==2.3.1", "torchvision==0.18.1", "xformers==0.0.27", "numpy==1.26.4"], 1200)
        probe = r'''
import json,torch,torchvision,xformers
from xformers.ops import memory_efficient_attention
from stormer.models.iterative_module import GlobalForecastIterativeModule
assert torch.cuda.is_available()
q=torch.randn(1,64,4,32,device='cuda',requires_grad=True)
o=memory_efficient_attention(q,q,q);o.square().sum().backward()
assert torch.isfinite(o).all() and torch.isfinite(q.grad).all()
print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,'xformers':xformers.__version__,'cuda':torch.version.cuda,'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'real_fp32_attention_forward_backward':True}))
'''
        execute("probe", [sys.executable, "-c", probe], 120)
        wrapper = str(source / "scripts/r4_run_strict_fp32.py")
        execute("export", [sys.executable, wrapper, str(source / "scripts/export_upstream_reference.py"),
                           "--config", str(a.run / "s0/gate_config.json")], 900)
        execute("gate", [sys.executable, wrapper, str(source / "scripts/s0_gate.py"),
                         "--config", str(a.run / "s0/gate_config.json"),
                         "--output-dir", str(a.run / "s0/gate_output")], 900)
        execute("reference", [sys.executable, str(source / "scripts/r4_value_pilot.py"),
                              "reference", "--run", str(a.run)], 600)
        records["status"] = "PASS"
    except Exception as exc:
        records["status"] = "BLOCKED"
        records["error"] = f"{type(exc).__name__}: {exc}"
    (a.out / "environment_stages.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2), flush=True)
    return 0 if records["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
