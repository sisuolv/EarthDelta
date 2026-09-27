#!/usr/bin/env python3
"""Minimal ACP environment smoke test for the approved probe.

This deliberately does not import zarr or open any data.  It only verifies the
Python packages required by the probe and a live CUDA device.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import socket
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()
    result = {
        "schema": "earthdelta.probe.smoke.v1",
        "engine": args.engine,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "imports": {},
    }
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if not result["cuda_available"]:
            raise RuntimeError("CUDA is unavailable")
        result["device_name"] = torch.cuda.get_device_name(0)
        result["device_count"] = int(torch.cuda.device_count())
        for name in ("timm", "earthdelta"):
            importlib.import_module(name)
            result["imports"][name] = True
        # Force one CUDA allocation and synchronize, catching broken runtime
        # bindings before any stage job is submitted.
        x = torch.ones((8,), device="cuda")
        result["cuda_sum"] = float(x.sum().detach().cpu())
        torch.cuda.synchronize()
        result["status"] = "SUCCEEDED"
        rc = 0
    except Exception as exc:
        result["status"] = "BLOCKED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        rc = 2
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
