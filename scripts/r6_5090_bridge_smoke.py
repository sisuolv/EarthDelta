#!/usr/bin/env python3
"""Engineering-only RTX 5090 probe for EarthDelta's SDPA bridge.

This is not an upstream xformers parity test.  It verifies that the bridge
implementation itself can execute and backpropagate on sm_120 when the native
5090 image does not provide xformers.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket
import sys


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    d = Path(__file__).resolve().parent
    request_bytes = (d / "request.json").read_bytes()
    req = json.loads(request_bytes)
    out = d / "result.json"
    if out.exists():
        raise RuntimeError("result.json already exists")
    result = {
        "schema": "earthdelta-r6-5090-bridge-smoke/1",
        "status": "failed",
        "run_id": req["run_id"],
        "source_cci_hostname": req["source_cci_hostname"],
        "worker_hostname": socket.gethostname(),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "script_sha256": sha256(Path(__file__)),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "official_xformers_parity": False,
    }
    try:
        if result["worker_hostname"] == result["source_cci_hostname"]:
            raise RuntimeError("not running on ACP worker")
        import torch
        result.update({
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        })
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
        prop = torch.cuda.get_device_properties(0)
        result.update({"gpu_name": prop.name,
                       "compute_capability": list(torch.cuda.get_device_capability(0)),
                       "memory_gib": round(prop.total_memory / 2**30, 3)})
        if "5090" not in prop.name.upper():
            raise RuntimeError(f"expected RTX 5090, got {prop.name}")
        repo = Path(req["repo_root"])
        sys.path.insert(0, str(repo))
        from earthdelta.bridge import Stormer as BridgeStormer
        variables = [
            "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure", "geopotential_50", "geopotential_100",
            "geopotential_150", "geopotential_200",
        ]
        torch.manual_seed(20260927)
        model = BridgeStormer(in_img_size=(16, 32), variables=variables,
                              patch_size=2, hidden_size=64, depth=2,
                              num_heads=4, mlp_ratio=2.0).cuda()
        x = torch.randn(2, len(variables), 16, 32, device="cuda", requires_grad=True)
        interval = torch.full((2,), 0.6, device="cuda")
        y = model(x, variables, interval)
        loss = y.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(y).all() and torch.isfinite(x.grad).all() and
                      all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        result.update({"bridge_forward": True, "bridge_backward": True,
                       "bridge_output_shape": list(y.shape),
                       "bridge_finite": finite,
                       "bridge_parameter_count": sum(p.numel() for p in model.parameters())})
        if not finite:
            raise RuntimeError("bridge output or gradients non-finite")
        result["status"] = "ok"
    except Exception as exc:
        result.update({"error_type": type(exc).__name__, "error": str(exc)})
    result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
