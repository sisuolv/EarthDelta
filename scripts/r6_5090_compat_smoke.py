#!/usr/bin/env python3
"""Strict native GPU compatibility probe for ACP workers.

The worker reads request.json next to this file and writes result.json exactly
once.  It intentionally uses the container's native torch/xformers stack and
the repository's official Stormer reference; no CUDA-site overlay is loaded.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> int:
    directory = Path(__file__).resolve().parent
    request_path = directory / "request.json"
    output_path = directory / "result.json"
    request_bytes = request_path.read_bytes()
    request = json.loads(request_bytes)
    if output_path.exists():
        raise RuntimeError("result.json already exists; use a fresh ACP run directory")

    result = {
        "schema": "earthdelta-r6-gpu-compat/1",
        "status": "failed",
        "run_id": request["run_id"],
        "expected_gpu": request["expected_gpu"],
        "source_cci_hostname": request["source_cci_hostname"],
        "worker_hostname": socket.gethostname(),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "script_sha256": _sha256(Path(__file__)),
        "started_utc": _utc(),
        "native_stack_only": True,
        "cuda_site_overlay_loaded": False,
    }
    try:
        if result["worker_hostname"] == result["source_cci_hostname"]:
            raise RuntimeError("probe ran on source CCI host rather than ACP worker")

        import torch

        result.update({
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "python_version": sys.version,
        })
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        count = torch.cuda.device_count()
        result["visible_gpu_count"] = count
        if count != 1:
            raise RuntimeError(f"expected exactly one visible GPU, found {count}")
        prop = torch.cuda.get_device_properties(0)
        name = str(prop.name)
        result.update({
            "gpu_name": name,
            "memory_gib": round(prop.total_memory / (1024 ** 3), 3),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_cuda_architectures": list(torch.cuda.get_arch_list()),
            "is_mig": "MIG" in name.upper(),
        })
        expected_gpu = str(request["expected_gpu"]).upper()
        if expected_gpu == "RTX 5090":
            if "RTX" not in name.upper() or "5090" not in name.upper():
                raise RuntimeError(f"expected RTX 5090, received {name}")
            if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
                raise RuntimeError("RTX 5090 compute capability is not sm_120")
        elif expected_gpu == "H100":
            if "H100" not in name.upper():
                raise RuntimeError(f"expected H100, received {name}")
            if prop.total_memory < 70 * (1024 ** 3):
                raise RuntimeError("H100 probe received less than 70 GiB of visible memory")
        else:
            raise RuntimeError(f"unsupported expected GPU {expected_gpu}")
        result["device_properties_ok"] = True

        # Record effective precision controls, then exercise native xformers.
        result["tf32_before"] = {
            "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
        }
        import xformers
        from xformers.ops import memory_efficient_attention

        result["xformers_version"] = str(getattr(xformers, "__version__", "unknown"))
        torch.manual_seed(20260925)
        q = torch.randn(2, 128, 4, 16, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        attn = memory_efficient_attention(q, k, v)
        attn_loss = attn.square().mean()
        attn_loss.backward()
        torch.cuda.synchronize()
        result.update({
            "xformers_attention_forward": True,
            "xformers_attention_backward": True,
            "xformers_attention_finite": bool(torch.isfinite(attn).all() and
                                                torch.isfinite(q.grad).all() and
                                                torch.isfinite(k.grad).all() and
                                                torch.isfinite(v.grad).all()),
            "xformers_attention_shape": list(attn.shape),
        })
        if not result["xformers_attention_finite"]:
            raise RuntimeError("xformers attention produced non-finite output/gradient")

        # Import the official reference from the checked-out repository.  This
        # path is deliberately independent of EarthDelta's CUDA-site overlay.
        repo_root = Path(request["repo_root"])
        reference_root = repo_root / "reference" / "stormer"
        if not reference_root.exists():
            raise RuntimeError(f"missing official reference at {reference_root}")
        sys.path.insert(0, str(reference_root))
        try:
            from stormer.models.hub.stormer import Stormer as OfficialStormer
            import stormer.models.hub.stormer as official_module
        finally:
            # Keep the imported module alive; only remove the path entry.
            sys.path.pop(0)
        result.update({
            "official_model_class": f"{OfficialStormer.__module__}.{OfficialStormer.__name__}",
            "official_stormer_sha256": _sha256(reference_root / "stormer" / "models" / "hub" / "stormer.py"),
            "official_iterative_sha256": _sha256(reference_root / "stormer" / "models" / "iterative_module.py"),
        })

        variables = [
            "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure", "geopotential_50", "geopotential_100",
            "geopotential_150", "geopotential_200",
        ]
        torch.manual_seed(20260926)
        model = OfficialStormer(
            in_img_size=[16, 32], variables=variables, patch_size=2,
            hidden_size=64, depth=2, num_heads=4, mlp_ratio=2.0,
        ).to("cuda")
        model.train()
        x = torch.randn(2, len(variables), 16, 32, device="cuda", dtype=torch.float32)
        interval = torch.full((2,), 0.6, device="cuda", dtype=torch.float32)
        model_out = model(x, variables, interval)
        model_loss = model_out.square().mean()
        model_loss.backward()
        torch.cuda.synchronize()
        finite_param_grads = all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        )
        result.update({
            "official_stormer_forward": True,
            "official_stormer_backward": True,
            "official_stormer_output_shape": list(model_out.shape),
            "official_stormer_output_finite": bool(torch.isfinite(model_out).all()),
            "official_stormer_gradients_finite": finite_param_grads,
            "official_stormer_parameter_count": sum(p.numel() for p in model.parameters()),
        })
        if not result["official_stormer_output_finite"] or not finite_param_grads:
            raise RuntimeError("official Stormer output or gradients are non-finite")

        torch.cuda.reset_peak_memory_stats()
        # A repeat forward checks deterministic native execution without using
        # output equality as a scientific parity claim.
        model.zero_grad(set_to_none=True)
        repeat = model(x.detach(), variables, interval)
        torch.cuda.synchronize()
        result["official_repeat_max_abs_diff"] = float((repeat - model_out.detach()).abs().max())
        result["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4)
        result["status"] = "ok"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    result["finished_utc"] = _utc()
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
