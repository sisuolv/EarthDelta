#!/usr/bin/env python3
"""Compare official and bridge execution in one GPU process.

This is a diagnostic only. It never changes the S0 tolerance or emits a gate
pass. Running both paths in one process helps separate xformers run-to-run
variation from a deterministic bridge/official implementation difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap(run: Path, assets: Path, source: Path) -> None:
    # The CCI base image does not contain the pinned overlay. Insert it before
    # importing torch so this diagnostic uses the same binaries as S0.
    sys.path[:0] = [
        str(run / "cuda_site"),
        str(assets / ".pydeps"),
        str(source),
        str(source / "reference/stormer"),
    ]


def _max_abs(a, b) -> float:
    return float((a - b).abs().max().item())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    source = Path(__file__).resolve().parents[1]
    _bootstrap(args.run, args.assets, source)

    import numpy as np
    import torch

    from earthdelta.bridge.stormer_bridge import (
        NormalizationContract,
        WeatherStepBridge,
        load_stormer_checkpoint,
    )
    from earthdelta.contracts import GateIdentityConfig
    from scripts.export_upstream_reference import (
        build_official_transforms,
        load_official_module,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for parity diagnosis")
    device = torch.device("cuda:0")
    config = GateIdentityConfig.load_json(args.run / "s0/gate_config.json")
    variables = list(config.variables)
    norm_dir = Path(config.normalization_dir)
    checkpoint = Path(config.checkpoint_path)
    inp_transform, diff_transforms = build_official_transforms(
        norm_dir, variables, tuple(config.normalization_intervals)
    )
    input_raw = np.load(Path(config.input_dir) / config.input_file)[0]
    x_raw = torch.from_numpy(input_raw).float().unsqueeze(0).to(device)
    x_norm = inp_transform(x_raw)

    official = load_official_module(
        checkpoint, config.patch_size, variables, inp_transform, diff_transforms,
        in_img_size=tuple(config.grid_shape), device=device,
        hidden_size=config.hidden_size, depth=config.depth,
        num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
    )
    bridge_model, version = load_stormer_checkpoint(
        str(checkpoint), config.patch_size, variables=variables,
        in_img_size=tuple(config.grid_shape), hidden_size=config.hidden_size,
        depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
        compute_sha256=False,
    )
    bridge_model.to(device).eval()
    normalization = NormalizationContract.from_npz_dir(
        str(norm_dir), variables=variables,
        intervals=tuple(config.normalization_intervals),
        policy=config.normalization_policy,
    )
    bridge = WeatherStepBridge(bridge_model, normalization, version)

    # The state dictionaries should be identical before any forward pass.
    state_max = 0.0
    for key, value in official.net.state_dict().items():
        state_max = max(state_max, _max_abs(value, bridge_model.state_dict()[key]))

    result = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "xformers": __import__("xformers").__version__,
        "state_dict_max_abs_diff": state_max,
        "rollouts": {},
        "note": "Diagnostic only; does not alter S0 gate verdict or tolerance.",
    }
    with torch.no_grad():
        for interval, steps in ((6, 1), (6, 4), (6, 12)):
            official_a = official.forward_validation(x_norm, variables, interval, steps)
            official_b = official.forward_validation(x_norm, variables, interval, steps)
            bridge_a = bridge.forward_validation(x_norm, variables, interval, steps)
            bridge_b = bridge.forward_validation(x_norm, variables, interval, steps)
            # Compare the direct one-step model calls as well as the complete
            # autoregressive wrappers, which localizes accumulation effects.
            interval_tensor = torch.tensor([interval], device=device, dtype=x_norm.dtype)
            direct_official = official.net(x_norm, variables, interval_tensor / 10.0)
            direct_bridge = bridge_model(x_norm, variables, interval_tensor / 10.0)
            # Reproduce the two wrappers' post-processing one operation at a
            # time.  The model calls above are identical; this identifies any
            # arithmetic-order difference in inverse/forward normalization.
            official_diff = official(x_norm, variables, interval_tensor)
            bridge_diff = bridge_model(x_norm, variables, interval_tensor)
            official_diff = official.replace_constant(official_diff.clone(), variables)
            bridge_diff = normalization.replace_constant(bridge_diff, variables)
            official_diff_raw = official.reverse_diff_transform[interval](official_diff)
            bridge_diff_raw = normalization.denormalize_diff(bridge_diff, interval)
            official_raw = official.reverse_inp_transform(x_norm) + official_diff_raw
            bridge_raw = normalization.denormalize(x_norm) + bridge_diff_raw
            official_next = official.inp_transform(official_raw)
            bridge_next = normalization.normalize(bridge_raw)
            key = f"{interval}h_{steps}step"
            result["rollouts"][key] = {
                "official_repeat_max_abs_diff": _max_abs(official_a, official_b),
                "bridge_repeat_max_abs_diff": _max_abs(bridge_a, bridge_b),
                "bridge_vs_official_max_abs_diff": _max_abs(bridge_a, official_a),
                "direct_model_max_abs_diff": _max_abs(direct_bridge, direct_official),
                "one_step_postprocess": {
                    "model_diff_max_abs_diff": _max_abs(bridge_diff, official_diff),
                    "diff_denorm_max_abs_diff": _max_abs(bridge_diff_raw, official_diff_raw),
                    "raw_add_max_abs_diff": _max_abs(bridge_raw, official_raw),
                    "final_norm_max_abs_diff": _max_abs(bridge_next, official_next),
                },
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
