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


def _tensor_stats(tensor) -> dict:
    import torch

    return {
        "max_abs": float(tensor.abs().max().item()),
        "mean_abs": float(tensor.abs().mean().item()),
        "finite": bool(torch.isfinite(tensor).all().item()),
    }


def _rollout_trace(
    model,
    replace_constant,
    x_norm,
    variables,
    interval,
    steps,
    inverse_input,
    inverse_diff,
    forward_input,
    label,
):
    """Run one wrapper arm and retain per-step states and local operators."""
    import torch

    interval_tensor = torch.tensor([interval], device=x_norm.device, dtype=x_norm.dtype)
    state = x_norm.clone()
    trace = []
    state_tensors = []
    norm_diff_tensors = []
    for step in range(1, steps + 1):
        norm_diff = model(state, variables, interval_tensor)
        norm_diff = replace_constant(norm_diff, variables)
        diff_raw = inverse_diff(norm_diff, interval)
        state_raw = inverse_input(state)
        next_raw = state_raw + diff_raw
        next_state = forward_input(next_raw)
        trace.append({
            "step": step,
            "state_norm": _tensor_stats(state),
            "norm_diff": _tensor_stats(norm_diff),
            "diff_raw": _tensor_stats(diff_raw),
            "state_raw": _tensor_stats(state_raw),
            "next_raw": _tensor_stats(next_raw),
            "next_norm": _tensor_stats(next_state),
        })
        state_tensors.append(state.detach().clone())
        norm_diff_tensors.append(norm_diff.detach().clone())
        state = next_state
    state_tensors.append(state.detach().clone())
    return {
        "label": label,
        "steps": trace,
        "final": state,
        "_state_tensors": state_tensors,
        "_norm_diff_tensors": norm_diff_tensors,
    }


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
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
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
            reference_dir = Path(config.reference_base_dir) / "upstream_reference_ps4_7fde884e_zd"
            reference_path = reference_dir / f"official_output_{key}.pt"
            official_reference_diff = None
            if reference_path.exists():
                reference = torch.load(reference_path, map_location=device)
                official_reference_diff = _max_abs(official_a, reference)
            result["rollouts"][key] = {
                "official_repeat_max_abs_diff": _max_abs(official_a, official_b),
                "bridge_repeat_max_abs_diff": _max_abs(bridge_a, bridge_b),
                "bridge_vs_official_max_abs_diff": _max_abs(bridge_a, official_a),
                "direct_model_max_abs_diff": _max_abs(direct_bridge, direct_official),
                "official_vs_frozen_reference_max_abs_diff": official_reference_diff,
                "one_step_postprocess": {
                    "model_diff_max_abs_diff": _max_abs(bridge_diff, official_diff),
                    "diff_denorm_max_abs_diff": _max_abs(bridge_diff_raw, official_diff_raw),
                    "raw_add_max_abs_diff": _max_abs(bridge_raw, official_raw),
                    "final_norm_max_abs_diff": _max_abs(bridge_next, official_next),
                },
            }
            if steps == 12:
                # Four arms share the same initial normalized tensor and model
                # weights.  This distinguishes model differences from
                # transform/accumulation differences at the first divergent
                # rollout step.
                official_arm = _rollout_trace(
                    official.net, official.replace_constant,
                    x_norm, variables, interval, steps,
                    official.reverse_inp_transform,
                    lambda y, i: official.reverse_diff_transform[i](y),
                    official.inp_transform, "official_model_official_transform",
                )
                official_bridge_transform = _rollout_trace(
                    official.net, bridge.normalization.replace_constant,
                    x_norm, variables, interval, steps,
                    bridge.normalization.denormalize,
                    bridge.normalization.denormalize_diff,
                    bridge.normalization.normalize, "official_model_bridge_transform",
                )
                bridge_official_transform = _rollout_trace(
                    bridge_model, official.replace_constant,
                    x_norm, variables, interval, steps,
                    official.reverse_inp_transform,
                    lambda y, i: official.reverse_diff_transform[i](y),
                    official.inp_transform, "bridge_model_official_transform",
                )
                bridge_arm = _rollout_trace(
                    bridge_model, bridge.normalization.replace_constant,
                    x_norm, variables, interval, steps,
                    bridge.normalization.denormalize,
                    bridge.normalization.denormalize_diff,
                    bridge.normalization.normalize, "bridge_model_bridge_transform",
                )
                arms = [official_arm, official_bridge_transform,
                        bridge_official_transform, bridge_arm]
                first_differences = {}
                for left in arms:
                    for right in arms:
                        if left["label"] >= right["label"]:
                            continue
                        diffs = []
                        first_state = None
                        first_norm_diff = None
                        for idx, (lstate, rstate) in enumerate(
                            zip(left["_state_tensors"], right["_state_tensors"])
                        ):
                            state_diff = _max_abs(lstate, rstate)
                            row = {
                                "state_step": idx,
                                "state_max_abs_diff": state_diff,
                            }
                            if idx > 0:
                                norm_diff = _max_abs(
                                    left["_norm_diff_tensors"][idx - 1],
                                    right["_norm_diff_tensors"][idx - 1],
                                )
                                row["norm_diff_max_abs_diff"] = norm_diff
                                if first_norm_diff is None and norm_diff > 0.0:
                                    first_norm_diff = idx
                            if first_state is None and state_diff > 0.0:
                                first_state = idx
                            diffs.append(row)
                        first_differences[f"{left['label']}__vs__{right['label']}"] = {
                            "first_state_divergence_step": first_state,
                            "first_norm_diff_divergence_step": first_norm_diff,
                            "steps": diffs,
                        }
                result["rollouts"][key]["four_arm_trace"] = {
                    "arms": [
                        {"label": arm["label"], "steps": arm["steps"]}
                        for arm in arms
                    ],
                    "comparison_note": (
                        "Per-step operator statistics and exact same-process tensor "
                        "differences are recorded for all four arms; this trace does "
                        "not change the gate verdict."
                    ),
                    "pair_index": first_differences,
                }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
