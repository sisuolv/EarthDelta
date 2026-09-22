#!/usr/bin/env python3
"""Exercise current synthetic Stormer/LoRA forward/backward on each H100, not official S0."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

from earthdelta.bridge import Stormer, NormalizationContract, WeatherStepBridge, controlled_rollout, DEFAULT_VARIABLES
from earthdelta.contracts import ArtifactVersion, EditPlan, reference_plan
from earthdelta.lowrank import ExpertLoRA


def run_device(index: int) -> dict:
    torch.manual_seed(1729)
    device = torch.device(f'cuda:{index}')
    variables = DEFAULT_VARIABLES[:8]
    model = Stormer(in_img_size=(16, 32), variables=variables, patch_size=2,
                    hidden_size=64, depth=2, num_heads=4, mlp_ratio=2.0).to(device).eval()
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
    torch.nn.init.normal_(model.head.linear.weight, std=0.02)
    torch.nn.init.normal_(model.head.linear.bias, std=0.02)
    torch.nn.init.normal_(model.head.adaLN_modulation[-1].weight, std=0.1)
    torch.nn.init.normal_(model.head.adaLN_modulation[-1].bias, std=0.1)
    model.requires_grad_(False)
    norm = NormalizationContract(torch.zeros(8), torch.ones(8), {6: torch.zeros(8)},
                                 {6: torch.ones(8)}, variables)
    identity = ArtifactVersion('synthetic', 'none', 'synthetic-bank', norm.digest,
                               '16x32', 'patch2', 'launch-check', 'reference_after_hold')
    bridge = WeatherStepBridge(model, norm, identity)
    bank = ExpertLoRA(64, 64, num_experts=2, rank_per_expert=4).to(device)
    for up in bank.up:
        torch.nn.init.normal_(up.weight, std=0.1)
    x = torch.randn(1, 8, 16, 32, device=device)
    noedit = reference_plan(2)
    edit = EditPlan('fixed-coefficient', 2, (0.25, 0.0), hold_steps=1, rho=0.25)
    kwargs = dict(interval=6, steps=2, expert_loras={0: bank}, target_blocks=(0,))
    with torch.no_grad():
        before = controlled_rollout(bridge, x, variables, plan=noedit, **kwargs)
    edited = controlled_rollout(bridge, x, variables, plan=edit, **kwargs)
    edited.square().mean().backward()
    with torch.no_grad():
        after = controlled_rollout(bridge, x, variables, plan=noedit, **kwargs)
    grads = [p.grad for p in bank.parameters() if p.grad is not None]
    report = {
        'device': index, 'name': torch.cuda.get_device_name(index),
        'edit_max_abs_change': float((edited.detach() - before).abs().max()),
        'zero_restore_max_abs_error': float((after - before).abs().max()),
        'bank_grad_abs_sum': sum(float(g.abs().sum()) for g in grads),
        'bank_grads_finite': bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads),
        'backbone_has_no_grads': all(p.grad is None for p in model.parameters()),
        'remaining_forward_hooks': sum(len(m._forward_hooks) for m in model.modules()),
    }
    report['passed'] = bool(report['edit_max_abs_change'] > 0
                            and report['zero_restore_max_abs_error'] == 0
                            and report['bank_grad_abs_sum'] > 0 and report['bank_grads_finite']
                            and report['backbone_has_no_grads'] and report['remaining_forward_hooks'] == 0)
    return report


def main() -> int:
    reports = [run_device(i) for i in range(torch.cuda.device_count())]
    result = {'scope': 'SYNTHETIC_CURRENT_BRIDGE_GPU_CHECK_ONLY', 'official_s0': False,
              'weather_skill_measured': False, 'devices': reports,
              'passed': len(reports) == 4 and all(r['passed'] for r in reports)}
    with Path(sys.argv[1]).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
