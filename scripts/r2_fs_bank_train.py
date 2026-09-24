#!/usr/bin/env python3
"""Stage-4 entry point: fit/freeze Fs, then train the K=4 dynamic expert bank.

Implements the runnable half of `plans/plans_v2_0921/CLAUDE_EXECUTION_PLAN.md`
sections 6.1-6.2. One script; every stage can be run on its own:

    --stage fs_fit          fit Fs on bank_fit-role admitted samples
    --stage fs_freeze       merge Fs into a backbone copy, re-record identity,
                            re-verify zero-edit equivalence AGAINST Fs, and
                            record F0's background score separately
    --stage bank_train      train exactly ONE dynamic expert (--expert-index)
                            on top of the frozen Fs-merged backbone
    --stage profile         measure one real training step on the target blocks
                            and derive the K=4/rank=4 vs K=2/rank=2 decision
    --stage horizon_check   test differentiable backprop through the 4 x 6h
                            hold window and emit a fallback spec if infeasible
    --stage bank_assemble   (FP-04) build bank.pt from the pinned initial bank
                            and the four BANK-QUAL-v1 PASS expert files
    --stage bank_verify     (FP-04, independent process) reload bank.pt and
                            run the exact assembly-equivalence checks A1-A5

FP-04: every stage past Fs consumes the ONE certified Fs reference (the FP-03
`certify/fs/` bundle). On its own -- real or synthetic -- such a stage runs only
after `authorize_certified_fs` binds the reference manifest, the FS_SELECTED
certify decision, the adapter and the merged backbone by SHA-256, and
`load_certified_fs` has re-verified the merged/base/static-adapter/version
digests of what it loaded. A worker can therefore not re-fit, or pick up, any
other Fs (the historical per-worker re-fit, e.g. job pt-cdj1s2le, is refused).

`--expert-index` is what makes the plan's "4 卡独立进程各训一个专家" work: four
processes, one per GPU, each training a single expert with every other expert's
factors frozen, each writing its own shard. This script never shares a bridge
between processes and never runs two rollouts against one backbone.

THIS SCRIPT NEVER SUBMITS OR POLLS AN ACP/CCI JOB. It is the payload a job
wraps, not a scheduler. Submission is the parent's responsibility; see
`plans/plans_v2_0921/cci/submit_job.py`.

Two data paths:

  * `--synthetic` builds a tiny CPU-testable Stormer (16x32 grid, 8 variables,
    hidden 64, depth 2, patch 2) matching the `cci/gpu_bank_check.py` precedent.
    Everything it produces is tagged `synthetic: true` and is evidence that the
    code paths run, NOT a weather result, NOT a capacity finding.
  * the real path takes `--config` (a `gate_config.json` in the same format
    `scripts/s0_gate.py` consumes) and `--admission` (the JSON record written by
    `scripts/r2_admission_gate.py`). Samples are read ONLY from admitted rows;
    there is no path here that opens a store the admission gate did not admit,
    which is what keeps the empty 2016/2017 stores and the absent 2021 out.

Examples:

    # CPU smoke over every stage, tiny synthetic model
    python scripts/r2_fs_bank_train.py --stage all --synthetic \\
        --mode gradient_check --max-updates 6 --output-dir artifacts/smoke

    # real Fs fit (GPU), 32-update gradient/throughput/direction check
    python scripts/r2_fs_bank_train.py --stage fs_fit \\
        --config artifacts/.../gate_config.json \\
        --admission artifacts/.../admission_bank_fit.json \\
        --data-role bank_fit --mode gradient_check \\
        --output-dir artifacts/round2_next/<run>/fs_fit

    # one expert per GPU process
    python scripts/r2_fs_bank_train.py --stage bank_train --expert-index 2 \\
        --num-experts 4 --rank-per-expert 4 --mode formal ...

Exit codes:
    0: every requested stage completed and passed its own checks
    1: a stage ran and failed a check
    2: the script could not run (bad arguments, missing inputs)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch  # noqa: E402

# Fs eligibility is a training-quality claim, so the optimizer must see the
# same strict FP32 arithmetic used by the certified S0/reference path.  H100
# TF32 is finite but can change the low-rank rollout gradients enough for a
# formal fit to diverge while still passing merge-equivalence.  Pin the
# process before model construction; this is recorded by the job environment
# and does not alter the S0 tolerance.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from earthdelta import bank_training as bt  # noqa: E402
from earthdelta import fs_protocol as fp  # noqa: E402
from earthdelta import static_adapter as sa  # noqa: E402
from earthdelta.bridge import (  # noqa: E402
    DEFAULT_VARIABLES,
    NormalizationContract,
    Stormer,
    WeatherStepBridge,
    _compute_file_sha256,
    load_stormer_checkpoint,
)
from earthdelta.contracts import ArtifactVersion, GateIdentityConfig  # noqa: E402
from earthdelta.pilot_contract import DATA_ROLE_VALUES, DataRole  # noqa: E402

#: Stated so a reader (and a grep) can confirm it: this entry point performs no
#: job submission, no job polling and no scheduler interaction of any kind.
SUBMITS_JOBS = False
POLLS_JOBS = False

STAGES = ("fs_fit", "fs_panel", "fs_freeze", "fs_verify", "bank_train", "profile",
          "horizon_check", "bank_assemble", "bank_verify", "all")

#: FP-04 (replaces FP-03's unconditional LOCKED_UNTIL_FP04 refusal): every stage
#: that consumes the certified Fs. Run on its own, each one is refused -- real
#: or synthetic -- unless `authorize_certified_fs` binds it to the certified
#: bundle; only the synthetic in-process `--stage all` chain may use the Fs it
#: just froze itself.
CERTIFIED_FS_STAGES = ("bank_train", "profile", "horizon_check", "bank_assemble",
                       "bank_verify")
BANK_STAGES = CERTIFIED_FS_STAGES

#: Protocol kinds: an Fs stage runs only under an Fs protocol, a bank stage only
#: under a bank protocol (FP-03 protocol files carry no kind and are "fs").
PROTOCOL_KIND_FS = "fs"
PROTOCOL_KIND_BANK = "bank"

REFERENCE_MANIFEST_SCHEMA = "ed-fs-reference-manifest/1"

_FS_REFERENCE_DECLARED = ("fs_reference_manifest", "fs_certify_decision", "fs_adapter",
                          "fs_merged_backbone")
_BANK_DECLARED = ("stage", "seed", "rank_per_expert", "num_experts", "target_blocks",
                  "hold_steps", "a0", "rho", "data_role", "device", "config", "admission",
                  "s0_certificate", "limit") + _FS_REFERENCE_DECLARED

#: Fields a protocol configuration MUST declare for each real stage, so that an
#: under-declared configuration cannot leave a free parameter to the CLI.
REQUIRED_DECLARED = {
    "fs_fit": ("stage", "mode", "max_updates", "learning_rate", "seed", "rank_per_expert",
               "target_blocks", "train_steps", "lead_steps", "panel_lead_steps",
               "hold_steps", "data_role", "device", "config", "admission",
               "panel_admission", "s0_certificate", "limit"),
    "fs_panel": ("stage", "seed", "rank_per_expert", "target_blocks", "train_steps",
                 "lead_steps", "panel_lead_steps", "data_role", "device", "config",
                 "admission", "panel_admission", "s0_certificate", "fs_adapter",
                 "fs_adapter_sha256", "limit"),
    "fs_freeze": ("stage", "seed", "rank_per_expert", "target_blocks", "num_experts",
                  "hold_steps", "data_role", "device", "config", "admission",
                  "s0_certificate", "limit"),
    "fs_verify": ("stage", "seed", "rank_per_expert", "target_blocks", "num_experts",
                  "hold_steps", "data_role", "device", "config", "admission",
                  "s0_certificate", "limit"),
    "bank_train": _BANK_DECLARED + ("mode", "max_updates", "learning_rate", "expert_index",
                                    "train_steps", "lead_steps", "panel_lead_steps"),
    "profile": _BANK_DECLARED + ("mode", "max_updates", "learning_rate", "expert_index",
                                 "train_steps", "lead_steps", "profile_steps", "profile_warmup",
                                 "memory_budget_gib", "memory_safety_fraction",
                                 "max_seconds_per_update"),
    "horizon_check": _BANK_DECLARED + ("mode", "max_updates", "learning_rate", "expert_index",
                                       "train_steps", "lead_steps", "memory_budget_gib",
                                       "memory_safety_fraction", "max_seconds_per_update",
                                       "max_differentiable_steps"),
    "bank_assemble": _BANK_DECLARED + ("panel_lead_steps",),
    "bank_verify": _BANK_DECLARED + ("panel_lead_steps",),
}


# =============================================================================
# Synthetic fixture (CPU smoke; matches cci/gpu_bank_check.py geometry)
# =============================================================================

SYNTHETIC_GRID = (16, 32)
SYNTHETIC_VARIABLES = 8
SYNTHETIC_HIDDEN = 64
SYNTHETIC_DEPTH = 2
SYNTHETIC_HEADS = 4
SYNTHETIC_PATCH = 2
SYNTHETIC_MLP_RATIO = 2.0


def build_synthetic_bridge(
    device: torch.device, seed: int = 20260921
) -> Tuple[WeatherStepBridge, List[str], torch.Tensor]:
    """A tiny Stormer + unit normalization, frozen exactly as production is."""
    torch.manual_seed(int(seed))
    variables = list(DEFAULT_VARIABLES[:SYNTHETIC_VARIABLES])
    model = Stormer(
        in_img_size=SYNTHETIC_GRID,
        variables=variables,
        patch_size=SYNTHETIC_PATCH,
        hidden_size=SYNTHETIC_HIDDEN,
        depth=SYNTHETIC_DEPTH,
        num_heads=SYNTHETIC_HEADS,
        mlp_ratio=SYNTHETIC_MLP_RATIO,
    )
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
    torch.nn.init.normal_(model.head.linear.weight, std=0.02)
    torch.nn.init.normal_(model.head.linear.bias, std=0.02)
    model.to(device)
    model.requires_grad_(False)
    model.eval()

    n = SYNTHETIC_VARIABLES
    normalization = NormalizationContract(
        inp_mean=torch.zeros(n),
        inp_std=torch.ones(n),
        diff_mean={6: torch.zeros(n)},
        diff_std={6: torch.ones(n)},
        variables=variables,
    )
    version = ArtifactVersion(
        backbone="stormer_synthetic_cpu_smoke",
        static_adapter="none",
        edit_bank="none",
        normalization="pending",
        grid=f"{SYNTHETIC_GRID[0]}x{SYNTHETIC_GRID[1]}",
        projection=f"patch{SYNTHETIC_PATCH}",
        split="synthetic",
        continuation="reference_after_hold",
    )
    bridge = WeatherStepBridge(model, normalization, version)
    lat = torch.linspace(-88.0, 88.0, SYNTHETIC_GRID[0])
    return bridge, variables, lat


def build_synthetic_samples(
    bridge: WeatherStepBridge,
    lead_steps: Sequence[int],
    *,
    n_samples: int = 4,
    device: Optional[torch.device] = None,
    seed: int = 20260921,
    months: Sequence[Tuple[int, int]] = ((2015, 2), (2015, 5), (2015, 8), (2015, 11)),
) -> List[sa.TrainingSample]:
    """Synthetic bank_fit samples spread over distinct calendar months.

    The months are spread so `assign_diversity_groups` has the >= K distinct
    months its pre-declared rule requires. The values are random: these samples
    exercise code paths and can never support a weather claim.
    """
    from datetime import datetime, timezone

    generator = torch.Generator().manual_seed(int(seed))
    samples: List[sa.TrainingSample] = []
    interval = 6 * 3600
    for i in range(int(n_samples)):
        year, month = months[i % len(months)]
        # mid-month, so history and a 72h window stay inside the month
        issue_dt = datetime(year, month, 15, 0, 0, 0, tzinfo=timezone.utc)
        issue_time = int(issue_dt.timestamp())
        x_raw = torch.randn(
            1, SYNTHETIC_VARIABLES, *SYNTHETIC_GRID, generator=generator
        )
        targets = {
            int(step): x_raw
            + 0.3 * torch.randn(
                1, SYNTHETIC_VARIABLES, *SYNTHETIC_GRID, generator=generator
            )
            for step in lead_steps
        }
        if device is not None:
            x_raw = x_raw.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
        samples.append(sa.TrainingSample(
            x_norm=bridge.normalization.normalize(x_raw),
            targets_raw=targets,
            issue_id=f"iss_synthetic_{i:03d}",
            issue_time=issue_time,
            valid_time=issue_time + max(int(s) for s in lead_steps) * interval,
            history_time=issue_time - interval,
            split_id="train",
            data_role=DataRole.BANK_FIT.value,
            source="synthetic",
            time_utc=issue_dt.isoformat(),
        ))
    return samples


# =============================================================================
# Real fixture: gate_config.json + admission record
# =============================================================================

def build_real_bridge(
    config_path: Path, device: torch.device
) -> Tuple[WeatherStepBridge, List[str], Optional[torch.Tensor], GateIdentityConfig]:
    """Load the production backbone from the same config the S0 gate consumes."""
    config = GateIdentityConfig.load_json(config_path)

    checkpoint = Path(config.checkpoint_path)
    if not checkpoint.exists():
        raise SystemExit(
            f"ERROR: checkpoint not found at {checkpoint}; the config names a "
            "backbone this machine does not have."
        )
    if config.expected_checkpoint_sha256:
        actual = _compute_file_sha256(str(checkpoint))
        if not config.validate_checkpoint_sha256(actual):
            raise SystemExit(
                "ERROR: checkpoint SHA-256 mismatch. Expected "
                f"{config.expected_checkpoint_sha256}, computed {actual}. "
                "Refusing to train Fs on a backbone that is not the one the "
                "frozen identity names."
            )

    model, version = load_stormer_checkpoint(
        str(checkpoint),
        patch_size=config.patch_size,
        variables=list(config.variables),
        in_img_size=tuple(config.grid_shape),
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
    )
    model.to(device)
    model.requires_grad_(False)
    model.eval()

    normalization = NormalizationContract.from_npz_dir(
        config.normalization_dir,
        variables=list(config.variables),
        intervals=tuple(config.normalization_intervals),
        policy=config.normalization_policy,
    )
    bridge = WeatherStepBridge(model, normalization, version)

    lat = _load_latitude(config)
    return bridge, list(config.variables), lat, config


def _load_latitude(config: GateIdentityConfig) -> Optional[torch.Tensor]:
    """Latitude for the area weight: the gate's own lat.npy, else the grid."""
    if config.input_dir:
        candidate = Path(config.input_dir) / "lat.npy"
        if candidate.exists():
            import numpy as np

            return torch.as_tensor(np.load(candidate), dtype=torch.float64)
    try:
        from earthdelta.data.pull_wb2 import get_stormer_target_grid

        lat, _ = get_stormer_target_grid()
        return torch.as_tensor(lat, dtype=torch.float64)
    except Exception:  # noqa: BLE001 - area weighting is then reported as absent
        return None


def load_real_samples(
    admission_path: Path,
    bridge: WeatherStepBridge,
    lead_steps: Sequence[int],
    *,
    limit: Optional[int],
    device: torch.device,
    certification_out: Optional[Dict[str, Any]] = None,
) -> List[sa.TrainingSample]:
    record = json.loads(Path(admission_path).read_text())
    # Every real path is fail-closed on the admission certificate (FP-03 Q0).
    return sa.load_admitted_samples(
        record, bridge, lead_steps=lead_steps, limit=limit, device=device,
        require_certified=True, certification_out=certification_out,
    )


# =============================================================================
# Stages
# =============================================================================

def _save_json(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote {path}", flush=True)


def _count_hooks(model: torch.nn.Module) -> int:
    return sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._backward_hooks)
        for m in model.modules()
    )


def _hidden_size(ctx: Dict[str, Any]) -> int:
    return int(ctx["bridge"].model.blocks[ctx["target_blocks"][0]].attn.proj.in_features)


def _device_of(ctx: Dict[str, Any]) -> torch.device:
    return next(ctx["bridge"].model.parameters()).device


def _compute_panels(ctx: Dict[str, Any], initial, final) -> Dict[str, Any]:
    """F0 / L0 (common initial checkpoint) / L1 panels on train and holdout."""
    leads = tuple(ctx["panel_lead_steps"])

    def panel(samples, adapters):
        return sa.fs_panel_losses(
            ctx["bridge"], samples, ctx["objective"], lead_steps=leads,
            fs_adapters=adapters, target_blocks=ctx["target_blocks"],
            variables=ctx["variables"],
        )

    out: Dict[str, Any] = {
        "schema": "ed-fs-panels/1",
        "lead_steps": list(leads),
        "lead_hours": [int(s) * 6 for s in leads],
        "reduction": "float64 full_objective_loss per issue per lead; no grad",
    }
    groups = {"train": ctx["samples"]}
    if ctx.get("panel_samples"):
        groups["holdout"] = ctx["panel_samples"]
    equal = {}
    for name, samples in groups.items():
        out[name] = {
            "issue_ids": [s.issue_id for s in samples],
            "F0": panel(samples, None),
            "L0": panel(samples, initial),
            "L1": panel(samples, final),
        }
        equal[name] = out[name]["L0"] == out[name]["F0"]
    out["L0_equals_F0_bitwise"] = equal
    return out


def _q2_thresholds(ctx: Dict[str, Any]) -> Dict[str, Any]:
    rule = ((ctx.get("protocol") or {}).get("qualification_rule") or {}).get("q2") or {}
    return {
        "clip_events_max": int(rule.get("clip_events_max", 0)),
        "per_visit_ratio_max": float(rule.get("per_visit_ratio_max", 1.25)),
        "epoch_onset_factor": float(rule.get("epoch_onset_factor", 1.02)),
    }


def stage_fs_fit(args, ctx: Dict[str, Any]) -> bool:
    bridge = ctx["bridge"]
    spec = ctx["objective"]
    samples = ctx["samples"]

    hidden = _hidden_size(ctx)
    adapters = sa.build_fs_adapter(
        hidden,
        ctx["target_blocks"],
        rank_per_expert=int(args.rank_per_expert),
        seed=int(args.seed),
    )
    # The common initial checkpoint L0, kept untouched for the panels.
    initial = {block: copy.deepcopy(lora) for block, lora in adapters.items()}
    initial_digest = sa.static_adapter_digest(adapters)
    expected_initial = ctx.get("expected_initial_adapter_digest")
    config = sa.FsFitConfig(
        mode=args.mode,
        max_updates=int(args.max_updates),
        learning_rate=float(args.learning_rate),
        train_steps=int(args.train_steps),
        lead_steps=tuple(ctx["lead_steps"]),
        target_blocks=tuple(ctx["target_blocks"]),
        rank_per_expert=int(args.rank_per_expert),
        seed=int(args.seed),
        config_id=args.config_id,
        protocol_sha256=args.protocol_sha256,
    )
    backbone_before = sa.state_dict_digest(bridge.model)
    hooks_before = _count_hooks(bridge.model)
    adapters, record = sa.fit_static_adapter(
        bridge, samples, spec, config, fs_adapters=adapters, variables=ctx["variables"]
    )
    backbone_after = sa.state_dict_digest(bridge.model)
    hooks_after = _count_hooks(bridge.model)
    ctx["fs_adapters"] = adapters
    ctx["fs_fit_record"] = record
    diagnostics: Dict[str, Any] = {
        "initial_adapter_digest": initial_digest,
        "expected_initial_adapter_digest": expected_initial,
        "initial_adapter_digest_matches": (
            None if expected_initial is None else initial_digest == expected_initial),
        "record_initial_digest_matches_built": record.initial_adapter_digest == initial_digest,
        "backbone_digest_before": backbone_before,
        "backbone_digest_after": backbone_after,
        "backbone_digest_unchanged": backbone_before == backbone_after,
        "hooks_before": hooks_before,
        "hooks_after": hooks_after,
        "no_leftover_hooks": hooks_before == hooks_after,
        "grad_A_update0": record.grad_norms_A[0] if record.grad_norms_A else None,
        "grad_B_update0": record.grad_norms_B[0] if record.grad_norms_B else None,
        "grad_A_update1": record.grad_norms_A[1] if len(record.grad_norms_A) > 1 else None,
        "clip_events": record.clip_events,
    }

    print(f"  updates       : {record.n_updates} (cap {config.max_updates})", flush=True)
    print(f"  loss          : {record.initial_loss} -> {record.final_loss}", flush=True)
    direction = record.loss_direction
    print(
        f"  direction     : {direction['n_decreased']}/"
        f"{direction['n_samples_with_repeats']} sample(s) improved "
        f"(per-sample first-vs-last; raw first/last update loss decreased="
        f"{record.loss_decreased})",
        flush=True,
    )
    print(f"  eligible      : {record.eligible} ({record.ineligible_reason})", flush=True)
    print(f"  clip events   : {record.clip_events}", flush=True)

    out = Path(args.output_dir) if args.output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        # CPU copies: the saved bytes must not depend on which device fitted them.
        torch.save({b: {k: v.detach().cpu() for k, v in lora.state_dict().items()}
                    for b, lora in initial.items()}, out / "fs_adapter_initial.pt")
        torch.save({b: {k: v.detach().cpu() for k, v in lora.state_dict().items()}
                    for b, lora in adapters.items()}, out / "fs_adapter.pt")
        print(f"  wrote {out / 'fs_adapter.pt'}", flush=True)
        adapter_sha = fp.file_sha256(out / "fs_adapter.pt")
        reloaded, _ = sa.load_fs_adapter(
            out / "fs_adapter.pt", hidden, ctx["target_blocks"],
            rank_per_expert=int(args.rank_per_expert), expected_sha256=adapter_sha,
        )
        diagnostics.update(
            adapter_sha256=adapter_sha,
            initial_adapter_sha256=fp.file_sha256(out / "fs_adapter_initial.pt"),
            reload_digest_matches=(sa.static_adapter_digest(reloaded)
                                   == record.final_adapter_digest),
        )

    panels = None
    if ctx.get("panel_lead_steps"):
        device = _device_of(ctx)
        initial_on_device = {b: lora.to(device) for b, lora in initial.items()}
        panels = _compute_panels(ctx, initial_on_device, adapters)
        stability = sa.training_stability(
            record, sa._panel_column(panels["train"]["L0"], 24),
            horizon=int(args.max_updates), **_q2_thresholds(ctx),
        )
        diagnostics["stability"] = stability
        diagnostics["L0_equals_F0_bitwise"] = panels["L0_equals_F0_bitwise"]
        diagnostics["nonzero_response"] = panels["train"]["L1"] != panels["train"]["L0"]
        train24 = sa.panel_ratio_summary(panels["train"]["L0"], panels["train"]["L1"], 24)
        diagnostics["train_panel_mean_ratio_24h"] = train24["mean_ratio"]
        print(f"  stability     : stable={stability['stable']} clip={stability['q2']['clip_events']} "
              f"max_ratio={stability['q2']['max_per_visit_ratio']:.4f} "
              f"onset_update={stability['q2']['onset_update']}", flush=True)
        print(f"  panel 24h     : mean L1/L0 = {train24['mean_ratio']:.6f} "
              f"max r_i = {train24['max_ratio']:.6f}", flush=True)
        _save_json(out / "fs_panels.json" if out else None, panels)

    hard = {
        "backbone_digest_unchanged": diagnostics["backbone_digest_unchanged"],
        "no_leftover_hooks": diagnostics["no_leftover_hooks"],
        "record_initial_digest_matches_built": diagnostics["record_initial_digest_matches_built"],
    }
    if expected_initial is not None:
        hard["initial_adapter_digest_matches"] = diagnostics["initial_adapter_digest_matches"]
    if out is not None:
        hard["reload_digest_matches"] = diagnostics["reload_digest_matches"]
    diagnostics["hard_checks"] = hard
    diagnostics["hard_checks_passed"] = all(v is True for v in hard.values())
    if not diagnostics["hard_checks_passed"]:
        print(f"  HARD CHECK FAILED: {hard}", flush=True)

    _save_json(out / "fs_fit_record.json" if out else None, record.to_dict())
    _save_json(out / "fs_diagnostics.json" if out else None, diagnostics)
    ctx["record"]["fs_fit"] = record.to_dict()
    ctx["record"]["fs_diagnostics"] = diagnostics
    return bool(record.eligible) and diagnostics["hard_checks_passed"]


def stage_fs_panel(args, ctx: Dict[str, Any]) -> bool:
    """Panels for a SAVED adapter (negative controls): F0 / L0 / loaded L1."""
    if args.fs_adapter is None:
        raise SystemExit("ERROR: --stage fs_panel needs --fs-adapter.")
    if not ctx.get("panel_lead_steps"):
        raise SystemExit("ERROR: --stage fs_panel needs --panel-lead-steps.")
    if not args.synthetic and args.fs_adapter_sha256 is None:
        raise SystemExit("ERROR: a real --stage fs_panel needs --fs-adapter-sha256.")
    hidden = _hidden_size(ctx)
    device = _device_of(ctx)
    loaded, sha = sa.load_fs_adapter(
        args.fs_adapter, hidden, ctx["target_blocks"],
        rank_per_expert=int(args.rank_per_expert),
        expected_sha256=args.fs_adapter_sha256, device=device,
    )
    initial = sa.build_fs_adapter(hidden, ctx["target_blocks"],
                                  rank_per_expert=int(args.rank_per_expert), seed=int(args.seed))
    initial_digest = sa.static_adapter_digest(initial)
    initial = {b: lora.to(device) for b, lora in initial.items()}
    panels = _compute_panels(ctx, initial, loaded)
    summary = {
        "train_24h": sa.panel_ratio_summary(panels["train"]["L0"], panels["train"]["L1"], 24),
        "train_72h": sa.panel_ratio_summary(panels["train"]["L0"], panels["train"]["L1"], 72),
        "train_6h": sa.panel_ratio_summary(panels["train"]["L0"], panels["train"]["L1"], 6),
    }
    if "holdout" in panels:
        summary["holdout_24h"] = sa.panel_ratio_summary(
            panels["holdout"]["L0"], panels["holdout"]["L1"], 24)
    record = {
        "stage": "fs_panel",
        "adapter": {"path": str(args.fs_adapter), "sha256": sha,
                    "static_adapter_digest": sa.static_adapter_digest(loaded)},
        "initial_adapter_digest": initial_digest,
        "expected_initial_adapter_digest": ctx.get("expected_initial_adapter_digest"),
        "summary": summary,
        "L0_equals_F0_bitwise": panels["L0_equals_F0_bitwise"],
    }
    print(f"  adapter       : {args.fs_adapter} sha256={sha[:16]}...", flush=True)
    print(f"  train 24h     : mean ratio {summary['train_24h']['mean_ratio']:.6f} "
          f"max r_i {summary['train_24h']['max_ratio']:.6f}", flush=True)
    out = Path(args.output_dir) if args.output_dir is not None else None
    _save_json(out / "fs_panels.json" if out else None, panels)
    _save_json(out / "fs_panel_record.json" if out else None, record)
    ctx["record"]["fs_panel"] = record
    values = [v for group in ("train", "holdout") if group in panels
              for tag in ("F0", "L0", "L1") for row in panels[group][tag].values()
              for v in row.values()]
    return all(v == v and abs(v) != float("inf") for v in values)


def authorize_freeze(args, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """FS-SELECT-v1: only a PASS decision for exactly this adapter may freeze.

    The decision must be a formal FS-SELECT-v1 decision (from
    `scripts/r4_fs_decide.py --kind formal`) whose designated adapter SHA-256
    equals both --fs-adapter-sha256 and the file's actual bytes, whose
    designated record is mode=formal with exactly the pre-registered number of
    updates, and (for real runs) which is bound to this protocol's SHA-256.
    """
    failures: List[str] = []
    if args.fs_adapter is None or args.fs_adapter_sha256 is None or args.fs_decision is None:
        raise fp.ProtocolViolation(
            "FREEZE_NOT_AUTHORIZED",
            "--stage fs_freeze needs --fs-adapter, --fs-adapter-sha256 and --fs-decision.",
        )
    decision = json.loads(Path(args.fs_decision).read_text())
    actual_sha = fp.file_sha256(args.fs_adapter)
    if decision.get("kind") != "formal" or decision.get("rule") != "FS-SELECT-v1":
        failures.append("decision is not a formal FS-SELECT-v1 decision")
    if decision.get("verdict") != "QUALITY_PASS_PENDING_DESIGNATED_GATES":
        failures.append(f"decision verdict is {decision.get('verdict')!r}")
    if decision.get("designated_adapter_sha256") != args.fs_adapter_sha256:
        failures.append("decision's designated adapter sha256 != --fs-adapter-sha256")
    if actual_sha != args.fs_adapter_sha256:
        failures.append(f"--fs-adapter bytes hash to {actual_sha}, not --fs-adapter-sha256")
    designated = decision.get("designated_record") or {}
    horizon = decision.get("horizon")
    protocol = ctx.get("protocol")
    if protocol is not None:
        horizon = protocol["qualification_rule"]["horizon"]
        if decision.get("protocol_sha256") != args.protocol_sha256:
            failures.append("decision is not bound to this protocol sha256")
    if designated.get("mode") != "formal":
        failures.append(f"designated record mode is {designated.get('mode')!r}, not 'formal'")
    if horizon is None or designated.get("n_updates") != horizon:
        failures.append(f"designated record has {designated.get('n_updates')} updates, "
                        f"not the pre-registered H_formal={horizon}")
    report = {"check": "freeze_authorization", "decision": str(args.fs_decision),
              "decision_sha256": fp.file_sha256(args.fs_decision),
              "adapter_sha256": actual_sha, "passed": not failures, "failures": failures}
    if failures:
        raise fp.ProtocolViolation("FREEZE_NOT_AUTHORIZED", "; ".join(failures), report)
    return report


def stage_fs_freeze_from_file(args, ctx: Dict[str, Any]) -> bool:
    """`--stage fs_freeze`: authorize, then load the adapter FROM FILE and merge."""
    authorization = authorize_freeze(args, ctx)
    adapters, _ = sa.load_fs_adapter(
        args.fs_adapter, _hidden_size(ctx), ctx["target_blocks"],
        rank_per_expert=int(args.rank_per_expert),
        expected_sha256=args.fs_adapter_sha256, device=_device_of(ctx),
    )
    ctx["fs_adapters"] = adapters
    ctx["record"]["freeze_authorization"] = authorization
    return stage_fs_freeze(args, ctx)


def stage_fs_freeze_in_chain(args, ctx: Dict[str, Any]) -> bool:
    """Synthetic `--stage all` only: freeze the just-fitted adapter IF it is a
    full-horizon formal fit that passed its own gate. Never for a
    gradient_check / stability_screen record or a short fit."""
    record = ctx.get("fs_fit_record")
    if (record is None or record.mode != "formal" or not record.eligible
            or record.n_updates != int(args.max_updates)):
        print("  BLOCKED: only a full-horizon, eligible formal fit may be frozen "
              f"(mode={getattr(record, 'mode', None)}, "
              f"n_updates={getattr(record, 'n_updates', None)}).", flush=True)
        return False
    return stage_fs_freeze(args, ctx)


def stage_fs_freeze(args, ctx: Dict[str, Any]) -> bool:
    bridge = ctx["bridge"]
    adapters = ctx.get("fs_adapters")
    if adapters is None:
        raise SystemExit(
            "ERROR: --stage fs_freeze needs Fs adapters. Run --stage fs_fit "
            "first (or --stage all), or pass --fs-adapter."
        )

    merged, artifact = sa.merge_static_adapter(
        bridge.model, adapters, target_blocks=ctx["target_blocks"]
    )
    fs_bridge = sa.make_fs_bridge(merged, bridge, artifact)
    ctx["fs_bridge"] = fs_bridge
    ctx["fs_artifact"] = artifact

    dynamic_bank = bt.build_dynamic_bank(
        bridge.model.blocks[ctx["target_blocks"][0]].attn.proj.in_features,
        ctx["target_blocks"],
        num_experts=int(args.num_experts),
        rank_per_expert=int(args.rank_per_expert),
        seed=int(args.seed),
    )
    # `build_dynamic_bank` always constructs on CPU. The zero-edit verification
    # below runs a real rollout that mixes these LoRAs with the Fs-merged
    # backbone, so they have to be re-homed onto the backbone's device first --
    # the same move `bt.train_expert` / `bt.profile_*` / `bt.horizon_check` each
    # make before their own rollouts. Without it a CUDA run dies with
    # "Expected all tensors to be on the same device".
    device = next(fs_bridge.model.parameters()).device
    for lora in dynamic_bank.values():
        lora.to(device)
    ctx["dynamic_bank"] = dynamic_bank

    verification = sa.record_post_freeze_reference(
        bridge,
        fs_bridge,
        adapters,
        artifact,
        ctx["samples"][0].x_norm,
        ctx["variables"],
        num_experts=int(args.num_experts),
        dynamic_bank=dynamic_bank,
        steps=int(args.hold_steps),
        target_blocks=ctx["target_blocks"],
        samples=ctx["samples"],
        spec=ctx["objective"],
    )
    ctx["post_freeze"] = verification

    merge = verification.merge_equivalence
    zero = verification.zero_edit_equivalence
    print(f"  merged digest : {artifact.merged_backbone_digest[:16]}...", flush=True)
    print(f"  static_adapter: {fs_bridge.version.static_adapter}", flush=True)
    print(
        f"  merge equiv   : passed={merge['passed']} "
        f"max_abs_diff={merge['max_abs_diff']:.3e}",
        flush=True,
    )
    print(
        f"  zero-edit vs Fs: passed={zero['passed']} "
        f"max_abs_diff={zero['max_abs_diff_vs_fs']:.3e}",
        flush=True,
    )
    print(
        f"  (reported separately) vs F0: {zero['max_abs_diff_vs_f0']} "
        f"discriminating={zero['discriminating']}",
        flush=True,
    )
    if verification.background_f0 is not None:
        print(
            f"  background_F0 : {verification.background_f0['loss']} "
            "[reporting only, never a gain baseline]",
            flush=True,
        )
        print(f"  Fs baseline   : {verification.fs_baseline['loss']}", flush=True)

    if args.output_dir is not None:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": merged.state_dict(),
                "artifact": artifact.to_dict(),
                "reference_identity": verification.reference_identity,
            },
            out / "fs_merged_backbone.pt",
        )
        print(f"  wrote {out / 'fs_merged_backbone.pt'}", flush=True)
    _save_json(
        Path(args.output_dir) / "fs_post_freeze.json" if args.output_dir else None,
        verification.to_dict(),
    )
    ctx["record"]["fs_freeze"] = verification.to_dict()
    return bool(verification.passed)


# =============================================================================
# FP-04: the ONE certified Fs reference -- authorize by hash, load, re-verify
# =============================================================================

def _read_json(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _try_json(path: Path | str, label: str, failures: List[str]) -> Dict[str, Any]:
    try:
        payload = _read_json(path)
    except (OSError, ValueError) as exc:
        failures.append(f"{label} {path} is not readable JSON: {type(exc).__name__}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"{label} {path} is not a JSON object")
        return {}
    return payload


def authorize_certified_fs(args, protocol: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Bind a post-Fs stage to the certified Fs by SHA-256, BEFORE anything loads.

    Replaces FP-03's blanket LOCKED_UNTIL_FP04 refusal with the real check,
    modelled on `authorize_freeze` (every identity recomputed from bytes):

      * --fs-reference-manifest hashes to --fs-reference-manifest-sha256 and
        is an ``ed-fs-reference-manifest/1`` whose independent-reload facts
        (merged / base / static-adapter digests) are all True;
      * --fs-adapter and --fs-merged-backbone hash to the manifest's values
        (and --fs-adapter-sha256, when given, to the same);
      * --fs-certify-decision is the FS-SELECT-v1 ``certify`` decision with
        verdict FS_SELECTED, every designated gate True, for exactly this
        adapter, bound to the manifest's Fs protocol;
      * real runs: the protocol's ``inputs.fs_reference`` pins every one of
        those SHA-256s and the Fs protocol, and both the protocol and the gate
        config name the manifest's checkpoint and normalization identity.

    Any mismatch raises ProtocolViolation FS_REFERENCE_NOT_AUTHORIZED (the CLI
    exits 2 before a backbone, a sample or an Fs byte is loaded).
    """
    missing = [flag for flag, value in (
        ("--fs-reference-manifest", args.fs_reference_manifest),
        ("--fs-reference-manifest-sha256", args.fs_reference_manifest_sha256),
        ("--fs-certify-decision", args.fs_certify_decision),
        ("--fs-adapter", args.fs_adapter),
        ("--fs-merged-backbone", args.fs_merged_backbone),
    ) if value is None]
    if missing:
        raise fp.ProtocolViolation(
            "FS_REFERENCE_NOT_AUTHORIZED",
            f"--stage {args.stage} consumes the certified Fs and needs {', '.join(missing)}. "
            "A post-Fs stage never fits, freezes or picks up an Fs of its own.",
            {"missing": missing})
    failures: List[str] = []
    manifest_sha = fp.file_sha256(args.fs_reference_manifest)
    if manifest_sha != args.fs_reference_manifest_sha256:
        failures.append(f"reference manifest hashes to {manifest_sha}, not "
                        f"--fs-reference-manifest-sha256 {args.fs_reference_manifest_sha256}")
    manifest = _try_json(args.fs_reference_manifest, "reference manifest", failures)
    if manifest.get("schema_version") != REFERENCE_MANIFEST_SCHEMA:
        failures.append(f"manifest schema {manifest.get('schema_version')!r} != "
                        f"{REFERENCE_MANIFEST_SCHEMA!r}")
    reload_facts = manifest.get("independent_reload") or {}
    for key in ("merged_backbone_digest_matches", "base_backbone_digest_matches",
                "static_adapter_digest_matches"):
        if reload_facts.get(key) is not True:
            failures.append(f"manifest independent_reload.{key} is not True")
    adapter_sha = fp.file_sha256(args.fs_adapter)
    if adapter_sha != manifest.get("fs_adapter_sha256"):
        failures.append(f"--fs-adapter hashes to {adapter_sha}, the manifest names "
                        f"{manifest.get('fs_adapter_sha256')}")
    if args.fs_adapter_sha256 is not None and args.fs_adapter_sha256 != adapter_sha:
        failures.append("--fs-adapter-sha256 does not match the adapter bytes")
    merged_sha = fp.file_sha256(args.fs_merged_backbone)
    if merged_sha != manifest.get("fs_merged_backbone_sha256"):
        failures.append(f"--fs-merged-backbone hashes to {merged_sha}, the manifest names "
                        f"{manifest.get('fs_merged_backbone_sha256')}")
    decision_sha = fp.file_sha256(args.fs_certify_decision)
    decision = _try_json(args.fs_certify_decision, "certify decision", failures)
    if decision.get("kind") != "certify" or decision.get("rule") != "FS-SELECT-v1":
        failures.append("--fs-certify-decision is not an FS-SELECT-v1 certify decision")
    if decision.get("verdict") != "FS_SELECTED" or decision.get("fs_selected") is not True:
        failures.append(f"certify decision verdict is {decision.get('verdict')!r}, not FS_SELECTED")
    gates = decision.get("designated_gates") or {}
    if not gates or not all(v is True for v in gates.values()):
        failures.append(f"certify decision designated gates are not all True: {gates}")
    if decision.get("designated_adapter_sha256") != manifest.get("fs_adapter_sha256"):
        failures.append("certify decision designates a different adapter than the manifest")
    if decision.get("protocol_sha256") != manifest.get("protocol_sha256"):
        failures.append("certify decision and manifest are bound to different Fs protocols")
    pinned: Dict[str, Any] = {}
    if protocol is not None:
        inputs = protocol.get("inputs") or {}
        pinned = dict(inputs.get("fs_reference") or {})
        if not pinned:
            failures.append("the protocol pins no inputs.fs_reference")
        else:
            for key, actual in (("reference_manifest", manifest_sha),
                                ("certify_decision", decision_sha),
                                ("fs_adapter", adapter_sha),
                                ("fs_merged_backbone", merged_sha)):
                if (pinned.get(key) or {}).get("sha256") != actual:
                    failures.append(f"{key} sha256 {actual} is not the protocol's pinned "
                                    f"{(pinned.get(key) or {}).get('sha256')}")
            if pinned.get("fs_protocol_sha256") != manifest.get("protocol_sha256"):
                failures.append("the manifest's Fs protocol is not the pinned one")
        if inputs.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
            failures.append("protocol checkpoint sha256 differs from the certified Fs's")
        if inputs.get("normalization_identity") != manifest.get("normalization_identity"):
            failures.append("protocol normalization identity differs from the certified Fs's")
        if args.config is not None:
            try:
                gate = GateIdentityConfig.load_json(args.config)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                failures.append(f"gate config {args.config} unreadable: {type(exc).__name__}")
                gate = None
            if gate is not None:
                if getattr(gate, "expected_checkpoint_sha256", None) != \
                        manifest.get("checkpoint_sha256"):
                    failures.append("gate config checkpoint sha256 differs from the certified Fs's")
                if getattr(gate, "expected_normalization_identity", None) != \
                        manifest.get("normalization_identity"):
                    failures.append("gate config normalization identity differs from the "
                                    "certified Fs's")
    report = {
        "check": "certified_fs_authorization",
        "reference_manifest": {"path": str(args.fs_reference_manifest), "sha256": manifest_sha},
        "certify_decision": {"path": str(args.fs_certify_decision), "sha256": decision_sha},
        "fs_adapter": {"path": str(args.fs_adapter), "sha256": adapter_sha},
        "fs_merged_backbone": {"path": str(args.fs_merged_backbone), "sha256": merged_sha},
        "fs_adapter_sha256": adapter_sha,
        "fs_protocol_sha256": manifest.get("protocol_sha256"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "normalization_identity": manifest.get("normalization_identity"),
        "pinned_digests": dict(pinned.get("digests") or {}),
        "bound_to_protocol": protocol is not None,
        "passed": not failures,
        "failures": failures,
    }
    if failures:
        raise fp.ProtocolViolation("FS_REFERENCE_NOT_AUTHORIZED", "; ".join(failures), report)
    return report


def load_certified_fs(args, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Load the AUTHORIZED Fs into ctx['fs_bridge'] / ctx['fs_artifact'] and re-verify it.

    The loading steps are those of `stage_fs_verify` (copied, not shared, so
    the FP-03 stage that certified the bundle stays byte-for-byte what it
    was). After loading, four digests must match the certified bundle: merged
    backbone, base (F0) backbone, static adapter, and the re-recorded
    ArtifactVersion; on real runs they must also equal the protocol's pins.
    """
    reference = ctx["record"]["binding"].get("fs_reference") or {}
    bridge = ctx["bridge"]
    device = _device_of(ctx)
    payload = torch.load(args.fs_merged_backbone, map_location="cpu", weights_only=True)
    artifact_dict = dict(payload["artifact"])
    merged = copy.deepcopy(bridge.model)
    merged.load_state_dict(payload["state_dict"], strict=True)
    merged.requires_grad_(False)
    merged.eval()
    adapters, adapter_sha = sa.load_fs_adapter(
        args.fs_adapter, _hidden_size(ctx), ctx["target_blocks"],
        rank_per_expert=int(args.rank_per_expert),
        expected_sha256=reference.get("fs_adapter_sha256"), device=device,
    )
    artifact = sa.MergedBackboneArtifact(
        **{**artifact_dict, "target_blocks": tuple(artifact_dict["target_blocks"])})
    fs_bridge = sa.make_fs_bridge(merged, bridge, artifact)
    recorded_identity = dict(payload.get("reference_identity") or {})
    digests = {
        "merged_backbone_digest": sa.state_dict_digest(merged),
        "base_backbone_digest": sa.state_dict_digest(bridge.model),
        "static_adapter_digest": sa.static_adapter_digest(adapters,
                                                          target_blocks=ctx["target_blocks"]),
        "artifact_version_digest": fs_bridge.version.digest,
    }
    identity = {
        "merged_backbone_digest_matches":
            digests["merged_backbone_digest"] == artifact_dict["merged_backbone_digest"],
        "base_backbone_digest_matches":
            digests["base_backbone_digest"] == artifact_dict["base_backbone_digest"],
        "static_adapter_digest_matches":
            digests["static_adapter_digest"] == artifact_dict["static_adapter_digest"],
        "version_digest_matches":
            digests["artifact_version_digest"] == recorded_identity.get("artifact_version_digest"),
        "target_blocks_match": [int(b) for b in artifact.target_blocks]
            == [int(b) for b in ctx["target_blocks"]],
        "adapter_sha256_matches": adapter_sha == reference.get("fs_adapter_sha256"),
    }
    pinned = dict(reference.get("pinned_digests") or {})
    if ctx.get("protocol") is not None:
        for key in ("merged_backbone_digest", "base_backbone_digest", "static_adapter_digest",
                    "artifact_version_digest"):
            identity[f"{key}_is_pinned"] = pinned.get(key) is not None and \
                digests[key] == pinned.get(key)
    report = {"check": "certified_fs_loaded", "digests": digests, "identity": identity,
              "identity_pass": all(identity.values()),
              "fs_adapter_sha256": adapter_sha,
              "merged_backbone_sha256": reference.get("fs_merged_backbone", {}).get("sha256"),
              "artifact": artifact.to_dict()}
    if not report["identity_pass"]:
        raise fp.ProtocolViolation(
            "FS_REFERENCE_IDENTITY_MISMATCH",
            f"the loaded Fs is not the certified one: "
            f"{sorted(k for k, v in identity.items() if v is not True)}", report)
    ctx["fs_bridge"] = fs_bridge
    ctx["fs_artifact"] = artifact
    ctx["fs_source"] = "certified_bundle"
    ctx["fs_identity"] = report
    return report


def _fs_source(args, ctx: Dict[str, Any], stage: str) -> str:
    """Which Fs this stage runs on; refuses anything but the certified bundle.

    The only exception is the synthetic in-process `--stage all` chain, whose
    bank rides the Fs `stage_fs_freeze_in_chain` just froze (CPU smoke only).
    """
    source = ctx.get("fs_source")
    if source == "certified_bundle":
        return source
    if args.synthetic and args.stage == "all" and ctx.get("fs_bridge") is not None:
        return "in_chain_synthetic"
    raise fp.ProtocolViolation(
        "FS_REFERENCE_NOT_AUTHORIZED",
        f"--stage {stage} runs only on the certified Fs loaded by load_certified_fs "
        f"(fs_source={source!r}).", {"fs_source": source, "stage": stage})


def _build_initial_bank(args, ctx: Dict[str, Any],
                        device: Optional[torch.device | str] = None) -> Dict[int, Any]:
    """The pinned initial bank: `build_dynamic_bank(seed=--seed)` on the Fs device."""
    fs_bridge = ctx.get("fs_bridge") or ctx["bridge"]
    target = device if device is not None else next(fs_bridge.model.parameters()).device
    return bt.build_dynamic_bank(
        _hidden_size(ctx), ctx["target_blocks"], num_experts=int(args.num_experts),
        rank_per_expert=int(args.rank_per_expert), seed=int(args.seed), device=target)


def _bank_train_config(args, ctx: Dict[str, Any]) -> bt.BankTrainConfig:
    return bt.BankTrainConfig(
        mode=args.mode,
        max_updates=int(args.max_updates),
        learning_rate=float(args.learning_rate),
        train_steps=int(args.train_steps),
        hold_steps=int(args.hold_steps),
        lead_steps=tuple(ctx["lead_steps"]),
        num_experts=int(args.num_experts),
        rank_per_expert=int(args.rank_per_expert),
        target_blocks=tuple(ctx["target_blocks"]),
        a0=float(args.a0),
        rho=float(args.rho),
        seed=int(args.seed),
    )


def _panel_steps(args, ctx: Dict[str, Any]) -> Tuple[int, ...]:
    """Panel / probe rollout steps: --panel-lead-steps (1 4 12 declared)."""
    steps = tuple(int(s) for s in (ctx.get("panel_lead_steps") or ()))
    return steps or (1, int(args.hold_steps))


def _probe_sample(ctx: Dict[str, Any]) -> sa.TrainingSample:
    """The fixed probe issue: the protocol's pinned id, else the first sample."""
    wanted = (ctx.get("bank_pins") or {}).get("probe_issue_id")
    samples = ctx["samples"]
    if wanted is None:
        return samples[0]
    for sample in samples:
        if sample.issue_id == wanted:
            return sample
    raise bt.BankTrainingViolation(
        "BANK_PROBE_ISSUE_MISSING",
        f"the pinned probe issue {wanted!r} is not among the admitted samples.",
        {"probe_issue_id": wanted})


def _pinned_grouping_matches(grouping, pins: Mapping[str, Any]) -> Optional[bool]:
    pinned = pins.get("grouping")
    if pinned is None:
        return None
    actual = {str(g.expert_index): list(g.issue_ids) for g in grouping.groups}
    return actual == {str(k): list(v) for k, v in dict(pinned).items()}


def _save_and_reload_expert(args, ctx: Dict[str, Any], bank, index: int, plan, probe,
                            probe_states, record, out: Optional[Path],
                            initial_bank_digest: str) -> Dict[str, Any]:
    """Save expert_{k}.pt + its probe, then prove the files reproduce it exactly.

    A FRESH initial bank is rebuilt from the seed, expert k is installed from
    the reloaded file, and the probe rollout is repeated: digest and every
    probe state must match the in-memory trained bank bit for bit.
    """
    if out is None:
        return {"saved": False, "file_sha256_matches": False, "expert_digest_matches": False,
                "probe_exact": False, "probe_max_abs_diff_by_step": {},
                "note": "no --output-dir: nothing saved, E5 cannot pass"}
    artifact = ctx.get("fs_artifact")
    metadata = {
        "config_id": args.config_id,
        "protocol_sha256": args.protocol_sha256,
        "mode": record.config.get("mode"),
        "learning_rate": record.config.get("learning_rate"),
        "n_updates": record.n_updates,
        "initial_bank_digest": initial_bank_digest,
        "initial_expert_digest": record.initial_expert_digest,
        "fs_merged_backbone_digest": artifact.merged_backbone_digest if artifact else None,
        "fs_source": ctx.get("fs_source"),
    }
    saved = bt.save_expert(bank, index, out / f"expert_{index}.pt", metadata=metadata)
    probe_path = out / f"bank_probe_expert{index}.pt"
    torch.save({
        "schema": bt.PROBE_FILE_SCHEMA,
        "expert_index": int(index),
        "probe_issue_id": probe.issue_id,
        "plan_id": plan.plan_id,
        "coefficients": [float(a) for a in plan.coefficients],
        "hold_steps": int(plan.hold_steps),
        "steps": sorted(int(s) for s in probe_states),
        "states": {str(s): t for s, t in probe_states.items()},
        "source_bank_digest": bt.bank_digest(bank),
        "fs_merged_backbone_digest": artifact.merged_backbone_digest if artifact else None,
    }, probe_path)
    loaded = bt.load_expert(saved["path"], expected_sha256=saved["sha256"],
                            expected_expert_index=index)
    fresh = _build_initial_bank(args, ctx)
    fresh_digest = bt.bank_digest(fresh)
    bt.install_expert(fresh, loaded, expert_index=index)
    reloaded_states = bt.singleton_probe_states(
        ctx["fs_bridge"], fresh, plan, probe.x_norm, ctx["variables"],
        probe_steps=sorted(probe_states), target_blocks=ctx["target_blocks"])
    reload_cmp = bt.compare_probe_states(probe_states, reloaded_states)
    stored = torch.load(probe_path, map_location="cpu", weights_only=True)
    file_cmp = bt.compare_probe_states(probe_states, stored["states"])
    return {
        "saved": True,
        "expert_file": {"path": saved["path"], "sha256": saved["sha256"]},
        "probe_file": {"path": str(probe_path), "sha256": fp.file_sha256(probe_path)},
        "expert_digest": loaded["expert_digest"],
        "file_sha256_matches": loaded["sha256"] == saved["sha256"] == fp.file_sha256(saved["path"]),
        "expert_digest_matches": (loaded["expert_digest"] == record.final_expert_digest
                                  == bt.expert_digest(fresh, index)),
        "fresh_initial_bank_digest": fresh_digest,
        "fresh_initial_bank_is_the_initial_bank": fresh_digest == initial_bank_digest,
        "probe_exact": bool(reload_cmp["exact"] and file_cmp["exact"]),
        "probe_max_abs_diff_by_step": reload_cmp["max_abs_diff_by_step"],
        "probe_file_roundtrip_exact": file_cmp["exact"],
        "probe_reload_comparison": reload_cmp,
    }


def stage_bank_train(args, ctx: Dict[str, Any]) -> bool:
    """Train one (or, in the synthetic chain, every) expert on the certified Fs.

    Before training: the initial bank digest (must equal the seed bank and the
    protocol's pin), the grouping (must equal the pin), the backbone digest
    (must be the certified merged Fs), and the untrained singleton panel,
    which must equal the pure-Fs panel bit for bit. Training: `train_expert`
    unchanged, now also recording A/B gradient norms, clip events and the
    expert / other-expert digests. After: the trained panel, the nonzero
    response, the probe states at the panel steps, expert_{k}.pt, and a
    reload that must reproduce digest and probe exactly; backbone unchanged
    and no hook left behind.
    """
    fs_bridge = ctx.get("fs_bridge")
    if fs_bridge is None:
        raise SystemExit(
            "ERROR: --stage bank_train needs the certified Fs (--fs-reference-manifest ...) "
            "or, synthetic only, the in-chain freeze of --stage all."
        )
    fs_source = _fs_source(args, ctx, "bank_train")
    device = next(fs_bridge.model.parameters()).device
    blocks = tuple(int(b) for b in ctx["target_blocks"])
    names = ctx["variables"]
    spec = ctx["objective"]
    pins = dict(ctx.get("bank_pins") or {})
    num_experts = int(args.num_experts)
    artifact = ctx.get("fs_artifact")
    fs_digest = artifact.merged_backbone_digest if artifact else ""

    bank = ctx.get("dynamic_bank")
    if bank is None:
        bank = _build_initial_bank(args, ctx)
        ctx["dynamic_bank"] = bank
    else:
        for lora in bank.values():
            lora.to(device)
    bt.check_bank_device(bank, device, context="bank_train")
    initial_bank_digest = bt.bank_digest(bank)
    initial_expert_digests = [bt.expert_digest(bank, k) for k in range(num_experts)]
    seed_bank_digest = bt.bank_digest(_build_initial_bank(args, ctx, device="cpu"))

    grouping = bt.assign_diversity_groups(ctx["samples"], num_experts,
                                          hold_steps=int(args.hold_steps))
    ctx["grouping"] = grouping
    print(f"  diversity rule: {grouping.rule}", flush=True)
    for group in grouping.groups:
        payload = group.to_dict()
        print(f"    expert {group.expert_index}: months={payload['months']} "
              f"n={payload['n_samples']}", flush=True)
    if grouping.purged:
        print(f"    purged: {len(grouping.purged)} sample(s)", flush=True)

    registry = bt.build_bank_registry(num_experts, rho=float(args.rho), a0=float(args.a0))
    ctx["registry"] = registry
    config = _bank_train_config(args, ctx)
    expert_indices = ([int(args.expert_index)] if args.expert_index is not None
                      else list(range(num_experts)))
    panel_leads = _panel_steps(args, ctx)
    probe = _probe_sample(ctx)
    out = Path(args.output_dir) if args.output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    common_checks: Dict[str, Any] = {
        "initial_bank_is_the_seed_bank": initial_bank_digest == seed_bank_digest,
    }
    if pins.get("initial_bank_digest") is not None:
        common_checks["initial_bank_digest_matches_pin"] = \
            initial_bank_digest == pins["initial_bank_digest"]
    grouping_ok = _pinned_grouping_matches(grouping, pins)
    if grouping_ok is not None:
        common_checks["grouping_matches_pin"] = grouping_ok
    if pins.get("probe_issue_id") is not None:
        common_checks["probe_issue_matches_pin"] = probe.issue_id == pins["probe_issue_id"]

    records: List[Dict[str, Any]] = []
    panels_out: Dict[str, Any] = {}
    diagnostics_out: Dict[str, Any] = {}
    passed = all(v is True for v in common_checks.values())
    for index in expert_indices:
        group = grouping.group_for(index)
        group_samples = grouping.samples_for(index, ctx["samples"])
        if not group_samples:
            print(f"  expert {index}: diversity group is empty after the guard purge; "
                  "recorded as NO_UPDATES_RUN rather than silently skipped.", flush=True)
            record = bt.ExpertTrainingRecord(
                expert_index=index, config=config.to_dict(), diversity_group=group.to_dict(),
                eligible=False, ineligible_reason="NO_UPDATES_RUN")
            records.append(record.to_dict())
            passed = False
            continue

        plan = bt.expert_plan(registry, index, hold_steps=int(args.hold_steps))
        backbone_before = sa.state_dict_digest(fs_bridge.model)
        hooks_before = _count_hooks(fs_bridge.model)
        fs_panel = sa.fs_panel_losses(
            fs_bridge, group_samples, spec, lead_steps=panel_leads, fs_adapters=None,
            target_blocks=blocks, variables=names)
        l0_panel = bt.bank_panel_losses(
            fs_bridge, bank, plan, group_samples, spec, lead_steps=panel_leads,
            target_blocks=blocks, variables=names)
        untrained_equal = l0_panel == fs_panel

        record = bt.train_expert(
            fs_bridge, bank, index, group_samples, spec, config,
            registry=registry, diversity_group=group, variables=names,
            fs_backbone_digest=fs_digest,
            fs_static_adapter_digest=artifact.static_adapter_digest if artifact else "",
        )
        l1_panel = bt.bank_panel_losses(
            fs_bridge, bank, plan, group_samples, spec, lead_steps=panel_leads,
            target_blocks=blocks, variables=names)
        probe_states = bt.singleton_probe_states(
            fs_bridge, bank, plan, probe.x_norm, names, probe_steps=panel_leads,
            target_blocks=blocks)
        reload = _save_and_reload_expert(args, ctx, bank, index, plan, probe, probe_states,
                                         record, out, initial_bank_digest)
        backbone_after = sa.state_dict_digest(fs_bridge.model)
        hooks_after = _count_hooks(fs_bridge.model)
        isolation = {
            "backbone_digest_before": backbone_before,
            "backbone_digest_after": backbone_after,
            "backbone_digest_unchanged": backbone_before == backbone_after,
            "backbone_is_certified_fs": (backbone_before == fs_digest) if artifact else None,
            "hooks_before": hooks_before,
            "hooks_after": hooks_after,
            "no_leftover_hooks": hooks_before == hooks_after,
            "other_experts_unchanged": record.other_experts_unchanged,
            "other_experts_grad_free": record.other_experts_grad_free,
        }
        hard = {
            "untrained_L0_equals_Fs_bitwise": untrained_equal,
            "backbone_digest_unchanged": isolation["backbone_digest_unchanged"],
            "no_leftover_hooks": isolation["no_leftover_hooks"],
            "other_experts_unchanged": record.other_experts_unchanged is True,
            "other_experts_grad_free": record.other_experts_grad_free is True,
        }
        if artifact is not None:
            hard["backbone_is_certified_fs"] = isolation["backbone_is_certified_fs"]
        if out is not None:
            hard["reload_file_sha256_matches"] = reload["file_sha256_matches"]
            hard["reload_expert_digest_matches"] = reload["expert_digest_matches"]
            hard["reload_probe_exact"] = reload["probe_exact"]
            hard["fresh_initial_bank_is_the_initial_bank"] = \
                reload["fresh_initial_bank_is_the_initial_bank"]
        diag = {
            "expert_index": index,
            "plan_id": plan.plan_id,
            "coefficients": list(plan.coefficients),
            "group_issue_ids": [s.issue_id for s in group_samples],
            "untrained_L0_equals_Fs_bitwise": untrained_equal,
            "nonzero_response": dict(record.nonzero_response),
            "reload": reload,
            "isolation": isolation,
            "grad_A_update0": record.grad_norms_A[0] if record.grad_norms_A else None,
            "grad_B_update0": record.grad_norms_B[0] if record.grad_norms_B else None,
            "grad_A_update1": record.grad_norms_A[1] if len(record.grad_norms_A) > 1 else None,
            "clip_events": record.clip_events,
            "probe_issue_id": probe.issue_id,
            "probe_steps": list(panel_leads),
            "hard_checks": hard,
            "hard_checks_passed": all(v is True for v in hard.values()),
        }
        records.append(record.to_dict())
        panels_out[str(index)] = {"issue_ids": [s.issue_id for s in group_samples],
                                  "Fs": fs_panel, "L0": l0_panel, "L1": l1_panel,
                                  "L0_equals_Fs_bitwise": untrained_equal}
        diagnostics_out[str(index)] = diag
        passed = passed and bool(record.eligible) and diag["hard_checks_passed"]
        print(f"  expert {index}: plan={record.plan_id} n={len(group_samples)} "
              f"updates={record.n_updates} clip={record.clip_events} "
              f"L0==Fs={untrained_equal} eligible={record.eligible} "
              f"nonzero={record.nonzero_response.get('nonzero')} "
              f"reload_exact={reload.get('probe_exact')} hard={diag['hard_checks_passed']}",
              flush=True)
        if not diag["hard_checks_passed"]:
            print(f"  HARD CHECK FAILED: {hard}", flush=True)

    ctx["expert_records"] = records
    suffix = f"_expert{int(args.expert_index)}" if args.expert_index is not None else ""
    payload = {
        "grouping": grouping.to_dict(),
        "registry": registry.to_dict(),
        "experts": records,
        "fs_source": fs_source,
        "initial_bank_digest": initial_bank_digest,
        "initial_expert_digests": initial_expert_digests,
        "config": config.to_dict(),
    }
    diagnostics = {
        "fs_source": fs_source,
        "fs_identity": ctx.get("fs_identity"),
        "initial_bank_digest": initial_bank_digest,
        "seed_bank_digest": seed_bank_digest,
        "expected_initial_bank_digest": pins.get("initial_bank_digest"),
        "initial_expert_digests": initial_expert_digests,
        "grouping_by_expert": {str(g.expert_index): list(g.issue_ids) for g in grouping.groups},
        "probe_issue_id": probe.issue_id,
        "common_checks": common_checks,
        "experts": diagnostics_out,
    }
    panels = {"schema": bt.BANK_PANELS_SCHEMA, "lead_steps": list(panel_leads),
              "lead_hours": [int(s) * 6 for s in panel_leads],
              "reduction": "float64 full_objective_loss per issue per lead; no grad; one "
                           "rollout per issue (Fs: plain; L0/L1: singleton plan, hold then Fs)",
              "experts": panels_out}
    _save_json(out / f"bank_training{suffix}.json" if out else None, payload)
    _save_json(out / f"bank_panels{suffix}.json" if out else None, panels)
    _save_json(out / f"bank_diagnostics{suffix}.json" if out else None, diagnostics)
    ctx["record"]["bank_train"] = {"experts": records, "diagnostics": diagnostics}
    return bool(passed)


def _measurement_bank(args, ctx: Dict[str, Any], stage: str):
    """The bridge and bank a profile / horizon stage measures (never mutated)."""
    if ctx.get("fs_bridge") is not None:
        _fs_source(args, ctx, stage)
        bridge = ctx["fs_bridge"]
    elif args.synthetic and args.stage == "all":
        bridge = ctx["bridge"]
    else:
        _fs_source(args, ctx, stage)  # raises: no certified Fs
        bridge = ctx["bridge"]
    bank = ctx.get("dynamic_bank")
    if bank is None:
        bank = _build_initial_bank(args, ctx)
        ctx["dynamic_bank"] = bank
    return bridge, bank


def _immutability(bank, digest_before: str, measured) -> Dict[str, Any]:
    """FP-02 gap closed: the measured bank is a deep copy; the real one is untouched."""
    digest_after = bt.bank_digest(bank)
    return {
        "bank_digest_before": digest_before,
        "bank_digest_after": digest_after,
        "bank_unchanged": digest_before == digest_after,
        "measured_on_a_deep_copy": True,
        "measured_copy_digest_after": bt.bank_digest(measured),
    }


def stage_profile(args, ctx: Dict[str, Any]) -> bool:
    bridge, bank = _measurement_bank(args, ctx, "profile")
    digest_before = bt.bank_digest(bank)
    # profile_training_step runs a real optimizer step: it measures a DEEP COPY
    # so the bank the pipeline carries (and later saves) is never moved.
    measured = {b: copy.deepcopy(lora) for b, lora in bank.items()}
    config = _bank_train_config(args, ctx)
    profile = bt.profile_training_step(
        bridge,
        measured,
        ctx["samples"][0],
        ctx["objective"],
        expert_index=int(args.expert_index or 0),
        config=config,
        variables=ctx["variables"],
        n_measured_steps=int(args.profile_steps),
        warmup_steps=int(args.profile_warmup),
        synthetic=bool(args.synthetic),
    )
    immutability = _immutability(bank, digest_before, measured)
    budget = bt.CapacityBudget(
        memory_budget_bytes=int(float(args.memory_budget_gib) * (1024 ** 3)),
        memory_safety_fraction=float(args.memory_safety_fraction),
        max_seconds_per_update=float(args.max_seconds_per_update),
    )
    decision = bt.decide_bank_capacity(profile, budget)
    ctx["profile"] = profile
    ctx["capacity_decision"] = decision

    print(f"  device        : {profile.device} ({profile.device_name})", flush=True)
    print(f"  step          : {profile.seconds_per_update:.4f} s/update over "
          f"{profile.n_measured_steps} step(s)", flush=True)
    print(f"  peak memory   : {profile.peak_memory_bytes} bytes ({profile.memory_source})",
          flush=True)
    print(f"  decision      : K={decision.decided_num_experts} "
          f"rank={decision.decided_rank_per_expert} shrink={decision.shrink_required} "
          f"provisional={decision.provisional}", flush=True)
    for reason in decision.reasons:
        print(f"    - {reason}", flush=True)
    if decision.provisional:
        print(f"    PROVISIONAL: {decision.provisional_reason}", flush=True)
    print(f"  bank unchanged: {immutability['bank_unchanged']} (measured on a deep copy)",
          flush=True)

    pins = dict(ctx.get("bank_pins") or {})
    payload = {"profile": profile.to_dict(), "decision": decision.to_dict(),
               "bank_immutability": immutability,
               "fs_source": ctx.get("fs_source"),
               "expected_initial_bank_digest": pins.get("initial_bank_digest"),
               "fs_identity": ctx.get("fs_identity")}
    _save_json(
        Path(args.output_dir) / "capacity_profile.json" if args.output_dir else None,
        payload,
    )
    ctx["record"]["profile"] = payload
    # A profile that ran is a successful profile even if it recommends
    # shrinking; an errored measurement or a moved bank is a failure.
    return profile.error is None and immutability["bank_unchanged"]


def stage_horizon_check(args, ctx: Dict[str, Any]) -> bool:
    bridge, bank = _measurement_bank(args, ctx, "horizon_check")
    digest_before = bt.bank_digest(bank)
    measured = {b: copy.deepcopy(lora) for b, lora in bank.items()}

    budget = bt.CapacityBudget(
        memory_budget_bytes=int(float(args.memory_budget_gib) * (1024 ** 3)),
        memory_safety_fraction=float(args.memory_safety_fraction),
        max_seconds_per_update=float(args.max_seconds_per_update),
    )
    sample = ctx["samples"][0]
    # Every horizon the check walks needs its own target, otherwise the shorter
    # horizons cannot be scored and would be reported as infeasible.
    horizons = tuple(range(1, int(args.hold_steps) + 1))
    enriched = ctx.get("horizon_sample")
    if enriched is None:
        targets = dict(sample.targets_raw)
        for step in horizons:
            targets.setdefault(step, sample.targets_raw[max(sample.targets_raw)])
        enriched = sa.TrainingSample(
            x_norm=sample.x_norm,
            targets_raw=targets,
            issue_id=sample.issue_id,
            issue_time=sample.issue_time,
            valid_time=sample.valid_time,
            history_time=sample.history_time,
            split_id=sample.split_id,
            data_role=sample.data_role,
            source=sample.source,
            time_utc=sample.time_utc,
        )
        ctx["horizon_sample"] = enriched

    report = bt.check_horizon_feasibility(
        bridge,
        measured,
        enriched,
        ctx["objective"],
        expert_index=int(args.expert_index or 0),
        plan_hold_steps=int(args.hold_steps),
        target_blocks=ctx["target_blocks"],
        budget=budget,
        max_differentiable_steps=args.max_differentiable_steps,
        variables=ctx["variables"],
        registry=ctx.get("registry"),
        updates_per_expert=int(args.max_updates),
        num_experts_for_cost=int(args.num_experts),
    )
    immutability = _immutability(bank, digest_before, measured)
    ctx["horizon"] = report

    print(f"  requested     : {report.requested_hold_steps} steps "
          f"({report.requested_hours}h)", flush=True)
    print(f"  feasible      : {report.feasible} (max "
          f"{report.max_feasible_differentiable_steps} differentiable steps)", flush=True)
    print(f"  activation ckpt: enabled={report.activation_checkpointing_enabled} "
          f"(never turned on to reach a longer horizon)", flush=True)
    print(f"  bank unchanged: {immutability['bank_unchanged']} (measured on a deep copy)",
          flush=True)
    if report.fallback is not None:
        spec = report.fallback
        print(f"  FALLBACK SPEC : {spec.reason}", flush=True)
        print(f"    train {spec.train_hours}h differentiable; evaluate "
              f"{list(spec.eval_hours)}h non-differentiably", flush=True)
        print(f"    new identity : {spec.experiment_identity}", flush=True)
        print(f"    supersedes   : {spec.superseded_identity}", flush=True)
        derived = spec.cost_estimate["derived"]
        print(f"    cost (derived): train {derived['training_gpu_hours_all_experts']} GPU-h, "
              f"eval {derived['evaluation_gpu_hours']} GPU-h", flush=True)

    payload = report.to_dict()
    payload["bank_immutability"] = immutability
    payload["fs_source"] = ctx.get("fs_source")
    payload["fs_identity"] = ctx.get("fs_identity")
    payload["expected_initial_bank_digest"] = (ctx.get("bank_pins") or {}).get("initial_bank_digest")
    _save_json(
        Path(args.output_dir) / "horizon_feasibility.json" if args.output_dir else None,
        payload,
    )
    ctx["record"]["horizon_check"] = payload
    # The check succeeds when it produced an answer (feasible, or an explicit
    # fallback spec) AND left the bank untouched. Silence is a failure.
    return bool((report.feasible or report.fallback is not None)
                and immutability["bank_unchanged"])


# =============================================================================
# FP-04: assembly and the independent exact verification
# =============================================================================

def authorize_bank_assembly(args, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """BANK-SELECT-v1: only a BANK_QUAL_PASS_PENDING_ASSEMBLY decision may assemble.

    The decision must be a formal BANK-SELECT-v1 decision naming exactly K
    experts (indices 0..K-1, distinct digests), every expert file must still
    hash to the decided SHA-256, and on real runs the decision must be bound
    to this protocol.
    """
    if args.bank_decision is None:
        raise fp.ProtocolViolation("ASSEMBLY_NOT_AUTHORIZED",
                                   f"--stage {args.stage} needs --bank-decision.")
    decision = _read_json(args.bank_decision)
    failures: List[str] = []
    num_experts = int(args.num_experts)
    if decision.get("kind") != "formal" or decision.get("rule") != bt.BANK_SELECT_RULE_ID:
        failures.append("decision is not a formal BANK-SELECT-v1 decision")
    if decision.get("verdict") != "BANK_QUAL_PASS_PENDING_ASSEMBLY" or \
            decision.get("bank_selected_for_assembly") is not True:
        failures.append(f"decision verdict is {decision.get('verdict')!r}")
    if ctx.get("protocol") is not None and decision.get("protocol_sha256") != args.protocol_sha256:
        failures.append("decision is not bound to this protocol sha256")
    experts = list(decision.get("experts_for_assembly") or [])
    indices = sorted(int(e.get("expert_index", -1)) for e in experts)
    if indices != list(range(num_experts)):
        failures.append(f"decision names experts {indices}, not exactly 0..{num_experts - 1}")
    digests = [e.get("expert_digest") for e in experts]
    if None in digests or len(set(digests)) != len(digests):
        failures.append("decided expert digests are missing or not distinct")
    checked = []
    for entry in experts:
        ref = entry.get("expert_file") or {}
        path = ref.get("path")
        actual = fp.file_sha256(path) if path and Path(path).is_file() else None
        if actual is None or actual != ref.get("sha256"):
            failures.append(f"expert {entry.get('expert_index')} file {path} hashes to "
                            f"{actual}, decided {ref.get('sha256')}")
        checked.append({"expert_index": entry.get("expert_index"), "path": path,
                        "sha256": actual})
    report = {"check": "assembly_authorization", "decision": str(args.bank_decision),
              "decision_sha256": fp.file_sha256(args.bank_decision),
              "experts": checked, "passed": not failures, "failures": failures}
    if failures:
        raise fp.ProtocolViolation("ASSEMBLY_NOT_AUTHORIZED", "; ".join(failures), report)
    return {**report, "decision_payload": decision}


def stage_bank_assemble(args, ctx: Dict[str, Any]) -> bool:
    """Build bank.pt: the pinned initial bank with all four decided experts installed."""
    _fs_source(args, ctx, "bank_assemble")
    authorization = authorize_bank_assembly(args, ctx)
    decision = authorization.pop("decision_payload")
    fs_bridge = ctx["fs_bridge"]
    device = next(fs_bridge.model.parameters()).device
    pins = dict(ctx.get("bank_pins") or {})
    initial = _build_initial_bank(args, ctx)
    initial_digest = bt.bank_digest(initial)
    experts: Dict[int, Dict[str, Any]] = {}
    per_expert = []
    for entry in decision["experts_for_assembly"]:
        k = int(entry["expert_index"])
        loaded = bt.load_expert(entry["expert_file"]["path"],
                                expected_sha256=entry["expert_file"]["sha256"],
                                expected_expert_index=k)
        experts[k] = loaded
        meta = loaded["payload"].get("metadata") or {}
        per_expert.append({
            "expert_index": k,
            "expert_file": dict(entry["expert_file"]),
            "expert_digest": loaded["expert_digest"],
            "decided_expert_digest": entry["expert_digest"],
            "digest_is_the_decided_one": loaded["expert_digest"] == entry["expert_digest"],
            "trained_from_the_initial_bank":
                meta.get("initial_expert_digest") == bt.expert_digest(initial, k)
                and meta.get("initial_bank_digest") == initial_digest,
        })
    bank = bt.assemble_bank(initial, experts)
    bt.check_bank_device(bank, device, context="bank_assemble")
    out = Path(args.output_dir) if args.output_dir is not None else None
    if out is None:
        raise SystemExit("ERROR: --stage bank_assemble needs --output-dir (bank.pt is its product).")
    out.mkdir(parents=True, exist_ok=True)
    artifact = ctx.get("fs_artifact")
    saved = bt.save_bank(bank, out / "bank.pt", metadata={
        "protocol_sha256": args.protocol_sha256, "config_id": args.config_id,
        "initial_bank_digest": initial_digest,
        "fs_merged_backbone_digest": artifact.merged_backbone_digest if artifact else None,
        "decision_sha256": authorization["decision_sha256"]})
    reloaded, meta = bt.load_bank(out / "bank.pt", expected_sha256=saved["sha256"], device=device)
    checks = {
        "initial_bank_is_the_seed_bank": True,
        "every_expert_is_the_decided_one": all(e["digest_is_the_decided_one"] for e in per_expert),
        "every_expert_trained_from_the_initial_bank":
            all(e["trained_from_the_initial_bank"] for e in per_expert),
        "bank_expert_digests_are_the_expert_files":
            saved["expert_digests"] == [experts[k]["expert_digest"] for k in sorted(experts)],
        "reload_self_check": meta["bank_digest"] == saved["bank_digest"]
            and meta["expert_digests"] == saved["expert_digests"],
    }
    if pins.get("initial_bank_digest") is not None:
        checks["initial_bank_digest_matches_pin"] = initial_digest == pins["initial_bank_digest"]
    record = {
        "check": "bank_assembly",
        "bank_file": {"path": saved["path"], "sha256": saved["sha256"]},
        "bank_digest": saved["bank_digest"],
        "expert_digests": saved["expert_digests"],
        "initial_bank_digest": initial_digest,
        "experts": per_expert,
        "decision": {"path": str(args.bank_decision),
                     "sha256": authorization["decision_sha256"]},
        "authorization": authorization,
        "fs_source": ctx.get("fs_source"),
        "fs_identity": ctx.get("fs_identity"),
        "checks": checks,
        "passed": all(v is True for v in checks.values()),
    }
    print(f"  bank.pt       : sha256={saved['sha256'][:16]}... digest={saved['bank_digest'][:16]}...",
          flush=True)
    print(f"  checks        : {checks}", flush=True)
    _save_json(out / "bank_assembly.json", record)
    ctx["record"]["bank_assemble"] = record
    return bool(record["passed"])


def stage_bank_verify(args, ctx: Dict[str, Any]) -> bool:
    """Independent process: reload bank.pt and run A1-A5 exactly (no tolerance)."""
    _fs_source(args, ctx, "bank_verify")
    if args.bank_assembly is None:
        raise SystemExit("ERROR: --stage bank_verify needs --bank-assembly (bank_assembly.json).")
    authorization = authorize_bank_assembly(args, ctx)
    decision = authorization.pop("decision_payload")
    assembly = _read_json(args.bank_assembly)
    fs_bridge = ctx["fs_bridge"]
    f0_bridge = ctx["bridge"]
    device = next(fs_bridge.model.parameters()).device
    names = ctx["variables"]
    blocks = tuple(int(b) for b in ctx["target_blocks"])
    num_experts = int(args.num_experts)
    probe_steps = _panel_steps(args, ctx)
    probe = _probe_sample(ctx)
    x_probe = probe.to(device).x_norm

    bank, meta = bt.load_bank(assembly["bank_file"]["path"], device=device)
    device_check = bt.check_bank_device(bank, device, context="bank_verify")
    initial = _build_initial_bank(args, ctx)
    registry = bt.build_bank_registry(num_experts, rho=float(args.rho), a0=float(args.a0))
    decided = {int(e["expert_index"]): e for e in decision["experts_for_assembly"]}

    a1: Dict[str, Any] = {}
    a2: Dict[str, Any] = {}
    a4: Dict[str, Any] = {}
    for k in range(num_experts):
        plan = bt.expert_plan(registry, k, hold_steps=int(args.hold_steps))
        assembled = bt.singleton_probe_states(fs_bridge, bank, plan, x_probe, names,
                                              probe_steps=probe_steps, target_blocks=blocks)
        loaded = bt.load_expert(decided[k]["expert_file"]["path"],
                                expected_sha256=decided[k]["expert_file"]["sha256"],
                                expected_expert_index=k)
        source = bt.source_bank_for_expert(initial, loaded)
        bt.check_bank_device(source, device, context=f"bank_verify source bank {k}")
        from_source = bt.singleton_probe_states(fs_bridge, source, plan, x_probe, names,
                                                probe_steps=probe_steps, target_blocks=blocks)
        a1[str(k)] = bt.compare_probe_states(from_source, assembled)
        probe_ref = dict(decided[k].get("probe_file") or {})
        probe_sha = (fp.file_sha256(probe_ref["path"])
                     if probe_ref.get("path") and Path(probe_ref["path"]).is_file() else None)
        comparison: Dict[str, Any] = {"exact": False, "max_abs_diff_by_step": {},
                                      "torch_equal_by_step": {}}
        probe_issue = None
        if probe_sha is not None:
            stored = torch.load(probe_ref["path"], map_location="cpu", weights_only=True)
            probe_issue = stored.get("probe_issue_id")
            comparison = bt.compare_probe_states(stored["states"], assembled)
            if probe_issue != probe.issue_id or int(stored.get("expert_index", -1)) != k:
                comparison["exact"] = False
                comparison["max_abs_diff_by_step"] = {}
        a2[str(k)] = {**comparison, "probe_file_sha256": probe_sha,
                      "expected_probe_file_sha256": probe_ref.get("sha256"),
                      "probe_issue_id": probe_issue}
        a4[str(k)] = sa.verify_continuation_is_fs(
            fs_bridge, f0_bridge, bank, x_probe, names, hold=int(args.hold_steps),
            total=max(probe_steps), active_expert=k, coefficient=float(args.a0),
            rho=float(args.rho), target_blocks=blocks)
    a3 = sa.verify_zero_edit_equivalence(
        fs_bridge, bank, num_experts, x_probe, names, steps=max(probe_steps),
        target_blocks=blocks, f0_bridge=f0_bridge)
    a5 = {
        "file_sha256": meta["sha256"],
        "expected_file_sha256": assembly["bank_file"]["sha256"],
        "bank_digest": meta["bank_digest"],
        "expected_bank_digest": assembly["bank_digest"],
        "expert_digests": meta["expert_digests"],
        "expected_expert_digests": assembly["expert_digests"],
        "decided_expert_digests": [decided[k]["expert_digest"] for k in range(num_experts)],
        "assembly_decision_sha256_matches":
            (assembly.get("decision") or {}).get("sha256") == authorization["decision_sha256"],
    }
    record = {
        "check": "bank_independent_verify",
        "probe_issue_id": probe.issue_id,
        "probe_steps": list(probe_steps),
        "device_check": device_check,
        "A1_source_bank": a1,
        "A2_training_probe": a2,
        "A3_zero_edit": a3,
        "A4_continuation": a4,
        "A5_reload": a5,
        "bank_assembly": {"path": str(args.bank_assembly),
                          "sha256": fp.file_sha256(args.bank_assembly)},
        "authorization": authorization,
        "fs_source": ctx.get("fs_source"),
        "fs_identity": ctx.get("fs_identity"),
    }
    evaluation = bt.evaluate_bank_assembly(record, num_experts=num_experts)
    consistency = (a5["assembly_decision_sha256_matches"]
                   and a5["decided_expert_digests"] == a5["expected_expert_digests"])
    record["evaluation"] = evaluation
    record["decision_consistency"] = consistency
    record["passed"] = bool(evaluation["passed"] and consistency)
    print(f"  A1..A5        : {evaluation['status']} consistency={consistency}", flush=True)
    _save_json(Path(args.output_dir) / "bank_verify.json" if args.output_dir else None, record)
    ctx["record"]["bank_verify"] = record
    return bool(record["passed"])


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit/freeze Fs and train the K=4 dynamic expert bank. "
            "This script never submits or polls an ACP/CCI job."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Job submission is out of scope here by design: this is the payload "
            "a job wraps. See plans/plans_v2_0921/cci/submit_job.py."
        ),
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--mode", choices=sorted(sa.MODE_UPDATE_CAPS), default="gradient_check",
                        help="gradient_check caps at 32 updates; formal caps at 500")
    parser.add_argument("--num-experts", type=int, default=bt.DESIGN_NUM_EXPERTS)
    parser.add_argument("--rank-per-expert", type=int, default=bt.DESIGN_RANK_PER_EXPERT)
    parser.add_argument("--expert-index", type=int, default=None,
                        help="train exactly this expert (one GPU process per expert)")
    parser.add_argument("--data-role", choices=list(DATA_ROLE_VALUES),
                        default=DataRole.BANK_FIT.value,
                        help="Fs and the bank may only consume bank_fit")
    parser.add_argument("--target-blocks", type=int, nargs="+", default=None,
                        help="default: 18-23 for real runs, 0..depth-1 for synthetic")
    parser.add_argument("--hold-steps", type=int, default=bt.DESIGN_HOLD_STEPS)
    parser.add_argument("--train-steps", type=int, default=None,
                        help="differentiable rollout horizon; defaults to --hold-steps")
    parser.add_argument("--lead-steps", type=int, nargs="+", default=None,
                        help="rollout steps forming the objective's H axis; "
                             "defaults to (--train-steps,)")
    parser.add_argument("--max-updates", type=int, default=None,
                        help="defaults to the cap for --mode")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--a0", type=float, default=bt.DESIGN_A0)
    parser.add_argument("--rho", type=float, default=bt.DESIGN_RHO)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", type=str, default=None,
                        help="default: cuda when available, else cpu")

    parser.add_argument("--synthetic", action="store_true",
                        help="tiny CPU-testable model and random samples; every "
                             "output is tagged synthetic and proves nothing "
                             "about weather or capacity")
    parser.add_argument("--synthetic-samples", type=int, default=4)
    parser.add_argument("--config", type=Path, default=None,
                        help="gate_config.json naming the real backbone identity")
    parser.add_argument("--admission", type=Path, default=None,
                        help="admission record from scripts/r2_admission_gate.py")
    parser.add_argument("--limit", type=int, default=None,
                        help="maximum admitted samples to load")

    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--profile-warmup", type=int, default=0)
    parser.add_argument("--memory-budget-gib", type=float, default=80.0,
                        help="declared per-GPU memory budget (H100 80GiB default)")
    parser.add_argument("--memory-safety-fraction", type=float, default=0.8)
    parser.add_argument("--max-seconds-per-update", type=float, default=30.0)
    parser.add_argument("--max-differentiable-steps", type=int, default=None,
                        help="force the horizon check to treat longer horizons "
                             "as infeasible; for exercising the fallback path")

    # FP-03 pre-registration binding (mandatory for real runs).
    parser.add_argument("--protocol", type=Path, default=None,
                        help="pre-registered FP-03 protocol JSON")
    parser.add_argument("--protocol-sha256", type=str, default=None,
                        help="SHA-256 the protocol file must hash to")
    parser.add_argument("--config-id", type=str, default=None,
                        help="protocol configuration id this process runs")
    parser.add_argument("--s0-certificate", type=Path, default=None,
                        help="committed S0 PASS s0_gate_result.json to consume first")
    parser.add_argument("--require-official-backend", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="refuse unless the certified xformers/official Stormer is "
                             "built (default: on for real runs)")
    parser.add_argument("--panel-admission", type=Path, default=None,
                        help="certified holdout-panel admission (never trained on)")
    parser.add_argument("--panel-lead-steps", type=int, nargs="+", default=None,
                        help="rollout steps of the fixed panels, e.g. 1 4 12")
    parser.add_argument("--fs-adapter", type=Path, default=None,
                        help="saved Fs adapter (fs_panel / fs_freeze / fs_verify)")
    parser.add_argument("--fs-adapter-sha256", type=str, default=None)
    parser.add_argument("--fs-merged-backbone", type=Path, default=None,
                        help="fs_merged_backbone.pt to reload (fs_verify)")
    parser.add_argument("--fs-decision", type=Path, default=None,
                        help="FS-SELECT-v1 decision authorizing fs_freeze")
    parser.add_argument("--screen-decision", type=Path, default=None,
                        help="FS-SCREEN-v1 decision a formal fit's lr is bound to")

    # FP-04: the certified Fs reference every post-Fs stage consumes, and the
    # bank decision / assembly record the assemble / verify stages consume.
    parser.add_argument("--fs-reference-manifest", type=Path, default=None,
                        help="certified Fs reference_manifest.json (FP-03 certify/fs/ bundle)")
    parser.add_argument("--fs-reference-manifest-sha256", type=str, default=None,
                        help="SHA-256 the reference manifest must hash to")
    parser.add_argument("--fs-certify-decision", type=Path, default=None,
                        help="FS_SELECTED decision from r4_fs_decide.py --kind certify")
    parser.add_argument("--bank-decision", type=Path, default=None,
                        help="BANK-SELECT-v1 formal decision authorizing bank_assemble/bank_verify")
    parser.add_argument("--bank-assembly", type=Path, default=None,
                        help="bank_assembly.json written by --stage bank_assemble (bank_verify)")

    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None,
                        help="write the combined run record here")
    return parser


def resolve_args(args) -> None:
    """Fill defaults that depend on other arguments, and refuse bad combos."""
    if args.train_steps is None:
        args.train_steps = int(args.hold_steps)
    if args.lead_steps is None:
        args.lead_steps = [int(args.train_steps)]
    if args.max_updates is None:
        args.max_updates = sa.MODE_UPDATE_CAPS[args.mode]


def stage_fs_verify(args, ctx: Dict[str, Any]) -> bool:
    """Independent process: reload the frozen Fs and re-verify identity/behaviour.

    Nothing from the freezing process is trusted: the merged backbone and the
    adapter are read back from their files, their digests are recomputed, and
    merge equivalence, zero-edit and the continuation-after-hold property are
    re-run against this process's own freshly loaded F0 bridge.
    """
    if args.fs_merged_backbone is None or args.fs_adapter is None or args.fs_adapter_sha256 is None:
        raise SystemExit("ERROR: --stage fs_verify needs --fs-merged-backbone, --fs-adapter "
                         "and --fs-adapter-sha256.")
    bridge = ctx["bridge"]
    device = _device_of(ctx)
    payload = torch.load(args.fs_merged_backbone, map_location="cpu", weights_only=True)
    artifact_dict = dict(payload["artifact"])
    merged = copy.deepcopy(bridge.model)
    merged.load_state_dict(payload["state_dict"], strict=True)
    merged.requires_grad_(False)
    merged.eval()
    adapters, adapter_sha = sa.load_fs_adapter(
        args.fs_adapter, _hidden_size(ctx), ctx["target_blocks"],
        rank_per_expert=int(args.rank_per_expert),
        expected_sha256=args.fs_adapter_sha256, device=device,
    )
    identity = {
        "merged_backbone_digest_matches": sa.state_dict_digest(merged)
        == artifact_dict["merged_backbone_digest"],
        "base_backbone_digest_matches": sa.state_dict_digest(bridge.model)
        == artifact_dict["base_backbone_digest"],
        "static_adapter_digest_matches": sa.static_adapter_digest(
            adapters, target_blocks=ctx["target_blocks"]) == artifact_dict["static_adapter_digest"],
    }
    artifact = sa.MergedBackboneArtifact(
        **{**artifact_dict, "target_blocks": tuple(artifact_dict["target_blocks"])})
    fs_bridge = sa.make_fs_bridge(merged, bridge, artifact)
    probe = sa.build_continuation_probe_bank(
        _hidden_size(ctx), ctx["target_blocks"], num_experts=int(args.num_experts),
        rank_per_expert=int(args.rank_per_expert), seed=int(args.seed))
    for lora in probe.values():
        lora.to(device)
    verification = sa.record_post_freeze_reference(
        bridge, fs_bridge, adapters, artifact, ctx["samples"][0].x_norm, ctx["variables"],
        num_experts=int(args.num_experts), dynamic_bank=probe, steps=int(args.hold_steps),
        target_blocks=ctx["target_blocks"], samples=ctx["samples"], spec=ctx["objective"],
    )
    continuation = sa.verify_continuation_is_fs(
        fs_bridge, bridge, probe, ctx["samples"][0].x_norm, ctx["variables"],
        hold=int(args.hold_steps), total=12, target_blocks=ctx["target_blocks"],
    )
    result = {
        "check": "fs_independent_reload",
        "adapter_sha256": adapter_sha,
        "merged_backbone_sha256": fp.file_sha256(args.fs_merged_backbone),
        "identity": identity,
        "identity_pass": all(identity.values()),
        "post_freeze": verification.to_dict(),
        "numerical_merge_pass": bool(verification.merge_equivalence.get("passed")),
        "zero_edit_pass": bool(verification.zero_edit_equivalence.get("passed")),
        "continuation": continuation,
        "merge_tolerance_source": (
            "record_post_freeze_reference defaults (atol=max(1e-5, 1.5e-3*delta_max_abs), "
            "rtol=1e-5); r4_fs_decide --kind certify checks they equal the protocol's "
            "pre-registered merge_equivalence_tolerance"),
    }
    result["reload_pass"] = bool(result["identity_pass"] and result["zero_edit_pass"]
                                 and continuation["passed"])
    print(f"  identity      : {identity}", flush=True)
    print(f"  merge/zero/continuation: {result['numerical_merge_pass']} / "
          f"{result['zero_edit_pass']} / {continuation['passed']}", flush=True)
    _save_json(Path(args.output_dir) / "independent_reload.json" if args.output_dir else None,
               result)
    ctx["record"]["fs_verify"] = result
    return bool(result["reload_pass"] and result["numerical_merge_pass"])


def _cli_view(args) -> Dict[str, Any]:
    """The command line in the vocabulary a protocol configuration declares."""
    return {
        "stage": args.stage,
        "mode": args.mode,
        "max_updates": int(args.max_updates),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "rank_per_expert": int(args.rank_per_expert),
        "num_experts": int(args.num_experts),
        "target_blocks": list(args.target_blocks) if args.target_blocks is not None
        else list(sa.DEFAULT_TARGET_BLOCKS),
        "train_steps": int(args.train_steps),
        "lead_steps": [int(s) for s in args.lead_steps],
        "panel_lead_steps": ([int(s) for s in args.panel_lead_steps]
                             if args.panel_lead_steps else None),
        "hold_steps": int(args.hold_steps),
        "data_role": args.data_role,
        "device": args.device,
        "config": args.config,
        "admission": args.admission,
        "panel_admission": args.panel_admission,
        "s0_certificate": args.s0_certificate,
        "fs_adapter": args.fs_adapter,
        "fs_adapter_sha256": args.fs_adapter_sha256,
        "fs_merged_backbone": args.fs_merged_backbone,
        "fs_decision": args.fs_decision,
        "screen_decision": args.screen_decision,
        "limit": args.limit,
        # FP-04 bank stages
        "expert_index": args.expert_index,
        "a0": float(args.a0),
        "rho": float(args.rho),
        "fs_reference_manifest": args.fs_reference_manifest,
        "fs_certify_decision": args.fs_certify_decision,
        "profile_steps": int(args.profile_steps),
        "profile_warmup": int(args.profile_warmup),
        "memory_budget_gib": float(args.memory_budget_gib),
        "memory_safety_fraction": float(args.memory_safety_fraction),
        "max_seconds_per_update": float(args.max_seconds_per_update),
        "max_differentiable_steps": args.max_differentiable_steps,
        "bank_decision": args.bank_decision,
        "bank_assembly": args.bank_assembly,
    }


def _backend_rule(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Certified backend versions: FS-QUAL's q0, or BANK-QUAL-v1's e0."""
    rule = protocol.get("qualification_rule") or {}
    return dict(rule.get("q0") or rule.get("e0") or {})


def _bind_screen_lr(args, declared: Dict[str, Any], protocol_sha256: str) -> Dict[str, Any]:
    """A formal lr declared as `{"from_screen_decision": field}` must equal the
    CONFIRMED FS-SCREEN-v1 decision's field, from a decision bound to this protocol."""
    field = declared["learning_rate"]["from_screen_decision"]
    if args.screen_decision is None:
        raise fp.ProtocolViolation("SCREEN_DECISION_REQUIRED",
                                   "this configuration's lr comes from --screen-decision.")
    decision = json.loads(Path(args.screen_decision).read_text())
    failures = []
    if decision.get("rule") != "FS-SCREEN-v1" or decision.get("verdict") != "CONFIRMED":
        failures.append(f"screen decision verdict is {decision.get('verdict')!r}")
    if decision.get("protocol_sha256") != protocol_sha256:
        failures.append("screen decision is not bound to this protocol")
    want = decision.get(field)
    if want is None or float(want) != float(args.learning_rate):
        failures.append(f"--learning-rate {args.learning_rate} != decision {field}={want}")
    if failures:
        raise fp.ProtocolViolation("SCREEN_DECISION_MISMATCH", "; ".join(failures))
    return {"screen_decision": str(args.screen_decision),
            "screen_decision_sha256": fp.file_sha256(args.screen_decision),
            "field": field, "learning_rate": float(want)}


def bind_real_run(args) -> Dict[str, Any]:
    """Everything a real run must establish before loading a backbone."""
    missing = [flag for flag, value in (
        ("--protocol", args.protocol), ("--protocol-sha256", args.protocol_sha256),
        ("--config-id", args.config_id), ("--s0-certificate", args.s0_certificate),
    ) if value is None]
    if missing:
        raise fp.ProtocolViolation(
            "PREREGISTRATION_MISSING",
            f"a real run needs {', '.join(missing)}; unregistered real fits are refused.")
    protocol = fp.load_protocol(args.protocol, args.protocol_sha256)
    kind = protocol.get("protocol_kind", PROTOCOL_KIND_FS)
    wanted_kind = PROTOCOL_KIND_BANK if args.stage in BANK_STAGES else PROTOCOL_KIND_FS
    if kind != wanted_kind:
        raise fp.ProtocolViolation(
            "PROTOCOL_KIND_MISMATCH",
            f"--stage {args.stage} runs under a {wanted_kind!r} protocol; "
            f"{args.protocol} is a {kind!r} protocol.", {"kind": kind, "wanted": wanted_kind})
    declared = fp.declared_config(protocol, args.config_id)
    required = REQUIRED_DECLARED.get(args.stage, ())
    under = [key for key in required if key not in declared]
    if under:
        raise fp.ProtocolViolation("CONFIG_UNDERDECLARED",
                                   f"{args.config_id} does not declare {under}.")
    binding: Dict[str, Any] = {"protocol": str(args.protocol),
                               "protocol_sha256": args.protocol_sha256,
                               "config_id": args.config_id, "declared": declared}
    check = dict(declared)
    if isinstance(declared.get("learning_rate"), dict):
        binding["screen_lr"] = _bind_screen_lr(args, declared, args.protocol_sha256)
        check.pop("learning_rate")
    mismatches = fp.check_cli_against_config(check, _cli_view(args))
    if mismatches:
        raise fp.ProtocolViolation("CLI_DIFFERS_FROM_DECLARED_CONFIG", "; ".join(mismatches),
                                   {"mismatches": mismatches})
    inputs = protocol.get("inputs") or {}
    if inputs.get("s0_certificate", {}).get("sha256") is None:
        raise fp.ProtocolViolation("S0_CERTIFICATE_UNDECLARED",
                                   "the protocol declares no S0 certificate sha256.")
    gate_config = GateIdentityConfig.load_json(args.config)
    binding["s0"] = fp.consume_s0_certificate(
        args.s0_certificate, expected_sha256=inputs["s0_certificate"]["sha256"],
        gate_config=gate_config)
    q0 = _backend_rule(protocol)
    if args.require_official_backend:
        binding["backend_precondition"] = fp.official_backend_precondition(
            expected_xformers=q0["xformers_version"], expected_torch=q0["torch_version"])
    return {"protocol": protocol, "binding": binding}


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv if argv is None else ["r2_fs_bank_train.py", *argv])
    args = build_parser().parse_args(argv)
    resolve_args(args)
    started_utc = fp.utc_now()
    real = not args.synthetic
    if args.require_official_backend is None:
        args.require_official_backend = real

    if args.data_role != DataRole.BANK_FIT.value:
        print(
            f"ERROR: --data-role {args.data_role!r} is refused. Fs and the "
            "expert bank are fit on bank_fit data only; training on policy_dev "
            "would let the bank see the blocks the cheap policies are evaluated "
            "on, and confirm is spent the first time it is read.",
            file=sys.stderr, flush=True,
        )
        return 2
    if real and (args.config is None or args.admission is None):
        print(
            "ERROR: a real run needs --config (gate_config.json) and "
            "--admission (the record written by scripts/r2_admission_gate.py). "
            "Use --synthetic for a CPU smoke run.",
            file=sys.stderr, flush=True,
        )
        return 2
    if real and args.stage == "all":
        print("ERROR: --stage all is refused for real runs: every real stage runs on its "
              "own and consumes its predecessor's certificate.", file=sys.stderr, flush=True)
        return 2
    if args.expert_index is not None and not (0 <= args.expert_index < args.num_experts):
        print(
            f"ERROR: --expert-index {args.expert_index} is outside "
            f"[0, {args.num_experts}).",
            file=sys.stderr, flush=True,
        )
        return 2

    bound: Dict[str, Any] = {"protocol": None, "binding": {}}
    if real:
        try:
            bound = bind_real_run(args)
        except (fp.ProtocolViolation, OSError, ValueError, KeyError) as exc:
            print(f"ERROR: pre-registration binding refused: {exc}", file=sys.stderr, flush=True)
            if args.output_dir is not None:
                _save_json(Path(args.output_dir) / "binding_refused.json", {
                    "error": str(exc), "code": getattr(exc, "code", type(exc).__name__),
                    "detail": getattr(exc, "detail", {}), "argv": raw_argv,
                    "utc": fp.utc_now()})
            return 2
    protocol = bound["protocol"]
    if args.stage in CERTIFIED_FS_STAGES:
        # FP-04: the certified Fs is bound by hash BEFORE any backbone, sample
        # or Fs byte is loaded; real or synthetic, there is no other way in.
        try:
            bound["binding"]["fs_reference"] = authorize_certified_fs(args, protocol)
        except (fp.ProtocolViolation, OSError, ValueError, KeyError) as exc:
            print(f"ERROR: certified Fs refused: {exc}", file=sys.stderr, flush=True)
            if args.output_dir is not None:
                _save_json(Path(args.output_dir) / "binding_refused.json", {
                    "error": str(exc), "code": getattr(exc, "code", type(exc).__name__),
                    "detail": getattr(exc, "detail", {}), "argv": raw_argv,
                    "utc": fp.utc_now()})
            return 2

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 68, flush=True)
    print("R2 Stage-4: Fs fit/freeze + dynamic expert-bank training", flush=True)
    print("=" * 68, flush=True)
    print(f"  stage={args.stage} mode={args.mode} device={device}", flush=True)
    print(
        f"  K={args.num_experts} rank={args.rank_per_expert} a0={args.a0} "
        f"rho={args.rho} max_active={bt.DESIGN_MAX_ACTIVE} "
        f"hold={args.hold_steps}",
        flush=True,
    )
    print(f"  synthetic={args.synthetic} submits_jobs={SUBMITS_JOBS} "
          f"config_id={args.config_id}", flush=True)

    panel_leads = [int(s) for s in args.panel_lead_steps] if args.panel_lead_steps else []
    sample_leads = sorted({int(s) for s in args.lead_steps} | set(panel_leads))
    binding = bound["binding"]
    try:
        if args.synthetic:
            bridge, variables, lat = build_synthetic_bridge(device, seed=args.seed)
            target_blocks = tuple(
                args.target_blocks if args.target_blocks is not None
                else range(len(bridge.model.blocks))
            )
            samples = build_synthetic_samples(
                bridge, sample_leads, n_samples=int(args.synthetic_samples),
                device=device, seed=args.seed,
            )
            panel_samples = []
            if args.panel_admission is not None or panel_leads:
                panel_samples = build_synthetic_samples(
                    bridge, sample_leads, n_samples=int(args.synthetic_samples),
                    device=device, seed=args.seed + 1,
                )
                for i, sample in enumerate(panel_samples):
                    sample.issue_id = f"iss_synthetic_holdout_{i:03d}"
            config_summary: Dict[str, Any] = {"synthetic": True}
        else:
            bridge, variables, lat, gate_config = build_real_bridge(args.config, device)
            if args.require_official_backend:
                binding["backend_postcondition"] = fp.official_backend_postcondition(bridge.model)
            target_blocks = tuple(
                args.target_blocks if args.target_blocks is not None
                else sa.DEFAULT_TARGET_BLOCKS
            )
            certification: Dict[str, Any] = {}
            samples = load_real_samples(
                args.admission, bridge, sample_leads,
                limit=args.limit, device=device, certification_out=certification,
            )
            binding["admission_consumer"] = certification
            panel_samples = []
            if args.panel_admission is not None:
                panel_certification: Dict[str, Any] = {}
                panel_samples = load_real_samples(
                    args.panel_admission, bridge, sample_leads, limit=None,
                    device=device, certification_out=panel_certification,
                )
                binding["panel_admission_consumer"] = panel_certification
                overlap = {s.issue_id for s in samples} & {s.issue_id for s in panel_samples}
                if overlap:
                    raise sa.StaticAdapterViolation(
                        "PANEL_OVERLAPS_TRAINING",
                        f"holdout panel shares issue(s) {sorted(overlap)[:4]} with training.")
            config_summary = {
                "synthetic": False,
                "gate_config": str(args.config),
                "gate_config_digest": gate_config.config_digest,
                "identity_tag": gate_config.identity_tag,
                "admission_record": str(args.admission),
                "panel_admission_record": str(args.panel_admission) if args.panel_admission else None,
            }

        objective = sa.build_objective_spec(
            bridge, lat, lead_steps=tuple(args.lead_steps), space="raw"
        )
        print(
            f"  samples={len(samples)} blocks={list(target_blocks)} "
            f"objective_leads={objective.lead_hours}h panel_leads={panel_leads} "
            f"holdout={len(panel_samples)}",
            flush=True,
        )
    except (sa.StaticAdapterViolation, bt.BankTrainingViolation, fp.ProtocolViolation,
            ValueError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if args.output_dir is not None:
            _save_json(Path(args.output_dir) / "binding_refused.json", {
                "error": str(exc), "code": getattr(exc, "code", type(exc).__name__),
                "detail": getattr(exc, "detail", {}), "argv": raw_argv, "utc": fp.utc_now()})
        return 2

    provenance = fp.collect_provenance(
        argv=raw_argv, device=device, source_roots=[SOURCE_ROOT], model=bridge.model,
        input_files={
            "config": args.config, "admission": args.admission,
            "panel_admission": args.panel_admission, "protocol": args.protocol,
            "s0_certificate": args.s0_certificate, "fs_adapter": args.fs_adapter,
            "fs_decision": args.fs_decision, "screen_decision": args.screen_decision,
            "fs_reference_manifest": args.fs_reference_manifest,
            "fs_certify_decision": args.fs_certify_decision,
            "bank_decision": args.bank_decision, "bank_assembly": args.bank_assembly,
        },
        started_utc=started_utc,
    )
    expected_initial = None
    bank_pins: Dict[str, Any] = {}
    if protocol is not None:
        expected_initial = (protocol.get("inputs") or {}).get("initial_adapter_digest")
        bank_pins = dict((protocol.get("inputs") or {}).get("bank") or {})

    ctx: Dict[str, Any] = {
        "bridge": bridge,
        "variables": variables,
        "samples": samples,
        "panel_samples": panel_samples,
        "objective": objective,
        "target_blocks": target_blocks,
        "lead_steps": tuple(args.lead_steps),
        "panel_lead_steps": tuple(panel_leads),
        "protocol": protocol,
        "expected_initial_adapter_digest": expected_initial,
        "bank_pins": bank_pins,
        "record": {
            "script": "r2_fs_bank_train",
            "stage": args.stage,
            "mode": args.mode,
            "device": str(device),
            "config_id": args.config_id,
            "protocol_sha256": args.protocol_sha256,
            "submits_jobs": SUBMITS_JOBS,
            "polls_jobs": POLLS_JOBS,
            "design": {
                "num_experts": int(args.num_experts),
                "rank_per_expert": int(args.rank_per_expert),
                "a0": float(args.a0),
                "rho": float(args.rho),
                "max_active": bt.DESIGN_MAX_ACTIVE,
                "hold_steps": int(args.hold_steps),
                "target_blocks": list(target_blocks),
            },
            "data": {
                "data_role": args.data_role,
                "n_samples": len(samples),
                "n_panel_samples": len(panel_samples),
                "training_issue_ids": [s.issue_id for s in samples],
                "panel_issue_ids": [s.issue_id for s in panel_samples],
                **config_summary,
            },
            "objective": objective.to_dict(),
            "binding": binding,
            "provenance": provenance,
        },
    }

    if args.stage in CERTIFIED_FS_STAGES:
        try:
            load_certified_fs(args, ctx)
        except (fp.ProtocolViolation, sa.StaticAdapterViolation, OSError, ValueError,
                KeyError, RuntimeError) as exc:
            print(f"ERROR: certified Fs could not be loaded as certified: {exc}",
                  file=sys.stderr, flush=True)
            if args.output_dir is not None:
                _save_json(Path(args.output_dir) / "binding_refused.json", {
                    "error": str(exc), "code": getattr(exc, "code", type(exc).__name__),
                    "detail": getattr(exc, "detail", {}), "argv": raw_argv,
                    "utc": fp.utc_now()})
            return 2
        print(f"  certified Fs  : merged digest "
              f"{ctx['fs_identity']['digests']['merged_backbone_digest'][:16]}... "
              f"identity_pass={ctx['fs_identity']['identity_pass']}", flush=True)

    stages = (
        ["fs_fit", "fs_freeze", "bank_train", "profile", "horizon_check"]
        if args.stage == "all" else [args.stage]
    )
    handlers = {
        "fs_fit": stage_fs_fit,
        "fs_panel": stage_fs_panel,
        "fs_freeze": stage_fs_freeze_in_chain if args.stage == "all" else stage_fs_freeze_from_file,
        "fs_verify": stage_fs_verify,
        "bank_train": stage_bank_train,
        "profile": stage_profile,
        "horizon_check": stage_horizon_check,
        "bank_assemble": stage_bank_assemble,
        "bank_verify": stage_bank_verify,
    }
    results: Dict[str, bool] = {}
    blocked: List[str] = []
    for position, name in enumerate(stages):
        print(f"\n[{name}]", flush=True)
        try:
            results[name] = bool(handlers[name](args, ctx))
        except (sa.StaticAdapterViolation, bt.BankTrainingViolation, fp.ProtocolViolation) as exc:
            print(f"  FAIL [{exc.code}] {exc.message[:300]}", file=sys.stderr, flush=True)
            ctx["record"].setdefault("errors", []).append(
                {"stage": name, "code": exc.code, "message": exc.message,
                 "detail": getattr(exc, "detail", {})}
            )
            results[name] = False
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            print(f"  ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            ctx["record"].setdefault("errors", []).append(
                {"stage": name, "code": type(exc).__name__, "message": str(exc)}
            )
            results[name] = False
        if not results[name]:
            blocked = stages[position + 1:]
            if blocked:
                print(f"  STOP: {name} failed; successors BLOCKED and not called: {blocked}",
                      flush=True)
            break

    ctx["record"]["results"] = results
    ctx["record"]["blocked_stages"] = blocked
    ctx["record"]["passed"] = bool(results) and all(results.values()) and not blocked
    ctx["record"]["provenance"]["finished_utc"] = fp.utc_now()

    print("", flush=True)
    for name, ok in results.items():
        print(f"  - {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    for name in blocked:
        print(f"  - {name}: BLOCKED (not called)", flush=True)
    print(f"\nRESULT: {'PASS' if ctx['record']['passed'] else 'FAIL'}", flush=True)
    if args.synthetic:
        print(
            "SCOPE: SYNTHETIC_CODE_PATH_CHECK_ONLY -- no weather skill, no "
            "capacity finding, no S0.",
            flush=True,
        )

    if args.output_dir is not None:
        _save_json(Path(args.output_dir) / "provenance.json", ctx["record"]["provenance"])
    _save_json(args.json, ctx["record"])
    if args.output_dir is not None and args.json is None:
        _save_json(Path(args.output_dir) / "run_record.json", ctx["record"])

    return 0 if ctx["record"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
