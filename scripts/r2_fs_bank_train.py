#!/usr/bin/env python3
"""Stage-4 entry point: fit/freeze Fs, then train the K=4 dynamic expert bank.

Implements the runnable half of `plans/plans_v2_0921/CLAUDE_EXECUTION_PLAN.md`
sections 6.1-6.2. One script, five stages, each of which can be run on its own:

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
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch  # noqa: E402

from earthdelta import bank_training as bt  # noqa: E402
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

STAGES = ("fs_fit", "fs_freeze", "bank_train", "profile", "horizon_check", "all")


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
) -> List[sa.TrainingSample]:
    record = json.loads(Path(admission_path).read_text())
    return sa.load_admitted_samples(
        record, bridge, lead_steps=lead_steps, limit=limit, device=device
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


def stage_fs_fit(args, ctx: Dict[str, Any]) -> bool:
    bridge = ctx["bridge"]
    spec = ctx["objective"]
    samples = ctx["samples"]

    hidden = bridge.model.blocks[ctx["target_blocks"][0]].attn.proj.in_features
    adapters = sa.build_fs_adapter(
        hidden,
        ctx["target_blocks"],
        rank_per_expert=int(args.rank_per_expert),
        seed=int(args.seed),
    )
    config = sa.FsFitConfig(
        mode=args.mode,
        max_updates=int(args.max_updates),
        learning_rate=float(args.learning_rate),
        train_steps=int(args.train_steps),
        lead_steps=tuple(ctx["lead_steps"]),
        target_blocks=tuple(ctx["target_blocks"]),
        rank_per_expert=int(args.rank_per_expert),
        seed=int(args.seed),
    )
    adapters, record = sa.fit_static_adapter(
        bridge, samples, spec, config, fs_adapters=adapters, variables=ctx["variables"]
    )
    ctx["fs_adapters"] = adapters
    ctx["fs_fit_record"] = record

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

    if args.output_dir is not None:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {block: lora.state_dict() for block, lora in adapters.items()},
            out / "fs_adapter.pt",
        )
        print(f"  wrote {out / 'fs_adapter.pt'}", flush=True)
    _save_json(
        Path(args.output_dir) / "fs_fit_record.json" if args.output_dir else None,
        record.to_dict(),
    )
    ctx["record"]["fs_fit"] = record.to_dict()
    return bool(record.eligible)


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


def stage_bank_train(args, ctx: Dict[str, Any]) -> bool:
    fs_bridge = ctx.get("fs_bridge")
    if fs_bridge is None:
        raise SystemExit(
            "ERROR: --stage bank_train needs a frozen Fs-merged backbone. Run "
            "--stage fs_freeze first (or --stage all)."
        )
    bank = ctx.get("dynamic_bank")
    if bank is None:
        bank = bt.build_dynamic_bank(
            fs_bridge.model.blocks[ctx["target_blocks"][0]].attn.proj.in_features,
            ctx["target_blocks"],
            num_experts=int(args.num_experts),
            rank_per_expert=int(args.rank_per_expert),
            seed=int(args.seed),
        )
        ctx["dynamic_bank"] = bank

    grouping = bt.assign_diversity_groups(
        ctx["samples"],
        int(args.num_experts),
        hold_steps=int(args.hold_steps),
    )
    ctx["grouping"] = grouping
    print(f"  diversity rule: {grouping.rule}", flush=True)
    for group in grouping.groups:
        payload = group.to_dict()
        print(
            f"    expert {group.expert_index}: months={payload['months']} "
            f"n={payload['n_samples']}",
            flush=True,
        )
    if grouping.purged:
        print(f"    purged: {len(grouping.purged)} sample(s)", flush=True)

    registry = bt.build_bank_registry(
        int(args.num_experts), rho=float(args.rho), a0=float(args.a0)
    )
    ctx["registry"] = registry

    expert_indices = (
        [int(args.expert_index)]
        if args.expert_index is not None
        else list(range(int(args.num_experts)))
    )
    config = bt.BankTrainConfig(
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

    artifact = ctx.get("fs_artifact")
    records: List[Dict[str, Any]] = []
    passed = True
    for index in expert_indices:
        group = grouping.group_for(index)
        group_samples = grouping.samples_for(index, ctx["samples"])
        if not group_samples:
            print(
                f"  expert {index}: diversity group is empty after the guard "
                "purge; nothing to train. Recorded as NO_UPDATES_RUN rather "
                "than silently skipped.",
                flush=True,
            )
            record = bt.ExpertTrainingRecord(
                expert_index=index,
                config=config.to_dict(),
                diversity_group=group.to_dict(),
                eligible=False,
                ineligible_reason="NO_UPDATES_RUN",
            )
            records.append(record.to_dict())
            passed = False
            continue

        record = bt.train_expert(
            fs_bridge,
            bank,
            index,
            group_samples,
            ctx["objective"],
            config,
            registry=registry,
            diversity_group=group,
            variables=ctx["variables"],
            fs_backbone_digest=artifact.merged_backbone_digest if artifact else "",
            fs_static_adapter_digest=artifact.static_adapter_digest if artifact else "",
        )
        records.append(record.to_dict())
        print(
            f"  expert {index}: plan={record.plan_id} "
            f"updates={record.n_updates} loss {record.initial_loss} -> "
            f"{record.final_loss} eligible={record.eligible} "
            f"nonzero_response={record.nonzero_response.get('nonzero')}",
            flush=True,
        )
        passed = passed and bool(record.eligible)

    ctx["expert_records"] = records
    payload = {
        "grouping": grouping.to_dict(),
        "registry": registry.to_dict(),
        "experts": records,
    }
    suffix = (
        f"_expert{int(args.expert_index)}" if args.expert_index is not None else ""
    )
    _save_json(
        Path(args.output_dir) / f"bank_training{suffix}.json"
        if args.output_dir else None,
        payload,
    )
    ctx["record"]["bank_train"] = payload
    return passed


def stage_profile(args, ctx: Dict[str, Any]) -> bool:
    bridge = ctx.get("fs_bridge") or ctx["bridge"]
    bank = ctx.get("dynamic_bank") or bt.build_dynamic_bank(
        bridge.model.blocks[ctx["target_blocks"][0]].attn.proj.in_features,
        ctx["target_blocks"],
        num_experts=int(args.num_experts),
        rank_per_expert=int(args.rank_per_expert),
        seed=int(args.seed),
    )
    ctx["dynamic_bank"] = bank

    config = bt.BankTrainConfig(
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
    profile = bt.profile_training_step(
        bridge,
        bank,
        ctx["samples"][0],
        ctx["objective"],
        expert_index=int(args.expert_index or 0),
        config=config,
        variables=ctx["variables"],
        n_measured_steps=int(args.profile_steps),
        warmup_steps=int(args.profile_warmup),
        synthetic=bool(args.synthetic),
    )
    budget = bt.CapacityBudget(
        memory_budget_bytes=int(float(args.memory_budget_gib) * (1024 ** 3)),
        memory_safety_fraction=float(args.memory_safety_fraction),
        max_seconds_per_update=float(args.max_seconds_per_update),
    )
    decision = bt.decide_bank_capacity(profile, budget)
    ctx["profile"] = profile
    ctx["capacity_decision"] = decision

    print(f"  device        : {profile.device} ({profile.device_name})", flush=True)
    print(
        f"  step          : {profile.seconds_per_update:.4f} s/update over "
        f"{profile.n_measured_steps} step(s)",
        flush=True,
    )
    print(
        f"  peak memory   : {profile.peak_memory_bytes} bytes "
        f"({profile.memory_source})",
        flush=True,
    )
    print(
        f"  decision      : K={decision.decided_num_experts} "
        f"rank={decision.decided_rank_per_expert} "
        f"shrink={decision.shrink_required} provisional={decision.provisional}",
        flush=True,
    )
    for reason in decision.reasons:
        print(f"    - {reason}", flush=True)
    if decision.provisional:
        print(f"    PROVISIONAL: {decision.provisional_reason}", flush=True)

    payload = {"profile": profile.to_dict(), "decision": decision.to_dict()}
    _save_json(
        Path(args.output_dir) / "capacity_profile.json" if args.output_dir else None,
        payload,
    )
    ctx["record"]["profile"] = payload
    # A profile that ran is a successful profile even if it recommends shrinking;
    # only an errored measurement is a failure of this stage.
    return profile.error is None


def stage_horizon_check(args, ctx: Dict[str, Any]) -> bool:
    bridge = ctx.get("fs_bridge") or ctx["bridge"]
    bank = ctx.get("dynamic_bank") or bt.build_dynamic_bank(
        bridge.model.blocks[ctx["target_blocks"][0]].attn.proj.in_features,
        ctx["target_blocks"],
        num_experts=int(args.num_experts),
        rank_per_expert=int(args.rank_per_expert),
        seed=int(args.seed),
    )
    ctx["dynamic_bank"] = bank

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
        bank,
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
    ctx["horizon"] = report

    print(
        f"  requested     : {report.requested_hold_steps} steps "
        f"({report.requested_hours}h)",
        flush=True,
    )
    print(
        f"  feasible      : {report.feasible} (max "
        f"{report.max_feasible_differentiable_steps} differentiable steps)",
        flush=True,
    )
    print(
        f"  activation ckpt: enabled={report.activation_checkpointing_enabled} "
        f"(never turned on to reach a longer horizon)",
        flush=True,
    )
    if report.fallback is not None:
        spec = report.fallback
        print(f"  FALLBACK SPEC : {spec.reason}", flush=True)
        print(
            f"    train {spec.train_hours}h differentiable; evaluate "
            f"{list(spec.eval_hours)}h non-differentiably",
            flush=True,
        )
        print(f"    new identity : {spec.experiment_identity}", flush=True)
        print(f"    supersedes   : {spec.superseded_identity}", flush=True)
        derived = spec.cost_estimate["derived"]
        print(
            f"    cost (derived): train "
            f"{derived['training_gpu_hours_all_experts']} GPU-h, eval "
            f"{derived['evaluation_gpu_hours']} GPU-h",
            flush=True,
        )

    _save_json(
        Path(args.output_dir) / "horizon_feasibility.json"
        if args.output_dir else None,
        report.to_dict(),
    )
    ctx["record"]["horizon_check"] = report.to_dict()
    # The check succeeds when it produced an answer: either the full horizon is
    # feasible, or an explicit fallback spec was written. Silence is the only
    # failure.
    return bool(report.feasible or report.fallback is not None)


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_args(args)

    if args.data_role != DataRole.BANK_FIT.value:
        print(
            f"ERROR: --data-role {args.data_role!r} is refused. Fs and the "
            "expert bank are fit on bank_fit data only; training on policy_dev "
            "would let the bank see the blocks the cheap policies are evaluated "
            "on, and confirm is spent the first time it is read.",
            file=sys.stderr, flush=True,
        )
        return 2
    if not args.synthetic and (args.config is None or args.admission is None):
        print(
            "ERROR: a real run needs --config (gate_config.json) and "
            "--admission (the record written by scripts/r2_admission_gate.py). "
            "Use --synthetic for a CPU smoke run.",
            file=sys.stderr, flush=True,
        )
        return 2
    if args.expert_index is not None and not (0 <= args.expert_index < args.num_experts):
        print(
            f"ERROR: --expert-index {args.expert_index} is outside "
            f"[0, {args.num_experts}).",
            file=sys.stderr, flush=True,
        )
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
    print(f"  synthetic={args.synthetic} submits_jobs={SUBMITS_JOBS}", flush=True)

    try:
        if args.synthetic:
            bridge, variables, lat = build_synthetic_bridge(device, seed=args.seed)
            target_blocks = tuple(
                args.target_blocks if args.target_blocks is not None
                else range(len(bridge.model.blocks))
            )
            samples = build_synthetic_samples(
                bridge, args.lead_steps, n_samples=int(args.synthetic_samples),
                device=device, seed=args.seed,
            )
            config_summary: Dict[str, Any] = {"synthetic": True}
        else:
            bridge, variables, lat, gate_config = build_real_bridge(args.config, device)
            target_blocks = tuple(
                args.target_blocks if args.target_blocks is not None
                else sa.DEFAULT_TARGET_BLOCKS
            )
            samples = load_real_samples(
                args.admission, bridge, args.lead_steps,
                limit=args.limit, device=device,
            )
            config_summary = {
                "synthetic": False,
                "gate_config": str(args.config),
                "gate_config_digest": gate_config.config_digest,
                "identity_tag": gate_config.identity_tag,
                "admission_record": str(args.admission),
            }

        objective = sa.build_objective_spec(
            bridge, lat, lead_steps=tuple(args.lead_steps), space="raw"
        )
        print(
            f"  samples={len(samples)} blocks={list(target_blocks)} "
            f"objective_leads={objective.lead_hours}h",
            flush=True,
        )
    except (sa.StaticAdapterViolation, bt.BankTrainingViolation, ValueError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2

    ctx: Dict[str, Any] = {
        "bridge": bridge,
        "variables": variables,
        "samples": samples,
        "objective": objective,
        "target_blocks": target_blocks,
        "lead_steps": tuple(args.lead_steps),
        "record": {
            "script": "r2_fs_bank_train",
            "stage": args.stage,
            "mode": args.mode,
            "device": str(device),
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
                **config_summary,
            },
            "objective": objective.to_dict(),
        },
    }

    stages = (
        ["fs_fit", "fs_freeze", "bank_train", "profile", "horizon_check"]
        if args.stage == "all" else [args.stage]
    )
    results: Dict[str, bool] = {}
    handlers = {
        "fs_fit": stage_fs_fit,
        "fs_freeze": stage_fs_freeze,
        "bank_train": stage_bank_train,
        "profile": stage_profile,
        "horizon_check": stage_horizon_check,
    }
    for name in stages:
        print(f"\n[{name}]", flush=True)
        try:
            results[name] = bool(handlers[name](args, ctx))
        except (sa.StaticAdapterViolation, bt.BankTrainingViolation) as exc:
            print(f"  FAIL [{exc.code}] {exc.message[:300]}", file=sys.stderr, flush=True)
            ctx["record"].setdefault("errors", []).append(
                {"stage": name, "code": exc.code, "message": exc.message}
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

    ctx["record"]["results"] = results
    ctx["record"]["passed"] = bool(results) and all(results.values())

    print("", flush=True)
    for name, ok in results.items():
        print(f"  - {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"\nRESULT: {'PASS' if ctx['record']['passed'] else 'FAIL'}", flush=True)
    if args.synthetic:
        print(
            "SCOPE: SYNTHETIC_CODE_PATH_CHECK_ONLY -- no weather skill, no "
            "capacity finding, no S0.",
            flush=True,
        )

    _save_json(args.json, ctx["record"])
    if args.output_dir is not None and args.json is None:
        _save_json(Path(args.output_dir) / "run_record.json", ctx["record"])

    return 0 if ctx["record"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
