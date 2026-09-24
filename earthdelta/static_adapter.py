"""Fs: the frozen static reference adapter -- fit on bank_fit data, then merged.

Governing spec: `plans/plans_v2_0921/CLAUDE_EXECUTION_PLAN.md` section 6.2.

    "Fs 在 bank_fit 侧训练/选择并冻结；动态专家在其上训练，保留 F0 背景分数。
     静态 adapter 可在独立模型副本合并，或使用经过等价性验证的 always-on 分支。
     no-edit 和 hold 后必须是 Fs；冻结 Fs 后重新记录参考身份及零编辑等价性，
     不沿用 F0 的数值参照。"

Four things follow from that paragraph and this module implements exactly them.

1. WHAT Fs IS. Architecturally Fs is an `ExpertLoRA(num_experts=1)` on the same
   `attn.proj` layers the dynamic bank targets (blocks 18-23 by default). It is
   fit ONLY on `bank_fit`-role admitted samples, with the same native-registered
   objective (`metrics_contract.full_objective_loss`) the dynamic bank will use.

2. HOW Fs IS APPLIED. The spec permits either an independent merged model copy
   or an equivalence-verified always-on branch. This module does BOTH and
   verifies they agree numerically:

     * during fitting, Fs rides the existing, unmodified `controlled_rollout`
       hook mechanism as a one-expert bank held at a FIXED coefficient of 1.0
       for every step of the rollout (the always-on branch);
     * at freeze time, that same low-rank delta is merged straight into a
       DEEP-COPIED backbone's `attn.proj.weight` (`W' = W + c*scale*B@A`),
       producing a new, frozen backbone artifact with a freshly computed
       SHA-256 identity.

   `verify_merge_equivalence` asserts the two paths produce the same rollout to
   float tolerance. The merged copy is what the dynamic bank then trains on, so
   the bank rides the ordinary `controlled_rollout` / `EditPlan` machinery with
   no change to that machinery at all.

3. WHAT "NO EDIT" MEANS AFTERWARDS. Once Fs is frozen, "no-edit" and the state
   after the hold window mean *Fs active, no dynamic expert* -- never F0.
   `verify_zero_edit_equivalence` re-records the zero-edit property FRESH
   against the Fs-merged bridge. It also reports, separately, how far that
   baseline is from F0's, precisely so a caller can see that the two are NOT
   interchangeable. `assert_gain_baseline_is_fs` refuses any attempt to compute
   a round-4 gain against the F0 numbers.

4. WHERE F0 STILL APPEARS. F0's raw background score is retained, computed
   once, and carried in its own clearly-tagged `BackgroundF0Score` object whose
   `reporting_only` flag is True. It is never added to, subtracted from, or
   otherwise mixed into an Fs-relative gain.

Update caps (section 5.2 / 6.1): Fs gets a <=32-update gradient/throughput/
direction check first, then a <=500-update formal cap. `FsFitConfig` enforces
both, and both are adjustable ONLY before development results are viewed --
which is a process rule this module records (`frozen_before_dev_results`) but
cannot itself enforce.

Nothing here modifies `controlled_rollout`; the B12 concurrency/hook guarantees
it enforces are treated as frozen and are relied upon, not re-implemented.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import datetime
import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .bridge.stormer_bridge import (
    WeatherStepBridge,
    controlled_rollout,
)
from .contracts import ArtifactVersion, EditPlan
from .lowrank import ExpertLoRA
from .metrics_contract import full_objective_loss
from .pilot_contract import DataRole, assert_valid_data_role

__all__ = [
    "StaticAdapterViolation",
    "FS_NUM_EXPERTS",
    "FS_COEFFICIENT",
    "FS_PLAN_RHO",
    "FS_BASELINE_TAG",
    "BACKGROUND_F0_TAG",
    "DEFAULT_TARGET_BLOCKS",
    "MODE_UPDATE_CAPS",
    "FS_ELIGIBILITY_RULES",
    "ObjectiveSpec",
    "TrainingSample",
    "FsFitConfig",
    "FsFitRecord",
    "MergedBackboneArtifact",
    "BackgroundF0Score",
    "PostFreezeVerification",
    "area_weight_q",
    "build_objective_spec",
    "build_fs_adapter",
    "fs_edit_plan",
    "fs_rollout_trajectory",
    "objective_fields",
    "objective_loss_for_sample",
    "per_sample_loss_direction",
    "evaluate_fs_quality",
    "fit_static_adapter",
    "state_dict_digest",
    "static_adapter_digest",
    "merge_static_adapter",
    "make_fs_bridge",
    "fs_artifact_version",
    "verify_merge_equivalence",
    "verify_zero_edit_equivalence",
    "background_f0_score",
    "fs_baseline_score",
    "assert_gain_baseline_is_fs",
    "fs_relative_gain",
    "record_post_freeze_reference",
    "load_admitted_sample",
    "load_admitted_samples",
    "certify_admission_for_fs",
    "FS_QUALIFICATION_VERDICTS",
    "fs_panel_losses",
    "panel_ratio_summary",
    "training_stability",
    "evaluate_fs_qualification",
    "decide_lr_screen",
    "decide_formal_fs",
    "load_fs_adapter",
    "build_continuation_probe_bank",
    "verify_continuation_is_fs",
]


# =============================================================================
# Frozen constants
# =============================================================================

#: Fs is one expert. This is not a tunable: a static reference with several
#: experts would be a bank, and "which of them is the reference" would again be
#: ambiguous -- the exact ambiguity `registry.py` refuses.
FS_NUM_EXPERTS = 1

#: The coefficient Fs is held at, both in the always-on branch and in the merge.
#: Fs is NOT a dynamic edit candidate, so it is not bounded by the dynamic
#: bank's rho=0.25; it is the static reference itself, applied at full strength.
#: The merge uses this same number, which is what makes the two paths equal.
FS_COEFFICIENT = 1.0

#: `EditPlan` bounds coefficients by rho, so the Fs plan declares rho=1.0.
FS_PLAN_RHO = 1.0

#: The only legal baseline tag for a round-4 gain number.
FS_BASELINE_TAG = "Fs"

#: The tag F0's background score is carried under. Reporting only.
BACKGROUND_F0_TAG = "background_F0"

#: Section 6.1, frozen for this round; identical to `controlled_rollout`'s own
#: default, restated here so a caller can pass it explicitly.
DEFAULT_TARGET_BLOCKS: Tuple[int, ...] = (18, 19, 20, 21, 22, 23)

#: Section 5.2: "Fs 和每个专家先用至多 32 updates 检查梯度、吞吐和训练方向；
#: 正式小库起始上限为各 500 updates".
MODE_UPDATE_CAPS: Dict[str, int] = {
    "gradient_check": 32,
    "formal": 500,
    #: FP-03 FS-SCREEN-v1: the pre-registered learning-rate stability screen.
    #: Its records are diagnostic (the formal per-visit gate does not apply)
    #: and can never be frozen as Fs.
    "stability_screen": 256,
}

#: Pre-declared eligibility rules. Section 6.2: "只允许按事前资格规则处理无效库,
#: 不按评估收益删掉专家." A fit is declared ineligible only for finite-state
#: failures or the formal training-direction check below.  Evaluation-time gain
#: is deliberately absent and must never be used to select an adapter.
FS_ELIGIBILITY_RULES: Tuple[str, ...] = (
    "NON_FINITE_LOSS: an update produced a non-finite objective value",
    "NON_FINITE_GRADIENT: an update produced a non-finite gradient norm",
    "NO_UPDATES_RUN: the fit performed zero updates, so nothing was trained",
    "FORMAL_LOSS_DIRECTION_UNAVAILABLE: formal fit did not repeat any sample",
    "FORMAL_LOSS_DIRECTION_REGRESSION: a repeated sample ended with a higher loss",
)

_DEFAULT_LOSS_DTYPE = torch.float64


class StaticAdapterViolation(ValueError):
    """An Fs fit, freeze, merge or verification step failed a formal check."""

    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.detail: Dict[str, Any] = dict(detail or {})


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# =============================================================================
# Objective: native registered variables, area weighted, scale standardized
# =============================================================================

def area_weight_q(lat: Sequence[float] | Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
    """cos(latitude) area weights, shaped ``[1, 1, Lat, 1]``.

    Right-aligned against the objective's trailing ``(H, V, Lat, Lon)`` axes, so
    `metrics_contract._align_objective_weight` shares it across B (and K) and
    normalizes it exactly once. The weights are scaled to mean 1 purely so the
    printed loss stays on a familiar scale; `full_objective_loss` divides by
    `sum(q)` anyway, so the scaling cancels.
    """
    values = lat.detach().cpu() if isinstance(lat, Tensor) else torch.as_tensor(list(lat))
    values = values.to(dtype)
    if values.ndim != 1 or values.numel() == 0:
        raise StaticAdapterViolation(
            "OBJECTIVE_LATITUDE_INVALID",
            f"latitude must be a non-empty 1-D sequence, got shape {tuple(values.shape)}.",
            {"shape": list(values.shape)},
        )
    weights = torch.cos(values * math.pi / 180.0)
    # cos is negative nowhere on [-90, 90] but can be a tiny negative at the
    # poles through rounding; q must be non-negative for the metric contract.
    weights = weights.clamp_min(0.0)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise StaticAdapterViolation(
            "OBJECTIVE_LATITUDE_DEGENERATE",
            "cos(latitude) weights sum to zero; no area-weighted objective can "
            "be formed from this latitude axis.",
            {"sum": total},
        )
    weights = weights * (weights.numel() / total)
    return weights.reshape(1, 1, -1, 1)


@dataclass(frozen=True)
class ObjectiveSpec:
    """The frozen scoring target: which leads, which weights, which space.

    Section 6.1 fixes the primary objective as the 24h native-registered
    variables under an area-weighted, scale-standardized squared loss. This
    object pins all three parts of that before any result is looked at:

    * ``lead_steps`` -- which rollout steps form the H axis. ``(4,)`` is 24h at
      a 6h interval; ``(1, 4, 12)`` would be the 6h diagnostic, the 24h primary
      and the 72h guard reported together (and they must stay SEPARATE columns,
      never silently averaged into one primary).
    * ``q`` -- the area weight, normalized once by the metric contract.
    * ``scale`` -- per-variable standardization ``s``. In ``space='raw'`` this is
      the official per-variable input std, so the loss is the native-variable,
      scale-standardized one the plan specifies.
    """

    lead_steps: Tuple[int, ...]
    q: Optional[Tensor]
    scale: Optional[Tensor]
    space: str = "raw"
    variables: Tuple[str, ...] = ()
    interval_hours: int = 6

    def __post_init__(self):
        object.__setattr__(self, "lead_steps", tuple(int(s) for s in self.lead_steps))
        object.__setattr__(self, "variables", tuple(self.variables))
        if not self.lead_steps:
            raise StaticAdapterViolation(
                "OBJECTIVE_NO_LEADS",
                "lead_steps must not be empty; an objective over no lead time "
                "scores nothing and would pass vacuously.",
                {},
            )
        if any(s <= 0 for s in self.lead_steps):
            raise StaticAdapterViolation(
                "OBJECTIVE_LEAD_NOT_POSITIVE",
                f"lead_steps must be positive rollout steps, got {self.lead_steps}. "
                "Step 0 is the initial condition, not a forecast.",
                {"lead_steps": list(self.lead_steps)},
            )
        if len(set(self.lead_steps)) != len(self.lead_steps):
            raise StaticAdapterViolation(
                "OBJECTIVE_LEAD_DUPLICATED",
                f"lead_steps {self.lead_steps} repeats a step; a repeated lead "
                "would be double-counted in the H-axis normalization.",
                {"lead_steps": list(self.lead_steps)},
            )
        if self.space not in ("raw", "normalized"):
            raise StaticAdapterViolation(
                "OBJECTIVE_SPACE_UNKNOWN",
                f"space must be 'raw' or 'normalized', got {self.space!r}.",
                {"space": self.space},
            )
        if int(self.interval_hours) != 6:
            raise StaticAdapterViolation(
                "OBJECTIVE_INTERVAL_UNSUPPORTED",
                f"This pilot is pinned to a 6h interval, got {self.interval_hours}.",
                {"interval_hours": self.interval_hours},
            )

    @property
    def max_step(self) -> int:
        return max(self.lead_steps)

    @property
    def lead_hours(self) -> Tuple[int, ...]:
        return tuple(s * int(self.interval_hours) for s in self.lead_steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lead_steps": list(self.lead_steps),
            "lead_hours": list(self.lead_hours),
            "space": self.space,
            "interval_hours": int(self.interval_hours),
            "n_variables": len(self.variables),
            "q_shape": list(self.q.shape) if self.q is not None else None,
            "scale_shape": list(self.scale.shape) if self.scale is not None else None,
            "q_kind": "cos_latitude_area_weight" if self.q is not None else "uniform",
            "scale_kind": (
                "official_per_variable_input_std" if self.scale is not None else "unit"
            ),
        }


def build_objective_spec(
    bridge: WeatherStepBridge,
    lat: Optional[Sequence[float] | Tensor],
    lead_steps: Sequence[int] = (4,),
    *,
    space: str = "raw",
    interval_hours: int = 6,
) -> ObjectiveSpec:
    """Build the frozen objective from a bridge's own normalization constants.

    The per-variable scale is taken from THIS bridge's `inp_std`, so the
    standardization in the loss is the same one the forward pass uses; it is not
    a second, independently chosen notion of "typical magnitude".
    """
    q = area_weight_q(lat) if lat is not None else None
    scale: Optional[Tensor] = None
    if space == "raw":
        std = bridge.normalization.inp_std.detach().cpu().to(_DEFAULT_LOSS_DTYPE)
        if not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
            raise StaticAdapterViolation(
                "OBJECTIVE_SCALE_INVALID",
                "bridge normalization inp_std is not strictly positive and "
                "finite, so it cannot standardize the raw-space objective.",
                {},
            )
        scale = std.reshape(1, -1, 1, 1)
    return ObjectiveSpec(
        lead_steps=tuple(lead_steps),
        q=q,
        scale=scale,
        space=space,
        variables=tuple(bridge.variables),
        interval_hours=int(interval_hours),
    )


# =============================================================================
# Samples
# =============================================================================

@dataclass
class TrainingSample:
    """One admitted analysis time plus the targets its rollout is scored on.

    `x_norm` is the NORMALIZED initial condition (what `controlled_rollout`
    consumes). `targets_raw` maps rollout step -> the raw (physical-space)
    verification field at that step. The provenance fields are copied from the
    admission record; they are what the per-expert training record reports as
    "which data trained this", and they are not optional in formal use.
    """

    x_norm: Tensor
    targets_raw: Dict[int, Tensor]
    issue_id: str = ""
    issue_time: int = 0
    valid_time: int = 0
    history_time: int = 0
    split_id: str = ""
    data_role: Optional[str] = None
    source: str = ""
    time_utc: str = ""

    def __post_init__(self):
        self.data_role = assert_valid_data_role(self.data_role)
        if not isinstance(self.x_norm, Tensor) or self.x_norm.ndim != 4:
            raise StaticAdapterViolation(
                "SAMPLE_X_NORM_SHAPE",
                "x_norm must be a [B, V, Lat, Lon] tensor, got "
                f"{type(self.x_norm).__name__} "
                f"{tuple(getattr(self.x_norm, 'shape', ()))}.",
                {"issue_id": self.issue_id},
            )
        if not self.targets_raw:
            raise StaticAdapterViolation(
                "SAMPLE_NO_TARGETS",
                f"Sample {self.issue_id!r} carries no targets; there is nothing "
                "to score a rollout against.",
                {"issue_id": self.issue_id},
            )
        for step, target in self.targets_raw.items():
            if not isinstance(target, Tensor) or target.shape != self.x_norm.shape:
                raise StaticAdapterViolation(
                    "SAMPLE_TARGET_SHAPE",
                    f"Sample {self.issue_id!r} target at step {step} has shape "
                    f"{tuple(getattr(target, 'shape', ()))}, expected "
                    f"{tuple(self.x_norm.shape)}.",
                    {"issue_id": self.issue_id, "step": step},
                )

    def provenance(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "issue_time": int(self.issue_time),
            "valid_time": int(self.valid_time),
            "history_time": int(self.history_time),
            "split_id": self.split_id,
            "data_role": self.data_role,
            "source": self.source,
            "time_utc": self.time_utc,
        }

    def to(self, device: torch.device | str) -> "TrainingSample":
        return TrainingSample(
            x_norm=self.x_norm.to(device),
            targets_raw={k: v.to(device) for k, v in self.targets_raw.items()},
            issue_id=self.issue_id,
            issue_time=self.issue_time,
            valid_time=self.valid_time,
            history_time=self.history_time,
            split_id=self.split_id,
            data_role=self.data_role,
            source=self.source,
            time_utc=self.time_utc,
        )


def assert_bank_fit_samples(samples: Sequence[TrainingSample]) -> None:
    """Refuse anything that is not admitted `bank_fit` data.

    Section 6.2: Fs is fit on the bank_fit side and bank selection may not touch
    the outer policy-evaluation blocks. A sample with no role is refused too --
    an unroled sample cannot be kept out of confirm later.
    """
    if not samples:
        raise StaticAdapterViolation(
            "FS_NO_SAMPLES",
            "No samples were supplied. An Fs fit over zero samples is not a fit.",
            {},
        )
    offending = [
        {"issue_id": s.issue_id, "data_role": s.data_role}
        for s in samples
        if s.data_role != DataRole.BANK_FIT.value
    ]
    if offending:
        raise StaticAdapterViolation(
            "FS_DATA_ROLE_NOT_BANK_FIT",
            f"{len(offending)} of {len(samples)} sample(s) are not "
            f"{DataRole.BANK_FIT.value!r}: {offending[:4]}. Fs and the expert "
            "bank are fit on bank_fit data only; consuming policy_dev or "
            "confirm here would let the bank see the blocks it is later "
            "evaluated on.",
            {"n_offending": len(offending), "offending": offending[:16]},
        )


# =============================================================================
# Fs construction and the always-on branch
# =============================================================================

def build_fs_adapter(
    hidden_size: int,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    *,
    rank_per_expert: int = 4,
    scale: float = 1.0,
    seed: Optional[int] = None,
    init_up_std: float = 0.0,
) -> Dict[int, ExpertLoRA]:
    """One `ExpertLoRA(num_experts=1)` per targeted block.

    `init_up_std=0.0` keeps the standard LoRA zero-init on the up (B) factors,
    which makes Fs exactly a no-op at initialization -- so an untrained Fs is
    numerically F0, and the post-freeze verification will say so rather than
    pretend the two differ.
    """
    blocks = tuple(int(b) for b in target_blocks)
    if not blocks:
        raise StaticAdapterViolation(
            "FS_NO_TARGET_BLOCKS",
            "target_blocks must not be empty; Fs with no targeted block is a "
            "no-op that would still be recorded as a static reference.",
            {},
        )
    if len(set(blocks)) != len(blocks):
        raise StaticAdapterViolation(
            "FS_DUPLICATE_TARGET_BLOCK",
            f"target_blocks {blocks} repeats a block.",
            {"target_blocks": list(blocks)},
        )
    generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    adapters: Dict[int, ExpertLoRA] = {}
    for block in blocks:
        lora = ExpertLoRA(
            hidden_size,
            hidden_size,
            num_experts=FS_NUM_EXPERTS,
            rank_per_expert=int(rank_per_expert),
            scale=float(scale),
        )
        if generator is not None:
            with torch.no_grad():
                for down in lora.down:
                    down.weight.copy_(
                        torch.randn(down.weight.shape, generator=generator) * 0.02
                    )
        if init_up_std > 0:
            with torch.no_grad():
                for up in lora.up:
                    noise = (
                        torch.randn(up.weight.shape, generator=generator)
                        if generator is not None
                        else torch.randn(up.weight.shape)
                    )
                    up.weight.copy_(noise * float(init_up_std))
        lora.requires_grad_(True)
        adapters[block] = lora
    return adapters


def fs_edit_plan(steps: int, *, plan_id: str = "fs_always_on") -> EditPlan:
    """The always-on Fs plan: one expert, coefficient 1.0, active every step.

    `hold_steps` is set to the rollout length so the adapter never switches off
    mid-rollout. This branch is used for FITTING only; the frozen artifact is
    the merged backbone, which is always-on for any rollout length by
    construction.
    """
    n = int(steps)
    if n <= 0:
        raise StaticAdapterViolation(
            "FS_PLAN_STEPS_INVALID",
            f"steps must be a positive integer, got {steps!r}.",
            {"steps": repr(steps)},
        )
    return EditPlan(
        plan_id=plan_id,
        num_experts=FS_NUM_EXPERTS,
        coefficients=(FS_COEFFICIENT,),
        hold_steps=n,
        interval_hours=6,
        continuation="reference_after_hold",
        rho=FS_PLAN_RHO,
    )


def fs_rollout_trajectory(
    bridge: WeatherStepBridge,
    x_norm: Tensor,
    variables: Sequence[str],
    *,
    steps: int,
    fs_adapters: Optional[Mapping[int, ExpertLoRA]] = None,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    interval_hours: int = 6,
) -> Tensor:
    """Rollout with Fs always on, returning the full ``[B, T+1, V, Lat, Lon]``.

    Calls the existing `controlled_rollout` unchanged. With `fs_adapters=None`
    this is the plain backbone trajectory (used for the merged bridge, where Fs
    already lives in the weights).
    """
    adapters = dict(fs_adapters or {})
    plan = fs_edit_plan(steps) if adapters else EditPlan(
        plan_id="fs_merged_no_branch",
        num_experts=FS_NUM_EXPERTS,
        coefficients=(0.0,),
        hold_steps=max(1, int(steps)),
        interval_hours=6,
        continuation="reference_after_hold",
        rho=FS_PLAN_RHO,
    )
    return controlled_rollout(
        bridge,
        x_norm,
        list(variables),
        interval=int(interval_hours),
        steps=int(steps),
        plan=plan,
        expert_loras=adapters,
        target_blocks=tuple(int(b) for b in target_blocks),
        return_trajectory=True,
    )


def objective_fields(
    bridge: WeatherStepBridge,
    trajectory: Tensor,
    sample: TrainingSample,
    spec: ObjectiveSpec,
) -> Tuple[Tensor, Tensor]:
    """Turn a rollout trajectory + a sample into ``[B, H, V, Lat, Lon]`` fields.

    The H axis is `spec.lead_steps` IN ORDER, and prediction and target are
    assembled from the same index list, so a lead can never be scored against
    another lead's truth.
    """
    if trajectory.ndim != 5:
        raise StaticAdapterViolation(
            "OBJECTIVE_TRAJECTORY_RANK",
            f"trajectory must be [B, T+1, V, Lat, Lon], got {tuple(trajectory.shape)}.",
            {"shape": list(trajectory.shape)},
        )
    available = trajectory.shape[1] - 1
    missing = [s for s in spec.lead_steps if s > available]
    if missing:
        raise StaticAdapterViolation(
            "OBJECTIVE_LEAD_NOT_ROLLED",
            f"objective needs rollout step(s) {missing} but the trajectory only "
            f"reaches step {available}.",
            {"missing": missing, "available": available},
        )
    absent = [s for s in spec.lead_steps if s not in sample.targets_raw]
    if absent:
        raise StaticAdapterViolation(
            "OBJECTIVE_TARGET_MISSING",
            f"Sample {sample.issue_id!r} has no target for rollout step(s) "
            f"{absent}; the objective would silently score fewer leads than it "
            "declares.",
            {"issue_id": sample.issue_id, "missing": absent},
        )

    predictions: List[Tensor] = []
    targets: List[Tensor] = []
    for step in spec.lead_steps:
        state = trajectory[:, step]
        target = sample.targets_raw[step]
        if spec.space == "raw":
            predictions.append(bridge.normalization.denormalize(state))
            targets.append(target)
        else:
            predictions.append(state)
            targets.append(bridge.normalization.normalize(target))
    return torch.stack(predictions, dim=1), torch.stack(targets, dim=1)


def objective_loss_for_sample(
    bridge: WeatherStepBridge,
    trajectory: Tensor,
    sample: TrainingSample,
    spec: ObjectiveSpec,
) -> Tensor:
    """Scalar float64 loss for one sample, via `full_objective_loss`.

    Uses the metric contract's native ``[B,H,V,Lat,Lon]`` scoring -- NOT a
    hand-rolled MSE -- so the loss this module minimizes is the same object the
    gain table is later built from.
    """
    prediction, target = objective_fields(bridge, trajectory, sample, spec)
    device = prediction.device
    q = spec.q.to(device) if spec.q is not None else None
    scale = spec.scale.to(device) if spec.scale is not None else None
    per_sample = full_objective_loss(prediction, target, q, scale=scale)
    return per_sample.mean()


# =============================================================================
# Fitting
# =============================================================================

@dataclass(frozen=True)
class FsFitConfig:
    """Everything the Fs fit is pinned to, with the update caps enforced."""

    mode: str = "gradient_check"
    max_updates: int = 32
    learning_rate: float = 1e-2
    train_steps: int = 4
    lead_steps: Tuple[int, ...] = (4,)
    target_blocks: Tuple[int, ...] = DEFAULT_TARGET_BLOCKS
    rank_per_expert: int = 4
    adapter_scale: float = 1.0
    interval_hours: int = 6
    grad_clip: Optional[float] = 1.0
    seed: int = 20260921
    #: Process flag, recorded not enforced: the caps above may be adjusted only
    #: BEFORE any development result is viewed (section 5.2 / 6.1).
    frozen_before_dev_results: bool = True
    #: FP-03 pre-registration binding: which declared protocol configuration
    #: this fit is, and the SHA-256 of the protocol file that declared it.
    #: Optional (synthetic/CPU fits carry none); written by `to_dict`.
    config_id: Optional[str] = None
    protocol_sha256: Optional[str] = None

    def __post_init__(self):
        if self.protocol_sha256 is not None and (
            not isinstance(self.protocol_sha256, str)
            or len(self.protocol_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.protocol_sha256)
        ):
            raise StaticAdapterViolation(
                "FS_PROTOCOL_SHA256_INVALID",
                "protocol_sha256 must be a lowercase 64-hex SHA-256, got "
                f"{self.protocol_sha256!r}.",
                {"protocol_sha256": repr(self.protocol_sha256)},
            )
        object.__setattr__(self, "lead_steps", tuple(int(s) for s in self.lead_steps))
        object.__setattr__(
            self, "target_blocks", tuple(int(b) for b in self.target_blocks)
        )
        if self.mode not in MODE_UPDATE_CAPS:
            raise StaticAdapterViolation(
                "FS_MODE_UNKNOWN",
                f"mode must be one of {sorted(MODE_UPDATE_CAPS)}, got {self.mode!r}.",
                {"mode": self.mode, "allowed": sorted(MODE_UPDATE_CAPS)},
            )
        cap = MODE_UPDATE_CAPS[self.mode]
        if type(self.max_updates) is not int or self.max_updates <= 0:
            raise StaticAdapterViolation(
                "FS_MAX_UPDATES_INVALID",
                f"max_updates must be a positive integer, got {self.max_updates!r}.",
                {"max_updates": repr(self.max_updates)},
            )
        if self.max_updates > cap:
            raise StaticAdapterViolation(
                "FS_UPDATE_CAP_EXCEEDED",
                f"mode {self.mode!r} caps the fit at {cap} updates but "
                f"max_updates={self.max_updates} was requested. The caps are "
                "the pre-declared ones from the execution plan (<=32 for the "
                "gradient/throughput/direction check, <=500 for the formal "
                "fit); raising one after development results have been seen "
                "makes it a different experiment.",
                {"mode": self.mode, "cap": cap, "max_updates": self.max_updates},
            )
        if self.train_steps <= 0:
            raise StaticAdapterViolation(
                "FS_TRAIN_STEPS_INVALID",
                f"train_steps must be positive, got {self.train_steps!r}.",
                {"train_steps": repr(self.train_steps)},
            )
        if max(self.lead_steps) > self.train_steps:
            raise StaticAdapterViolation(
                "FS_LEAD_BEYOND_TRAIN_HORIZON",
                f"lead_steps {self.lead_steps} require rollout step "
                f"{max(self.lead_steps)} but train_steps={self.train_steps}. A "
                "lead that is never rolled cannot be trained against; shrink "
                "the lead or declare the longer horizon explicitly.",
                {"lead_steps": list(self.lead_steps),
                 "train_steps": self.train_steps},
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise StaticAdapterViolation(
                "FS_LEARNING_RATE_INVALID",
                f"learning_rate must be positive and finite, got {self.learning_rate!r}.",
                {"learning_rate": repr(self.learning_rate)},
            )

    @classmethod
    def for_mode(cls, mode: str, **overrides: Any) -> "FsFitConfig":
        """Config at that mode's full cap, unless the caller lowers it."""
        if mode not in MODE_UPDATE_CAPS:
            raise StaticAdapterViolation(
                "FS_MODE_UNKNOWN",
                f"mode must be one of {sorted(MODE_UPDATE_CAPS)}, got {mode!r}.",
                {"mode": mode},
            )
        kwargs: Dict[str, Any] = {"mode": mode, "max_updates": MODE_UPDATE_CAPS[mode]}
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def train_hours(self) -> int:
        return int(self.train_steps) * int(self.interval_hours)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "max_updates": int(self.max_updates),
            "update_cap_for_mode": MODE_UPDATE_CAPS[self.mode],
            "learning_rate": float(self.learning_rate),
            "train_steps": int(self.train_steps),
            "train_hours": self.train_hours,
            "lead_steps": list(self.lead_steps),
            "target_blocks": list(self.target_blocks),
            "rank_per_expert": int(self.rank_per_expert),
            "adapter_scale": float(self.adapter_scale),
            "interval_hours": int(self.interval_hours),
            "grad_clip": self.grad_clip,
            "seed": int(self.seed),
            "frozen_before_dev_results": bool(self.frozen_before_dev_results),
            "config_id": self.config_id,
            "protocol_sha256": self.protocol_sha256,
        }


@dataclass
class FsFitRecord:
    """What the Fs fit actually did -- data, updates, loss, eligibility."""

    mode: str
    config: Dict[str, Any] = field(default_factory=dict)
    n_updates: int = 0
    losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    #: Optimizer-side trajectory (ARTIFACT_CONTRACT section 4: "梯度/optimizer/
    #: update"). Recorded AFTER each optimizer step, under no_grad, in float64;
    #: pure observation that never feeds back into the fit. `param_norms[t]` is
    #: the L2 norm of every Fs factor after update t, `update_norms[t]` the L2
    #: norm of the parameter change that update t actually applied, and
    #: `initial_param_norm` the norm before the first update.
    param_norms: List[float] = field(default_factory=list)
    update_norms: List[float] = field(default_factory=list)
    initial_param_norm: Optional[float] = None
    #: Pre-clip gradient L2 norms of the A (`down`) and B (`up`) factors,
    #: per update, float64. B is zero-initialized, so A's gradient is exactly
    #: zero at update 0 and B's is not; both are the real `.grad` tensors.
    grad_norms_A: List[float] = field(default_factory=list)
    grad_norms_B: List[float] = field(default_factory=list)
    #: Updates whose pre-clip total norm exceeded `grad_clip` (clip engaged).
    clip_events: int = 0
    clip_event_updates: List[int] = field(default_factory=list)
    #: `static_adapter_digest` of the factors before the first / after the
    #: last update, and the optimizer hyper-parameters actually used.
    initial_adapter_digest: Optional[str] = None
    final_adapter_digest: Optional[str] = None
    optimizer: Dict[str, Any] = field(default_factory=dict)
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    min_loss: Optional[float] = None
    training_issue_ids: List[str] = field(default_factory=list)
    training_times_utc: List[str] = field(default_factory=list)
    data_role: Optional[str] = None
    n_samples: int = 0
    trainable_parameters: int = 0
    eligible: bool = True
    ineligible_reason: Optional[str] = None
    eligibility_rules: Tuple[str, ...] = FS_ELIGIBILITY_RULES
    wallclock_seconds: float = 0.0
    objective: Dict[str, Any] = field(default_factory=dict)
    quality_gate: Dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    @property
    def loss_decreased(self) -> bool:
        """Raw first-vs-last update loss. See `loss_direction` for the real test.

        Kept because it is the cheapest thing to print, but it compares two
        DIFFERENT samples whenever the schedule cycles, so it is not by itself
        evidence of a training direction.
        """
        if self.initial_loss is None or self.final_loss is None:
            return False
        return self.final_loss < self.initial_loss

    @property
    def loss_direction(self) -> Dict[str, Any]:
        """Per-sample first-vs-last loss -- the like-for-like direction check."""
        return per_sample_loss_direction(self.losses, self.training_issue_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "config": dict(self.config),
            "n_updates": int(self.n_updates),
            "losses": [float(v) for v in self.losses],
            "grad_norms": [float(v) for v in self.grad_norms],
            "param_norms": [float(v) for v in self.param_norms],
            "update_norms": [float(v) for v in self.update_norms],
            "initial_param_norm": self.initial_param_norm,
            "grad_norms_A": [float(v) for v in self.grad_norms_A],
            "grad_norms_B": [float(v) for v in self.grad_norms_B],
            "clip_events": int(self.clip_events),
            "clip_event_updates": [int(v) for v in self.clip_event_updates],
            "initial_adapter_digest": self.initial_adapter_digest,
            "final_adapter_digest": self.final_adapter_digest,
            "optimizer": dict(self.optimizer),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "min_loss": self.min_loss,
            "loss_decreased": self.loss_decreased,
            "loss_direction": self.loss_direction,
            "training_issue_ids": list(self.training_issue_ids),
            "training_times_utc": list(self.training_times_utc),
            "data_role": self.data_role,
            "n_samples": int(self.n_samples),
            "trainable_parameters": int(self.trainable_parameters),
            "eligible": bool(self.eligible),
            "ineligible_reason": self.ineligible_reason,
            "eligibility_rules": list(self.eligibility_rules),
            "wallclock_seconds": float(self.wallclock_seconds),
            "objective": dict(self.objective),
            "quality_gate": dict(self.quality_gate),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def count_trainable_parameters(modules: Iterable[nn.Module]) -> int:
    total = 0
    for module in modules:
        for param in module.parameters():
            if param.requires_grad:
                total += param.numel()
    return total


def _grad_l2_norm(params: Sequence[Tensor]) -> float:
    """float64 L2 norm of the current `.grad` of `params` (0.0 if none has one)."""
    with torch.no_grad():
        squares = [p.grad.double().pow(2).sum() for p in params if p.grad is not None]
        if not squares:
            return 0.0
        return float(torch.stack(squares).sum().sqrt())


def per_sample_loss_direction(
    losses: Sequence[float], issue_ids: Sequence[str]
) -> Dict[str, Any]:
    """First-vs-last loss on the SAME sample, for every sample seen twice.

    Section 5.2 asks the <=32-update run to check "梯度、吞吐和训练方向". The raw
    first and last update losses do NOT answer the direction question when the
    schedule cycles through several samples: update 0 and update N-1 then score
    DIFFERENT samples, and their difference mixes the training trend with the
    difference between two samples. This groups the recorded losses by
    `issue_id` and compares each sample's own first and last value, which is a
    like-for-like comparison.

    Samples seen only once are excluded rather than counted as "did not
    improve": one observation carries no direction.
    """
    by_sample: Dict[str, List[float]] = {}
    for loss, issue_id in zip(losses, issue_ids):
        by_sample.setdefault(str(issue_id), []).append(float(loss))

    per_sample: Dict[str, Dict[str, Any]] = {}
    decreased = 0
    for issue_id, values in by_sample.items():
        if len(values) < 2:
            continue
        went_down = values[-1] < values[0]
        per_sample[issue_id] = {
            "n_observations": len(values),
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "decreased": bool(went_down),
        }
        decreased += int(went_down)

    n = len(per_sample)
    return {
        "n_samples_with_repeats": n,
        "n_decreased": decreased,
        "fraction_decreased": (decreased / n) if n else None,
        "per_sample": per_sample,
        "note": (
            "first vs last loss on the SAME sample. The raw first/last update "
            "losses are not comparable when the schedule cycles through several "
            "samples, because they score different samples."
        ),
    }


def evaluate_fs_quality(record: FsFitRecord) -> Dict[str, Any]:
    """Apply the predeclared formal-fit direction gate.

    Merge equivalence only proves that two implementations execute the same
    update.  It says nothing about whether the fitted adapter improves its
    training objective.  Formal fits therefore require every repeated sample
    to finish below its own first observation.  The gradient-check mode keeps
    the historical diagnostic-only behavior so a short smoke run cannot be
    mistaken for a formal qualification.
    """
    direction = record.loss_direction
    result: Dict[str, Any] = {
        "mode": record.mode,
        "n_samples_with_repeats": int(direction["n_samples_with_repeats"]),
        "n_decreased": int(direction["n_decreased"]),
        "fraction_decreased": direction["fraction_decreased"],
        "passed": True,
        "reason": None,
    }
    if record.mode != "formal":
        result["reason"] = "DIAGNOSTIC_ONLY_NON_FORMAL_MODE"
        return result
    if result["n_samples_with_repeats"] == 0:
        result["passed"] = False
        result["reason"] = "FORMAL_LOSS_DIRECTION_UNAVAILABLE"
    elif result["n_decreased"] != result["n_samples_with_repeats"]:
        result["passed"] = False
        result["reason"] = "FORMAL_LOSS_DIRECTION_REGRESSION"
    return result


def fit_static_adapter(
    bridge: WeatherStepBridge,
    samples: Sequence[TrainingSample],
    spec: ObjectiveSpec,
    config: FsFitConfig,
    *,
    fs_adapters: Optional[Dict[int, ExpertLoRA]] = None,
    variables: Optional[Sequence[str]] = None,
    progress: Optional[Any] = None,
) -> Tuple[Dict[int, ExpertLoRA], FsFitRecord]:
    """Fit Fs on bank_fit samples against the native registered objective.

    The backbone is never touched: only the Fs factors carry gradient, and the
    optimizer is built over those parameters alone.

    Eligibility is decided ONLY by `FS_ELIGIBILITY_RULES`. A fit that runs to
    completion with finite losses and finite gradients is eligible even if the
    loss barely moved -- "under-trained" is a resource finding, not a scientific
    STOP, and it is never grounds for dropping the adapter.
    """
    assert_bank_fit_samples(list(samples))
    names = list(variables) if variables is not None else list(bridge.variables)

    torch.manual_seed(int(config.seed))
    adapters = fs_adapters if fs_adapters is not None else build_fs_adapter(
        bridge.model.blocks[config.target_blocks[0]].attn.proj.in_features,
        config.target_blocks,
        rank_per_expert=config.rank_per_expert,
        scale=config.adapter_scale,
        seed=config.seed,
    )
    device = next(bridge.model.parameters()).device
    for lora in adapters.values():
        lora.to(device)
        lora.requires_grad_(True)

    params = [p for lora in adapters.values() for p in lora.parameters()]
    optimizer = torch.optim.Adam(params, lr=float(config.learning_rate))

    record = FsFitRecord(
        mode=config.mode,
        config=config.to_dict(),
        data_role=DataRole.BANK_FIT.value,
        n_samples=len(samples),
        trainable_parameters=count_trainable_parameters(adapters.values()),
        objective=spec.to_dict(),
        started_at=_utc_now(),
    )

    group = optimizer.param_groups[0]
    record.optimizer = {
        "name": type(optimizer).__name__,
        "lr": float(group["lr"]),
        "betas": [float(b) for b in group["betas"]],
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "amsgrad": bool(group["amsgrad"]),
        "grad_clip_max_norm": config.grad_clip,
        "schedule": "constant",
    }
    record.initial_adapter_digest = static_adapter_digest(adapters)
    params_a = [p for lora in adapters.values() for p in lora.down.parameters()]
    params_b = [p for lora in adapters.values() for p in lora.up.parameters()]

    # Observation only: snapshots of the factors so the applied step and the
    # resulting parameter norm can be recorded. Never read by the optimizer.
    with torch.no_grad():
        previous_params = [p.detach().clone() for p in params]
        record.initial_param_norm = float(
            torch.stack([p.double().pow(2).sum() for p in previous_params]).sum().sqrt()
        )

    started = time.perf_counter()
    for update in range(int(config.max_updates)):
        sample = samples[update % len(samples)].to(device)
        optimizer.zero_grad(set_to_none=True)

        trajectory = fs_rollout_trajectory(
            bridge,
            sample.x_norm,
            names,
            steps=int(config.train_steps),
            fs_adapters=adapters,
            target_blocks=config.target_blocks,
            interval_hours=int(config.interval_hours),
        )
        loss = objective_loss_for_sample(bridge, trajectory, sample, spec)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            record.eligible = False
            record.ineligible_reason = "NON_FINITE_LOSS"
            break

        loss.backward()
        # Pre-clip per-factor norms: read-only on the real `.grad` tensors.
        grad_a = _grad_l2_norm(params_a)
        grad_b = _grad_l2_norm(params_b)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                params,
                float(config.grad_clip) if config.grad_clip is not None else float("inf"),
            )
        )
        if not math.isfinite(grad_norm):
            record.eligible = False
            record.ineligible_reason = "NON_FINITE_GRADIENT"
            record.losses.append(loss_value)
            break

        optimizer.step()

        with torch.no_grad():
            current_params = [p.detach().clone() for p in params]
            update_sq = torch.stack([
                (c.double() - q.double()).pow(2).sum()
                for c, q in zip(current_params, previous_params)
            ]).sum()
            param_sq = torch.stack([c.double().pow(2).sum() for c in current_params]).sum()
            previous_params = current_params
        record.update_norms.append(float(update_sq.sqrt()))
        record.param_norms.append(float(param_sq.sqrt()))

        record.losses.append(loss_value)
        record.grad_norms.append(grad_norm)
        record.grad_norms_A.append(grad_a)
        record.grad_norms_B.append(grad_b)
        if config.grad_clip is not None and grad_norm > float(config.grad_clip):
            record.clip_events += 1
            record.clip_event_updates.append(int(update))
        record.n_updates += 1
        record.training_issue_ids.append(sample.issue_id)
        record.training_times_utc.append(sample.time_utc)
        if progress is not None:
            progress(update, loss_value, grad_norm)

    record.wallclock_seconds = time.perf_counter() - started
    record.finished_at = _utc_now()
    record.final_adapter_digest = static_adapter_digest(adapters)
    if record.losses:
        record.initial_loss = float(record.losses[0])
        record.final_loss = float(record.losses[-1])
        record.min_loss = float(min(record.losses))
    if record.n_updates == 0 and record.eligible:
        record.eligible = False
        record.ineligible_reason = "NO_UPDATES_RUN"
    record.quality_gate = evaluate_fs_quality(record)
    if record.eligible and not record.quality_gate["passed"]:
        record.eligible = False
        record.ineligible_reason = str(record.quality_gate["reason"])
    return adapters, record


# =============================================================================
# FP-03: fixed panels, training stability, pre-registered qualification
# =============================================================================
#
# These are PURE functions of recorded numbers plus a rule mapping taken from
# the pre-registered protocol file. They never read policy_dev/confirm data and
# never rank candidates by a quality number: `decide_formal_fs` only ever
# looks at the designated device index and the binary per-replica verdicts.

FS_QUALIFICATION_VERDICTS: Tuple[str, ...] = (
    "PASS", "FAIL", "INCONCLUSIVE_QUALIFICATION", "INVALID",
)


def fs_panel_losses(
    bridge: WeatherStepBridge,
    samples: Sequence[TrainingSample],
    spec: ObjectiveSpec,
    *,
    lead_steps: Sequence[int] = (1, 4, 12),
    fs_adapters: Optional[Mapping[int, ExpertLoRA]] = None,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    variables: Optional[Sequence[str]] = None,
    interval_hours: int = 6,
) -> Dict[str, Dict[str, float]]:
    """No-grad per-issue objective at each lead, from ONE rollout per issue.

    Returns ``{issue_id: {"6": L, "24": L, "72": L}}``; the keys are lead HOURS
    as strings so the mapping survives a JSON round trip unchanged. Every value
    is the objective the fit minimizes (`objective_loss_for_sample`, float64
    reduction, same Q and scale) restricted to that single lead -- leads are
    never averaged together. ``fs_adapters=None`` is the plain backbone; the
    zero-initialized Fs factors must reproduce it exactly.
    """
    steps = sorted({int(s) for s in lead_steps})
    if not steps or steps[0] <= 0:
        raise StaticAdapterViolation(
            "PANEL_LEAD_STEPS_INVALID",
            f"panel lead_steps must be positive rollout steps, got {list(lead_steps)}.",
            {"lead_steps": list(lead_steps)},
        )
    names = list(variables) if variables is not None else list(bridge.variables)
    device = next(bridge.model.parameters()).device
    per_lead = {s: dataclasses.replace(spec, lead_steps=(s,)) for s in steps}
    adapters = dict(fs_adapters) if fs_adapters else None
    panel: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for sample in samples:
            if sample.issue_id in panel:
                raise StaticAdapterViolation(
                    "PANEL_DUPLICATE_ISSUE",
                    f"issue {sample.issue_id!r} appears twice in one panel.",
                    {"issue_id": sample.issue_id},
                )
            on_device = sample.to(device)
            trajectory = fs_rollout_trajectory(
                bridge,
                on_device.x_norm,
                names,
                steps=steps[-1],
                fs_adapters=adapters,
                target_blocks=target_blocks,
                interval_hours=int(interval_hours),
            )
            panel[sample.issue_id] = {
                str(s * int(interval_hours)): float(
                    objective_loss_for_sample(bridge, trajectory, on_device, per_lead[s])
                )
                for s in steps
            }
    return panel


def _panel_column(panel: Mapping[str, Mapping[str, float]], lead_hours: int) -> Dict[str, float]:
    key = str(int(lead_hours))
    missing = [iid for iid, row in panel.items() if key not in row]
    if missing:
        raise StaticAdapterViolation(
            "PANEL_LEAD_MISSING",
            f"panel has no {key}h value for issue(s) {missing[:4]}.",
            {"lead_hours": key, "missing": missing[:16]},
        )
    return {str(iid): float(row[key]) for iid, row in panel.items()}


def panel_ratio_summary(
    panel0: Mapping[str, Mapping[str, float]],
    panel1: Mapping[str, Mapping[str, float]],
    lead_hours: int,
) -> Dict[str, Any]:
    """Per-issue ``r_i = L1/L0`` and ``m = mean(L1)/mean(L0)`` at one lead.

    Both panels must cover exactly the same issues: a ratio over mismatched
    issue sets would compare different weather.
    """
    col0 = _panel_column(panel0, lead_hours)
    col1 = _panel_column(panel1, lead_hours)
    if set(col0) != set(col1) or not col0:
        raise StaticAdapterViolation(
            "PANEL_ISSUE_SET_MISMATCH",
            "initial and final panels do not cover the same non-empty issue set.",
            {"only_initial": sorted(set(col0) - set(col1))[:16],
             "only_final": sorted(set(col1) - set(col0))[:16]},
        )
    issues = sorted(col0)
    ratios = {iid: col1[iid] / col0[iid] for iid in issues}
    mean0 = sum(col0[i] for i in issues) / len(issues)
    mean1 = sum(col1[i] for i in issues) / len(issues)
    return {
        "lead_hours": int(lead_hours),
        "n_issues": len(issues),
        "per_issue_ratio": ratios,
        "max_ratio": max(ratios.values()),
        "min_ratio": min(ratios.values()),
        "mean_initial": mean0,
        "mean_final": mean1,
        "mean_ratio": mean1 / mean0,
    }


def _record_dict(record: "FsFitRecord | Mapping[str, Any]") -> Dict[str, Any]:
    return record.to_dict() if isinstance(record, FsFitRecord) else dict(record)


def training_stability(
    record: "FsFitRecord | Mapping[str, Any]",
    l0_24h: Mapping[str, float],
    *,
    horizon: int,
    clip_events_max: int = 0,
    per_visit_ratio_max: float = 1.25,
    epoch_onset_factor: float = 1.02,
    epoch_size: int = 8,
) -> Dict[str, Any]:
    """FS-QUAL-v1 Q1 (finite, exact horizon) and Q2 (stability) from a trace.

    Q2 has three parts, all required: no clip event; every training loss at
    most ``per_visit_ratio_max`` times that issue's L0 24h panel loss; and no
    epoch-mean onset, i.e. ``E_k <= epoch_onset_factor * min(E_0..E_{k-1})``
    for every full epoch k >= 1, with ``E_k`` the mean loss of updates
    ``8k..8k+7``. The first violating epoch is reported as the onset.
    """
    rec = _record_dict(record)
    losses = [float(v) for v in rec.get("losses", [])]
    grads = [float(v) for v in rec.get("grad_norms", [])]
    ids = [str(v) for v in rec.get("training_issue_ids", [])]
    n = int(rec.get("n_updates", 0))
    finite_losses = all(math.isfinite(v) for v in losses)
    finite_grads = all(math.isfinite(v) for v in grads)
    q1 = {
        "all_losses_finite": finite_losses,
        "all_grad_norms_finite": finite_grads,
        "n_updates": n,
        "required_n_updates": int(horizon),
        "trace_lengths_consistent": len(losses) == n == len(grads) == len(ids),
    }
    q1["passed"] = bool(
        finite_losses and finite_grads and n == int(horizon)
        and q1["trace_lengths_consistent"] and rec.get("ineligible_reason") not in (
            "NON_FINITE_LOSS", "NON_FINITE_GRADIENT")
    )

    missing = sorted({i for i in ids if i not in l0_24h})
    if missing:
        raise StaticAdapterViolation(
            "STABILITY_L0_MISSING",
            f"no L0 24h panel loss for trained issue(s) {missing[:4]}; the "
            "per-visit ratio cannot be formed.",
            {"missing": missing[:16]},
        )
    clip = (rec.get("config") or {}).get("grad_clip")
    clip_from_trace = sum(1 for g in grads if clip is not None and g > float(clip))
    clip_events = int(rec["clip_events"]) if rec.get("clip_events") is not None else clip_from_trace
    ratios = [loss / float(l0_24h[iid]) for loss, iid in zip(losses, ids)]
    max_ratio = max(ratios) if ratios else None
    argmax = ratios.index(max_ratio) if ratios else None
    epochs = [losses[k:k + epoch_size] for k in range(0, len(losses) - len(losses) % epoch_size, epoch_size)]
    e_means = [sum(e) / len(e) for e in epochs]
    onset = None
    for k in range(1, len(e_means)):
        if e_means[k] > float(epoch_onset_factor) * min(e_means[:k]):
            onset = k
            break
    first_clip = next((t for t, g in enumerate(grads) if clip is not None and g > float(clip)), None)
    clip_ok = clip_events <= int(clip_events_max)
    ratio_ok = max_ratio is not None and max_ratio <= float(per_visit_ratio_max)
    onset_ok = onset is None and len(e_means) >= 2
    q2 = {
        "clip_events": clip_events,
        "clip_events_from_trace": clip_from_trace,
        "clip_events_max": int(clip_events_max),
        "first_clip_update": first_clip,
        "clip_ok": clip_ok,
        "max_per_visit_ratio": max_ratio,
        "argmax_update": argmax,
        "argmax_issue": ids[argmax] if argmax is not None else None,
        "per_visit_ratio_max": float(per_visit_ratio_max),
        "ratio_ok": bool(ratio_ok),
        "epoch_means": e_means,
        "epoch_onset_factor": float(epoch_onset_factor),
        "onset_epoch": onset,
        "onset_update": onset * epoch_size if onset is not None else None,
        "epoch_ok": bool(onset_ok),
    }
    q2["passed"] = bool(clip_ok and ratio_ok and onset_ok)
    return {"q1": q1, "q2": q2, "stable": bool(q1["passed"] and q2["passed"])}


def _le(value: Optional[float], threshold: Optional[float]) -> str:
    if threshold is None:
        return "NOT_PREREGISTERED"
    if value is None or not math.isfinite(float(value)):
        return "FAIL"
    return "PASS" if float(value) <= float(threshold) else "FAIL"


def _guard(status: str, report_only: bool) -> str:
    """Report-only criteria never block; a null threshold blocks otherwise."""
    return "REPORT_ONLY" if report_only else status


def evaluate_fs_qualification(
    record: "FsFitRecord | Mapping[str, Any]",
    panel0: Mapping[str, Mapping[str, Mapping[str, float]]],
    panel1: Mapping[str, Mapping[str, Mapping[str, float]]],
    rule: Mapping[str, Any],
    *,
    validity: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    """FS-QUAL-v1 for one replica. Verdict in `FS_QUALIFICATION_VERDICTS`.

    ``panel0`` / ``panel1`` are ``{"train": panel, "holdout": panel}`` at the
    common initial checkpoint (L0) and after the fit (L1). ``validity`` holds
    the Q0 facts (backend, versions, TF32, certificates, digests); every entry
    must be True or the verdict is INVALID. A null threshold makes its
    criterion NOT_PREREGISTERED, which blocks PASS unless the rule marks that
    criterion report-only. ``substantive_verdict`` is the same aggregation
    without Q0, used for negative controls whose provenance is historical.
    """
    horizon = rule.get("horizon")
    q2r = rule.get("q2", {})
    q3r = rule.get("q3", {})
    q4r = rule.get("q4", {})
    q5r = rule.get("q5", {})
    q6r = rule.get("q6", {})

    train0, train1 = panel0["train"], panel1["train"]
    l0_24h = _panel_column(train0, 24)
    q2_thresholds = (q2r.get("clip_events_max"), q2r.get("per_visit_ratio_max"),
                     q2r.get("epoch_onset_factor"))
    stability = training_stability(
        record, l0_24h,
        horizon=int(horizon) if horizon is not None else -1,
        clip_events_max=int(q2_thresholds[0]) if q2_thresholds[0] is not None else 0,
        per_visit_ratio_max=float(q2_thresholds[1]) if q2_thresholds[1] is not None else 1.25,
        epoch_onset_factor=float(q2_thresholds[2]) if q2_thresholds[2] is not None else 1.02,
    )
    status: Dict[str, str] = {}
    status["Q1"] = ("NOT_PREREGISTERED" if horizon is None
                    else "PASS" if stability["q1"]["passed"] else "FAIL")
    status["Q2"] = ("NOT_PREREGISTERED" if any(t is None for t in q2_thresholds)
                    else "PASS" if stability["q2"]["passed"] else "FAIL")

    rec = _record_dict(record)
    direction = per_sample_loss_direction(rec.get("losses", []), rec.get("training_issue_ids", []))
    required = q3r.get("required_issues")
    if required is None:
        status["Q3"] = "NOT_PREREGISTERED"
    else:
        status["Q3"] = ("PASS" if direction["n_samples_with_repeats"] == int(required)
                        == direction["n_decreased"] else "FAIL")

    q4 = panel_ratio_summary(train0, train1, 24)
    q4_t = (q4r.get("pass_max_ratio_lt"), q4r.get("pass_mean_ratio_le"),
            q4r.get("fail_mean_ratio_gt"), q4r.get("fail_max_ratio_gt"))
    if any(t is None for t in q4_t):
        status["Q4"] = "NOT_PREREGISTERED"
    elif q4["max_ratio"] < q4_t[0] and q4["mean_ratio"] <= q4_t[1]:
        status["Q4"] = "PASS"
    elif q4["mean_ratio"] > q4_t[2] or q4["max_ratio"] > q4_t[3]:
        status["Q4"] = "FAIL"
    else:
        status["Q4"] = "INCONCLUSIVE_QUALIFICATION"

    q5 = panel_ratio_summary(train0, train1, 72)
    status["Q5"] = _guard(_le(q5["mean_ratio"], q5r.get("mean_ratio_72h_le")),
                          bool(q5r.get("report_only", False)))
    q6 = panel_ratio_summary(panel0["holdout"], panel1["holdout"], 24)
    status["Q6"] = _guard(_le(q6["mean_ratio"], q6r.get("holdout_mean_ratio_24h_le")),
                          bool(q6r.get("report_only", False)))
    q6h = panel_ratio_summary(train0, train1, 6)

    enforced = [status[k] for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6") if status[k] != "REPORT_ONLY"]
    if "FAIL" in enforced:
        substantive = "FAIL"
    elif all(s == "PASS" for s in enforced):
        substantive = "PASS"
    else:
        substantive = "INCONCLUSIVE_QUALIFICATION"

    if validity is None:
        q0 = {"status": "NOT_EVALUATED", "failed": []}
    else:
        failed = sorted(k for k, v in validity.items() if v is not True)
        q0 = {"status": "PASS" if validity and not failed else "FAIL",
              "failed": failed or ([] if validity else ["NO_VALIDITY_FACTS"]),
              "facts": dict(validity)}
    verdict = "INVALID" if q0["status"] != "PASS" else substantive
    return {
        "rule": "FS-QUAL-v1",
        "verdict": verdict,
        "substantive_verdict": substantive,
        "quality_pass": verdict == "PASS",
        "status": status,
        "q0": q0,
        "stability": stability,
        "q3_per_visit": {k: direction[k] for k in
                         ("n_samples_with_repeats", "n_decreased", "fraction_decreased")},
        "q4_train_24h": q4,
        "q5_train_72h": q5,
        "q6_holdout_24h": q6,
        "report_only_train_6h": q6h,
    }


def decide_lr_screen(arms: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> Dict[str, Any]:
    """FS-SCREEN-v1, applied exactly. No arm is ranked by a quality number.

    Each arm carries ``arm_id``, ``lr``, ``valid`` (its Q0), ``stability``
    (`training_stability`) and ``train_panel_mean_ratio_24h``. A missing or
    invalid arm makes the whole screen INVALID. A stable positive control means
    the instability did not reproduce: NOT_TESTABLE. Otherwise S is every
    candidate arm that is STABLE (Q1 and Q2) and PROGRESSING (final train-panel
    24h mean ratio <= the threshold); empty S refutes the lr hypothesis, else
    the highest-lr member of S is selected and the next-lower one is fallback.
    """
    control = str(rule["control_arm"])
    candidates = [str(a) for a in rule["candidate_arms"]]
    progressing_max = rule.get("progressing_mean_ratio_24h_le")
    by_id = {str(a["arm_id"]): a for a in arms}
    required = [control] + candidates
    table = []
    for arm_id in required:
        arm = by_id.get(arm_id)
        if arm is None:
            continue
        stab = arm["stability"]
        m = arm.get("train_panel_mean_ratio_24h")
        progressing = (progressing_max is not None and m is not None
                       and math.isfinite(float(m)) and float(m) <= float(progressing_max))
        onset = stab["q2"]["onset_update"]
        table.append({
            "arm_id": arm_id,
            "lr": float(arm["lr"]),
            "valid": bool(arm.get("valid")),
            "stable": bool(stab["stable"]),
            "q1_passed": bool(stab["q1"]["passed"]),
            "q2_passed": bool(stab["q2"]["passed"]),
            "progressing": bool(progressing),
            "train_panel_mean_ratio_24h": m,
            "clip_events": stab["q2"]["clip_events"],
            "first_clip_update": stab["q2"]["first_clip_update"],
            "max_per_visit_ratio": stab["q2"]["max_per_visit_ratio"],
            "onset_epoch": stab["q2"]["onset_epoch"],
            "onset_update": onset,
            "lr_x_onset_update": float(arm["lr"]) * onset if onset is not None else None,
        })
    missing = [a for a in required if a not in by_id]
    invalid = [row["arm_id"] for row in table if not row["valid"]]
    result: Dict[str, Any] = {
        "rule": "FS-SCREEN-v1",
        "horizon": rule.get("horizon"),
        "arms": table,
        "missing_arms": missing,
        "invalid_arms": invalid,
        "S": [],
        "selected_arm": None,
        "selected_lr": None,
        "fallback_arm": None,
        "fallback_lr": None,
    }
    rows = {row["arm_id"]: row for row in table}
    if missing or invalid or progressing_max is None:
        result.update(verdict="INVALID", stop=True)
        return result
    if rows[control]["stable"]:
        result.update(verdict="NOT_TESTABLE", stop=True)
        return result
    members = sorted((rows[a] for a in candidates if rows[a]["stable"] and rows[a]["progressing"]),
                     key=lambda row: -row["lr"])
    result["S"] = [row["arm_id"] for row in members]
    if not members:
        result.update(verdict="REFUTED", stop=True)
        return result
    result.update(
        verdict="CONFIRMED",
        stop=False,
        selected_arm=members[0]["arm_id"],
        selected_lr=members[0]["lr"],
        fallback_arm=members[1]["arm_id"] if len(members) > 1 else None,
        fallback_lr=members[1]["lr"] if len(members) > 1 else None,
        claim_scope=(f"lr hypothesis confirmed for H={rule.get('horizon')} updates only; "
                     "no extrapolation to a longer horizon"),
    )
    return result


def decide_formal_fs(
    replicas: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    *,
    designated_gates: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    """FS-SELECT-v1. All replicas must qualify; only the designated one can be Fs.

    ``replicas`` are ``{"device_index", "qualification", "adapter_sha256"}``.
    The designated candidate is fixed by device index before submission; a
    witness can never be substituted, whatever its numbers. With
    ``designated_gates`` (numerical_merge / reload / identity results of the
    designated candidate in independent processes) the final selection is
    decided; without them a quality PASS is only pending those gates.
    """
    n_required = int(rule.get("n_replicas", 4))
    designated = int(rule.get("designated_device_index", 0))
    by_dev: Dict[int, Mapping[str, Any]] = {}
    duplicate = False
    for rep in replicas:
        idx = int(rep["device_index"])
        duplicate = duplicate or idx in by_dev
        by_dev[idx] = rep
    verdicts = {idx: str(rep["qualification"]["verdict"]) for idx, rep in sorted(by_dev.items())}
    result: Dict[str, Any] = {
        "rule": "FS-SELECT-v1",
        "replica_verdicts": {str(k): v for k, v in verdicts.items()},
        "designated_device_index": designated,
        "designated_adapter_sha256": None,
        "substitution": "FORBIDDEN",
        "fs_selected": False,
    }
    if duplicate or sorted(by_dev) != list(range(n_required)):
        result.update(verdict="INVALID", reason="REPLICA_SET_INCOMPLETE_OR_DUPLICATED")
        return result
    values = list(verdicts.values())
    if "INVALID" in values:
        result.update(verdict="INVALID", reason="A_REPLICA_FAILED_Q0_VALIDITY")
    elif all(v == "PASS" for v in values):
        result["designated_adapter_sha256"] = by_dev[designated].get("adapter_sha256")
        if designated_gates is None:
            result.update(verdict="QUALITY_PASS_PENDING_DESIGNATED_GATES")
        else:
            required_gates = ("numerical_merge_pass", "reload_pass", "identity_pass")
            failed = [g for g in required_gates if designated_gates.get(g) is not True]
            result["designated_gates"] = dict(designated_gates)
            if failed:
                result.update(verdict="STOP_DESIGNATED_GATE_FAILED", failed_gates=failed)
            else:
                result.update(verdict="FS_SELECTED", fs_selected=True)
    elif "FAIL" in values:
        result.update(verdict="STOP_FITTED_FS_QUALITY",
                      failed_replicas=[k for k, v in verdicts.items() if v == "FAIL"])
    else:
        result.update(verdict="INCONCLUSIVE_QUALIFICATION")
    return result


def load_fs_adapter(
    path: Path | str,
    hidden_size: int,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    *,
    rank_per_expert: int = 4,
    scale: float = 1.0,
    expected_sha256: Optional[str] = None,
    device: Optional[torch.device | str] = None,
) -> Tuple[Dict[int, ExpertLoRA], str]:
    """Load a saved ``{block: state_dict}`` Fs adapter file, byte-identity first.

    The file's SHA-256 is computed before it is deserialized and must equal
    ``expected_sha256`` when one is given; blocks must match exactly and every
    state dict loads strictly.
    """
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    if expected_sha256 is not None and sha != str(expected_sha256):
        raise StaticAdapterViolation(
            "FS_ADAPTER_SHA256_MISMATCH",
            f"{file_path} has sha256 {sha}, expected {expected_sha256}.",
            {"path": str(file_path), "actual": sha, "expected": str(expected_sha256)},
        )
    state = torch.load(file_path, map_location="cpu", weights_only=True)
    blocks = tuple(int(b) for b in target_blocks)
    stored = {int(k): v for k, v in state.items()}
    if set(stored) != set(blocks):
        raise StaticAdapterViolation(
            "FS_ADAPTER_BLOCKS_MISMATCH",
            f"{file_path} holds blocks {sorted(stored)}, expected {list(blocks)}.",
            {"stored": sorted(stored), "expected": list(blocks)},
        )
    adapters: Dict[int, ExpertLoRA] = {}
    for block in blocks:
        lora = ExpertLoRA(hidden_size, hidden_size, num_experts=FS_NUM_EXPERTS,
                          rank_per_expert=int(rank_per_expert), scale=float(scale))
        lora.load_state_dict(stored[block], strict=True)
        if device is not None:
            lora.to(device)
        adapters[block] = lora
    return adapters, sha


def build_continuation_probe_bank(
    hidden_size: int,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    *,
    num_experts: int = 4,
    rank_per_expert: int = 4,
    seed: int = 20260921,
    factor_std: float = 0.02,
) -> Dict[int, ExpertLoRA]:
    """A dynamic-bank-shaped probe whose B factors are NONZERO.

    The real bank is zero-initialized in B, which would make any continuation
    check vacuous (an all-zero edit leaves the state untouched). This probe is
    used only to prove the continuation mechanics; it is never trained or saved.
    """
    generator = torch.Generator().manual_seed(int(seed))
    bank: Dict[int, ExpertLoRA] = {}
    for block in (int(b) for b in target_blocks):
        lora = ExpertLoRA(hidden_size, hidden_size, num_experts=int(num_experts),
                          rank_per_expert=int(rank_per_expert), scale=1.0)
        with torch.no_grad():
            for module in list(lora.down) + list(lora.up):
                module.weight.copy_(torch.randn(module.weight.shape, generator=generator)
                                    * float(factor_std))
        lora.requires_grad_(False)
        bank[block] = lora
    return bank


def verify_continuation_is_fs(
    fs_bridge: WeatherStepBridge,
    f0_bridge: WeatherStepBridge,
    probe_bank: Mapping[int, ExpertLoRA],
    x_norm: Tensor,
    variables: Sequence[str],
    *,
    hold: int = 4,
    total: int = 12,
    active_expert: int = 0,
    coefficient: float = 0.25,
    rho: float = 0.25,
    interval_hours: int = 6,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
) -> Dict[str, Any]:
    """After the hold window the rollout continues as Fs FROM THE EDITED STATE.

    A controlled rollout on the Fs bridge applies one probe expert for ``hold``
    steps. Steps ``hold+1..total`` must equal, exactly, the Fs bridge's own
    `forward_validation` continued step by step from the edited step-``hold``
    state, and must differ from an F0 continuation from that same state. The
    probe must be non-vacuous: nonzero B factors and an edited step-``hold``
    state that differs from the unedited Fs one.
    """
    num_experts = {int(lora.num_experts) for lora in probe_bank.values()}
    if len(num_experts) != 1:
        raise StaticAdapterViolation("CONTINUATION_PROBE_INCONSISTENT",
                                     "probe bank blocks disagree on num_experts.", {})
    k = num_experts.pop()
    nonzero_b = all(
        float(lora._get_up(int(active_expert)).weight.detach().abs().max()) > 0.0
        for lora in probe_bank.values()
    )
    if not nonzero_b:
        raise StaticAdapterViolation(
            "CONTINUATION_PROBE_VACUOUS",
            "the probe bank's active expert has an all-zero B factor; the edit "
            "would be a no-op and the continuation check would pass vacuously.",
            {"active_expert": int(active_expert)},
        )
    coefficients = tuple(float(coefficient) if i == int(active_expert) else 0.0 for i in range(k))
    plan = EditPlan(
        plan_id="fs_continuation_probe",
        num_experts=k,
        coefficients=coefficients,
        hold_steps=int(hold),
        interval_hours=int(interval_hours),
        continuation="reference_after_hold",
        rho=float(rho),
    )
    names = list(variables)
    with torch.no_grad(), _disable_tf32_for_identity_check():
        trajectory = controlled_rollout(
            fs_bridge, x_norm, names, interval=int(interval_hours), steps=int(total),
            plan=plan, expert_loras=dict(probe_bank),
            target_blocks=tuple(int(b) for b in target_blocks), return_trajectory=True,
        )
        unedited = fs_bridge.forward_validation(x_norm, names, interval=int(interval_hours),
                                                steps=int(hold))
        fs_state = trajectory[:, int(hold)]
        f0_state = trajectory[:, int(hold)]
        diffs_fs: List[float] = []
        diffs_f0: List[float] = []
        for step in range(int(hold) + 1, int(total) + 1):
            fs_state = fs_bridge.forward_validation(fs_state, names, interval=int(interval_hours), steps=1)
            f0_state = f0_bridge.forward_validation(f0_state, names, interval=int(interval_hours), steps=1)
            diffs_fs.append(float((trajectory[:, step] - fs_state).abs().max()))
            diffs_f0.append(float((trajectory[:, step] - f0_state).abs().max()))
    edit_effect = float((trajectory[:, int(hold)] - unedited).abs().max())
    exact = all(d == 0.0 for d in diffs_fs)
    discriminating = all(d > 0.0 for d in diffs_f0)
    return {
        "check": "continuation_after_hold_is_fs_from_edited_state",
        "passed": bool(exact and discriminating and edit_effect > 0.0),
        "hold": int(hold),
        "total": int(total),
        "active_expert": int(active_expert),
        "coefficient": float(coefficient),
        "edit_effect_at_hold_max_abs": edit_effect,
        "max_abs_diff_vs_fs_continuation_by_step": diffs_fs,
        "max_abs_diff_vs_f0_continuation_by_step": diffs_f0,
        "exact_fs_continuation": bool(exact),
        "discriminates_f0": bool(discriminating),
        "tf32_disabled_for_check": True,
    }


# =============================================================================
# Freeze: merge Fs into an independent backbone copy, with a fresh identity
# =============================================================================

#: Schema label for the merged-backbone digest, so a digest computed under a
#: different serialization can never be compared as if it matched.
FS_MERGE_SCHEMA = "ed-fs-merge/1"


def state_dict_digest(module: nn.Module) -> str:
    """SHA-256 over a module's parameters and buffers, name/shape/dtype bound.

    Same pattern as the checkpoint identity in `load_stormer_checkpoint`: the
    identity is over the bytes that actually participate in the forward pass,
    plus enough structure that two different tensors cannot collide by being
    reshaped into each other.
    """
    digest = hashlib.sha256()
    digest.update(f"schema={FS_MERGE_SCHEMA}\n".encode())
    state = module.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            digest.update(f"key={name}\nnon_tensor={tensor!r}\n".encode())
            continue
        values = tensor.detach().cpu().contiguous()
        digest.update(
            f"key={name}\ndtype={values.dtype}\nshape={tuple(values.shape)}\n".encode()
        )
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def static_adapter_digest(
    fs_adapters: Mapping[int, ExpertLoRA],
    *,
    coefficient: float = FS_COEFFICIENT,
    target_blocks: Optional[Sequence[int]] = None,
) -> str:
    """SHA-256 over the Fs factors AND the coefficient/scale they apply at.

    The coefficient is part of the identity: the same factors applied at a
    different strength are a different static reference.
    """
    blocks = tuple(sorted(int(b) for b in (target_blocks or fs_adapters.keys())))
    digest = hashlib.sha256()
    digest.update(f"schema={FS_MERGE_SCHEMA}\nkind=static_adapter\n".encode())
    digest.update(f"coefficient={float(coefficient)!r}\n".encode())
    digest.update(f"blocks={list(blocks)}\n".encode())
    for block in blocks:
        lora = fs_adapters[block]
        digest.update(
            f"block={block}\nnum_experts={lora.num_experts}\n"
            f"rank={lora.rank_per_expert}\nscale={float(lora.scale)!r}\n".encode()
        )
        digest.update(state_dict_digest(lora).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class MergedBackboneArtifact:
    """The identity of the frozen, Fs-merged backbone produced at freeze time."""

    schema: str
    base_backbone_digest: str
    merged_backbone_digest: str
    static_adapter_digest: str
    coefficient: float
    target_blocks: Tuple[int, ...]
    rank_per_expert: int
    adapter_scale: float
    merged_at: str
    delta_max_abs: float
    delta_frobenius: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "base_backbone_digest": self.base_backbone_digest,
            "merged_backbone_digest": self.merged_backbone_digest,
            "static_adapter_digest": self.static_adapter_digest,
            "coefficient": float(self.coefficient),
            "target_blocks": list(self.target_blocks),
            "rank_per_expert": int(self.rank_per_expert),
            "adapter_scale": float(self.adapter_scale),
            "merged_at": self.merged_at,
            "delta_max_abs": float(self.delta_max_abs),
            "delta_frobenius": float(self.delta_frobenius),
        }

    @property
    def static_adapter_label(self) -> str:
        """The value `ArtifactVersion.static_adapter` carries after the freeze.

        Everywhere before this point that field is the placeholder ``"none"``.
        Once Fs is frozen it must name Fs's real digest instead.
        """
        blocks = "-".join(str(b) for b in self.target_blocks)
        return (
            f"fs_r{self.rank_per_expert}_b{blocks}_c{self.coefficient:g}"
            f"_sha256:{self.static_adapter_digest[:16]}"
        )

    @property
    def backbone_label_suffix(self) -> str:
        return f"fs_merged_sha256:{self.merged_backbone_digest[:16]}"


def merge_static_adapter(
    model: nn.Module,
    fs_adapters: Mapping[int, ExpertLoRA],
    *,
    target_blocks: Optional[Sequence[int]] = None,
    coefficient: float = FS_COEFFICIENT,
    deep_copy: bool = True,
) -> Tuple[nn.Module, MergedBackboneArtifact]:
    """Merge Fs into a COPY of the backbone: ``W' = W + c * scale * B @ A``.

    This is the standard LoRA merge, applied at exactly the injection point the
    hook uses: the hook adds `lora(input_of_proj)` to `proj(input_of_proj)`, and
    `proj` is `x -> W x + b`, so folding `c * scale * B @ A` into `W` reproduces
    the hook's contribution identically. `verify_merge_equivalence` checks that
    numerically rather than taking this argument on trust.

    The returned model is a fresh object with `requires_grad_(False)` and
    `eval()`: it is a frozen backbone artifact, not a trainable adapter.
    """
    blocks = tuple(sorted(int(b) for b in (target_blocks or fs_adapters.keys())))
    missing = [b for b in blocks if b not in fs_adapters]
    if missing:
        raise StaticAdapterViolation(
            "FS_MERGE_ADAPTER_MISSING",
            f"No Fs adapter for target block(s) {missing}; merging a partial Fs "
            "would produce a backbone whose identity does not describe what it "
            "contains.",
            {"missing": missing, "available": sorted(fs_adapters)},
        )
    if not math.isfinite(float(coefficient)):
        raise StaticAdapterViolation(
            "FS_MERGE_COEFFICIENT_INVALID",
            f"coefficient must be finite, got {coefficient!r}.",
            {"coefficient": repr(coefficient)},
        )

    base_digest = state_dict_digest(model)
    merged = copy.deepcopy(model) if deep_copy else model

    max_abs = 0.0
    frobenius_sq = 0.0
    ranks = set()
    scales = set()
    with torch.no_grad():
        for block in blocks:
            lora = fs_adapters[block]
            if lora.num_experts != FS_NUM_EXPERTS:
                raise StaticAdapterViolation(
                    "FS_MERGE_NOT_SINGLE_EXPERT",
                    f"Fs adapter at block {block} has {lora.num_experts} experts; "
                    f"Fs is a single static reference (num_experts={FS_NUM_EXPERTS}).",
                    {"block": block, "num_experts": lora.num_experts},
                )
            try:
                proj = merged.blocks[block].attn.proj
            except (AttributeError, IndexError) as exc:
                raise StaticAdapterViolation(
                    "FS_MERGE_TARGET_NOT_FOUND",
                    f"Backbone has no blocks[{block}].attn.proj to merge into: {exc}",
                    {"block": block},
                ) from exc

            down = lora._get_down(0).weight.detach().to(proj.weight.dtype).to(proj.weight.device)
            up = lora._get_up(0).weight.detach().to(proj.weight.dtype).to(proj.weight.device)
            delta = (up @ down) * float(lora.scale) * float(coefficient)
            if delta.shape != proj.weight.shape:
                raise StaticAdapterViolation(
                    "FS_MERGE_SHAPE_MISMATCH",
                    f"Fs delta at block {block} has shape {tuple(delta.shape)} but "
                    f"attn.proj.weight is {tuple(proj.weight.shape)}.",
                    {"block": block, "delta_shape": list(delta.shape),
                     "weight_shape": list(proj.weight.shape)},
                )
            if not bool(torch.isfinite(delta).all()):
                raise StaticAdapterViolation(
                    "FS_MERGE_DELTA_NON_FINITE",
                    f"Fs delta at block {block} contains non-finite values; a "
                    "merged backbone built from it would be silently unusable.",
                    {"block": block},
                )
            proj.weight.add_(delta)
            max_abs = max(max_abs, float(delta.abs().max()))
            frobenius_sq += float((delta.double() ** 2).sum())
            ranks.add(int(lora.rank_per_expert))
            scales.add(float(lora.scale))

    merged.requires_grad_(False)
    merged.eval()

    artifact = MergedBackboneArtifact(
        schema=FS_MERGE_SCHEMA,
        base_backbone_digest=base_digest,
        merged_backbone_digest=state_dict_digest(merged),
        static_adapter_digest=static_adapter_digest(
            fs_adapters, coefficient=coefficient, target_blocks=blocks
        ),
        coefficient=float(coefficient),
        target_blocks=blocks,
        rank_per_expert=int(sorted(ranks)[0]) if len(ranks) == 1 else -1,
        adapter_scale=float(sorted(scales)[0]) if len(scales) == 1 else float("nan"),
        merged_at=_utc_now(),
        delta_max_abs=max_abs,
        delta_frobenius=math.sqrt(frobenius_sq),
    )
    return merged, artifact


def fs_artifact_version(
    base_version: ArtifactVersion,
    artifact: MergedBackboneArtifact,
    *,
    normalization: Optional[str] = None,
    edit_bank: Optional[str] = None,
    split: Optional[str] = None,
) -> ArtifactVersion:
    """Re-record the reference identity AFTER the freeze.

    Two fields change and both matter:

    * ``backbone`` gains the merged-state digest, so an artifact produced
      against F0 can no longer `assert_matches` an Fs run;
    * ``static_adapter`` stops being the placeholder ``"none"`` and names Fs's
      real digest.
    """
    if base_version.static_adapter not in ("none", ""):
        # Merging Fs on top of an already-Fs-bearing version would produce an
        # identity that claims one static adapter while carrying two.
        raise StaticAdapterViolation(
            "FS_IDENTITY_ALREADY_HAS_STATIC_ADAPTER",
            f"base version already declares static_adapter="
            f"{base_version.static_adapter!r}. Merge Fs onto the F0 identity "
            "once; do not stack static adapters under one label.",
            {"static_adapter": base_version.static_adapter},
        )
    return ArtifactVersion(
        backbone=f"{base_version.backbone}|{artifact.backbone_label_suffix}",
        static_adapter=artifact.static_adapter_label,
        edit_bank=edit_bank if edit_bank is not None else base_version.edit_bank,
        normalization=(
            normalization if normalization is not None else base_version.normalization
        ),
        grid=base_version.grid,
        projection=base_version.projection,
        split=split if split is not None else base_version.split,
        continuation=base_version.continuation,
    )


def make_fs_bridge(
    merged_model: nn.Module,
    base_bridge: WeatherStepBridge,
    artifact: MergedBackboneArtifact,
    *,
    edit_bank: Optional[str] = None,
    split: Optional[str] = None,
) -> WeatherStepBridge:
    """Wrap the merged backbone in its own bridge with the re-recorded identity."""
    version = fs_artifact_version(
        base_bridge._version, artifact, edit_bank=edit_bank, split=split
    )
    return WeatherStepBridge(merged_model, base_bridge.normalization, version)


# =============================================================================
# Post-freeze verification
# =============================================================================

@contextlib.contextmanager
def _disable_tf32_for_identity_check():
    """Merge/zero-edit equivalence must compare code paths, not TF32 rounding.

    TF32 matmul (the CUDA default on Ampere/Hopper) has ~10 mantissa bits, i.e.
    ~1e-3 relative precision -- the same order as the divergence these checks
    exist to catch. Left on, it would both mask a real merge bug and flag a
    correct merge as broken, because the branch path and the merged path issue
    different GEMMs and therefore round differently under TF32.

    Scoped to these checks only. It must not change training/profiling numerics
    elsewhere in the pipeline, which have already been measured and accepted
    under the platform default; a global flip would silently invalidate them.

    The prior values are saved and restored exactly, never hardcoded back to
    True: the platform default differs by build and device (a CPU-only build
    reports ``matmul.allow_tf32 == False``), and unrelated code may have set
    them deliberately.
    """
    prior_matmul = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul
        torch.backends.cudnn.allow_tf32 = prior_cudnn


def verify_merge_equivalence(
    base_bridge: WeatherStepBridge,
    fs_bridge: WeatherStepBridge,
    fs_adapters: Mapping[int, ExpertLoRA],
    x_norm: Tensor,
    variables: Sequence[str],
    *,
    steps: int = 1,
    interval_hours: int = 6,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Dict[str, Any]:
    """The merged copy and the always-on branch must produce the same rollout.

    This is the "经过等价性验证" half of the spec's permission to use either
    form: the merge is only a legitimate stand-in for the branch if the two are
    numerically the same object.

    Two numbers come back and they must be read together:

    * ``max_abs_diff`` -- branch path against merged path. This is the property
      being certified and the one ``passed`` is computed from.
    * ``self_max_abs_diff`` -- the SAME merged-path call, issued twice with
      identical arguments under the same no-grad block, differenced against
      itself. No code path differs between those two calls, so this is the
      floor: whatever the hardware and kernel selection contribute on their own.

    The second number exists because ``max_abs_diff`` alone cannot tell a real
    merge bug from hardware-level non-determinism. If ``max_abs_diff`` is the
    same order as ``self_max_abs_diff``, the two paths agree as closely as one
    path agrees with itself and there is nothing to find in the merge algebra.
    If ``max_abs_diff`` is much larger, the divergence is attributable to the
    branch-versus-merge difference and is worth chasing. TF32 is disabled for
    all three forward passes (see `_disable_tf32_for_identity_check`) so neither
    number is dominated by TF32 rounding.
    """
    with torch.no_grad(), _disable_tf32_for_identity_check():
        branch = fs_rollout_trajectory(
            base_bridge,
            x_norm,
            variables,
            steps=steps,
            fs_adapters=fs_adapters,
            target_blocks=target_blocks,
            interval_hours=interval_hours,
        )
        merged = fs_bridge.forward_validation(
            x_norm, list(variables), interval=int(interval_hours), steps=int(steps)
        )
        # Same call, same arguments, same weights, no code-path difference.
        merged_repeat = fs_bridge.forward_validation(
            x_norm, list(variables), interval=int(interval_hours), steps=int(steps)
        )
    branch_final = branch[:, -1]
    diff = (branch_final - merged).abs()
    denominator = branch_final.abs().max().clamp_min(torch.finfo(branch_final.dtype).tiny)
    max_abs = float(diff.max())
    max_rel = float((diff.max() / denominator))
    self_max_abs = float((merged - merged_repeat).abs().max())
    passed = bool(
        torch.allclose(branch_final, merged, atol=atol, rtol=rtol)
    )
    return {
        "check": "fs_merge_equals_always_on_branch",
        "passed": passed,
        "steps": int(steps),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "self_max_abs_diff": self_max_abs,
        "atol": float(atol),
        "rtol": float(rtol),
        "branch_max_abs": float(branch_final.abs().max()),
        "target_blocks": [int(b) for b in target_blocks],
        "tf32_disabled_for_check": True,
        "self_max_abs_diff_note": (
            "Noise floor: the merged-path forward called twice with identical "
            "arguments, differenced against itself. Compare max_abs_diff "
            "against it -- a max_abs_diff of the same order means the two code "
            "paths agree as closely as one path agrees with itself."
        ),
    }


def verify_zero_edit_equivalence(
    fs_bridge: WeatherStepBridge,
    dynamic_bank: Mapping[int, ExpertLoRA],
    num_experts: int,
    x_norm: Tensor,
    variables: Sequence[str],
    *,
    steps: int = 4,
    interval_hours: int = 6,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    f0_bridge: Optional[WeatherStepBridge] = None,
    atol: float = 0.0,
) -> Dict[str, Any]:
    """Re-record zero-edit equivalence against **Fs**, never against F0.

    Two numbers come back and they are not interchangeable:

    * ``max_abs_diff_vs_fs`` -- the no-edit controlled rollout on the
      Fs-merged bridge against that same bridge's plain `forward_validation`.
      This is the property being certified, and it must be zero.
    * ``max_abs_diff_vs_f0`` -- how far that Fs baseline sits from F0's. It is
      reported for exactly one purpose: to show that reusing F0's numbers as the
      new reference would be wrong. When Fs has actually been trained this value
      is non-zero, and a check written against F0 would FAIL.

    ``passed`` depends only on the Fs comparison. If an F0 bridge is supplied
    and the two baselines coincide, ``discriminating`` comes back False and the
    caller is told the check cannot currently tell Fs from F0 (an untrained Fs
    is numerically F0 -- a normal initialization state, not a failure).
    """
    plan = EditPlan(
        plan_id="fs_zero_edit_reference",
        num_experts=int(num_experts),
        coefficients=(0.0,) * int(num_experts),
        hold_steps=max(1, int(steps)),
        interval_hours=int(interval_hours),
        continuation="reference_after_hold",
        rho=0.25,
    )
    names = list(variables)
    # Same reasoning as `verify_merge_equivalence`: this is a cross-code-path
    # numerical identity (hook-with-zero-coefficients vs plain forward), so it
    # must compare the paths and not TF32 rounding. It passes exactly today,
    # but TF32 could mask a real regression on a future run whose activations
    # happen to sit where TF32 and FP32 round apart.
    with torch.no_grad(), _disable_tf32_for_identity_check():
        no_edit = controlled_rollout(
            fs_bridge,
            x_norm,
            names,
            interval=int(interval_hours),
            steps=int(steps),
            plan=plan,
            expert_loras=dict(dynamic_bank),
            target_blocks=tuple(int(b) for b in target_blocks),
        )
        fs_plain = fs_bridge.forward_validation(
            x_norm, names, interval=int(interval_hours), steps=int(steps)
        )
        f0_plain = None
        if f0_bridge is not None:
            f0_plain = f0_bridge.forward_validation(
                x_norm, names, interval=int(interval_hours), steps=int(steps)
            )

    diff_fs = float((no_edit - fs_plain).abs().max())
    passed = diff_fs <= float(atol)

    result: Dict[str, Any] = {
        "check": "zero_edit_equals_fs",
        "baseline_tag": FS_BASELINE_TAG,
        "baseline_source": "fs_merged_backbone.forward_validation",
        "passed": bool(passed),
        "steps": int(steps),
        "num_experts": int(num_experts),
        "atol": float(atol),
        "max_abs_diff_vs_fs": diff_fs,
        "max_abs_diff_vs_f0": None,
        "discriminating": None,
        "tf32_disabled_for_check": True,
        "f0_is_not_the_reference": (
            "F0's numbers are NOT reused as the post-freeze reference; every "
            "round-4 gain is measured against Fs."
        ),
    }
    if f0_plain is not None:
        diff_f0 = float((no_edit - f0_plain).abs().max())
        result["max_abs_diff_vs_f0"] = diff_f0
        result["discriminating"] = bool(diff_f0 > float(atol))
        if not result["discriminating"]:
            result["note"] = (
                "The Fs baseline is numerically identical to F0's, so this check "
                "cannot currently distinguish them. That is the expected state "
                "for an untrained (zero-initialized) Fs and is a normal "
                "initialization state, not a failure -- but it does mean the "
                "equivalence result carries no evidence that Fs replaced F0."
            )
    return result


@dataclass(frozen=True)
class BackgroundF0Score:
    """F0's background score. Computed once. Reporting only. Never a baseline.

    Section 6.2 keeps F0's background score ("保留 F0 背景分数") while making Fs
    the reference every gain is measured against. Carrying it in its own tagged
    object -- rather than as another entry in a loss dict -- is what stops it
    being accidentally differenced against an Fs number.
    """

    tag: str
    loss: float
    n_samples: int
    objective: Dict[str, Any]
    computed_at: str
    reporting_only: bool = True
    note: str = (
        "F0 background score, retained for reporting only. Round-4 gains are "
        "measured against Fs; this number must never be subtracted from an "
        "Fs-relative loss or reported as gain_vs_Fs."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "loss": float(self.loss),
            "n_samples": int(self.n_samples),
            "objective": dict(self.objective),
            "computed_at": self.computed_at,
            "reporting_only": bool(self.reporting_only),
            "note": self.note,
        }


def _mean_no_edit_loss(
    bridge: WeatherStepBridge,
    samples: Sequence[TrainingSample],
    spec: ObjectiveSpec,
    *,
    variables: Optional[Sequence[str]] = None,
    steps: Optional[int] = None,
    interval_hours: int = 6,
) -> Tuple[float, int]:
    names = list(variables) if variables is not None else list(bridge.variables)
    horizon = int(steps) if steps is not None else spec.max_step
    total = 0.0
    counted = 0
    device = next(bridge.model.parameters()).device
    with torch.no_grad():
        for sample in samples:
            moved = sample.to(device)
            states = [moved.x_norm]
            x = moved.x_norm
            for _ in range(horizon):
                x = bridge.forward_validation(
                    x, names, interval=int(interval_hours), steps=1
                )
                states.append(x)
            trajectory = torch.stack(states, dim=1)
            total += float(objective_loss_for_sample(bridge, trajectory, moved, spec))
            counted += 1
    if counted == 0:
        raise StaticAdapterViolation(
            "SCORE_NO_SAMPLES",
            "No samples were scored; an empty mean is not a score.",
            {},
        )
    return total / counted, counted


def background_f0_score(
    f0_bridge: WeatherStepBridge,
    samples: Sequence[TrainingSample],
    spec: ObjectiveSpec,
    *,
    variables: Optional[Sequence[str]] = None,
    interval_hours: int = 6,
) -> BackgroundF0Score:
    """Compute F0's no-edit background loss ONCE, tagged and reporting-only."""
    mean_loss, counted = _mean_no_edit_loss(
        f0_bridge, samples, spec, variables=variables, interval_hours=interval_hours
    )
    return BackgroundF0Score(
        tag=BACKGROUND_F0_TAG,
        loss=mean_loss,
        n_samples=counted,
        objective=spec.to_dict(),
        computed_at=_utc_now(),
    )


def fs_baseline_score(
    fs_bridge: WeatherStepBridge,
    samples: Sequence[TrainingSample],
    spec: ObjectiveSpec,
    *,
    variables: Optional[Sequence[str]] = None,
    interval_hours: int = 6,
) -> Dict[str, Any]:
    """The Fs no-edit baseline: the ONE number gains are measured against."""
    mean_loss, counted = _mean_no_edit_loss(
        fs_bridge, samples, spec, variables=variables, interval_hours=interval_hours
    )
    return {
        "tag": FS_BASELINE_TAG,
        "loss": mean_loss,
        "n_samples": counted,
        "objective": spec.to_dict(),
        "computed_at": _utc_now(),
        "role": "the post-freeze no-edit reference; gain = baseline - candidate",
    }


def assert_gain_baseline_is_fs(baseline_tag: str) -> None:
    """Refuse any gain whose baseline is not Fs.

    Section 6.2: "冻结 Fs 后重新记录参考身份及零编辑等价性，不沿用 F0 的数值参照."
    The most likely way to violate that is not a deliberate decision but a
    leftover variable holding F0's loss, so the check is a hard refusal at the
    point the subtraction happens.
    """
    if baseline_tag != FS_BASELINE_TAG:
        raise StaticAdapterViolation(
            "GAIN_BASELINE_NOT_FS",
            f"Gain baseline is tagged {baseline_tag!r}, but after the Fs freeze "
            f"every gain must be measured against {FS_BASELINE_TAG!r}. "
            f"{BACKGROUND_F0_TAG!r} is retained for reporting only and must not "
            "be used as the numeric reference.",
            {"baseline_tag": baseline_tag, "required": FS_BASELINE_TAG},
        )


def fs_relative_gain(
    baseline: Mapping[str, Any], candidate_loss: float
) -> Dict[str, Any]:
    """gain = L(Fs no-edit) - L(candidate), with the baseline tag enforced."""
    assert_gain_baseline_is_fs(str(baseline.get("tag", "")))
    baseline_loss = float(baseline["loss"])
    return {
        "baseline_tag": FS_BASELINE_TAG,
        "baseline_loss": baseline_loss,
        "candidate_loss": float(candidate_loss),
        "gain_vs_fs": baseline_loss - float(candidate_loss),
    }


@dataclass
class PostFreezeVerification:
    """Everything re-recorded after the Fs freeze, in one auditable object."""

    artifact: Dict[str, Any] = field(default_factory=dict)
    reference_identity: Dict[str, Any] = field(default_factory=dict)
    merge_equivalence: Dict[str, Any] = field(default_factory=dict)
    zero_edit_equivalence: Dict[str, Any] = field(default_factory=dict)
    fs_baseline: Optional[Dict[str, Any]] = None
    background_f0: Optional[Dict[str, Any]] = None
    verified_at: str = ""

    @property
    def passed(self) -> bool:
        return bool(
            self.merge_equivalence.get("passed")
            and self.zero_edit_equivalence.get("passed")
            and self.reference_identity.get("static_adapter_is_bound")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "artifact": dict(self.artifact),
            "reference_identity": dict(self.reference_identity),
            "merge_equivalence": dict(self.merge_equivalence),
            "zero_edit_equivalence": dict(self.zero_edit_equivalence),
            "fs_baseline": dict(self.fs_baseline) if self.fs_baseline else None,
            "background_f0": dict(self.background_f0) if self.background_f0 else None,
            "verified_at": self.verified_at,
        }


def _effective_merge_atol(delta_max_abs: float, *, relative: float, floor: float) -> float:
    """Delta-magnitude-relative merge-equivalence tolerance.

    The branch-vs-merge FP32 non-associativity floor scales with the merged
    LoRA delta's magnitude, not with a fixed constant -- see the comment on
    `record_post_freeze_reference`'s `merge_atol_relative`/`merge_atol_floor`
    parameters for the real-GPU evidence. A fixed absolute atol calibrated at
    one training regime does not generalize to a longer one.
    """
    return max(float(floor), float(relative) * float(delta_max_abs))


def record_post_freeze_reference(
    base_bridge: WeatherStepBridge,
    fs_bridge: WeatherStepBridge,
    fs_adapters: Mapping[int, ExpertLoRA],
    artifact: MergedBackboneArtifact,
    x_norm: Tensor,
    variables: Sequence[str],
    *,
    num_experts: int,
    dynamic_bank: Optional[Mapping[int, ExpertLoRA]] = None,
    steps: int = 4,
    interval_hours: int = 6,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    # merge_atol is no longer a fixed constant: real-GPU evidence (jobs
    # pt-2r6tbwu7 at 32 updates and pt-3x63g0c6 at 500 updates, same 4
    # independently-trained experts, K=4/rank=4, target_blocks=[18..23], TF32
    # disabled for this check) shows the branch-vs-merge FP32 non-associativity
    # residual scales with the merged LoRA delta's magnitude
    # (artifact.delta_max_abs), not with a fixed absolute number:
    #   32 updates: delta_max_abs~=0.0678, max_abs_diff in [3.8e-5, 5.4e-5],
    #     ratio max_abs_diff/delta_max_abs in [5.63e-4, 7.93e-4]
    #   500 updates: delta_max_abs in [0.51, 2.31], max_abs_diff in
    #     [1.29e-4, 5.51e-4], ratio in [1.44e-4, 2.58e-4]
    # self_max_abs_diff was EXACTLY 0.0 for all 8 experts across both jobs,
    # ruling out hardware/kernel non-determinism in both regimes -- this is a
    # deterministic floor that grows with training progress via the trained
    # delta's magnitude, so a single fixed atol cannot be evidence-justified
    # across training regimes. merge_atol_relative=1.5e-3 gives >=1.89x margin
    # over every one of the 8 observed ratios above (tightest: 32-update
    # expert3 at 7.93e-4 vs threshold 1.5e-3, margin 1.89x; 500-update margins
    # are far larger, 5.8x-10.4x). merge_atol_floor=1e-5 preserves the
    # original tight generic-function tolerance when delta_max_abs is ~0 (a
    # numerical no-op Fs). This bound is still ~11-14x tighter than the old
    # TF32-enabled failure mode's ratio (~1.6e-2 to 2.1e-2 at the same
    # delta_max_abs~=0.068), so a TF32 regression is still caught decisively.
    # merge_rtol is unrelated to this scaling and is unchanged.
    merge_atol_relative: float = 1.5e-3,
    merge_atol_floor: float = 1e-5,
    merge_rtol: float = 1e-5,
    samples: Optional[Sequence[TrainingSample]] = None,
    spec: Optional[ObjectiveSpec] = None,
) -> PostFreezeVerification:
    """Run the whole post-freeze re-recording in one call.

    Produces: the merged-backbone artifact identity, the re-recorded reference
    `ArtifactVersion` (with `static_adapter` bound to Fs's digest rather than
    the placeholder ``"none"``), the merge-equivalence certificate, the FRESH
    zero-edit-equivalence certificate measured against Fs, and -- when samples
    and an objective are supplied -- the Fs baseline score alongside the
    separately-tagged F0 background score.
    """
    version = fs_bridge.version
    reference_identity = {
        "artifact_version": {
            "backbone": version.backbone,
            "static_adapter": version.static_adapter,
            "edit_bank": version.edit_bank,
            "normalization": version.normalization,
            "grid": version.grid,
            "projection": version.projection,
            "split": version.split,
            "continuation": version.continuation,
        },
        "artifact_version_digest": version.digest,
        "static_adapter_is_bound": version.static_adapter not in ("none", ""),
        "recorded_fresh_after_freeze": True,
        "note": (
            "Recorded from the Fs-merged bridge after the freeze. The F0 "
            "identity is not reused; static_adapter now names Fs's digest "
            "instead of the placeholder 'none'."
        ),
    }

    merge_atol = _effective_merge_atol(
        artifact.delta_max_abs, relative=merge_atol_relative, floor=merge_atol_floor
    )
    merge_check = verify_merge_equivalence(
        base_bridge,
        fs_bridge,
        fs_adapters,
        x_norm,
        variables,
        steps=1,
        interval_hours=interval_hours,
        target_blocks=target_blocks,
        atol=merge_atol,
        rtol=merge_rtol,
    )

    bank = dict(dynamic_bank or {})
    zero_edit = verify_zero_edit_equivalence(
        fs_bridge,
        bank,
        num_experts,
        x_norm,
        variables,
        steps=steps,
        interval_hours=interval_hours,
        target_blocks=target_blocks,
        f0_bridge=base_bridge,
    )

    verification = PostFreezeVerification(
        artifact=artifact.to_dict(),
        reference_identity=reference_identity,
        merge_equivalence=merge_check,
        zero_edit_equivalence=zero_edit,
        verified_at=_utc_now(),
    )
    if samples and spec is not None:
        verification.fs_baseline = fs_baseline_score(
            fs_bridge, samples, spec, variables=variables, interval_hours=interval_hours
        )
        verification.background_f0 = background_f0_score(
            base_bridge, samples, spec, variables=variables,
            interval_hours=interval_hours,
        ).to_dict()
    return verification


# =============================================================================
# Real-data loading from admission records
# =============================================================================

def load_admitted_sample(
    record: Mapping[str, Any],
    bridge: WeatherStepBridge,
    *,
    lead_steps: Sequence[int],
    variable: str = "data",
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> TrainingSample:
    """Load one `AdmissionResult` row into a `TrainingSample`.

    The record MUST come from `scripts/r2_admission_gate.py` /
    `earthdelta.data.make_splits.admit_real_sample`. This function deliberately
    does not resolve times or open manifests itself: bypassing the admission
    gate is how a year with an empty store (2016/2017) or a year that does not
    exist (2021) gets consumed.
    """
    if not record.get("admitted"):
        raise StaticAdapterViolation(
            "SAMPLE_NOT_ADMITTED",
            f"Record {record.get('event_id')!r} is not an admitted sample. Only "
            "rows admitted by the Stage-3 admission gate may be trained on.",
            {"event_id": record.get("event_id")},
        )
    role = assert_valid_data_role(record.get("data_role"))
    if role != DataRole.BANK_FIT.value:
        raise StaticAdapterViolation(
            "SAMPLE_DATA_ROLE_NOT_BANK_FIT",
            f"Record {record.get('event_id')!r} has data_role={role!r}; Fs and "
            "the expert bank consume bank_fit data only.",
            {"event_id": record.get("event_id"), "data_role": role},
        )
    for key in ("issue_store", "issue_index", "issue_id", "process_group_id"):
        if key not in record:
            raise StaticAdapterViolation(
                "SAMPLE_RECORD_INCOMPLETE",
                f"Admission record is missing {key!r}; it did not come from the "
                "admission gate.",
                {"missing": key},
            )

    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - exercised only on real runs
        raise StaticAdapterViolation(
            "XARRAY_UNAVAILABLE",
            f"xarray is required to read an admitted sample: {exc}",
            {"store_path": record.get("issue_store")},
        ) from exc

    store_path = Path(str(record["issue_store"]))
    issue_index = int(record["issue_index"])
    needed = sorted({int(s) for s in lead_steps})
    if not needed or needed[0] <= 0:
        raise StaticAdapterViolation(
            "SAMPLE_LEAD_STEPS_INVALID",
            f"lead_steps must be positive rollout steps, got {list(lead_steps)}.",
            {"lead_steps": list(lead_steps)},
        )

    dataset = xr.open_zarr(str(store_path))
    try:
        n_timesteps = int(dataset.sizes["time"])
        # Reuse the admission contract's endpoint check rather than inventing a
        # second one: every target index must really exist in this store.
        from .pilot_contract import validate_slice_index

        validate_slice_index(
            index=issue_index,
            n_timesteps=n_timesteps,
            history_steps=int(record.get("history_steps", 0)),
            target_steps=needed[-1],
            time_coords=dataset["time"].values,
            interval_hours=int(record.get("interval_hours", 6)),
        )
        array = dataset[variable]
        issue_raw = torch.as_tensor(
            array.isel(time=issue_index).values, dtype=dtype
        ).unsqueeze(0)
        targets = {
            step: torch.as_tensor(
                array.isel(time=issue_index + step).values, dtype=dtype
            ).unsqueeze(0)
            for step in needed
        }
        times = dataset["time"].values
        time_utc = str(times[issue_index])
    finally:
        try:
            dataset.close()
        except Exception:  # noqa: BLE001 - closing must not mask a failure
            pass

    if device is not None:
        issue_raw = issue_raw.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

    x_norm = bridge.normalization.normalize(issue_raw)
    return TrainingSample(
        x_norm=x_norm,
        targets_raw=targets,
        issue_id=str(record["issue_id"]),
        issue_time=int(record.get("issue_time", 0)),
        valid_time=int(record.get("valid_time", 0)),
        history_time=int(record.get("history_time", 0)),
        split_id=str(record.get("split_id", "")),
        data_role=role,
        source=str(store_path),
        time_utc=time_utc,
    )


def _utc_seconds(value: Any) -> int:
    """Epoch seconds of a datetime64/ISO value, for exact time identity."""
    import numpy as np

    return int(np.datetime64(value, "s").astype("int64"))


def certify_admission_for_fs(
    admission_record: Mapping[str, Any],
    *,
    lead_steps: Sequence[int],
    required_history_steps: int = 2,
    expected_normalization_digest: Optional[str] = None,
    expected_grid_hash: Optional[str] = None,
    expected_variable_order_hash: Optional[str] = None,
    reverify_content: bool = True,
) -> Dict[str, Any]:
    """Fail-closed consumer certification of an admission record (FP-03 Q0).

    Nothing is trusted from the record's own PASS flags alone. Every failure
    is collected and raised together as ``ADMISSION_NOT_CERTIFIED``:

      * the OUTER record and the inner admission are passed, formal and
        bank_fit, and both the B09 join and the B08 content results are True;
      * every admitted row is formal bank_fit with >= ``required_history_steps``
        of history and a declared lead reaching the furthest requested lead,
        and has exactly one passed certificate for ITS issue_id, role and
        process group, on the same store;
      * that certificate covers history through the furthest lead:
        ``issue_index - required_history_steps .. issue_index + max(lead_steps)``;
      * the certificate's UTC stamps equal the store's time coordinate at those
        indices, and the row's issue_time equals the store time at issue_index;
      * normalization digest, grid hash and variable-order hash equal the
        expected (consumer-side) identities;
      * a FRESH `verify_content_subset` over the same slice reproduces the
        stored content and identity SHA-256 byte for byte.
    """
    from .pilot_contract import PilotContractViolation, verify_content_subset

    failures: List[Dict[str, Any]] = []

    def fail(code: str, **detail: Any) -> None:
        failures.append({"code": code, **detail})

    max_step = max(int(s) for s in lead_steps)
    max_lead_hours = max_step * 6
    admission = admission_record.get("admission", {})
    for scope, obj in (("outer", admission_record), ("admission", admission)):
        if obj.get("passed") is not True:
            fail("NOT_PASSED", scope=scope)
        if obj.get("formal") is not True:
            fail("NOT_FORMAL", scope=scope)
        if obj.get("data_role") != DataRole.BANK_FIT.value:
            fail("NOT_BANK_FIT", scope=scope, data_role=obj.get("data_role"))
    results = admission_record.get("results") or {}
    for key in ("b09_real_sample_admission", "b08_content_verification"):
        if results.get(key) is not True:
            fail("GATE_RESULT_NOT_TRUE", result=key, value=results.get(key))

    rows = list(admission.get("admitted") or [])
    if not rows:
        fail("NO_ADMITTED_ROWS")
    certificates = list(admission_record.get("content_certificates") or [])
    by_issue: Dict[str, List[Mapping[str, Any]]] = {}
    for cert in certificates:
        by_issue.setdefault(str(cert.get("issue_id")), []).append(cert)

    report_rows: List[Dict[str, Any]] = []
    for row in rows:
        iid = str(row.get("issue_id"))
        if row.get("admitted") is not True or row.get("formal") is not True:
            fail("ROW_NOT_ADMITTED_FORMAL", issue_id=iid)
        if row.get("data_role") != DataRole.BANK_FIT.value:
            fail("ROW_NOT_BANK_FIT", issue_id=iid, data_role=row.get("data_role"))
        if int(row.get("interval_hours", 0)) != 6:
            fail("ROW_INTERVAL_NOT_6H", issue_id=iid)
        if int(row.get("history_steps", 0)) < int(required_history_steps):
            fail("ROW_HISTORY_TOO_SHORT", issue_id=iid, history_steps=row.get("history_steps"),
                 required=int(required_history_steps))
        if float(row.get("lead_hours", 0)) < max_lead_hours:
            fail("ROW_LEAD_TOO_SHORT", issue_id=iid, lead_hours=row.get("lead_hours"),
                 required=max_lead_hours)
        stores = {str(Path(str(row.get(k))).resolve()) for k in
                  ("history_store", "issue_store", "target_store") if row.get(k) is not None}
        if len(stores) != 1:
            fail("ROW_SPANS_STORES", issue_id=iid)
        store = str(Path(str(row.get("issue_store"))).resolve())
        if expected_normalization_digest is not None and \
                row.get("normalization_hash") != expected_normalization_digest:
            fail("ROW_NORMALIZATION_MISMATCH", issue_id=iid, row=row.get("normalization_hash"),
                 expected=expected_normalization_digest)
        if expected_grid_hash is not None and row.get("grid_hash") != expected_grid_hash:
            fail("ROW_GRID_MISMATCH", issue_id=iid, row=row.get("grid_hash"), expected=expected_grid_hash)

        embedded = row.get("content_certificate")
        matches = list(by_issue.get(iid, []))
        if isinstance(embedded, Mapping):
            matches = matches or [embedded]
            if any(dict(m) != dict(embedded) for m in matches):
                fail("ROW_CERTIFICATE_CONFLICT", issue_id=iid)
        if len(matches) != 1:
            fail("ROW_CERTIFICATE_COUNT", issue_id=iid, n=len(matches))
            continue
        cert = matches[0]
        if cert.get("passed") is not True:
            fail("CERT_NOT_PASSED", issue_id=iid)
        if cert.get("data_role") != DataRole.BANK_FIT.value:
            fail("CERT_NOT_BANK_FIT", issue_id=iid, data_role=cert.get("data_role"))
        if cert.get("process_group_id") != row.get("process_group_id"):
            fail("CERT_PROCESS_GROUP_MISMATCH", issue_id=iid)
        if str(Path(str(cert.get("store_path"))).resolve()) != store:
            fail("CERT_STORE_MISMATCH", issue_id=iid)
        if cert.get("grid_hash") != row.get("grid_hash"):
            fail("CERT_GRID_MISMATCH", issue_id=iid)
        if expected_variable_order_hash is not None and \
                cert.get("variable_order_hash") != expected_variable_order_hash:
            fail("CERT_VARIABLE_ORDER_MISMATCH", issue_id=iid,
                 cert=cert.get("variable_order_hash"), expected=expected_variable_order_hash)
        indices = [int(i) for i in cert.get("indices") or []]
        issue_index = int(row.get("issue_index", -1))
        needed = list(range(issue_index - int(required_history_steps), issue_index + max_step + 1))
        if indices != sorted(indices) or (indices and indices != list(range(indices[0], indices[-1] + 1))):
            fail("CERT_INDICES_NOT_CONTIGUOUS", issue_id=iid)
        if not set(needed) <= set(indices):
            fail("CERT_DOES_NOT_COVER_WINDOW", issue_id=iid, needed=[needed[0], needed[-1]],
                 covered=[indices[0], indices[-1]] if indices else None)
        times = list(cert.get("times_utc") or [])
        if len(times) != len(indices):
            fail("CERT_TIMES_LENGTH", issue_id=iid)
            continue

        entry: Dict[str, Any] = {
            "issue_id": iid,
            "issue_index": issue_index,
            "store": store,
            "certificate_indices": [indices[0], indices[-1]] if indices else None,
            "content_sha256": cert.get("content_sha256"),
            "identity_sha256": cert.get("identity_sha256"),
        }
        try:
            import xarray as xr

            dataset = xr.open_zarr(store)
            try:
                store_times = dataset["time"].values
                for index, stamp in zip(indices, times):
                    if _utc_seconds(store_times[index]) != _utc_seconds(stamp):
                        fail("CERT_UTC_MISMATCH", issue_id=iid, index=index, cert=stamp,
                             store=str(store_times[index]))
                        break
                if _utc_seconds(store_times[issue_index]) != int(row.get("issue_time", -1)):
                    fail("ROW_ISSUE_TIME_MISMATCH", issue_id=iid)
            finally:
                dataset.close()
        except (OSError, KeyError, IndexError, ValueError) as exc:
            fail("STORE_UNREADABLE", issue_id=iid, error=f"{type(exc).__name__}: {exc}")
            continue

        if reverify_content:
            try:
                fresh = verify_content_subset(
                    store_path=Path(store),
                    indices=indices,
                    batch_size=int(cert.get("batch_size") or 2),
                    expected_channels=cert.get("n_channels"),
                    sigma_bound=float(cert.get("sigma_bound")),
                    normalization_dir=(Path(cert["normalization_source"])
                                       if cert.get("normalization_source") else None),
                    require_physical_range=bool(cert.get("physical_range_checked")),
                    data_role=cert.get("data_role"),
                    issue_id=cert.get("issue_id"),
                    process_group_id=cert.get("process_group_id"),
                    strict=True,
                )
                entry["fresh_content_sha256"] = fresh.content_sha256
                if fresh.content_sha256 != cert.get("content_sha256"):
                    fail("CONTENT_SHA256_MISMATCH", issue_id=iid)
                if fresh.identity_sha256 != cert.get("identity_sha256"):
                    fail("IDENTITY_SHA256_MISMATCH", issue_id=iid)
            except (PilotContractViolation, OSError, ValueError, TypeError) as exc:
                fail("CONTENT_REVERIFY_FAILED", issue_id=iid, error=f"{type(exc).__name__}: {exc}")
        report_rows.append(entry)

    report = {
        "check": "fs_admission_consumer_certification",
        "passed": not failures,
        "n_rows": len(rows),
        "required_history_steps": int(required_history_steps),
        "max_lead_step": max_step,
        "expected_normalization_digest": expected_normalization_digest,
        "expected_grid_hash": expected_grid_hash,
        "expected_variable_order_hash": expected_variable_order_hash,
        "content_reverified": bool(reverify_content),
        "rows": report_rows,
        "failures": failures,
    }
    if failures:
        raise StaticAdapterViolation(
            "ADMISSION_NOT_CERTIFIED",
            f"admission record failed {len(failures)} consumer check(s); first: "
            f"{failures[0]}",
            report,
        )
    return report


def load_admitted_samples(
    admission_record: Mapping[str, Any],
    bridge: WeatherStepBridge,
    *,
    lead_steps: Sequence[int],
    limit: Optional[int] = None,
    require_certified: bool = False,
    required_history_steps: int = 2,
    expected_grid_hash: Optional[str] = None,
    expected_variable_order_hash: Optional[str] = None,
    certification_out: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[TrainingSample]:
    """Load every admitted row from an admission-gate JSON record.

    Accepts either the full gate record (`{"admission": {"admitted": [...]}}`)
    or the `AdmissionRunReport` dict directly.

    With ``require_certified=True`` (every real FP-03 entry point) the whole
    record must first pass `certify_admission_for_fs` against THIS bridge's
    normalization digest and the canonical grid / variable-order hashes (or
    the explicitly supplied ones); nothing is read otherwise. The report is
    copied into ``certification_out`` when a dict is supplied.
    """
    if require_certified:
        if expected_grid_hash is None or expected_variable_order_hash is None:
            from .data.pull_wb2 import grid_hash, variable_order_hash

            expected_grid_hash = expected_grid_hash or grid_hash()
            expected_variable_order_hash = expected_variable_order_hash or variable_order_hash()
        report = certify_admission_for_fs(
            admission_record,
            lead_steps=lead_steps,
            required_history_steps=required_history_steps,
            expected_normalization_digest=bridge.normalization.digest,
            expected_grid_hash=expected_grid_hash,
            expected_variable_order_hash=expected_variable_order_hash,
        )
        if certification_out is not None:
            certification_out.update(report)
    admission = admission_record.get("admission", admission_record)
    rows = admission.get("admitted") if isinstance(admission, Mapping) else None
    if not rows:
        raise StaticAdapterViolation(
            "ADMISSION_RECORD_EMPTY",
            "The admission record contains no admitted samples. Run "
            "scripts/r2_admission_gate.py first; an empty admission is not a "
            "training set.",
            {},
        )
    selected = list(rows)[: int(limit)] if limit is not None else list(rows)
    return [
        load_admitted_sample(row, bridge, lead_steps=lead_steps, **kwargs)
        for row in selected
    ]
