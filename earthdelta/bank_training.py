"""K=4 / rank=4 dynamic expert-bank training on top of a frozen Fs backbone.

Governing spec: `plans/plans_v2_0921/CLAUDE_EXECUTION_PLAN.md` sections 6.1-6.2.

The design values this module implements are FROZEN for this round and are not
re-derived here:

    dynamic bank        K = 4, rank_per_expert = 4, target blocks 18-23
    candidates          explicit no-edit + K singletons, max_active = 1
    magnitude           a0 = 0.25, rho = 0.25, explicitly constructed
    hold                4 x 6h steps, then continue from the edited state as Fs

Three obligations from section 6.2 shape the code:

1. DIVERSITY MUST BE DECLARED, NOT SEARCHED FOR.

       "bank 先说明专家差异的训练依据，优先使用 fit-only 的合法 regime/数据分组
        与共同目标，不做一轮无边界专家构造搜索。"

   `assign_diversity_groups` implements ONE pre-declared, fit-only rule --
   disjoint calendar-month blocks over `bank_fit`-role admitted samples
   (`DIVERSITY_RULE_ID`). The month of an analysis time is an input-side
   quantity: it is known before any forecast is run and depends on no truth, no
   outcome and no candidate response, so grouping on it cannot leak evaluation
   information into the bank. The rule is stated in code, applied to every
   expert identically, and produces provably disjoint groups.

2. EVERY EXPERT MUST BE RECORDED, AND NONE MAY BE DROPPED FOR LOSING.

       "每个专家记录训练数据、更新数、loss、非零响应与多样性；零 B 初始状态是
        正常初始化，未训练零库不是科学失败。只允许按事前资格规则处理无效库，
        不按评估收益删掉专家。"

   `ExpertTrainingRecord` carries exactly those fields. `ELIGIBILITY_RULES` is
   the complete, pre-declared list of reasons an expert may be excluded -- all
   of them numerical failures (non-finite loss / gradient, zero updates). Low
   gain is deliberately absent, and `assert_eligibility_rule_is_pre_declared`
   refuses any reason that is not on the list. A bank still at its zero-init
   `B` factors reports `zero_initialized_untrained=True` and stays ELIGIBLE.

3. THE HORIZON AND THE CAPACITY DECISION MUST COME FROM MEASUREMENTS.

       "固定系数训练验证 A/B 梯度即可，不等待 B03。若 24h 反传在当前串行/no-replay
        模式下不可行，先提供 6h 训练、24h/72h 评估的明确规格调整和成本；不能暗中开
        activation checkpoint 或改变训练时域后仍引用旧实验身份。"

   `profile_training_step` measures one real training step (memory +
   throughput) and `decide_bank_capacity` turns that measurement -- and nothing
   else -- into a K=4/rank=4 vs K=2/lower-rank decision, tagging the decision
   `provisional` unless it came from a real CUDA profile. `check_horizon_
   feasibility` walks the differentiable rollout out to the full 4 x 6h hold
   window and, if that is not feasible, emits a `HorizonFallbackSpec` that
   states the reduced training horizon, the separate non-differentiable
   24h/72h evaluation, the cost estimate, and a NEW experiment identity.
   Neither function ever enables activation checkpointing;
   `assert_activation_checkpointing_disabled` asserts it stayed off.

Training itself uses the existing, unmodified `controlled_rollout` with fixed
(non-differentiable) coefficients, so gradients flow to the bank's A/B factors
only -- which is all section 6.2 asks for while B03 stays deferred.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import datetime as _datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .bridge.stormer_bridge import (
    _ACTIVATION_CHECKPOINT_FLAGS,
    CONTROLLED_ROLLOUT_SUPPORTED_MODES,
    WeatherStepBridge,
    controlled_rollout,
)
from .contracts import EditPlan
from .data.make_splits import compute_guard_boundaries, is_in_guard_window
from .lowrank import ExpertLoRA
from .pilot_contract import DataRole
from .registry import (
    CandidateRegistry,
    DEFAULT_A0,
    DEFAULT_RHO,
    build_pilot_registry,
)
from .static_adapter import (
    DEFAULT_TARGET_BLOCKS,
    MODE_UPDATE_CAPS,
    ObjectiveSpec,
    TrainingSample,
    assert_bank_fit_samples,
    objective_loss_for_sample,
    panel_ratio_summary,
    per_sample_loss_direction,
    state_dict_digest,
    training_stability,
)

__all__ = [
    "BankTrainingViolation",
    "DESIGN_NUM_EXPERTS",
    "DESIGN_RANK_PER_EXPERT",
    "FALLBACK_NUM_EXPERTS",
    "FALLBACK_RANK_PER_EXPERT",
    "DESIGN_A0",
    "DESIGN_RHO",
    "DESIGN_MAX_ACTIVE",
    "DESIGN_HOLD_STEPS",
    "DIVERSITY_RULE_ID",
    "DIVERSITY_RULE_DOC",
    "ELIGIBILITY_RULES",
    "DiversityGroup",
    "GroupingReport",
    "BankTrainConfig",
    "ExpertTrainingRecord",
    "TrainingStepProfile",
    "CapacityBudget",
    "BankCapacityDecision",
    "HorizonFallbackSpec",
    "HorizonFeasibilityReport",
    "assign_diversity_groups",
    "build_bank_registry",
    "expert_plan",
    "assert_max_active",
    "assert_eligibility_rule_is_pre_declared",
    "evaluate_eligibility",
    "nonzero_response_check",
    "build_dynamic_bank",
    "train_expert",
    "profile_training_step",
    "decide_bank_capacity",
    "assert_activation_checkpointing_disabled",
    "check_horizon_feasibility",
    # FP-04: certified-Fs bank artifacts and the pre-registered rules
    "bank_digest",
    "expert_digest",
    "other_experts_digest",
    "check_bank_device",
    "bank_panel_losses",
    "singleton_probe_states",
    "compare_probe_states",
    "save_expert",
    "load_expert",
    "install_expert",
    "save_bank",
    "load_bank",
    "assemble_bank",
    "source_bank_for_expert",
    "BANK_QUAL_RULE_ID",
    "BANK_SELECT_RULE_ID",
    "BANK_QUAL_VERDICTS",
    "evaluate_bank_expert_qualification",
    "decide_bank_formal",
    "evaluate_bank_assembly",
]


# =============================================================================
# Frozen design values (section 6.1)
# =============================================================================

DESIGN_NUM_EXPERTS = 4
DESIGN_RANK_PER_EXPERT = 4

#: The single pre-declared shrink target, named BEFORE any profile is run so
#: that the capacity decision is a choice between two announced options rather
#: than a free search: "不足时在看开发结果前缩为 K=2 或降低 rank".
FALLBACK_NUM_EXPERTS = 2
FALLBACK_RANK_PER_EXPERT = 2

DESIGN_A0 = DEFAULT_A0        # 0.25
DESIGN_RHO = DEFAULT_RHO      # 0.25
DESIGN_MAX_ACTIVE = 1
DESIGN_HOLD_STEPS = 4         # 4 x 6h = a 24h hold, then continue as Fs
DESIGN_INTERVAL_HOURS = 6


class BankTrainingViolation(ValueError):
    """A bank grouping, training, profiling or horizon check failed a check."""

    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.detail: Dict[str, Any] = dict(detail or {})


def _utc_now() -> str:
    return datetime.datetime.now(timezone.utc).isoformat()


def _utc_dt(timestamp: int) -> _datetime:
    return _datetime.fromtimestamp(int(timestamp), tz=timezone.utc)


def _month_key(timestamp: int) -> Tuple[int, int]:
    dt = _utc_dt(timestamp)
    return (dt.year, dt.month)


def _month_start(year: int, month: int) -> int:
    return int(_datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


def _next_month(year: int, month: int) -> Tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


# =============================================================================
# 1. Pre-declared, fit-only diversity grouping
# =============================================================================

DIVERSITY_RULE_ID = "ed-bank-diversity/1:disjoint-calendar-month-blocks"

DIVERSITY_RULE_DOC = """\
Expert k trains on block k of the bank_fit calendar.

Rule, stated in full and fixed before any expert is trained:

  1. Only samples whose frozen data_role is `bank_fit` are eligible. Fs, the
     bank and this grouping never read policy_dev or confirm.
  2. A sample is purged if its analysis time falls in its split's guard window
     (`make_splits.compute_guard_boundaries` / `is_in_guard_window`, reused
     unchanged), so a sample whose history or lead reaches into an adjacent
     split never trains anything.
  3. The distinct (year, month) keys present among the survivors are sorted and
     partitioned into K contiguous, non-overlapping blocks of as-equal size as
     the month count allows. Block k is expert k's training regime.
  4. A sample is admitted to block k only if its ENTIRE window -- from its
     history endpoint through the later of its verification time and the end of
     its hold window -- lies inside block k's half-open time interval. A sample
     whose window straddles a block boundary is purged, not truncated and not
     shared, so the two experts either side of that boundary see disjoint data.

Why the calendar month: it is a property of the analysis time alone. It is
known before any forecast is produced, it depends on no truth field, no
realized loss and no candidate response, and it is identical for every expert,
so it is a legal fit-only regime split rather than a post-hoc search for a
grouping that happens to make the bank look good. It also gives a defensible
meteorological reading -- different months are different large-scale regimes --
without introducing a learned or outcome-dependent regime labeller.

What this rule is NOT: it is not a claim that calendar months are the optimal
partition, and it must not be re-run with a different K, a different month
assignment or a different calendar unit after development results have been
seen. Changing it after that point makes a new experiment, not a better
grouping.
"""


@dataclass(frozen=True)
class DiversityGroup:
    """One expert's pre-declared training regime."""

    expert_index: int
    rule: str
    month_keys: Tuple[Tuple[int, int], ...]
    start_utc: int
    end_utc: int
    issue_ids: Tuple[str, ...]
    sample_positions: Tuple[int, ...]

    @property
    def n_samples(self) -> int:
        return len(self.sample_positions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_index": int(self.expert_index),
            "rule": self.rule,
            "months": [f"{y:04d}-{m:02d}" for y, m in self.month_keys],
            "start_utc": int(self.start_utc),
            "end_utc": int(self.end_utc),
            "start_iso": _utc_dt(self.start_utc).isoformat(),
            "end_iso": _utc_dt(self.end_utc).isoformat(),
            "n_samples": self.n_samples,
            "issue_ids": list(self.issue_ids),
        }


@dataclass
class GroupingReport:
    """Every input sample, and which group it landed in or why it was purged."""

    rule: str
    rule_doc: str
    num_experts: int
    groups: Tuple[DiversityGroup, ...] = ()
    purged: List[Dict[str, Any]] = field(default_factory=list)
    n_input: int = 0
    n_assigned: int = 0
    max_history_hours: int = 24
    max_lead_hours: int = 168
    hold_steps: int = DESIGN_HOLD_STEPS
    interval_hours: int = DESIGN_INTERVAL_HOURS
    created_at: str = ""

    def group_for(self, expert_index: int) -> DiversityGroup:
        for group in self.groups:
            if group.expert_index == int(expert_index):
                return group
        raise BankTrainingViolation(
            "BANK_GROUP_NOT_FOUND",
            f"No diversity group for expert {expert_index}; the grouping holds "
            f"{[g.expert_index for g in self.groups]}.",
            {"expert_index": int(expert_index)},
        )

    def samples_for(
        self, expert_index: int, samples: Sequence[TrainingSample]
    ) -> List[TrainingSample]:
        group = self.group_for(expert_index)
        return [samples[i] for i in group.sample_positions]

    def assert_disjoint(self) -> None:
        """Groups must not share an issue_id or overlap in time."""
        seen: Dict[str, int] = {}
        for group in self.groups:
            for issue_id in group.issue_ids:
                if issue_id in seen:
                    raise BankTrainingViolation(
                        "BANK_GROUPS_OVERLAP",
                        f"issue_id {issue_id!r} appears in both expert "
                        f"{seen[issue_id]}'s and expert {group.expert_index}'s "
                        "group; the diversity rule must produce disjoint groups.",
                        {"issue_id": issue_id,
                         "experts": [seen[issue_id], group.expert_index]},
                    )
                seen[issue_id] = group.expert_index
        ordered = sorted(self.groups, key=lambda g: g.start_utc)
        for left, right in zip(ordered, ordered[1:]):
            if left.end_utc > right.start_utc:
                raise BankTrainingViolation(
                    "BANK_GROUP_INTERVALS_OVERLAP",
                    f"expert {left.expert_index}'s interval ends at "
                    f"{_utc_dt(left.end_utc).isoformat()} after expert "
                    f"{right.expert_index}'s begins at "
                    f"{_utc_dt(right.start_utc).isoformat()}.",
                    {"left": left.expert_index, "right": right.expert_index},
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "rule_doc": self.rule_doc,
            "num_experts": int(self.num_experts),
            "groups": [g.to_dict() for g in self.groups],
            "purged": list(self.purged),
            "n_input": int(self.n_input),
            "n_assigned": int(self.n_assigned),
            "n_purged": len(self.purged),
            "max_history_hours": int(self.max_history_hours),
            "max_lead_hours": int(self.max_lead_hours),
            "hold_steps": int(self.hold_steps),
            "interval_hours": int(self.interval_hours),
            "created_at": self.created_at,
        }


def _sample_attr(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(name, default)
    return getattr(sample, name, default)


def assign_diversity_groups(
    samples: Sequence[Any],
    num_experts: int = DESIGN_NUM_EXPERTS,
    *,
    hold_steps: int = DESIGN_HOLD_STEPS,
    interval_hours: int = DESIGN_INTERVAL_HOURS,
    max_history_hours: int = 24,
    max_lead_hours: int = 168,
    require_data_role: str = DataRole.BANK_FIT.value,
) -> GroupingReport:
    """Partition bank_fit samples into K disjoint calendar-month regimes.

    See `DIVERSITY_RULE_DOC` for the rule in full. Accepts `TrainingSample`
    objects or plain admission-record dicts; both expose the same field names.

    Raises:
        BankTrainingViolation: If any sample carries a role other than
            `bank_fit`, if the surviving samples span fewer distinct months than
            experts (diversity cannot be fabricated from one month), or if the
            resulting groups are not disjoint.
    """
    if type(num_experts) is not int or num_experts <= 0:
        raise BankTrainingViolation(
            "BANK_NUM_EXPERTS_INVALID",
            f"num_experts must be a positive integer, got {num_experts!r}.",
            {"num_experts": repr(num_experts)},
        )

    report = GroupingReport(
        rule=DIVERSITY_RULE_ID,
        rule_doc=DIVERSITY_RULE_DOC,
        num_experts=int(num_experts),
        n_input=len(samples),
        max_history_hours=int(max_history_hours),
        max_lead_hours=int(max_lead_hours),
        hold_steps=int(hold_steps),
        interval_hours=int(interval_hours),
        created_at=_utc_now(),
    )

    # -- 1. role -----------------------------------------------------------
    offending = [
        {
            "position": i,
            "issue_id": str(_sample_attr(s, "issue_id", "")),
            "data_role": _sample_attr(s, "data_role"),
        }
        for i, s in enumerate(samples)
        if _sample_attr(s, "data_role") != require_data_role
    ]
    if offending:
        raise BankTrainingViolation(
            "BANK_FIT_ROLE_REQUIRED",
            f"{len(offending)} of {len(samples)} sample(s) are not "
            f"{require_data_role!r}: {offending[:4]}. The bank is grouped and "
            "trained on bank_fit data only; a policy_dev or confirm sample here "
            "would let the bank see the blocks it is evaluated on.",
            {"n_offending": len(offending), "offending": offending[:16]},
        )

    # -- 2. split guard window (reused, not re-invented) --------------------
    boundaries = compute_guard_boundaries(
        max_history_hours=int(max_history_hours), max_lead_hours=int(max_lead_hours)
    )
    survivors: List[int] = []
    for position, sample in enumerate(samples):
        issue_time = int(_sample_attr(sample, "issue_time", 0) or 0)
        split_id = str(_sample_attr(sample, "split_id", "") or "")
        issue_dt = _utc_dt(issue_time)
        if is_in_guard_window(issue_dt, split_id, boundaries):
            report.purged.append({
                "position": position,
                "issue_id": str(_sample_attr(sample, "issue_id", "")),
                "issue_time_utc": issue_dt.isoformat(),
                "split_id": split_id,
                "reason": "split_guard_window",
                "detail": (
                    "issue time lies in the split's guard window "
                    "(compute_guard_boundaries / is_in_guard_window), so its "
                    "history or lead would cross into an adjacent split"
                ),
            })
            continue
        survivors.append(position)

    if not survivors:
        raise BankTrainingViolation(
            "BANK_NO_SAMPLES_AFTER_GUARD",
            f"All {len(samples)} sample(s) were purged by the split guard "
            "window; there is nothing left to group.",
            {"n_input": len(samples), "n_purged": len(report.purged)},
        )

    # -- 3. month blocks ----------------------------------------------------
    month_keys = sorted({
        _month_key(int(_sample_attr(samples[p], "issue_time", 0) or 0))
        for p in survivors
    })
    if len(month_keys) < int(num_experts):
        raise BankTrainingViolation(
            "BANK_DIVERSITY_INSUFFICIENT_MONTHS",
            f"The surviving bank_fit samples span {len(month_keys)} distinct "
            f"calendar month(s) but {num_experts} disjoint expert regimes were "
            "requested. The pre-declared rule cannot manufacture diversity from "
            "fewer months than experts; admit a wider bank_fit range, or lower "
            "K before any development result is viewed.",
            {"n_months": len(month_keys), "num_experts": int(num_experts),
             "months": [f"{y:04d}-{m:02d}" for y, m in month_keys]},
        )

    blocks: List[List[Tuple[int, int]]] = []
    base, extra = divmod(len(month_keys), int(num_experts))
    cursor = 0
    for k in range(int(num_experts)):
        size = base + (1 if k < extra else 0)
        blocks.append(month_keys[cursor:cursor + size])
        cursor += size

    # -- 4. window-containment purge ---------------------------------------
    groups: List[DiversityGroup] = []
    for k, block in enumerate(blocks):
        start_utc = _month_start(*block[0])
        end_year, end_month = _next_month(*block[-1])
        end_utc = _month_start(end_year, end_month)

        members: List[int] = []
        for position in survivors:
            sample = samples[position]
            issue_time = int(_sample_attr(sample, "issue_time", 0) or 0)
            if not (start_utc <= issue_time < end_utc):
                continue
            history_time = int(
                _sample_attr(sample, "history_time", issue_time) or issue_time
            )
            valid_time = int(_sample_attr(sample, "valid_time", issue_time) or issue_time)
            hold_end = issue_time + int(hold_steps) * int(interval_hours) * 3600
            window_start = min(history_time, issue_time)
            window_end = max(valid_time, hold_end)
            if window_start < start_utc or window_end > end_utc:
                report.purged.append({
                    "position": position,
                    "issue_id": str(_sample_attr(sample, "issue_id", "")),
                    "issue_time_utc": _utc_dt(issue_time).isoformat(),
                    "window_start_utc": _utc_dt(window_start).isoformat(),
                    "window_end_utc": _utc_dt(window_end).isoformat(),
                    "group_start_utc": _utc_dt(start_utc).isoformat(),
                    "group_end_utc": _utc_dt(end_utc).isoformat(),
                    "expert_index": k,
                    "reason": "group_boundary_overlap",
                    "detail": (
                        "the sample's history/target/hold window crosses this "
                        "diversity group's boundary, so admitting it would make "
                        "two experts' training data overlap in time"
                    ),
                })
                continue
            members.append(position)

        groups.append(DiversityGroup(
            expert_index=k,
            rule=DIVERSITY_RULE_ID,
            month_keys=tuple(block),
            start_utc=start_utc,
            end_utc=end_utc,
            issue_ids=tuple(
                str(_sample_attr(samples[p], "issue_id", "")) for p in members
            ),
            sample_positions=tuple(members),
        ))

    report.groups = tuple(groups)
    report.n_assigned = sum(g.n_samples for g in groups)
    report.assert_disjoint()
    return report


# =============================================================================
# 2. Registry: explicit no-edit + K singletons, max_active = 1, a0 = rho = 0.25
# =============================================================================

def build_bank_registry(
    num_experts: int = DESIGN_NUM_EXPERTS,
    *,
    name: str = "r2_dynamic_bank",
    rho: float = DESIGN_RHO,
    a0: float = DESIGN_A0,
    data_role: Optional[Any] = DataRole.BANK_FIT.value,
    artifact_refs: Optional[Dict[str, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> CandidateRegistry:
    """The round-4 candidate set: explicit no-edit plus K singleton experts.

    Built through `registry.build_pilot_registry`, which routes the singletons
    through `register_single_expert_plans` and asserts the REALIZED coefficient
    equals a0 -- so `single_expert_plans`' silent `min(coefficient, rho)` clip
    cannot leave the experiment running at a weaker magnitude than specified.

    Deliberately NOT `contracts.pilot_plans()`: that legacy table is the
    combinatorial g0/g1/g01/g23/all set at rho=1.0, which is not this round's
    candidate set.
    """
    payload = {
        "design": "round4_section_6.1",
        "num_experts": int(num_experts),
        "a0": float(a0),
        "rho": float(rho),
        "max_active": DESIGN_MAX_ACTIVE,
        "hold_steps": DESIGN_HOLD_STEPS,
        "candidates": "explicit no-edit + K singletons",
        "not_used": "contracts.pilot_plans() combinatorial table",
    }
    payload.update(metadata or {})
    registry = build_pilot_registry(
        int(num_experts),
        name=name,
        rho=float(rho),
        expected_coefficient=float(a0),
        data_role=data_role,
        artifact_refs=artifact_refs,
        metadata=payload,
    )
    assert_max_active(registry, max_active=DESIGN_MAX_ACTIVE)
    expected = int(num_experts) + 1
    if len(registry) != expected:
        raise BankTrainingViolation(
            "BANK_REGISTRY_SIZE_UNEXPECTED",
            f"registry holds {len(registry)} entries; expected {expected} "
            f"(1 no-edit reference + {num_experts} singletons).",
            {"n_entries": len(registry), "expected": expected},
        )
    return registry


def assert_max_active(registry: CandidateRegistry, *, max_active: int = DESIGN_MAX_ACTIVE) -> None:
    """Every non-reference candidate must activate at most `max_active` experts."""
    offending = [
        {"plan_id": entry.plan_id, "support": list(entry.support)}
        for entry in registry.candidate_entries()
        if len(entry.support) > int(max_active)
    ]
    if offending:
        raise BankTrainingViolation(
            "BANK_MAX_ACTIVE_EXCEEDED",
            f"{len(offending)} candidate(s) activate more than {max_active} "
            f"expert(s): {offending[:4]}. Section 6.1 fixes max_active="
            f"{max_active} for this round; a pair or combination candidate is "
            "outside the frozen action set.",
            {"max_active": int(max_active), "offending": offending[:16]},
        )


def expert_plan(
    registry: CandidateRegistry,
    expert_index: int,
    *,
    hold_steps: int = DESIGN_HOLD_STEPS,
) -> EditPlan:
    """The registered singleton plan that activates exactly `expert_index`.

    Read out of the registry rather than reconstructed, so the coefficients that
    train an expert are the same registered numbers that will later be run.
    """
    matches = [
        entry for entry in registry.candidate_entries()
        if entry.support == (int(expert_index),)
    ]
    if len(matches) != 1:
        raise BankTrainingViolation(
            "BANK_EXPERT_PLAN_NOT_UNIQUE",
            f"Expected exactly one registered singleton plan for expert "
            f"{expert_index}, found {len(matches)}: "
            f"{[m.plan_id for m in matches]}.",
            {"expert_index": int(expert_index),
             "matches": [m.plan_id for m in matches]},
        )
    entry = matches[0]
    plan = entry.to_edit_plan()
    if int(hold_steps) == plan.hold_steps:
        return plan
    return EditPlan(
        plan_id=plan.plan_id,
        num_experts=plan.num_experts,
        coefficients=plan.coefficients,
        hold_steps=int(hold_steps),
        interval_hours=plan.interval_hours,
        continuation=plan.continuation,
        rho=plan.rho,
    )


# =============================================================================
# 3. Pre-declared eligibility
# =============================================================================

#: The complete list of reasons an expert may be excluded. Section 6.2 permits
#: only PRE-DECLARED eligibility rules and forbids dropping an expert because it
#: did not pay off at evaluation time, so every entry here is a numerical
#: failure of the fit itself. Nothing about gain, rank or usefulness appears,
#: and `assert_eligibility_rule_is_pre_declared` refuses any reason not listed.
ELIGIBILITY_RULES: Tuple[str, ...] = (
    "NON_FINITE_LOSS",
    "NON_FINITE_GRADIENT",
    "NO_UPDATES_RUN",
)

ELIGIBILITY_RULE_DOC: Dict[str, str] = {
    "NON_FINITE_LOSS": (
        "an update produced a non-finite objective value, so the expert's "
        "parameters are not defined by the fit"
    ),
    "NON_FINITE_GRADIENT": (
        "an update produced a non-finite gradient norm, so the update applied "
        "to the expert is not defined"
    ),
    "NO_UPDATES_RUN": (
        "the fit performed zero updates, so the recorded expert was never "
        "trained by this run at all"
    ),
}

#: Reasons that are explicitly NOT grounds for exclusion, kept in the source so
#: the prohibition is visible next to the permission.
FORBIDDEN_EXCLUSION_REASONS: Tuple[str, ...] = (
    "LOW_EVALUATION_GAIN",
    "NEGATIVE_GAIN",
    "LOSES_TO_NO_EDIT",
    "UNDER_TRAINED",
    "ZERO_INITIALIZED",
)


def assert_eligibility_rule_is_pre_declared(reason: Optional[str]) -> Optional[str]:
    """Refuse an exclusion reason that was not declared in advance."""
    if reason is None:
        return None
    if reason in FORBIDDEN_EXCLUSION_REASONS:
        raise BankTrainingViolation(
            "BANK_EXCLUSION_REASON_FORBIDDEN",
            f"{reason!r} is not a legal reason to exclude an expert. Section "
            "6.2 permits only pre-declared eligibility rules "
            f"({list(ELIGIBILITY_RULES)}) and explicitly forbids dropping an "
            "expert for its evaluation-time gain. An under-trained or "
            "zero-initialized bank is reported as a resource finding, not a "
            "scientific failure.",
            {"reason": reason, "allowed": list(ELIGIBILITY_RULES),
             "forbidden": list(FORBIDDEN_EXCLUSION_REASONS)},
        )
    if reason not in ELIGIBILITY_RULES:
        raise BankTrainingViolation(
            "BANK_EXCLUSION_REASON_UNKNOWN",
            f"{reason!r} is not among the pre-declared eligibility rules "
            f"{list(ELIGIBILITY_RULES)}. New rules may not be introduced after "
            "the fact.",
            {"reason": reason, "allowed": list(ELIGIBILITY_RULES)},
        )
    return reason


def evaluate_eligibility(
    losses: Sequence[float], grad_norms: Sequence[float], n_updates: int
) -> Tuple[bool, Optional[str]]:
    """Apply `ELIGIBILITY_RULES` and nothing else."""
    if any(not math.isfinite(float(v)) for v in losses):
        return False, "NON_FINITE_LOSS"
    if any(not math.isfinite(float(v)) for v in grad_norms):
        return False, "NON_FINITE_GRADIENT"
    if int(n_updates) <= 0:
        return False, "NO_UPDATES_RUN"
    return True, None


# =============================================================================
# 4. Bank construction and per-expert training
# =============================================================================

def build_dynamic_bank(
    hidden_size: int,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    *,
    num_experts: int = DESIGN_NUM_EXPERTS,
    rank_per_expert: int = DESIGN_RANK_PER_EXPERT,
    scale: float = 1.0,
    seed: Optional[int] = None,
    device: Optional[torch.device | str] = None,
) -> Dict[int, ExpertLoRA]:
    """One K-expert `ExpertLoRA` per targeted block, at the standard LoRA init.

    The up (B) factors start at zero, which makes the whole bank a numerical
    no-op until it is trained. Section 6.2 calls that out explicitly -- "零 B
    初始状态是正常初始化，未训练零库不是科学失败" -- so nothing here treats a
    zero bank as an error.

    The factors are always DRAWN on CPU from a CPU generator, so the values
    (and `bank_digest`) do not depend on the device. ``device`` (FP-04,
    additive; default None keeps the historical CPU-resident result) then
    moves the finished bank and asserts it arrived there, so a caller that
    rolls the bank out against a CUDA backbone cannot forget the move.
    """
    blocks = tuple(int(b) for b in target_blocks)
    if not blocks:
        raise BankTrainingViolation(
            "BANK_NO_TARGET_BLOCKS",
            "target_blocks must not be empty.",
            {},
        )
    generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    bank: Dict[int, ExpertLoRA] = {}
    for block in blocks:
        lora = ExpertLoRA(
            hidden_size,
            hidden_size,
            num_experts=int(num_experts),
            rank_per_expert=int(rank_per_expert),
            scale=float(scale),
        )
        if generator is not None:
            with torch.no_grad():
                for down in lora.down:
                    down.weight.copy_(
                        torch.randn(down.weight.shape, generator=generator) * 0.02
                    )
        bank[block] = lora
    if device is not None:
        for lora in bank.values():
            lora.to(device)
        check_bank_device(bank, device, context="build_dynamic_bank")
    return bank


def bank_is_zero_initialized(bank: Mapping[int, ExpertLoRA], expert_index: int) -> bool:
    """True when this expert's up (B) factors are still exactly zero."""
    for lora in bank.values():
        weight = lora._get_up(int(expert_index)).weight
        if bool(torch.any(weight != 0)):
            return False
    return True


def _select_expert_parameters(
    bank: Mapping[int, ExpertLoRA], expert_index: int
) -> List[torch.nn.Parameter]:
    """Only expert `expert_index`'s factors train; every other expert is frozen.

    Section 5.1 runs the four experts as "4 卡独立进程各训一个专家". Freezing the
    other experts' parameters is what makes one process's optimizer step a
    no-op for the experts another process owns, so the four shards stay
    independent even though they share a bank object layout.
    """
    params: List[torch.nn.Parameter] = []
    for lora in bank.values():
        for k in range(lora.num_experts):
            trainable = k == int(expert_index)
            down = lora._get_down(k)
            up = lora._get_up(k)
            down.weight.requires_grad_(trainable)
            up.weight.requires_grad_(trainable)
            if trainable:
                params.extend([down.weight, up.weight])
        if lora.shared_down or lora.shared_up:
            raise BankTrainingViolation(
                "BANK_SHARED_FACTORS_UNSUPPORTED",
                "This round trains one expert per process, which requires "
                "independent A/B factors; a bank with shared_down/shared_up "
                "cannot isolate a single expert's gradient.",
                {"shared_down": lora.shared_down, "shared_up": lora.shared_up},
            )
    if not params:
        raise BankTrainingViolation(
            "BANK_EXPERT_HAS_NO_PARAMETERS",
            f"Expert {expert_index} contributed no trainable parameters; check "
            "the expert index against the bank's num_experts.",
            {"expert_index": int(expert_index)},
        )
    return params


def nonzero_response_check(
    bridge: WeatherStepBridge,
    bank: Mapping[int, ExpertLoRA],
    plan: EditPlan,
    x_norm: Tensor,
    variables: Sequence[str],
    *,
    expert_index: int,
    steps: int = DESIGN_HOLD_STEPS,
    interval_hours: int = DESIGN_INTERVAL_HOURS,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
) -> Dict[str, Any]:
    """Does this expert actually change the forecast at its registered a0?

    Reports the magnitude, and -- when the answer is no -- whether that is
    simply the untrained zero-init state. A zero response from a zero-init bank
    is recorded as such and is NOT an eligibility failure.
    """
    names = list(variables)
    blocks = tuple(int(b) for b in target_blocks)
    no_edit = EditPlan(
        plan_id="no_edit_reference",
        num_experts=plan.num_experts,
        coefficients=(0.0,) * plan.num_experts,
        hold_steps=plan.hold_steps,
        interval_hours=plan.interval_hours,
        continuation=plan.continuation,
        rho=plan.rho,
    )
    with torch.no_grad():
        baseline = controlled_rollout(
            bridge, x_norm, names, interval=int(interval_hours), steps=int(steps),
            plan=no_edit, expert_loras=dict(bank), target_blocks=blocks,
        )
        edited = controlled_rollout(
            bridge, x_norm, names, interval=int(interval_hours), steps=int(steps),
            plan=plan, expert_loras=dict(bank), target_blocks=blocks,
        )
    delta = (edited - baseline).abs()
    max_abs = float(delta.max())
    zero_init = bank_is_zero_initialized(bank, expert_index)
    return {
        "expert_index": int(expert_index),
        "plan_id": plan.plan_id,
        "coefficients": list(plan.coefficients),
        "steps": int(steps),
        "max_abs_response": max_abs,
        "mean_abs_response": float(delta.mean()),
        "nonzero": bool(max_abs > 0.0),
        "zero_initialized_untrained": bool(zero_init),
        "note": (
            "zero response with zero-initialized B factors is the normal LoRA "
            "initialization state, not a failure"
            if (max_abs == 0.0 and zero_init)
            else ""
        ),
    }


@dataclass(frozen=True)
class BankTrainConfig:
    """Pinned configuration for one expert's training run."""

    mode: str = "gradient_check"
    max_updates: int = 32
    learning_rate: float = 1e-2
    train_steps: int = DESIGN_HOLD_STEPS
    hold_steps: int = DESIGN_HOLD_STEPS
    lead_steps: Tuple[int, ...] = (DESIGN_HOLD_STEPS,)
    num_experts: int = DESIGN_NUM_EXPERTS
    rank_per_expert: int = DESIGN_RANK_PER_EXPERT
    target_blocks: Tuple[int, ...] = DEFAULT_TARGET_BLOCKS
    interval_hours: int = DESIGN_INTERVAL_HOURS
    a0: float = DESIGN_A0
    rho: float = DESIGN_RHO
    grad_clip: Optional[float] = 1.0
    seed: int = 20260921
    frozen_before_dev_results: bool = True

    def __post_init__(self):
        object.__setattr__(self, "lead_steps", tuple(int(s) for s in self.lead_steps))
        object.__setattr__(
            self, "target_blocks", tuple(int(b) for b in self.target_blocks)
        )
        if self.mode not in MODE_UPDATE_CAPS:
            raise BankTrainingViolation(
                "BANK_MODE_UNKNOWN",
                f"mode must be one of {sorted(MODE_UPDATE_CAPS)}, got {self.mode!r}.",
                {"mode": self.mode},
            )
        cap = MODE_UPDATE_CAPS[self.mode]
        if type(self.max_updates) is not int or self.max_updates <= 0:
            raise BankTrainingViolation(
                "BANK_MAX_UPDATES_INVALID",
                f"max_updates must be a positive integer, got {self.max_updates!r}.",
                {"max_updates": repr(self.max_updates)},
            )
        if self.max_updates > cap:
            raise BankTrainingViolation(
                "BANK_UPDATE_CAP_EXCEEDED",
                f"mode {self.mode!r} caps each expert at {cap} updates but "
                f"max_updates={self.max_updates} was requested. The caps come "
                "from the execution plan (<=32 for the gradient/throughput/"
                "direction check, <=500 for the formal fit) and may be adjusted "
                "only before development results are viewed.",
                {"mode": self.mode, "cap": cap, "max_updates": self.max_updates},
            )
        if max(self.lead_steps) > self.train_steps:
            raise BankTrainingViolation(
                "BANK_LEAD_BEYOND_TRAIN_HORIZON",
                f"lead_steps {self.lead_steps} need rollout step "
                f"{max(self.lead_steps)} but train_steps={self.train_steps}.",
                {"lead_steps": list(self.lead_steps),
                 "train_steps": int(self.train_steps)},
            )
        if abs(float(self.a0) - float(self.rho)) > 1e-12:
            # Not fatal in general, but this round fixes a0 == rho == 0.25 and a
            # silent divergence between them is exactly the bug the registry's
            # realized-coefficient assertion exists to catch.
            raise BankTrainingViolation(
                "BANK_A0_RHO_MISMATCH",
                f"This round fixes a0 == rho == {DESIGN_A0}; got a0={self.a0}, "
                f"rho={self.rho}.",
                {"a0": float(self.a0), "rho": float(self.rho)},
            )

    @classmethod
    def for_mode(cls, mode: str, **overrides: Any) -> "BankTrainConfig":
        if mode not in MODE_UPDATE_CAPS:
            raise BankTrainingViolation(
                "BANK_MODE_UNKNOWN",
                f"mode must be one of {sorted(MODE_UPDATE_CAPS)}, got {mode!r}.",
                {"mode": mode},
            )
        kwargs: Dict[str, Any] = {"mode": mode, "max_updates": MODE_UPDATE_CAPS[mode]}
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def train_hours(self) -> int:
        return int(self.train_steps) * int(self.interval_hours)

    @property
    def hold_hours(self) -> int:
        return int(self.hold_steps) * int(self.interval_hours)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "max_updates": int(self.max_updates),
            "update_cap_for_mode": MODE_UPDATE_CAPS[self.mode],
            "learning_rate": float(self.learning_rate),
            "train_steps": int(self.train_steps),
            "train_hours": self.train_hours,
            "hold_steps": int(self.hold_steps),
            "hold_hours": self.hold_hours,
            "lead_steps": list(self.lead_steps),
            "num_experts": int(self.num_experts),
            "rank_per_expert": int(self.rank_per_expert),
            "target_blocks": list(self.target_blocks),
            "interval_hours": int(self.interval_hours),
            "a0": float(self.a0),
            "rho": float(self.rho),
            "max_active": DESIGN_MAX_ACTIVE,
            "grad_clip": self.grad_clip,
            "seed": int(self.seed),
            "frozen_before_dev_results": bool(self.frozen_before_dev_results),
        }


@dataclass
class ExpertTrainingRecord:
    """Per-expert record: data, updates, loss, non-zero response, diversity."""

    expert_index: int
    plan_id: str = ""
    coefficients: List[float] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    diversity_group: Dict[str, Any] = field(default_factory=dict)
    diversity_rule: str = DIVERSITY_RULE_ID
    n_updates: int = 0
    losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    min_loss: Optional[float] = None
    training_issue_ids: List[str] = field(default_factory=list)
    training_times_utc: List[str] = field(default_factory=list)
    training_sources: List[str] = field(default_factory=list)
    data_role: Optional[str] = DataRole.BANK_FIT.value
    n_samples: int = 0
    trainable_parameters: int = 0
    nonzero_response: Dict[str, Any] = field(default_factory=dict)
    zero_initialized_untrained: bool = False
    eligible: bool = True
    ineligible_reason: Optional[str] = None
    eligibility_rules: Tuple[str, ...] = ELIGIBILITY_RULES
    forbidden_exclusion_reasons: Tuple[str, ...] = FORBIDDEN_EXCLUSION_REASONS
    fs_backbone_digest: str = ""
    fs_static_adapter_digest: str = ""
    objective: Dict[str, Any] = field(default_factory=dict)
    wallclock_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    #: FP-04 observation fields. Pure read-outs taken around the unchanged
    #: update (Adam + clip at `grad_clip`); nothing here feeds back into it.
    #: `grad_norms_A` / `grad_norms_B` are the PRE-clip float64 L2 norms of
    #: the trained expert's down (A) and up (B) factors' real `.grad`. B
    #: starts at zero, so A's gradient is exactly zero at update 0.
    grad_norms_A: List[float] = field(default_factory=list)
    grad_norms_B: List[float] = field(default_factory=list)
    #: Updates whose pre-clip total norm exceeded `grad_clip` (clip engaged).
    clip_events: int = 0
    clip_event_updates: List[int] = field(default_factory=list)
    #: `expert_digest` of the trained expert before the first / after the
    #: last update, and `other_experts_digest` of every OTHER expert before /
    #: after -- the isolation evidence that this process moved only its own.
    initial_expert_digest: Optional[str] = None
    final_expert_digest: Optional[str] = None
    other_experts_digest_before: Optional[str] = None
    other_experts_digest_after: Optional[str] = None
    other_experts_unchanged: Optional[bool] = None
    other_experts_grad_free: Optional[bool] = None
    optimizer: Dict[str, Any] = field(default_factory=dict)

    @property
    def loss_decreased(self) -> bool:
        """Raw first-vs-last update loss; see `loss_direction` for the real test."""
        if self.initial_loss is None or self.final_loss is None:
            return False
        return self.final_loss < self.initial_loss

    @property
    def loss_direction(self) -> Dict[str, Any]:
        """Per-sample first-vs-last loss -- the like-for-like direction check."""
        return per_sample_loss_direction(self.losses, self.training_issue_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_index": int(self.expert_index),
            "plan_id": self.plan_id,
            "coefficients": [float(a) for a in self.coefficients],
            "config": dict(self.config),
            "diversity_rule": self.diversity_rule,
            "diversity_group": dict(self.diversity_group),
            "n_updates": int(self.n_updates),
            "losses": [float(v) for v in self.losses],
            "grad_norms": [float(v) for v in self.grad_norms],
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "min_loss": self.min_loss,
            "loss_decreased": self.loss_decreased,
            "loss_direction": self.loss_direction,
            "training_issue_ids": list(self.training_issue_ids),
            "training_times_utc": list(self.training_times_utc),
            "training_sources": sorted(set(self.training_sources)),
            "data_role": self.data_role,
            "n_samples": int(self.n_samples),
            "trainable_parameters": int(self.trainable_parameters),
            "nonzero_response": dict(self.nonzero_response),
            "zero_initialized_untrained": bool(self.zero_initialized_untrained),
            "eligible": bool(self.eligible),
            "ineligible_reason": self.ineligible_reason,
            "eligibility_rules": list(self.eligibility_rules),
            "forbidden_exclusion_reasons": list(self.forbidden_exclusion_reasons),
            "fs_backbone_digest": self.fs_backbone_digest,
            "fs_static_adapter_digest": self.fs_static_adapter_digest,
            "objective": dict(self.objective),
            "wallclock_seconds": float(self.wallclock_seconds),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "grad_norms_A": [float(v) for v in self.grad_norms_A],
            "grad_norms_B": [float(v) for v in self.grad_norms_B],
            "clip_events": int(self.clip_events),
            "clip_event_updates": [int(v) for v in self.clip_event_updates],
            "initial_expert_digest": self.initial_expert_digest,
            "final_expert_digest": self.final_expert_digest,
            "other_experts_digest_before": self.other_experts_digest_before,
            "other_experts_digest_after": self.other_experts_digest_after,
            "other_experts_unchanged": self.other_experts_unchanged,
            "other_experts_grad_free": self.other_experts_grad_free,
            "optimizer": dict(self.optimizer),
        }


def _grad_norm_f64(params: Sequence[Tensor]) -> float:
    """float64 L2 norm of the current `.grad` of `params` (0.0 if none has one).

    Read-only: `.double()` copies, so the gradient the optimizer consumes is
    never touched.
    """
    with torch.no_grad():
        squares = [p.grad.double().pow(2).sum() for p in params if p.grad is not None]
        if not squares:
            return 0.0
        return float(torch.stack(squares).sum().sqrt())


def train_expert(
    fs_bridge: WeatherStepBridge,
    bank: Mapping[int, ExpertLoRA],
    expert_index: int,
    samples: Sequence[TrainingSample],
    spec: ObjectiveSpec,
    config: BankTrainConfig,
    *,
    registry: Optional[CandidateRegistry] = None,
    diversity_group: Optional[DiversityGroup] = None,
    variables: Optional[Sequence[str]] = None,
    fs_backbone_digest: str = "",
    fs_static_adapter_digest: str = "",
    progress: Optional[Any] = None,
) -> ExpertTrainingRecord:
    """Train ONE dynamic expert on top of the frozen Fs-merged backbone.

    Coefficients are FIXED at the registered a0 (`differentiable=False`), which
    is what section 6.2 asks for while B03 (caller-coefficient gradients) stays
    deferred: gradients reach the expert's A and B factors, and nothing else.
    The Fs-merged backbone is frozen, so it receives no gradient at all.
    """
    assert_bank_fit_samples(list(samples))
    names = list(variables) if variables is not None else list(fs_bridge.variables)
    reg = registry if registry is not None else build_bank_registry(
        config.num_experts, rho=config.rho, a0=config.a0
    )
    plan = expert_plan(reg, expert_index, hold_steps=config.hold_steps)

    torch.manual_seed(int(config.seed) + int(expert_index))
    device = next(fs_bridge.model.parameters()).device
    for lora in bank.values():
        lora.to(device)

    # The Fs-merged backbone must be frozen before anything trains on it.
    trainable_backbone = [
        name for name, param in fs_bridge.model.named_parameters() if param.requires_grad
    ]
    if trainable_backbone:
        raise BankTrainingViolation(
            "BANK_BACKBONE_NOT_FROZEN",
            f"{len(trainable_backbone)} Fs-merged backbone parameter(s) still "
            f"require grad (first: {trainable_backbone[:4]}). Fs is frozen once "
            "merged; training the bank must not move it.",
            {"n_trainable": len(trainable_backbone),
             "first": trainable_backbone[:8]},
        )

    params = _select_expert_parameters(bank, expert_index)
    optimizer = torch.optim.Adam(params, lr=float(config.learning_rate))
    # Observation-only handles on the trained expert's A (down) and B (up)
    # factors. `params` itself -- and therefore the optimizer and the clip --
    # is exactly what it was before FP-04.
    index = int(expert_index)
    params_a = [bank[b]._get_down(index).weight for b in sorted(bank)]
    params_b = [bank[b]._get_up(index).weight for b in sorted(bank)]
    other_params = [
        p for b in sorted(bank) for k in range(bank[b].num_experts) if k != index
        for p in (bank[b]._get_down(k).weight, bank[b]._get_up(k).weight)
    ]
    # A stale `.grad` left on another expert by an EARLIER expert trained in
    # this same process (synthetic `--stage all`) is not a weight and is never
    # read; clear it so `other_experts_grad_free` measures only THIS run. The
    # other experts are frozen above, so a grad appearing on them now is a leak.
    for p in other_params:
        p.grad = None

    record = ExpertTrainingRecord(
        expert_index=int(expert_index),
        plan_id=plan.plan_id,
        coefficients=[float(a) for a in plan.coefficients],
        config=config.to_dict(),
        diversity_group=diversity_group.to_dict() if diversity_group else {},
        n_samples=len(samples),
        trainable_parameters=sum(p.numel() for p in params),
        objective=spec.to_dict(),
        fs_backbone_digest=fs_backbone_digest,
        fs_static_adapter_digest=fs_static_adapter_digest,
        started_at=_utc_now(),
    )
    record.zero_initialized_untrained = bank_is_zero_initialized(bank, expert_index)
    record.initial_expert_digest = expert_digest(bank, index)
    record.other_experts_digest_before = other_experts_digest(bank, index)
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

    started = time.perf_counter()
    for update in range(int(config.max_updates)):
        sample = samples[update % len(samples)].to(device)
        optimizer.zero_grad(set_to_none=True)

        trajectory = controlled_rollout(
            fs_bridge,
            sample.x_norm,
            names,
            interval=int(config.interval_hours),
            steps=int(config.train_steps),
            plan=plan,
            expert_loras=dict(bank),
            target_blocks=config.target_blocks,
            return_trajectory=True,
        )
        loss = objective_loss_for_sample(fs_bridge, trajectory, sample, spec)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            record.losses.append(loss_value)
            break

        loss.backward()
        # Pre-clip per-factor norms: read-only on the real `.grad` tensors.
        grad_a = _grad_norm_f64(params_a)
        grad_b = _grad_norm_f64(params_b)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                params,
                float(config.grad_clip) if config.grad_clip is not None else float("inf"),
            )
        )
        if not math.isfinite(grad_norm):
            record.losses.append(loss_value)
            record.grad_norms.append(grad_norm)
            break

        optimizer.step()
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
        record.training_sources.append(sample.source)
        if progress is not None:
            progress(update, loss_value, grad_norm)

    record.wallclock_seconds = time.perf_counter() - started
    record.finished_at = _utc_now()
    record.final_expert_digest = expert_digest(bank, index)
    record.other_experts_digest_after = other_experts_digest(bank, index)
    record.other_experts_unchanged = (
        record.other_experts_digest_before == record.other_experts_digest_after
    )
    record.other_experts_grad_free = all(p.grad is None for p in other_params)
    if record.losses:
        record.initial_loss = float(record.losses[0])
        record.final_loss = float(record.losses[-1])
        finite = [v for v in record.losses if math.isfinite(v)]
        record.min_loss = float(min(finite)) if finite else None

    eligible, reason = evaluate_eligibility(
        record.losses, record.grad_norms, record.n_updates
    )
    record.eligible = eligible
    record.ineligible_reason = assert_eligibility_rule_is_pre_declared(reason)

    if samples:
        record.nonzero_response = nonzero_response_check(
            fs_bridge,
            bank,
            plan,
            samples[0].to(device).x_norm,
            names,
            expert_index=expert_index,
            steps=int(config.hold_steps),
            interval_hours=int(config.interval_hours),
            target_blocks=config.target_blocks,
        )
        record.zero_initialized_untrained = bool(
            record.nonzero_response.get("zero_initialized_untrained", False)
        )
    return record


# =============================================================================
# 5. Single-training-step profile and the K/rank capacity decision
# =============================================================================

def _peak_memory_reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_read(device: torch.device) -> Tuple[int, str]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device)), "cuda_max_memory_allocated"
    try:
        import resource

        # ru_maxrss is a process high-water mark in kibibytes on Linux. It is
        # NOT resettable, so it is reported as what it is and is explicitly not
        # treated as equivalent to a CUDA peak-allocation measurement.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "host_rss_high_water"
    except Exception:  # noqa: BLE001 - measurement, not correctness
        return -1, "unavailable"


@dataclass(frozen=True)
class TrainingStepProfile:
    """One real forward+backward+step, measured. The basis for the K decision."""

    device: str
    device_name: str
    num_experts: int
    rank_per_expert: int
    target_blocks: Tuple[int, ...]
    train_steps: int
    hold_steps: int
    batch_size: int
    grid: Tuple[int, int]
    n_variables: int
    hidden_size: int
    depth: int
    n_measured_steps: int
    forward_seconds: float
    backward_seconds: float
    total_seconds: float
    seconds_per_update: float
    peak_memory_bytes: int
    memory_source: str
    trainable_parameters: int
    synthetic: bool
    measured_at: str
    error: Optional[str] = None

    @property
    def is_real_gpu_profile(self) -> bool:
        return self.device.startswith("cuda") and not self.synthetic and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "device_name": self.device_name,
            "num_experts": int(self.num_experts),
            "rank_per_expert": int(self.rank_per_expert),
            "target_blocks": list(self.target_blocks),
            "train_steps": int(self.train_steps),
            "hold_steps": int(self.hold_steps),
            "batch_size": int(self.batch_size),
            "grid": list(self.grid),
            "n_variables": int(self.n_variables),
            "hidden_size": int(self.hidden_size),
            "depth": int(self.depth),
            "n_measured_steps": int(self.n_measured_steps),
            "forward_seconds": float(self.forward_seconds),
            "backward_seconds": float(self.backward_seconds),
            "total_seconds": float(self.total_seconds),
            "seconds_per_update": float(self.seconds_per_update),
            "peak_memory_bytes": int(self.peak_memory_bytes),
            "peak_memory_gib": (
                float(self.peak_memory_bytes) / (1024 ** 3)
                if self.peak_memory_bytes >= 0 else None
            ),
            "memory_source": self.memory_source,
            "trainable_parameters": int(self.trainable_parameters),
            "synthetic": bool(self.synthetic),
            "is_real_gpu_profile": self.is_real_gpu_profile,
            "measured_at": self.measured_at,
            "error": self.error,
        }


def profile_training_step(
    bridge: WeatherStepBridge,
    bank: Mapping[int, ExpertLoRA],
    sample: TrainingSample,
    spec: ObjectiveSpec,
    *,
    expert_index: int = 0,
    plan: Optional[EditPlan] = None,
    config: Optional[BankTrainConfig] = None,
    variables: Optional[Sequence[str]] = None,
    n_measured_steps: int = 1,
    warmup_steps: int = 0,
    synthetic: bool = False,
) -> TrainingStepProfile:
    """Measure one real training step on the declared target blocks.

    Section 5.2: "单次训练步 profile 单列为 bank 准备测量，记录显存/速度与更新内容."
    This runs the SAME code path training uses -- `controlled_rollout` with the
    registered singleton plan, the native objective, a backward pass and an
    optimizer step -- and times/measures it. It makes no decision itself;
    `decide_bank_capacity` does, from these numbers.
    """
    cfg = config if config is not None else BankTrainConfig()
    names = list(variables) if variables is not None else list(bridge.variables)
    device = next(bridge.model.parameters()).device
    any_lora = next(iter(bank.values()))
    active_plan = plan if plan is not None else expert_plan(
        build_bank_registry(any_lora.num_experts, rho=cfg.rho, a0=cfg.a0),
        expert_index,
        hold_steps=cfg.hold_steps,
    )

    for lora in bank.values():
        lora.to(device)
    params = _select_expert_parameters(bank, expert_index)
    optimizer = torch.optim.Adam(params, lr=float(cfg.learning_rate))
    moved = sample.to(device)

    forward_total = 0.0
    backward_total = 0.0
    error: Optional[str] = None

    def one_step() -> None:
        nonlocal forward_total, backward_total
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        trajectory = controlled_rollout(
            bridge, moved.x_norm, names, interval=int(cfg.interval_hours),
            steps=int(cfg.train_steps), plan=active_plan, expert_loras=dict(bank),
            target_blocks=cfg.target_blocks, return_trajectory=True,
        )
        loss = objective_loss_for_sample(bridge, trajectory, moved, spec)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        forward_total += t1 - t0
        backward_total += t2 - t1

    try:
        for _ in range(max(0, int(warmup_steps))):
            one_step()
        forward_total = 0.0
        backward_total = 0.0
        _peak_memory_reset(device)
        for _ in range(max(1, int(n_measured_steps))):
            one_step()
    except RuntimeError as exc:  # OOM or any runtime failure IS the measurement
        error = f"{type(exc).__name__}: {exc}"

    peak_bytes, memory_source = _peak_memory_read(device)
    measured = max(1, int(n_measured_steps))
    total = forward_total + backward_total
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    )
    return TrainingStepProfile(
        device=str(device),
        device_name=device_name,
        num_experts=int(any_lora.num_experts),
        rank_per_expert=int(any_lora.rank_per_expert),
        target_blocks=tuple(int(b) for b in cfg.target_blocks),
        train_steps=int(cfg.train_steps),
        hold_steps=int(cfg.hold_steps),
        batch_size=int(moved.x_norm.shape[0]),
        grid=(int(moved.x_norm.shape[-2]), int(moved.x_norm.shape[-1])),
        n_variables=int(moved.x_norm.shape[1]),
        hidden_size=int(any_lora.in_features),
        depth=len(bridge.model.blocks),
        n_measured_steps=measured,
        forward_seconds=forward_total,
        backward_seconds=backward_total,
        total_seconds=total,
        seconds_per_update=total / measured if total > 0 else 0.0,
        peak_memory_bytes=peak_bytes,
        memory_source=memory_source,
        trainable_parameters=sum(p.numel() for p in params),
        synthetic=bool(synthetic),
        measured_at=_utc_now(),
        error=error,
    )


@dataclass(frozen=True)
class CapacityBudget:
    """The resource envelope the profile is compared against.

    Declared before the profile is run so the comparison is not retrofitted to
    whatever was measured.
    """

    memory_budget_bytes: int
    memory_safety_fraction: float = 0.8
    max_seconds_per_update: float = 30.0

    def __post_init__(self):
        if int(self.memory_budget_bytes) <= 0:
            raise BankTrainingViolation(
                "CAPACITY_MEMORY_BUDGET_INVALID",
                f"memory_budget_bytes must be positive, got "
                f"{self.memory_budget_bytes!r}.",
                {"memory_budget_bytes": repr(self.memory_budget_bytes)},
            )
        if not (0 < float(self.memory_safety_fraction) <= 1):
            raise BankTrainingViolation(
                "CAPACITY_SAFETY_FRACTION_INVALID",
                f"memory_safety_fraction must be in (0, 1], got "
                f"{self.memory_safety_fraction!r}.",
                {"memory_safety_fraction": repr(self.memory_safety_fraction)},
            )
        if not math.isfinite(float(self.max_seconds_per_update)) or self.max_seconds_per_update <= 0:
            raise BankTrainingViolation(
                "CAPACITY_TIME_BUDGET_INVALID",
                f"max_seconds_per_update must be positive and finite, got "
                f"{self.max_seconds_per_update!r}.",
                {"max_seconds_per_update": repr(self.max_seconds_per_update)},
            )

    @property
    def usable_memory_bytes(self) -> float:
        return float(self.memory_budget_bytes) * float(self.memory_safety_fraction)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_budget_bytes": int(self.memory_budget_bytes),
            "memory_budget_gib": float(self.memory_budget_bytes) / (1024 ** 3),
            "memory_safety_fraction": float(self.memory_safety_fraction),
            "usable_memory_bytes": self.usable_memory_bytes,
            "max_seconds_per_update": float(self.max_seconds_per_update),
        }


#: The decision rule, fixed in advance. Stated as a string so it can be written
#: into the run record verbatim alongside the numbers it was applied to.
CAPACITY_DECISION_RULE = (
    "Keep the section-6.1 design (K=4, rank=4) unless the measured single "
    "training step on the declared target blocks exceeds the declared budget: "
    "peak memory above memory_budget_bytes * memory_safety_fraction, wallclock "
    "above max_seconds_per_update, or a runtime failure (e.g. OOM) during the "
    "step. Any of those triggers the single pre-declared shrink to K=2 / "
    "rank=2. A decision derived from a profile that is not a real CUDA profile "
    "is PROVISIONAL and may not freeze the design."
)


@dataclass(frozen=True)
class BankCapacityDecision:
    """K=4/rank=4 vs the pre-declared K=2/rank=2 shrink, decided from a profile."""

    decided_num_experts: int
    decided_rank_per_expert: int
    designed_num_experts: int
    designed_rank_per_expert: int
    fallback_num_experts: int
    fallback_rank_per_expert: int
    shrink_required: bool
    reasons: Tuple[str, ...]
    rule: str
    basis: Dict[str, Any]
    budget: Dict[str, Any]
    provisional: bool
    provisional_reason: Optional[str]
    decided_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decided_num_experts": int(self.decided_num_experts),
            "decided_rank_per_expert": int(self.decided_rank_per_expert),
            "designed_num_experts": int(self.designed_num_experts),
            "designed_rank_per_expert": int(self.designed_rank_per_expert),
            "fallback_num_experts": int(self.fallback_num_experts),
            "fallback_rank_per_expert": int(self.fallback_rank_per_expert),
            "shrink_required": bool(self.shrink_required),
            "reasons": list(self.reasons),
            "rule": self.rule,
            "basis": dict(self.basis),
            "budget": dict(self.budget),
            "provisional": bool(self.provisional),
            "provisional_reason": self.provisional_reason,
            "decided_at": self.decided_at,
        }


def decide_bank_capacity(
    profile: TrainingStepProfile,
    budget: CapacityBudget,
    *,
    designed_num_experts: int = DESIGN_NUM_EXPERTS,
    designed_rank_per_expert: int = DESIGN_RANK_PER_EXPERT,
    fallback_num_experts: int = FALLBACK_NUM_EXPERTS,
    fallback_rank_per_expert: int = FALLBACK_RANK_PER_EXPERT,
) -> BankCapacityDecision:
    """Turn a measured profile into the K/rank decision. Measurements only.

    The decision is `provisional` unless `profile.is_real_gpu_profile` -- a CPU
    or synthetic profile cannot settle whether blocks 18-23 of the real ps4
    backbone fit on an H100, and a decision drawn from one must not be reported
    as the capacity finding.
    """
    reasons: List[str] = []
    if profile.error is not None:
        reasons.append(
            f"the measured training step failed with {profile.error}, which is "
            "itself evidence the configuration does not run here"
        )
    if profile.peak_memory_bytes >= 0 and profile.peak_memory_bytes > budget.usable_memory_bytes:
        reasons.append(
            f"peak memory {profile.peak_memory_bytes / (1024 ** 3):.3f} GiB "
            f"({profile.memory_source}) exceeds the usable budget "
            f"{budget.usable_memory_bytes / (1024 ** 3):.3f} GiB"
        )
    if profile.seconds_per_update > budget.max_seconds_per_update:
        reasons.append(
            f"{profile.seconds_per_update:.3f} s/update exceeds the declared "
            f"{budget.max_seconds_per_update:.3f} s/update"
        )

    shrink = bool(reasons)
    provisional = not profile.is_real_gpu_profile
    provisional_reason = None
    if provisional:
        provisional_reason = (
            f"profile ran on device {profile.device!r} with synthetic="
            f"{profile.synthetic}. Only a real CUDA profile of the production "
            "backbone on the declared target blocks can settle the K/rank "
            "capacity question; this decision records the machinery working, "
            "not the answer."
        )

    return BankCapacityDecision(
        decided_num_experts=fallback_num_experts if shrink else designed_num_experts,
        decided_rank_per_expert=(
            fallback_rank_per_expert if shrink else designed_rank_per_expert
        ),
        designed_num_experts=int(designed_num_experts),
        designed_rank_per_expert=int(designed_rank_per_expert),
        fallback_num_experts=int(fallback_num_experts),
        fallback_rank_per_expert=int(fallback_rank_per_expert),
        shrink_required=shrink,
        reasons=tuple(reasons) if reasons else (
            "measured peak memory and throughput are both inside the declared "
            "budget, so the designed K=4 / rank=4 stands",
        ),
        rule=CAPACITY_DECISION_RULE,
        basis=profile.to_dict(),
        budget=budget.to_dict(),
        provisional=provisional,
        provisional_reason=provisional_reason,
        decided_at=_utc_now(),
    )


# =============================================================================
# 6. Horizon feasibility and the written fallback spec
# =============================================================================

def assert_activation_checkpointing_disabled(model: nn.Module) -> List[str]:
    """Assert no module advertises activation checkpointing. Never enables it.

    Section 6.2 forbids quietly switching activation checkpointing on to make a
    longer differentiable horizon fit. `controlled_rollout` already REJECTS a
    checkpointed backbone (B12); this is the complementary assertion on the
    caller side, so a horizon report can state positively that the measurement
    ran without it.
    """
    enabled: List[str] = []
    for module_name, module in model.named_modules():
        for flag in _ACTIVATION_CHECKPOINT_FLAGS:
            if getattr(module, flag, False):
                enabled.append(f"{module_name or '<backbone>'}.{flag}")
    if enabled:
        raise BankTrainingViolation(
            "HORIZON_ACTIVATION_CHECKPOINTING_ENABLED",
            f"Activation checkpointing is enabled on {enabled[:4]}. The horizon "
            "measurement must run in the same serial / no-replay mode the "
            "experiment declares; enabling checkpointing to fit a longer "
            "differentiable horizon and then reporting it under the same "
            "experiment identity is exactly what section 6.2 forbids.",
            {"enabled": enabled[:16],
             "supported_modes": dict(CONTROLLED_ROLLOUT_SUPPORTED_MODES)},
        )
    return enabled


@dataclass(frozen=True)
class HorizonFallbackSpec:
    """The reduced-horizon specification, with its cost and a NEW identity."""

    reason: str
    train_interval_hours: int
    train_steps: int
    train_hours: int
    designed_train_steps: int
    designed_train_hours: int
    eval_steps: Tuple[int, ...]
    eval_hours: Tuple[int, ...]
    eval_differentiable: bool
    experiment_identity: str
    superseded_identity: str
    cost_estimate: Dict[str, Any]
    activation_checkpointing_used: bool
    note: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "train_interval_hours": int(self.train_interval_hours),
            "train_steps": int(self.train_steps),
            "train_hours": int(self.train_hours),
            "designed_train_steps": int(self.designed_train_steps),
            "designed_train_hours": int(self.designed_train_hours),
            "eval_steps": list(self.eval_steps),
            "eval_hours": list(self.eval_hours),
            "eval_differentiable": bool(self.eval_differentiable),
            "experiment_identity": self.experiment_identity,
            "superseded_identity": self.superseded_identity,
            "cost_estimate": dict(self.cost_estimate),
            "activation_checkpointing_used": bool(self.activation_checkpointing_used),
            "note": self.note,
            "created_at": self.created_at,
        }


def _fallback_cost_estimate(
    *,
    measurements: Sequence[Dict[str, Any]],
    train_steps: int,
    eval_steps: Sequence[int],
    updates_per_expert: int,
    num_experts: int,
    n_eval_issues: int,
    n_candidates: int,
) -> Dict[str, Any]:
    """Cost of the reduced-horizon plan, with measured vs assumed marked.

    Every number says where it came from. A measured per-step cost is scaled;
    an unmeasured one is reported as null rather than invented, because a
    fabricated hour count is worse than a missing one.
    """
    by_steps = {int(m["steps"]): m for m in measurements if m.get("ok")}
    train_seconds_per_update = None
    if int(train_steps) in by_steps:
        train_seconds_per_update = float(by_steps[int(train_steps)]["total_seconds"])

    # Per-step forward cost, from the cheapest successful differentiable
    # measurement; a non-differentiable evaluation forward is strictly cheaper,
    # so this is an UPPER bound and is labelled as such.
    per_step_forward_upper_bound = None
    if by_steps:
        cheapest = by_steps[min(by_steps)]
        per_step_forward_upper_bound = (
            float(cheapest["forward_seconds"]) / max(1, int(cheapest["steps"]))
        )

    training_seconds = (
        train_seconds_per_update * int(updates_per_expert) * int(num_experts)
        if train_seconds_per_update is not None else None
    )
    eval_seconds = None
    if per_step_forward_upper_bound is not None:
        total_eval_steps = sum(int(s) for s in eval_steps)
        eval_seconds = (
            per_step_forward_upper_bound
            * total_eval_steps
            * int(n_eval_issues)
            * int(n_candidates)
        )

    return {
        "measured": {
            "differentiable_step_measurements": list(measurements),
            "train_seconds_per_update": train_seconds_per_update,
            "per_step_forward_seconds_upper_bound": per_step_forward_upper_bound,
        },
        "assumed": {
            "updates_per_expert": int(updates_per_expert),
            "num_experts": int(num_experts),
            "n_eval_issues": int(n_eval_issues),
            "n_candidates": int(n_candidates),
            "eval_is_non_differentiable": True,
            "note": (
                "update/issue/candidate counts are the plan's declared caps, "
                "not measurements"
            ),
        },
        "derived": {
            "training_seconds_all_experts": training_seconds,
            "training_gpu_hours_all_experts": (
                training_seconds / 3600.0 if training_seconds is not None else None
            ),
            "evaluation_seconds": eval_seconds,
            "evaluation_gpu_hours": (
                eval_seconds / 3600.0 if eval_seconds is not None else None
            ),
            "total_gpu_hours": (
                (training_seconds + eval_seconds) / 3600.0
                if training_seconds is not None and eval_seconds is not None
                else None
            ),
            "caveat": (
                "derived hours scale a measurement taken on THIS device and "
                "THIS model size; they are not a hardware measurement of the "
                "production run and must be re-measured there"
            ),
        },
    }


@dataclass
class HorizonFeasibilityReport:
    """Can the 4 x 6h hold window be backpropagated through, as configured?"""

    requested_hold_steps: int
    requested_hours: int
    interval_hours: int
    feasible: bool = False
    max_feasible_differentiable_steps: int = 0
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    activation_checkpointing_enabled: bool = False
    activation_checkpoint_flags_seen: List[str] = field(default_factory=list)
    supported_modes: Dict[str, bool] = field(
        default_factory=lambda: dict(CONTROLLED_ROLLOUT_SUPPORTED_MODES)
    )
    budget: Dict[str, Any] = field(default_factory=dict)
    fallback: Optional[HorizonFallbackSpec] = None
    checked_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_hold_steps": int(self.requested_hold_steps),
            "requested_hours": int(self.requested_hours),
            "interval_hours": int(self.interval_hours),
            "feasible": bool(self.feasible),
            "max_feasible_differentiable_steps": int(
                self.max_feasible_differentiable_steps
            ),
            "max_feasible_differentiable_hours": int(
                self.max_feasible_differentiable_steps * self.interval_hours
            ),
            "measurements": list(self.measurements),
            "activation_checkpointing_enabled": bool(
                self.activation_checkpointing_enabled
            ),
            "activation_checkpoint_flags_seen": list(
                self.activation_checkpoint_flags_seen
            ),
            "supported_modes": dict(self.supported_modes),
            "budget": dict(self.budget),
            "fallback": self.fallback.to_dict() if self.fallback else None,
            "checked_at": self.checked_at,
        }


DESIGNED_EXPERIMENT_IDENTITY = "ed-bank-train/1:hold-4x6h-differentiable-24h"
FALLBACK_EXPERIMENT_IDENTITY = "ed-bank-train/1:train-6h-eval-24h-72h-nondiff"


def check_horizon_feasibility(
    bridge: WeatherStepBridge,
    bank: Mapping[int, ExpertLoRA],
    sample: TrainingSample,
    spec: ObjectiveSpec,
    *,
    expert_index: int = 0,
    plan_hold_steps: int = DESIGN_HOLD_STEPS,
    interval_hours: int = DESIGN_INTERVAL_HOURS,
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    budget: Optional[CapacityBudget] = None,
    max_differentiable_steps: Optional[int] = None,
    variables: Optional[Sequence[str]] = None,
    registry: Optional[CandidateRegistry] = None,
    updates_per_expert: int = MODE_UPDATE_CAPS["formal"],
    num_experts_for_cost: int = DESIGN_NUM_EXPERTS,
    n_eval_issues: int = 128,
    n_candidates: int = DESIGN_NUM_EXPERTS + 1,
    fallback_train_steps: int = 1,
    fallback_eval_steps: Sequence[int] = (4, 12),
) -> HorizonFeasibilityReport:
    """Walk the differentiable rollout out to the full hold window, and measure.

    For each horizon 1..`plan_hold_steps` this runs a real differentiable
    rollout and backward pass through `controlled_rollout`, recording time and
    peak memory. A horizon is infeasible when the step raises (OOM being the
    case that matters), when it exceeds the declared budget, or when
    `max_differentiable_steps` forbids it -- that last argument exists so the
    machinery can be exercised without provoking a real OOM.

    When the full window is not reachable, a `HorizonFallbackSpec` is produced:
    train at the reduced horizon (6h by default), evaluate 24h/72h separately
    with NON-differentiable rollouts, cost it from the measurements taken here,
    and carry a NEW experiment identity. Activation checkpointing is never
    enabled to make a longer horizon fit; the report asserts it stayed off.
    """
    names = list(variables) if variables is not None else list(bridge.variables)
    blocks = tuple(int(b) for b in target_blocks)
    device = next(bridge.model.parameters()).device
    any_lora = next(iter(bank.values()))
    reg = registry if registry is not None else build_bank_registry(
        any_lora.num_experts, rho=DESIGN_RHO, a0=DESIGN_A0
    )

    report = HorizonFeasibilityReport(
        requested_hold_steps=int(plan_hold_steps),
        requested_hours=int(plan_hold_steps) * int(interval_hours),
        interval_hours=int(interval_hours),
        budget=budget.to_dict() if budget is not None else {},
        checked_at=_utc_now(),
    )

    # The forbidden shortcut must be off before AND after the measurement.
    report.activation_checkpoint_flags_seen = assert_activation_checkpointing_disabled(
        bridge.model
    )

    for lora in bank.values():
        lora.to(device)
    params = _select_expert_parameters(bank, expert_index)
    moved = sample.to(device)

    max_feasible = 0
    for steps in range(1, int(plan_hold_steps) + 1):
        if max_differentiable_steps is not None and steps > int(max_differentiable_steps):
            report.measurements.append({
                "steps": steps,
                "hours": steps * int(interval_hours),
                "ok": False,
                "reason": "forced_step_limit",
                "detail": (
                    f"max_differentiable_steps={max_differentiable_steps} "
                    "forbids this horizon"
                ),
            })
            continue

        step_plan = expert_plan(reg, expert_index, hold_steps=steps)
        # The objective must score a lead that is actually rolled at this
        # horizon, otherwise the measurement would be of a different graph.
        step_spec = ObjectiveSpec(
            lead_steps=(steps,),
            q=spec.q,
            scale=spec.scale,
            space=spec.space,
            variables=spec.variables,
            interval_hours=spec.interval_hours,
        )
        if steps not in moved.targets_raw:
            report.measurements.append({
                "steps": steps,
                "hours": steps * int(interval_hours),
                "ok": False,
                "reason": "no_target_for_horizon",
                "detail": (
                    f"the sample carries targets for steps "
                    f"{sorted(moved.targets_raw)}, so horizon {steps} cannot be "
                    "scored"
                ),
            })
            continue

        for param in params:
            param.grad = None
        _peak_memory_reset(device)
        started = time.perf_counter()
        try:
            trajectory = controlled_rollout(
                bridge, moved.x_norm, names, interval=int(interval_hours),
                steps=steps, plan=step_plan, expert_loras=dict(bank),
                target_blocks=blocks, return_trajectory=True,
            )
            loss = objective_loss_for_sample(bridge, trajectory, moved, step_spec)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - started
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_seconds = time.perf_counter() - started
        except RuntimeError as exc:
            peak_bytes, memory_source = _peak_memory_read(device)
            report.measurements.append({
                "steps": steps,
                "hours": steps * int(interval_hours),
                "ok": False,
                "reason": "runtime_error",
                "detail": f"{type(exc).__name__}: {exc}",
                "peak_memory_bytes": peak_bytes,
                "memory_source": memory_source,
            })
            break

        peak_bytes, memory_source = _peak_memory_read(device)
        entry: Dict[str, Any] = {
            "steps": steps,
            "hours": steps * int(interval_hours),
            "ok": True,
            "forward_seconds": forward_seconds,
            "total_seconds": total_seconds,
            "peak_memory_bytes": peak_bytes,
            "memory_source": memory_source,
            "loss": float(loss.detach()),
        }
        if budget is not None:
            if peak_bytes >= 0 and peak_bytes > budget.usable_memory_bytes:
                entry["ok"] = False
                entry["reason"] = "peak_memory_over_budget"
            elif total_seconds > budget.max_seconds_per_update:
                entry["ok"] = False
                entry["reason"] = "seconds_per_update_over_budget"
        report.measurements.append(entry)
        if entry["ok"]:
            max_feasible = steps
        else:
            break

    for param in params:
        param.grad = None

    # Re-assert: nothing in this routine may have switched the shortcut on.
    assert_activation_checkpointing_disabled(bridge.model)
    report.activation_checkpointing_enabled = False
    report.max_feasible_differentiable_steps = max_feasible
    report.feasible = max_feasible >= int(plan_hold_steps)

    if not report.feasible:
        train_steps = min(int(fallback_train_steps), max(1, max_feasible))
        reason_parts = [
            m.get("reason", "unknown")
            for m in report.measurements
            if not m.get("ok")
        ]
        report.fallback = HorizonFallbackSpec(
            reason=(
                f"differentiable backprop reached only "
                f"{max_feasible * int(interval_hours)}h "
                f"({max_feasible} of {plan_hold_steps} hold steps); first "
                f"blocking reason(s): {reason_parts[:3]}"
            ),
            train_interval_hours=int(interval_hours),
            train_steps=train_steps,
            train_hours=train_steps * int(interval_hours),
            designed_train_steps=int(plan_hold_steps),
            designed_train_hours=int(plan_hold_steps) * int(interval_hours),
            eval_steps=tuple(int(s) for s in fallback_eval_steps),
            eval_hours=tuple(int(s) * int(interval_hours) for s in fallback_eval_steps),
            eval_differentiable=False,
            experiment_identity=FALLBACK_EXPERIMENT_IDENTITY,
            superseded_identity=DESIGNED_EXPERIMENT_IDENTITY,
            cost_estimate=_fallback_cost_estimate(
                measurements=report.measurements,
                train_steps=train_steps,
                eval_steps=fallback_eval_steps,
                updates_per_expert=int(updates_per_expert),
                num_experts=int(num_experts_for_cost),
                n_eval_issues=int(n_eval_issues),
                n_candidates=int(n_candidates),
            ),
            activation_checkpointing_used=False,
            note=(
                "Reduced training horizon with a SEPARATE non-differentiable "
                "24h/72h evaluation. This is a different experiment identity "
                f"({FALLBACK_EXPERIMENT_IDENTITY}) and results obtained under it "
                f"must not be reported under {DESIGNED_EXPERIMENT_IDENTITY}. "
                "Activation checkpointing was NOT enabled to reach the longer "
                "horizon: controlled_rollout removes its injection hooks before "
                "backward, so a replayed forward would omit the edit entirely "
                "and produce gradients for a model that was never evaluated."
            ),
            created_at=_utc_now(),
        )
    return report


# =============================================================================
# 7. FP-04: bank artifacts on the certified Fs (digests, files, panels, probes)
# =============================================================================
#
# Everything below is additive. Training itself is still `train_expert`
# (unchanged update); these helpers only identify, save, reload, compare and
# qualify what it produced.

BANK_SCHEMA = "ed-bank/1"
EXPERT_FILE_SCHEMA = "ed-bank-expert-file/1"
BANK_FILE_SCHEMA = "ed-bank-file/1"
PROBE_FILE_SCHEMA = "ed-bank-probe/1"
BANK_PANELS_SCHEMA = "ed-bank-panels/1"


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tensor(digest: Any, name: str, tensor: Tensor) -> None:
    values = tensor.detach().cpu().contiguous()
    digest.update(f"key={name}\ndtype={values.dtype}\nshape={tuple(values.shape)}\n".encode())
    digest.update(values.numpy().tobytes())


def _bank_blocks(bank: Mapping[int, ExpertLoRA],
                 target_blocks: Optional[Sequence[int]] = None) -> Tuple[int, ...]:
    blocks = tuple(sorted(int(b) for b in (target_blocks if target_blocks is not None
                                            else bank.keys())))
    if not blocks:
        raise BankTrainingViolation("BANK_NO_TARGET_BLOCKS", "the bank has no blocks.", {})
    return blocks


def _uniform_geometry(bank: Mapping[int, ExpertLoRA], blocks: Sequence[int]) -> Dict[str, Any]:
    """K / rank / scale / widths shared by every block, or a refusal."""
    geometry = {
        (int(bank[b].num_experts), int(bank[b].rank_per_expert), float(bank[b].scale),
         int(bank[b].in_features), int(bank[b].out_features),
         bool(bank[b].shared_down), bool(bank[b].shared_up))
        for b in blocks
    }
    if len(geometry) != 1:
        raise BankTrainingViolation(
            "BANK_GEOMETRY_INCONSISTENT",
            f"bank blocks disagree on (K, rank, scale, in, out, shared): {sorted(geometry)}.",
            {"geometry": [list(g) for g in sorted(geometry)]})
    k, rank, scale, fan_in, fan_out, shared_down, shared_up = geometry.pop()
    if shared_down or shared_up:
        raise BankTrainingViolation(
            "BANK_SHARED_FACTORS_UNSUPPORTED",
            "per-expert digests need independent A/B factors.",
            {"shared_down": shared_down, "shared_up": shared_up})
    return {"num_experts": k, "rank_per_expert": rank, "scale": scale,
            "in_features": fan_in, "out_features": fan_out}


def _expert_digest_from_factors(
    factors: Mapping[int, Tuple[Tensor, Tensor]], *, rank: int, scale: float
) -> str:
    """SHA-256 over one expert's (down, up) factors in every block.

    The expert INDEX is deliberately not part of the digest: two experts with
    identical factors must collide, which is what the distinct-digest rule of
    BANK-SELECT-v1 relies on.
    """
    blocks = sorted(int(b) for b in factors)
    digest = hashlib.sha256()
    digest.update(f"schema={BANK_SCHEMA}\nkind=expert\nrank={int(rank)}\n"
                  f"scale={float(scale)!r}\nblocks={blocks}\n".encode())
    for block in blocks:
        down, up = factors[block]
        _hash_tensor(digest, f"{block}.down", down)
        _hash_tensor(digest, f"{block}.up", up)
    return digest.hexdigest()


def _expert_factors(bank: Mapping[int, ExpertLoRA], expert_index: int,
                    blocks: Sequence[int]) -> Dict[int, Tuple[Tensor, Tensor]]:
    k = int(expert_index)
    num_experts = int(bank[blocks[0]].num_experts)
    if not 0 <= k < num_experts:
        raise BankTrainingViolation(
            "BANK_EXPERT_INDEX_OUT_OF_RANGE",
            f"expert {k} is outside [0, {num_experts}).", {"expert_index": k})
    return {int(b): (bank[b]._get_down(k).weight, bank[b]._get_up(k).weight) for b in blocks}


def expert_digest(bank: Mapping[int, ExpertLoRA], expert_index: int, *,
                  target_blocks: Optional[Sequence[int]] = None) -> str:
    """Identity of ONE expert's factors (A and B, every block)."""
    blocks = _bank_blocks(bank, target_blocks)
    geometry = _uniform_geometry(bank, blocks)
    return _expert_digest_from_factors(
        _expert_factors(bank, expert_index, blocks),
        rank=geometry["rank_per_expert"], scale=geometry["scale"])


def other_experts_digest(bank: Mapping[int, ExpertLoRA], expert_index: int, *,
                         target_blocks: Optional[Sequence[int]] = None) -> str:
    """Identity of every expert EXCEPT `expert_index` (the isolation evidence)."""
    blocks = _bank_blocks(bank, target_blocks)
    geometry = _uniform_geometry(bank, blocks)
    digest = hashlib.sha256()
    digest.update(f"schema={BANK_SCHEMA}\nkind=other_experts\nexcluded={int(expert_index)}\n".encode())
    for k in range(int(geometry["num_experts"])):
        if k != int(expert_index):
            digest.update(f"expert={k}:{expert_digest(bank, k, target_blocks=blocks)}\n".encode())
    return digest.hexdigest()


def bank_digest(bank: Mapping[int, ExpertLoRA], *,
                target_blocks: Optional[Sequence[int]] = None) -> str:
    """Identity of the whole bank: geometry plus every block's full state dict.

    Recomputable on CPU from `build_dynamic_bank(seed=...)` alone, which is how
    the FP-04 protocol pins the initial bank before any worker runs.
    """
    blocks = _bank_blocks(bank, target_blocks)
    geometry = _uniform_geometry(bank, blocks)
    digest = hashlib.sha256()
    digest.update(f"schema={BANK_SCHEMA}\nkind=dynamic_bank\nblocks={list(blocks)}\n".encode())
    digest.update(("geometry=" + ",".join(f"{k}={geometry[k]!r}" for k in sorted(geometry))
                   + "\n").encode())
    for block in blocks:
        digest.update(f"block={block}\n".encode())
        digest.update(state_dict_digest(bank[block]).encode())
    return digest.hexdigest()


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    if actual.type != expected.type:
        return False
    return expected.index is None or actual.index == expected.index


def check_bank_device(bank: Mapping[int, ExpertLoRA], device: torch.device | str, *,
                      context: str = "") -> Dict[str, Any]:
    """Every factor of the bank must live on ``device`` before a rollout.

    `controlled_rollout` would otherwise fail deep inside a hook with a generic
    "Expected all tensors to be on the same device" -- or, worse, a caller
    that compares a CPU-resident reloaded bank against a CUDA backbone could
    silently compare the wrong objects. This makes the precondition explicit.
    """
    expected = torch.device(device)
    offending: List[Dict[str, Any]] = []
    for block, lora in bank.items():
        for name, tensor in list(lora.named_parameters()) + list(lora.named_buffers()):
            if not _same_device(tensor.device, expected):
                offending.append({"block": int(block), "tensor": name, "device": str(tensor.device)})
    if offending:
        raise BankTrainingViolation(
            "BANK_DEVICE_MISMATCH",
            f"{len(offending)} bank tensor(s) are not on {expected} "
            f"({context or 'bank rollout'}); first: {offending[:3]}. Move the bank "
            "(build_dynamic_bank(device=...) / load_bank(device=...)) before comparing "
            "or rolling it out against a backbone on that device.",
            {"expected": str(expected), "offending": offending[:16], "context": context})
    return {"check": "bank_device", "device": str(expected), "context": context, "passed": True}


def _panel_leads(lead_steps: Sequence[int]) -> List[int]:
    steps = sorted({int(s) for s in lead_steps})
    if not steps or steps[0] <= 0:
        raise BankTrainingViolation(
            "BANK_PANEL_LEAD_STEPS_INVALID",
            f"panel lead steps must be positive rollout steps, got {list(lead_steps)}.",
            {"lead_steps": list(lead_steps)})
    return steps


def bank_panel_losses(
    fs_bridge: WeatherStepBridge,
    bank: Mapping[int, ExpertLoRA],
    plan: EditPlan,
    samples: Sequence[TrainingSample],
    spec: ObjectiveSpec,
    *,
    lead_steps: Sequence[int] = (1, 4, 12),
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    variables: Optional[Sequence[str]] = None,
    interval_hours: int = DESIGN_INTERVAL_HOURS,
) -> Dict[str, Dict[str, float]]:
    """No-grad per-issue objective at each lead under ``plan``, ONE rollout per issue.

    The same shape and reduction as `static_adapter.fs_panel_losses`
    (``{issue_id: {"6": L, "24": L, "72": L}}``, float64 objective restricted to
    one lead), but the rollout is `controlled_rollout(plan, bank)` on the
    frozen Fs bridge: a singleton plan applies its expert at a0 for the hold
    window and continues as Fs afterwards. With the expert's B still at its
    zero init every hook adds an exact zero, so the panel must equal the pure
    Fs panel bit for bit.
    """
    steps = _panel_leads(lead_steps)
    names = list(variables) if variables is not None else list(fs_bridge.variables)
    device = next(fs_bridge.model.parameters()).device
    check_bank_device(bank, device, context="bank_panel_losses")
    per_lead = {s: dataclasses.replace(spec, lead_steps=(s,)) for s in steps}
    blocks = tuple(int(b) for b in target_blocks)
    panel: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for sample in samples:
            if sample.issue_id in panel:
                raise BankTrainingViolation(
                    "BANK_PANEL_DUPLICATE_ISSUE",
                    f"issue {sample.issue_id!r} appears twice in one panel.",
                    {"issue_id": sample.issue_id})
            on_device = sample.to(device)
            trajectory = controlled_rollout(
                fs_bridge, on_device.x_norm, names, interval=int(interval_hours),
                steps=steps[-1], plan=plan, expert_loras=dict(bank), target_blocks=blocks,
                return_trajectory=True,
            )
            panel[sample.issue_id] = {
                str(s * int(interval_hours)): float(
                    objective_loss_for_sample(fs_bridge, trajectory, on_device, per_lead[s]))
                for s in steps
            }
    return panel


def singleton_probe_states(
    fs_bridge: WeatherStepBridge,
    bank: Mapping[int, ExpertLoRA],
    plan: EditPlan,
    x_norm: Tensor,
    variables: Sequence[str],
    *,
    probe_steps: Sequence[int] = (1, 4, 12),
    target_blocks: Sequence[int] = DEFAULT_TARGET_BLOCKS,
    interval_hours: int = DESIGN_INTERVAL_HOURS,
) -> Dict[int, Tensor]:
    """The rollout STATES of one plan at the probe steps, as CPU tensors.

    Saved by the training process and recomputed by the assembly verifier:
    the two must agree exactly (FP-04 A1/A2).
    """
    steps = _panel_leads(probe_steps)
    device = next(fs_bridge.model.parameters()).device
    check_bank_device(bank, device, context="singleton_probe_states")
    with torch.no_grad():
        trajectory = controlled_rollout(
            fs_bridge, x_norm.to(device), list(variables), interval=int(interval_hours),
            steps=steps[-1], plan=plan, expert_loras=dict(bank),
            target_blocks=tuple(int(b) for b in target_blocks), return_trajectory=True,
        )
    return {s: trajectory[:, s].detach().cpu().clone() for s in steps}


def compare_probe_states(reference: Mapping[Any, Tensor],
                         candidate: Mapping[Any, Tensor]) -> Dict[str, Any]:
    """Exact comparison of two probe-state maps (no tolerance).

    ``exact`` requires the same steps, shapes and dtypes, ``torch.equal`` at
    every step and a max |difference| of exactly 0.0 (a NaN anywhere fails).
    """
    ref = {int(k): v for k, v in reference.items()}
    cand = {int(k): v for k, v in candidate.items()}
    steps = sorted(ref)
    diffs: Dict[str, Optional[float]] = {}
    equal: Dict[str, bool] = {}
    same_steps = steps == sorted(cand)
    for step in steps:
        a = ref[step].detach().cpu()
        b = cand.get(step)
        if b is None or b.shape != a.shape or b.dtype != a.dtype:
            diffs[str(step)] = None
            equal[str(step)] = False
            continue
        b = b.detach().cpu()
        diffs[str(step)] = float((a.double() - b.double()).abs().max())
        equal[str(step)] = bool(torch.equal(a, b))
    exact = bool(same_steps and steps and all(equal.values())
                 and all(d is not None and d == 0.0 for d in diffs.values()))
    return {"steps": steps, "same_steps": same_steps, "max_abs_diff_by_step": diffs,
            "torch_equal_by_step": equal, "exact": exact}


def _json_primitives(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Metadata stored inside a ``weights_only`` file must be plain JSON types."""
    import json

    return json.loads(json.dumps(dict(metadata or {}), default=str))


def save_expert(bank: Mapping[int, ExpertLoRA], expert_index: int, path: Path | str, *,
                metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Write ONE expert's factors (every block) to ``path``; return its identity.

    CPU copies only, so the bytes do not depend on the training device; the
    payload loads with ``torch.load(weights_only=True)``.
    """
    blocks = _bank_blocks(bank)
    geometry = _uniform_geometry(bank, blocks)
    k = int(expert_index)
    factors = _expert_factors(bank, k, blocks)
    payload = {
        "schema": EXPERT_FILE_SCHEMA,
        "expert_index": k,
        **geometry,
        "blocks": list(blocks),
        "factors": {str(b): {"down": down.detach().cpu().clone(), "up": up.detach().cpu().clone()}
                    for b, (down, up) in factors.items()},
        "expert_digest": expert_digest(bank, k),
        "metadata": _json_primitives(metadata),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return {"path": str(out), "sha256": _file_sha256(out), "expert_index": k,
            "expert_digest": payload["expert_digest"]}


def load_expert(path: Path | str, *, expected_sha256: Optional[str] = None,
                expected_expert_index: Optional[int] = None) -> Dict[str, Any]:
    """Read an expert file: byte identity first, then schema and digest.

    Refuses a file whose SHA-256 differs from ``expected_sha256``, whose
    expert index is not the expected one, or whose factors do not hash to the
    digest recorded inside it.
    """
    file_path = Path(path)
    sha = _file_sha256(file_path)
    if expected_sha256 is not None and sha != str(expected_sha256):
        raise BankTrainingViolation(
            "BANK_EXPERT_FILE_SHA256_MISMATCH",
            f"{file_path} has sha256 {sha}, expected {expected_sha256}.",
            {"path": str(file_path), "actual": sha, "expected": str(expected_sha256)})
    payload = torch.load(file_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != EXPERT_FILE_SCHEMA:
        raise BankTrainingViolation(
            "BANK_EXPERT_FILE_SCHEMA", f"{file_path} is not an {EXPERT_FILE_SCHEMA} file.",
            {"path": str(file_path)})
    index = int(payload["expert_index"])
    if expected_expert_index is not None and index != int(expected_expert_index):
        raise BankTrainingViolation(
            "BANK_EXPERT_INDEX_MISMATCH",
            f"{file_path} holds expert {index}, expected expert {expected_expert_index}.",
            {"path": str(file_path), "stored": index, "expected": int(expected_expert_index)})
    factors = {int(b): (f["down"], f["up"]) for b, f in payload["factors"].items()}
    recomputed = _expert_digest_from_factors(
        factors, rank=int(payload["rank_per_expert"]), scale=float(payload["scale"]))
    if recomputed != payload.get("expert_digest"):
        raise BankTrainingViolation(
            "BANK_EXPERT_DIGEST_MISMATCH",
            f"{file_path}: factors hash to {recomputed}, the file records "
            f"{payload.get('expert_digest')}.",
            {"path": str(file_path), "recomputed": recomputed,
             "recorded": payload.get("expert_digest")})
    return {"path": str(file_path), "sha256": sha, "expert_index": index,
            "expert_digest": recomputed, "factors": factors, "payload": payload}


def install_expert(bank: Mapping[int, ExpertLoRA], loaded: Mapping[str, Any], *,
                   expert_index: Optional[int] = None) -> str:
    """Copy a loaded expert's factors into slot ``expert_index`` of ``bank``.

    Geometry (K, rank, scale, widths, blocks) must match exactly; the installed
    expert must then hash to the loaded digest. Returns that digest.
    """
    payload = loaded["payload"]
    k = int(loaded["expert_index"])
    if expert_index is not None and int(expert_index) != k:
        raise BankTrainingViolation(
            "BANK_EXPERT_INDEX_MISMATCH",
            f"expert file holds expert {k}; it cannot be installed as expert {expert_index}.",
            {"stored": k, "requested": int(expert_index)})
    blocks = _bank_blocks(bank)
    geometry = _uniform_geometry(bank, blocks)
    for key in ("num_experts", "rank_per_expert", "scale", "in_features", "out_features"):
        if payload.get(key) != geometry[key]:
            raise BankTrainingViolation(
                "BANK_EXPERT_GEOMETRY_MISMATCH",
                f"expert file {key}={payload.get(key)!r} but the bank has {geometry[key]!r}.",
                {"key": key, "file": payload.get(key), "bank": geometry[key]})
    factors = loaded["factors"]
    if sorted(factors) != list(blocks):
        raise BankTrainingViolation(
            "BANK_EXPERT_BLOCKS_MISMATCH",
            f"expert file covers blocks {sorted(factors)}, the bank {list(blocks)}.",
            {"file": sorted(factors), "bank": list(blocks)})
    with torch.no_grad():
        for block in blocks:
            down_w = bank[block]._get_down(k).weight
            up_w = bank[block]._get_up(k).weight
            down, up = factors[block]
            if tuple(down.shape) != tuple(down_w.shape) or tuple(up.shape) != tuple(up_w.shape):
                raise BankTrainingViolation(
                    "BANK_EXPERT_SHAPE_MISMATCH",
                    f"block {block}: file shapes {tuple(down.shape)}/{tuple(up.shape)} vs "
                    f"bank {tuple(down_w.shape)}/{tuple(up_w.shape)}.", {"block": block})
            down_w.copy_(down.to(device=down_w.device, dtype=down_w.dtype))
            up_w.copy_(up.to(device=up_w.device, dtype=up_w.dtype))
    installed = expert_digest(bank, k)
    if installed != loaded["expert_digest"]:
        raise BankTrainingViolation(
            "BANK_EXPERT_INSTALL_MISMATCH",
            f"installed expert {k} hashes to {installed}, not {loaded['expert_digest']}.",
            {"installed": installed, "loaded": loaded["expert_digest"]})
    return installed


def save_bank(bank: Mapping[int, ExpertLoRA], path: Path | str, *,
              metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Write the whole bank (every block's state dict) with its digests."""
    blocks = _bank_blocks(bank)
    geometry = _uniform_geometry(bank, blocks)
    payload = {
        "schema": BANK_FILE_SCHEMA,
        **geometry,
        "blocks": list(blocks),
        "state_dicts": {str(b): {name: t.detach().cpu().clone()
                                 for name, t in bank[b].state_dict().items()} for b in blocks},
        "bank_digest": bank_digest(bank),
        "expert_digests": [expert_digest(bank, k) for k in range(geometry["num_experts"])],
        "metadata": _json_primitives(metadata),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return {"path": str(out), "sha256": _file_sha256(out), "bank_digest": payload["bank_digest"],
            "expert_digests": list(payload["expert_digests"])}


def load_bank(path: Path | str, *, expected_sha256: Optional[str] = None,
              device: Optional[torch.device | str] = None
              ) -> Tuple[Dict[int, ExpertLoRA], Dict[str, Any]]:
    """Reload a bank file: bytes, schema, strict state dicts, recomputed digests."""
    file_path = Path(path)
    sha = _file_sha256(file_path)
    if expected_sha256 is not None and sha != str(expected_sha256):
        raise BankTrainingViolation(
            "BANK_FILE_SHA256_MISMATCH",
            f"{file_path} has sha256 {sha}, expected {expected_sha256}.",
            {"path": str(file_path), "actual": sha, "expected": str(expected_sha256)})
    payload = torch.load(file_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != BANK_FILE_SCHEMA:
        raise BankTrainingViolation(
            "BANK_FILE_SCHEMA", f"{file_path} is not an {BANK_FILE_SCHEMA} file.",
            {"path": str(file_path)})
    bank: Dict[int, ExpertLoRA] = {}
    for block in payload["blocks"]:
        lora = ExpertLoRA(int(payload["in_features"]), int(payload["out_features"]),
                          num_experts=int(payload["num_experts"]),
                          rank_per_expert=int(payload["rank_per_expert"]),
                          scale=float(payload["scale"]))
        lora.load_state_dict(payload["state_dicts"][str(block)], strict=True)
        lora.requires_grad_(False)
        bank[int(block)] = lora
    recomputed = bank_digest(bank)
    experts = [expert_digest(bank, k) for k in range(int(payload["num_experts"]))]
    if recomputed != payload.get("bank_digest") or experts != list(payload.get("expert_digests") or []):
        raise BankTrainingViolation(
            "BANK_FILE_DIGEST_MISMATCH",
            f"{file_path}: reloaded bank/expert digests differ from the recorded ones.",
            {"recomputed_bank": recomputed, "recorded_bank": payload.get("bank_digest"),
             "recomputed_experts": experts, "recorded_experts": payload.get("expert_digests")})
    if device is not None:
        for lora in bank.values():
            lora.to(device)
        check_bank_device(bank, device, context="load_bank")
    return bank, {"path": str(file_path), "sha256": sha, "bank_digest": recomputed,
                  "expert_digests": experts, "metadata": payload.get("metadata") or {}}


def assemble_bank(initial_bank: Mapping[int, ExpertLoRA],
                  experts: Mapping[int, Mapping[str, Any]]) -> Dict[int, ExpertLoRA]:
    """The K=4 bank: a copy of the pinned initial bank with EVERY expert replaced.

    Exactly K loaded experts, one per index, pairwise distinct digests. A
    bank with a missing expert is not "the K=4 bank minus one" -- it is
    refused (BANK-SELECT-v1: no substitution, no partial assembly).
    """
    blocks = _bank_blocks(initial_bank)
    geometry = _uniform_geometry(initial_bank, blocks)
    num_experts = int(geometry["num_experts"])
    indices = sorted(int(k) for k in experts)
    if indices != list(range(num_experts)):
        raise BankTrainingViolation(
            "BANK_ASSEMBLY_INCOMPLETE",
            f"assembly needs exactly experts {list(range(num_experts))}, got {indices}. "
            "BANK-SELECT-v1 forbids assembling a bank from fewer experts or "
            "substituting one.", {"indices": indices, "num_experts": num_experts})
    digests = [str(experts[k]["expert_digest"]) for k in indices]
    if len(set(digests)) != num_experts:
        raise BankTrainingViolation(
            "BANK_ASSEMBLY_DUPLICATE_EXPERT",
            "two expert files carry the same factors; a bank of duplicated experts "
            "is not K distinct experts.", {"expert_digests": digests})
    bank = {int(b): copy.deepcopy(lora) for b, lora in initial_bank.items()}
    for k in indices:
        install_expert(bank, experts[k], expert_index=k)
    for lora in bank.values():
        lora.requires_grad_(False)
    return bank


def source_bank_for_expert(initial_bank: Mapping[int, ExpertLoRA],
                           loaded: Mapping[str, Any]) -> Dict[int, ExpertLoRA]:
    """What expert k's own training process held: the initial bank with ONLY k trained."""
    bank = {int(b): copy.deepcopy(lora) for b, lora in initial_bank.items()}
    install_expert(bank, loaded)
    for lora in bank.values():
        lora.requires_grad_(False)
    return bank


# =============================================================================
# 8. FP-04 pre-registered rules: BANK-QUAL-v1, BANK-SELECT-v1, assembly A1-A5
# =============================================================================
#
# PURE functions of recorded numbers plus a rule mapping taken from the frozen
# protocol. They never read policy_dev/confirm data, never rank experts by a
# quality number, and never return a partial bank: a failure of any expert
# stops the whole batch (see `decide_bank_formal`).

BANK_QUAL_RULE_ID = "BANK-QUAL-v1"
BANK_SELECT_RULE_ID = "BANK-SELECT-v1"
BANK_QUAL_VERDICTS: Tuple[str, ...] = ("PASS", "FAIL", "INVALID")


def _panel_col(panel: Mapping[str, Mapping[str, float]], lead_hours: int) -> Dict[str, float]:
    key = str(int(lead_hours))
    return {str(i): float(row[key]) for i, row in panel.items() if key in row}


def _all_finite(values: Sequence[Any]) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def evaluate_bank_expert_qualification(
    record: Mapping[str, Any],
    panels: Mapping[str, Mapping[str, Mapping[str, float]]],
    diagnostics: Mapping[str, Any],
    rule: Mapping[str, Any],
    *,
    validity: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    """BANK-QUAL-v1 for ONE formal expert. Verdict in `BANK_QUAL_VERDICTS`.

    ``record`` is the expert's `ExpertTrainingRecord.to_dict()`; ``panels``
    holds its own-group panels ``{"Fs", "L0", "L1"}`` (pure Fs, untrained
    singleton, trained singleton; per issue, per lead hour); ``diagnostics``
    carries the worker's ``nonzero_response``, ``reload`` and ``isolation``
    facts; ``validity`` the E0 facts (each must be True). Criteria:

      E0 validity    every E0 fact True; untrained panel == Fs bit for bit
      E1 numerics    n_updates == horizon, every loss / grad norm (total, A, B)
                     finite, traces consistent, eligible
      E2 stability   `training_stability` pattern: clip events <= max, every
                     training loss <= per_visit_ratio_max x its issue's Fs 24h,
                     no epoch-mean onset (epoch = one pass over the group)
      E3 direction   own-group 24h mean(L1) / mean(Fs) < threshold (hard)
      E4 response    nonzero response at a0, B off zero, the expert moved, and
                     the trained panel differs from the untrained one
      E5 reload      saved file re-hashes, reloaded digest and probe states exact
      E6 isolation   other experts unchanged and gradient-free, backbone digest
                     unchanged, no leftover hooks

    Verdict: INVALID if E0 fails or a threshold is not pre-registered; else
    FAIL if any E1..E6 fails; else PASS. 6h / 72h own-group ratios and the
    per-issue 24h detail are REPORT-ONLY: out-of-sample judgement is FP-05's.
    """
    rec = dict(record)
    horizon = rule.get("horizon")
    e2r = dict(rule.get("e2") or {})
    e3r = dict(rule.get("e3") or {})
    fs, l0, l1 = panels.get("Fs") or {}, panels.get("L0") or {}, panels.get("L1") or {}
    n_group = int(rec.get("n_samples") or 0)
    epoch_size = e2r.get("epoch_size")
    if epoch_size == "group_size":
        epoch_size = n_group
    status: Dict[str, str] = {}

    # -- E0 ------------------------------------------------------------------
    facts = {str(k): v for k, v in (validity or {}).items()}
    trained = [str(i) for i in rec.get("training_issue_ids") or []]
    facts["untrained_L0_equals_Fs_bitwise"] = bool(fs) and dict(l0) == dict(fs)
    facts["panels_cover_the_training_issues"] = bool(fs) and (
        set(trained) <= set(fs) and set(fs) == set(l0) == set(l1))
    failed = sorted(k for k, v in facts.items() if v is not True)
    if validity is None:
        failed = ["NO_VALIDITY_FACTS"] + failed
    e0 = {"status": "PASS" if not failed else "FAIL", "failed": failed, "facts": facts}
    status["E0"] = e0["status"]

    thresholds = {
        "horizon": horizon,
        "e2.clip_events_max": e2r.get("clip_events_max"),
        "e2.per_visit_ratio_max": e2r.get("per_visit_ratio_max"),
        "e2.epoch_onset_factor": e2r.get("epoch_onset_factor"),
        "e2.epoch_size": epoch_size,
        "e3.own_group_mean_ratio_24h_lt": e3r.get("own_group_mean_ratio_24h_lt"),
    }
    not_preregistered = sorted(k for k, v in thresholds.items() if v is None)

    # -- E1 / E2: the FS-QUAL Q1/Q2 stability pattern, per expert ------------
    stability: Optional[Dict[str, Any]] = None
    stability_error = None
    if not not_preregistered and facts["panels_cover_the_training_issues"]:
        try:
            stability = training_stability(
                rec, _panel_col(fs, 24), horizon=int(horizon),
                clip_events_max=int(e2r["clip_events_max"]),
                per_visit_ratio_max=float(e2r["per_visit_ratio_max"]),
                epoch_onset_factor=float(e2r["epoch_onset_factor"]),
                epoch_size=int(epoch_size))
        except (ValueError, KeyError, TypeError) as exc:
            stability_error = f"{type(exc).__name__}: {exc}"
    grads_a = list(rec.get("grad_norms_A") or [])
    grads_b = list(rec.get("grad_norms_B") or [])
    n_updates = int(rec.get("n_updates") or 0)
    e1 = {
        "q1": stability["q1"] if stability else None,
        "ab_grad_norms_finite": _all_finite(grads_a + grads_b),
        "ab_trace_lengths_consistent": len(grads_a) == len(grads_b) == n_updates,
        "eligible": rec.get("eligible") is True,
        "error": stability_error,
    }
    e1["passed"] = bool(stability and stability["q1"]["passed"] and e1["ab_grad_norms_finite"]
                        and e1["ab_trace_lengths_consistent"] and e1["eligible"])
    status["E1"] = "PASS" if e1["passed"] else "FAIL"
    e2 = dict(stability["q2"]) if stability else {"passed": False, "error": stability_error}
    if stability:
        e2["epoch_size"] = int(epoch_size)
        e2["n_full_epochs"] = len(stability["q2"]["epoch_means"])
    status["E2"] = "PASS" if e2.get("passed") else "FAIL"

    # -- E3: own training group must improve (hard) ------------------------
    own24 = None
    if facts["panels_cover_the_training_issues"]:
        own24 = panel_ratio_summary(fs, l1, 24)
    threshold = e3r.get("own_group_mean_ratio_24h_lt")
    e3 = {"own_group_24h": own24, "threshold_lt": threshold}
    if threshold is None:
        status["E3"] = "NOT_PREREGISTERED"
    else:
        ratio = own24["mean_ratio"] if own24 else None
        ok = ratio is not None and math.isfinite(float(ratio)) and float(ratio) < float(threshold)
        status["E3"] = "PASS" if ok else "FAIL"

    # -- E4: nonzero response ----------------------------------------------
    nz = dict(diagnostics.get("nonzero_response") or {})
    e4 = {
        "nonzero_response": nz.get("nonzero") is True,
        "not_zero_initialized": nz.get("zero_initialized_untrained") is False,
        "expert_moved": (rec.get("initial_expert_digest") is not None
                         and rec.get("final_expert_digest") is not None
                         and rec.get("initial_expert_digest") != rec.get("final_expert_digest")),
        "trained_panel_differs_from_untrained": bool(l1) and dict(l1) != dict(l0),
        "max_abs_response": nz.get("max_abs_response"),
    }
    status["E4"] = "PASS" if all(e4[k] for k in ("nonzero_response", "not_zero_initialized",
                                               "expert_moved",
                                               "trained_panel_differs_from_untrained")) else "FAIL"

    # -- E5: reload --------------------------------------------------------
    rl = dict(diagnostics.get("reload") or {})
    probe_diffs = dict(rl.get("probe_max_abs_diff_by_step") or {})
    e5 = {
        "file_sha256_matches": rl.get("file_sha256_matches") is True,
        "expert_digest_matches": rl.get("expert_digest_matches") is True,
        "probe_exact": rl.get("probe_exact") is True,
        "probe_max_abs_diff_all_zero": bool(probe_diffs) and all(
            v is not None and float(v) == 0.0 for v in probe_diffs.values()),
    }
    status["E5"] = "PASS" if all(e5.values()) else "FAIL"

    # -- E6: isolation -----------------------------------------------------
    iso = dict(diagnostics.get("isolation") or {})
    e6 = {
        "other_experts_unchanged": rec.get("other_experts_unchanged") is True,
        "other_experts_grad_free": rec.get("other_experts_grad_free") is True,
        "backbone_digest_unchanged": iso.get("backbone_digest_unchanged") is True,
        "no_leftover_hooks": iso.get("no_leftover_hooks") is True,
    }
    status["E6"] = "PASS" if all(e6.values()) else "FAIL"

    substantive = [status[k] for k in ("E1", "E2", "E3", "E4", "E5", "E6")]
    if status["E0"] != "PASS" or not_preregistered or "NOT_PREREGISTERED" in substantive:
        verdict = "INVALID"
    elif "FAIL" in substantive:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    report_only: Dict[str, Any] = {
        "per_visit_direction": per_sample_loss_direction(rec.get("losses") or [], trained),
    }
    if facts["panels_cover_the_training_issues"]:
        report_only["own_group_6h"] = panel_ratio_summary(fs, l1, 6)
        report_only["own_group_72h"] = panel_ratio_summary(fs, l1, 72)
        report_only["own_group_24h_per_issue"] = own24["per_issue_ratio"] if own24 else None
    return {
        "rule": BANK_QUAL_RULE_ID,
        "expert_index": rec.get("expert_index"),
        "verdict": verdict,
        "quality_pass": verdict == "PASS",
        "status": status,
        "not_preregistered": not_preregistered,
        "e0": e0, "e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6,
        "report_only": report_only,
    }


def decide_bank_formal(experts: Sequence[Mapping[str, Any]],
                       rule: Mapping[str, Any]) -> Dict[str, Any]:
    """BANK-SELECT-v1: all K experts of ONE batch PASS, or the bank stops.

    ``experts`` are ``{"expert_index", "qualification", "expert_digest",
    "expert_file": {"path", "sha256"}, "probe_file": {...}, "batch_id"}``. There
    is no substitution, no "keep the passing three", and no mixing of experts
    from different batches (jobs / learning rates). A failing expert is never
    EXCLUDED -- the whole batch is STOP_CURRENT_BANK -- so no gain-based
    exclusion reason can ever be exercised here.

    ``rule["fallback"]["requires_failed_criterion"]`` (e.g. ``"E2"``) sets
    ``fallback_authorized`` on a STOP_CURRENT_BANK: True only if EVERY failed
    expert failed that criterion (other criteria may fail alongside it). No
    fallback is ever authorized after INVALID or when no rule is declared.
    """
    n_required = int(rule.get("num_experts", DESIGN_NUM_EXPERTS))
    by_index: Dict[int, Mapping[str, Any]] = {}
    duplicate = False
    for entry in experts:
        k = int(entry["expert_index"])
        duplicate = duplicate or k in by_index
        by_index[k] = entry
    verdicts = {k: str((e.get("qualification") or {}).get("verdict")) for k, e in sorted(by_index.items())}
    batches = sorted({str(e.get("batch_id")) for e in experts})
    result: Dict[str, Any] = {
        "rule": BANK_SELECT_RULE_ID,
        "num_experts": n_required,
        "expert_verdicts": {str(k): v for k, v in verdicts.items()},
        "batches": batches,
        "substitution": "FORBIDDEN",
        "partial_bank": "FORBIDDEN",
        "cross_batch_mixing": "FORBIDDEN",
        "bank_selected_for_assembly": False,
        "experts_for_assembly": None,
        "fallback_authorized": False,
    }
    if duplicate or sorted(by_index) != list(range(n_required)):
        result.update(verdict="INVALID", reason="EXPERT_SET_INCOMPLETE_OR_DUPLICATED")
        return result
    if len(batches) != 1 or batches[0] in ("None", ""):
        result.update(verdict="INVALID", reason="CROSS_BATCH_MIXING_FORBIDDEN")
        return result
    values = list(verdicts.values())
    if "INVALID" in values or any(v not in BANK_QUAL_VERDICTS for v in values):
        result.update(verdict="INVALID", reason="AN_EXPERT_FAILED_E0_VALIDITY",
                      invalid_experts=[k for k, v in verdicts.items() if v != "PASS" and v != "FAIL"])
        return result
    if all(v == "PASS" for v in values):
        digests = [by_index[k].get("expert_digest") for k in range(n_required)]
        files = [(by_index[k].get("expert_file") or {}).get("sha256") for k in range(n_required)]
        if None in digests or None in files or len(set(digests)) != n_required \
                or len(set(files)) != n_required:
            result.update(verdict="INVALID", reason="EXPERT_DIGESTS_MISSING_OR_NOT_DISTINCT",
                          expert_digests=digests)
            return result
        result.update(
            verdict="BANK_QUAL_PASS_PENDING_ASSEMBLY",
            bank_selected_for_assembly=True,
            experts_for_assembly=[{
                "expert_index": k,
                "expert_digest": by_index[k]["expert_digest"],
                "expert_file": dict(by_index[k]["expert_file"]),
                "probe_file": dict(by_index[k].get("probe_file") or {}),
            } for k in range(n_required)])
        return result
    failed = [k for k, v in verdicts.items() if v == "FAIL"]
    failed_criteria = {
        str(k): sorted(c for c, s in ((by_index[k].get("qualification") or {}).get("status") or {}).items()
                       if s == "FAIL")
        for k in failed}
    fallback_rule = dict(rule.get("fallback") or {})
    required = fallback_rule.get("requires_failed_criterion")
    authorized = bool(
        required is not None and failed
        and all(required in failed_criteria[str(k)] for k in failed))
    result.update(verdict="STOP_CURRENT_BANK",
                  failed_experts=failed,
                  failed_criteria_by_expert=failed_criteria,
                  fallback_rule=fallback_rule or None,
                  fallback_authorized=authorized,
                  note=("BANK-SELECT-v1: one or more experts FAILED BANK-QUAL-v1, so the whole "
                        "batch stops. The passing experts are NOT kept, NOT assembled as a "
                        "smaller bank and NOT mixed with any other batch. The lr fallback is "
                        "authorized only if EVERY failed expert failed "
                        f"{required or '(no fallback declared)'}; a failure without it (e.g. "
                        "E3 alone) is a final STOP_CURRENT_BANK."))
    return result


def _exact_zero(values: Any) -> bool:
    items = list(values.values()) if isinstance(values, Mapping) else list(values or [])
    return bool(items) and all(v is not None and float(v) == 0.0 for v in items)


def _all_positive(values: Any) -> bool:
    items = list(values.values()) if isinstance(values, Mapping) else list(values or [])
    return bool(items) and all(v is not None and math.isfinite(float(v)) and float(v) > 0.0
                               for v in items)


def evaluate_bank_assembly(verify_record: Mapping[str, Any], *,
                           num_experts: int = DESIGN_NUM_EXPERTS) -> Dict[str, Any]:
    """A1-A5 recomputed from the RAW numbers of `bank_verify` (no tolerance).

    A1 assembled singleton == same-process source bank (exact, every probe step)
    A2 assembled singleton == the training process's saved probe (exact)
    A3 whole-bank zero edit == Fs (exactly 0.0) and != F0 (discriminating)
    A4 per expert: continuation after the hold == Fs exactly at every step,
       != F0 at every step, with a nonzero edit at the hold
    A5 reloaded bank file and digests == the assembly record's

    Every "passed" flag in the record is ignored; only the numbers count.
    Exactness is justified by `ExpertLoRA.forward_dense`: an inactive expert
    contributes ``expert_out * 0.0``, an exact zero for finite outputs.
    """
    rec = dict(verify_record)
    experts = [str(k) for k in range(int(num_experts))]
    a1 = rec.get("A1_source_bank") or {}
    a2 = rec.get("A2_training_probe") or {}
    a3 = rec.get("A3_zero_edit") or {}
    a4 = rec.get("A4_continuation") or {}
    a5 = rec.get("A5_reload") or {}
    status: Dict[str, bool] = {}
    status["A1"] = sorted(a1) == experts and all(
        _exact_zero(a1[k].get("max_abs_diff_by_step")) and all(
            (a1[k].get("torch_equal_by_step") or {}).values()) for k in experts)
    status["A2"] = sorted(a2) == experts and all(
        _exact_zero(a2[k].get("max_abs_diff_by_step"))
        and all((a2[k].get("torch_equal_by_step") or {}).values())
        and a2[k].get("probe_file_sha256") is not None
        and a2[k].get("probe_file_sha256") == a2[k].get("expected_probe_file_sha256")
        for k in experts)
    diff_fs = a3.get("max_abs_diff_vs_fs")
    diff_f0 = a3.get("max_abs_diff_vs_f0")
    status["A3"] = (diff_fs is not None and float(diff_fs) == 0.0 and diff_f0 is not None
                    and math.isfinite(float(diff_f0)) and float(diff_f0) > 0.0)
    status["A4"] = sorted(a4) == experts and all(
        _exact_zero(a4[k].get("max_abs_diff_vs_fs_continuation_by_step"))
        and _all_positive(a4[k].get("max_abs_diff_vs_f0_continuation_by_step"))
        and a4[k].get("edit_effect_at_hold_max_abs") is not None
        and float(a4[k]["edit_effect_at_hold_max_abs"]) > 0.0
        for k in experts)
    status["A5"] = bool(
        a5.get("file_sha256") is not None
        and a5.get("file_sha256") == a5.get("expected_file_sha256")
        and a5.get("bank_digest") is not None
        and a5.get("bank_digest") == a5.get("expected_bank_digest")
        and list(a5.get("expert_digests") or []) == list(a5.get("expected_expert_digests") or [])
        and len(a5.get("expert_digests") or []) == int(num_experts))
    passed = all(status.values())
    return {"check": "bank_assembly_equivalence", "status": status, "passed": passed,
            "failed": sorted(k for k, v in status.items() if not v),
            "tolerance": "none: every comparison is exact (max |diff| == 0.0)"}
