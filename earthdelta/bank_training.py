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

import datetime
import math
import time
from dataclasses import dataclass, field
from datetime import datetime as _datetime, timezone
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
    per_sample_loss_direction,
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
) -> Dict[int, ExpertLoRA]:
    """One K-expert `ExpertLoRA` per targeted block, at the standard LoRA init.

    The up (B) factors start at zero, which makes the whole bank a numerical
    no-op until it is trained. Section 6.2 calls that out explicitly -- "零 B
    初始状态是正常初始化，未训练零库不是科学失败" -- so nothing here treats a
    zero bank as an error.
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
        }


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
        record.n_updates += 1
        record.training_issue_ids.append(sample.issue_id)
        record.training_times_utc.append(sample.time_utc)
        record.training_sources.append(sample.source)
        if progress is not None:
            progress(update, loss_value, grad_norm)

    record.wallclock_seconds = time.perf_counter() - started
    record.finished_at = _utc_now()
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
