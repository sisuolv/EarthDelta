"""Dynamic expert-bank training tests (CPU, tiny synthetic).

Same scale as `tests/test_fs_static_adapter.py` and the
`plans/plans_v2_0921/cci/gpu_bank_check.py` precedent. Everything is synthetic:
these tests show the code paths are structurally correct and the contracts are
enforced. They are not a capacity finding, not a horizon decision and not a
weather result -- all three of those require the real GPU profile.

What is asserted:

  1. GRADIENTS. A dynamic expert's A/B factors receive non-zero gradients
     exactly when that expert's coefficient is non-zero, every other expert
     gets exactly zero, and the frozen Fs-merged backbone receives no gradient
     at all (and `train_expert` refuses to run if the backbone is not frozen).
  2. DIVERSITY GROUPING. The pre-declared fit-only rule produces disjoint
     groups over bank_fit samples, purges samples in the split guard window,
     and purges samples whose history/target/hold window straddles a group
     boundary. Non-bank_fit input is refused outright.
  3. REGISTRY. The candidate set is the explicit no-edit reference plus K
     singletons at the realized a0=0.25, with max_active=1.
  4. ELIGIBILITY. Only the pre-declared numerical rules may exclude an expert;
     "low gain" and friends are refused as exclusion reasons, and a zero-init
     untrained bank stays eligible.
  5. PROFILER + HORIZON. The single-training-step profiler runs and returns a
     decision object (tagged provisional off real GPU), and the horizon check
     produces a written fallback spec when forced to a tiny step limit --
     without ever enabling activation checkpointing.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta import bank_training as bt  # noqa: E402
from earthdelta import static_adapter as sa  # noqa: E402
from earthdelta.bridge import (  # noqa: E402
    DEFAULT_VARIABLES,
    NormalizationContract,
    Stormer,
    WeatherStepBridge,
    controlled_rollout,
)
from earthdelta.contracts import ArtifactVersion  # noqa: E402
from earthdelta.lowrank import ExpertLoRA  # noqa: E402
from earthdelta.registry import RegistryViolation  # noqa: E402


GRID = (16, 32)
N_VARS = 8
HIDDEN = 32
DEPTH = 2
TARGET_BLOCKS: Tuple[int, ...] = (0, 1)
K = 4
RANK = 2

FIXED_A0 = 0.25
ACTIVE_EXPERT = 1
INACTIVE_EXPERTS = (0, 2, 3)


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(20260921)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def variables() -> List[str]:
    return list(DEFAULT_VARIABLES[:N_VARS])


def _backbone(variables: List[str], seed: int = 11) -> Stormer:
    torch.manual_seed(seed)
    model = Stormer(
        in_img_size=GRID, variables=variables, patch_size=2,
        hidden_size=HIDDEN, depth=DEPTH, num_heads=4, mlp_ratio=2.0,
    )
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
    torch.nn.init.normal_(model.head.linear.weight, std=0.02)
    torch.nn.init.normal_(model.head.linear.bias, std=0.02)
    model.requires_grad_(False)
    model.eval()
    return model


def _bridge(model: Stormer, variables: List[str], *, static_adapter: str = "none"):
    normalization = NormalizationContract(
        inp_mean=torch.zeros(N_VARS), inp_std=torch.ones(N_VARS),
        diff_mean={6: torch.zeros(N_VARS)}, diff_std={6: torch.ones(N_VARS)},
        variables=variables,
    )
    version = ArtifactVersion(
        backbone="stormer_test_f0", static_adapter=static_adapter, edit_bank="none",
        normalization="pending", grid="16x32", projection="patch2",
        split="test", continuation="reference_after_hold",
    )
    return WeatherStepBridge(model, normalization, version)


@pytest.fixture
def fs_bridge(variables):
    """A frozen Fs-MERGED backbone: Fs already folded into attn.proj weights."""
    base = _backbone(variables)
    adapters = sa.build_fs_adapter(
        HIDDEN, TARGET_BLOCKS, rank_per_expert=2, seed=3, init_up_std=0.05
    )
    merged, artifact = sa.merge_static_adapter(
        base, adapters, target_blocks=TARGET_BLOCKS
    )
    bridge = _bridge(merged, variables, static_adapter=artifact.static_adapter_label)
    assert artifact.delta_max_abs > 0, "the Fs fixture must actually change the weights"
    return bridge


@pytest.fixture
def objective(fs_bridge) -> sa.ObjectiveSpec:
    lat = torch.linspace(-88.0, 88.0, GRID[0])
    return sa.build_objective_spec(fs_bridge, lat, lead_steps=(1,), space="raw")


def _bank(seed: int = 7, *, zero_init: bool = False) -> Dict[int, ExpertLoRA]:
    """A K-expert bank whose B factors are off the origin unless asked otherwise.

    `ExpertLoRA` zero-initializes the up (B) projections, which makes dL/dA
    identically zero; a gradient test must therefore start from a bank that has
    already moved off that origin. `zero_init=True` keeps the pristine state so
    the "untrained bank is normal" property can be tested too.
    """
    generator = torch.Generator().manual_seed(seed)
    bank: Dict[int, ExpertLoRA] = {}
    for block in TARGET_BLOCKS:
        lora = ExpertLoRA(HIDDEN, HIDDEN, num_experts=K, rank_per_expert=RANK)
        with torch.no_grad():
            for down in lora.down:
                down.weight.copy_(torch.randn(down.weight.shape, generator=generator) * 0.1)
            if not zero_init:
                for up in lora.up:
                    up.weight.copy_(torch.randn(up.weight.shape, generator=generator) * 0.1)
        for param in lora.parameters():
            param.requires_grad_(True)
        bank[block] = lora
    return bank


def _training_samples(bridge, n: int = 2, seed: int = 5) -> List[sa.TrainingSample]:
    generator = torch.Generator().manual_seed(seed)
    out: List[sa.TrainingSample] = []
    for i in range(n):
        x_raw = torch.randn(1, N_VARS, *GRID, generator=generator)
        issue = int(datetime(2015, 3, 10 + i, 0, 0, tzinfo=timezone.utc).timestamp())
        out.append(sa.TrainingSample(
            x_norm=bridge.normalization.normalize(x_raw),
            targets_raw={1: x_raw + 0.4 * torch.randn(1, N_VARS, *GRID, generator=generator)},
            issue_id=f"iss_{i:03d}",
            issue_time=issue,
            valid_time=issue + 6 * 3600,
            history_time=issue - 6 * 3600,
            split_id="train",
            data_role="bank_fit",
            source="synthetic",
            time_utc=datetime.fromtimestamp(issue, tz=timezone.utc).isoformat(),
        ))
    return out


# =============================================================================
# 1. Gradients: active experts only; the frozen Fs backbone stays untouched
# =============================================================================

def _rollout_and_backward(fs_bridge, variables, bank, expert_index, steps=1):
    registry = bt.build_bank_registry(K)
    plan = bt.expert_plan(registry, expert_index, hold_steps=max(1, steps))
    x_norm = torch.randn(1, N_VARS, *GRID)
    out = controlled_rollout(
        fs_bridge, x_norm, variables, interval=6, steps=steps, plan=plan,
        expert_loras=bank, target_blocks=TARGET_BLOCKS,
    )
    out.sum().backward()
    return plan


def test_only_the_active_dynamic_expert_receives_gradients(fs_bridge, variables):
    """Non-zero gradient exactly where the coefficient is non-zero."""
    bank = _bank()
    plan = _rollout_and_backward(fs_bridge, variables, bank, ACTIVE_EXPERT)
    assert plan.coefficients[ACTIVE_EXPERT] == pytest.approx(FIXED_A0)

    for block, lora in bank.items():
        for label, module in (
            ("down/A", lora.down[ACTIVE_EXPERT]), ("up/B", lora.up[ACTIVE_EXPERT])
        ):
            grad = module.weight.grad
            assert grad is not None, f"block {block} expert {ACTIVE_EXPERT} {label}"
            assert torch.isfinite(grad).all()
            assert grad.abs().max().item() > 0.0, (
                f"block {block} expert {ACTIVE_EXPERT} {label} has an identically "
                f"zero gradient despite coefficient {plan.coefficients[ACTIVE_EXPERT]}"
            )
        for k in INACTIVE_EXPERTS:
            for label, module in (("down/A", lora.down[k]), ("up/B", lora.up[k])):
                grad = module.weight.grad
                assert grad is not None, (
                    f"block {block} expert {k} {label} has no grad at all; the "
                    "dense path must still build the graph for inactive experts"
                )
                assert torch.count_nonzero(grad).item() == 0, (
                    f"block {block} expert {k} {label} received a gradient while "
                    "its coefficient was zero"
                )


def test_frozen_fs_merged_backbone_receives_zero_gradient(fs_bridge, variables):
    """Training the bank must not move -- or even touch -- the frozen Fs."""
    bank = _bank()
    before = {
        name: param.detach().clone()
        for name, param in fs_bridge.model.named_parameters()
    }

    _rollout_and_backward(fs_bridge, variables, bank, ACTIVE_EXPERT)

    assert [n for n, p in fs_bridge.model.named_parameters() if p.requires_grad] == []
    offenders = [
        name for name, param in fs_bridge.model.named_parameters()
        if param.grad is not None
    ]
    assert offenders == [], f"frozen Fs backbone received gradients: {offenders}"

    optimizer = torch.optim.SGD(
        [p for lora in bank.values() for p in lora.parameters()], lr=0.5
    )
    optimizer.step()
    for name, param in fs_bridge.model.named_parameters():
        torch.testing.assert_close(param.detach(), before[name], rtol=0.0, atol=0.0)


def test_zero_coefficients_give_every_expert_a_zero_gradient(fs_bridge, variables):
    """The no-edit reference plan leaves the whole bank gradient-free."""
    bank = _bank()
    registry = bt.build_bank_registry(K)
    reference = registry.reference_entry.to_edit_plan()
    x_norm = torch.randn(1, N_VARS, *GRID)

    controlled_rollout(
        fs_bridge, x_norm, variables, interval=6, steps=1, plan=reference,
        expert_loras=bank, target_blocks=TARGET_BLOCKS,
    ).sum().backward()

    for lora in bank.values():
        for name, param in lora.named_parameters():
            assert torch.count_nonzero(param.grad).item() == 0, name


def test_train_expert_freezes_the_other_experts(fs_bridge, variables, objective):
    """One process trains one expert; the other three must not move."""
    bank = _bank()
    samples = _training_samples(fs_bridge)
    before = {
        (block, k): bank[block].up[k].weight.detach().clone()
        for block in TARGET_BLOCKS for k in range(K)
    }
    config = bt.BankTrainConfig(
        mode="gradient_check", max_updates=3, train_steps=1, hold_steps=1,
        lead_steps=(1,), num_experts=K, rank_per_expert=RANK,
        target_blocks=TARGET_BLOCKS,
    )
    record = bt.train_expert(
        fs_bridge, bank, ACTIVE_EXPERT, samples, objective, config,
        variables=variables,
    )

    assert record.n_updates == 3
    assert record.eligible is True
    moved = [
        (block, k) for block in TARGET_BLOCKS for k in range(K)
        if not torch.equal(bank[block].up[k].weight.detach(), before[(block, k)])
    ]
    assert moved, "the training run did not move the expert at all"
    assert all(k == ACTIVE_EXPERT for _, k in moved), (
        f"training expert {ACTIVE_EXPERT} also moved {sorted(set(k for _, k in moved))}"
    )


def test_train_expert_refuses_an_unfrozen_backbone(fs_bridge, variables, objective):
    fs_bridge.model.requires_grad_(True)
    try:
        with pytest.raises(bt.BankTrainingViolation) as excinfo:
            bt.train_expert(
                fs_bridge, _bank(), 0, _training_samples(fs_bridge), objective,
                bt.BankTrainConfig(
                    mode="gradient_check", max_updates=1, train_steps=1,
                    hold_steps=1, lead_steps=(1,), num_experts=K,
                    rank_per_expert=RANK, target_blocks=TARGET_BLOCKS,
                ),
                variables=variables,
            )
        assert excinfo.value.code == "BANK_BACKBONE_NOT_FROZEN"
    finally:
        fs_bridge.model.requires_grad_(False)


def test_expert_record_carries_the_required_fields(fs_bridge, variables, objective):
    """Section 6.2: training data, update count, loss, non-zero response, group."""
    bank = _bank()
    samples = _training_samples(fs_bridge)
    grouping = bt.assign_diversity_groups(samples, 1, hold_steps=1)
    config = bt.BankTrainConfig(
        mode="gradient_check", max_updates=2, train_steps=1, hold_steps=1,
        lead_steps=(1,), num_experts=K, rank_per_expert=RANK,
        target_blocks=TARGET_BLOCKS,
    )
    record = bt.train_expert(
        fs_bridge, bank, 0, samples, objective, config,
        diversity_group=grouping.group_for(0), variables=variables,
        fs_backbone_digest="deadbeef", fs_static_adapter_digest="cafef00d",
    )
    payload = record.to_dict()

    assert payload["training_issue_ids"] == ["iss_000", "iss_001"]
    assert payload["data_role"] == "bank_fit"
    assert payload["n_updates"] == 2
    assert len(payload["losses"]) == 2
    assert payload["nonzero_response"]["nonzero"] is True
    assert payload["diversity_rule"] == bt.DIVERSITY_RULE_ID
    assert payload["diversity_group"]["expert_index"] == 0
    assert payload["fs_backbone_digest"] == "deadbeef"
    assert payload["fs_static_adapter_digest"] == "cafef00d"
    assert payload["eligibility_rules"] == list(bt.ELIGIBILITY_RULES)


def test_zero_initialized_bank_is_eligible_and_reported_as_such(
    fs_bridge, variables, objective
):
    """"未训练零库不是科学失败" -- a zero bank is normal init, not a failure."""
    bank = _bank(zero_init=True)
    registry = bt.build_bank_registry(K)
    plan = bt.expert_plan(registry, 0, hold_steps=1)
    x_norm = _training_samples(fs_bridge)[0].x_norm

    report = bt.nonzero_response_check(
        fs_bridge, bank, plan, x_norm, variables, expert_index=0, steps=1,
        target_blocks=TARGET_BLOCKS,
    )
    assert report["nonzero"] is False
    assert report["zero_initialized_untrained"] is True
    assert "not a failure" in report["note"]

    eligible, reason = bt.evaluate_eligibility([1.0, 0.9], [0.1, 0.1], 2)
    assert eligible is True and reason is None


# =============================================================================
# 2. Diversity grouping
# =============================================================================

def _row(
    issue: datetime,
    *,
    issue_id: str,
    lead_hours: int = 6,
    history_hours: int = 6,
    data_role: str = "bank_fit",
    split_id: str = "train",
) -> Dict[str, Any]:
    ts = int(issue.timestamp())
    return {
        "issue_id": issue_id,
        "issue_time": ts,
        "valid_time": ts + lead_hours * 3600,
        "history_time": ts - history_hours * 3600,
        "split_id": split_id,
        "data_role": data_role,
    }


def _four_month_rows() -> List[Dict[str, Any]]:
    return [
        _row(datetime(2015, month, 15, tzinfo=timezone.utc), issue_id=f"iss_{month:02d}")
        for month in (2, 5, 8, 11)
    ]


def test_diversity_groups_are_disjoint_and_cover_distinct_months():
    report = bt.assign_diversity_groups(_four_month_rows(), 4, hold_steps=4)

    assert report.rule == bt.DIVERSITY_RULE_ID
    assert len(report.groups) == 4
    assert report.n_assigned == 4
    assert report.purged == []

    months = [g.to_dict()["months"] for g in report.groups]
    assert months == [["2015-02"], ["2015-05"], ["2015-08"], ["2015-11"]]

    all_ids = [i for g in report.groups for i in g.issue_ids]
    assert len(all_ids) == len(set(all_ids)), "an issue_id landed in two groups"
    report.assert_disjoint()  # must not raise

    ordered = sorted(report.groups, key=lambda g: g.start_utc)
    for left, right in zip(ordered, ordered[1:]):
        assert left.end_utc <= right.start_utc


def test_sample_whose_window_straddles_a_group_boundary_is_purged():
    """History/target/hold overlap across a boundary is purged, not shared.

    The straddling row's analysis time is inside February but its 72h lead
    reaches into March, which belongs to a different expert's regime. Admitting
    it would make two experts' training windows overlap in time.
    """
    rows = _four_month_rows()
    straddler = _row(
        datetime(2015, 2, 27, 0, tzinfo=timezone.utc),
        issue_id="iss_straddle", lead_hours=72,
    )
    report = bt.assign_diversity_groups(rows + [straddler], 4, hold_steps=4)

    purged_ids = {p["issue_id"] for p in report.purged}
    assert "iss_straddle" in purged_ids
    offender = next(p for p in report.purged if p["issue_id"] == "iss_straddle")
    assert offender["reason"] == "group_boundary_overlap"
    assert "crosses this diversity group's boundary" in offender["detail"]

    assert report.n_assigned == 4
    assert all("iss_straddle" not in g.issue_ids for g in report.groups)
    report.assert_disjoint()


def test_sample_whose_history_reaches_into_the_previous_group_is_purged():
    """The purge is symmetric: a history endpoint before the block also counts."""
    rows = _four_month_rows()
    backward = _row(
        datetime(2015, 5, 1, 0, tzinfo=timezone.utc),
        issue_id="iss_backward", history_hours=12,
    )
    report = bt.assign_diversity_groups(rows + [backward], 4, hold_steps=4)
    offender = next(
        (p for p in report.purged if p["issue_id"] == "iss_backward"), None
    )
    assert offender is not None and offender["reason"] == "group_boundary_overlap"


def test_sample_in_the_split_guard_window_is_purged_by_the_reused_guard_logic():
    """`compute_guard_boundaries` / `is_in_guard_window` are reused, not re-invented.

    The train split ends 168h (the max lead) before 2019-01-01, so an issue time
    on 2018-12-31 lies in the guard window and must be purged before grouping.
    """
    rows = _four_month_rows()
    guarded = _row(
        datetime(2018, 12, 31, 0, tzinfo=timezone.utc), issue_id="iss_guarded"
    )
    report = bt.assign_diversity_groups(rows + [guarded], 4, hold_steps=4)

    offender = next(
        (p for p in report.purged if p["issue_id"] == "iss_guarded"), None
    )
    assert offender is not None, [p["issue_id"] for p in report.purged]
    assert offender["reason"] == "split_guard_window"
    assert all("iss_guarded" not in g.issue_ids for g in report.groups)


def test_grouping_refuses_a_sample_that_is_not_bank_fit():
    rows = _four_month_rows()
    rows.append(_row(
        datetime(2015, 6, 15, tzinfo=timezone.utc),
        issue_id="iss_dev", data_role="policy_dev",
    ))
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.assign_diversity_groups(rows, 4, hold_steps=4)
    assert excinfo.value.code == "BANK_FIT_ROLE_REQUIRED"


def test_grouping_refuses_to_manufacture_diversity_from_too_few_months():
    single_month = [
        _row(datetime(2015, 2, day, tzinfo=timezone.utc), issue_id=f"iss_{day}")
        for day in (10, 12, 14, 16)
    ]
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.assign_diversity_groups(single_month, 4, hold_steps=4)
    assert excinfo.value.code == "BANK_DIVERSITY_INSUFFICIENT_MONTHS"
    assert excinfo.value.detail["n_months"] == 1


def test_diversity_rule_is_documented_in_code():
    assert "fit-only" in bt.DIVERSITY_RULE_DOC or "bank_fit" in bt.DIVERSITY_RULE_DOC
    assert "compute_guard_boundaries" in bt.DIVERSITY_RULE_DOC
    assert bt.DIVERSITY_RULE_ID in bt.assign_diversity_groups(
        _four_month_rows(), 4, hold_steps=4
    ).to_dict()["rule"]


# =============================================================================
# 3. Registry: explicit no-edit + K singletons at a0, max_active = 1
# =============================================================================

def test_bank_registry_is_the_declared_candidate_set():
    registry = bt.build_bank_registry(K)
    summary = registry.validate()

    assert summary["n_entries"] == K + 1
    assert summary["reference_plan_id"] == "reference"
    assert registry.reference_entry.is_reference is True
    assert all(a == 0.0 for a in registry.reference_entry.coefficients)

    for entry in registry.candidate_entries():
        assert len(entry.support) == bt.DESIGN_MAX_ACTIVE == 1
        active = [a for a in entry.coefficients if a != 0.0]
        assert active == [pytest.approx(FIXED_A0)], entry.plan_id
        assert entry.rho == pytest.approx(0.25)
        assert entry.data_role == "bank_fit"


def test_bank_registry_refuses_a_weaker_realized_coefficient():
    """`single_expert_plans` clips with min(coefficient, rho); the registry checks.

    A nominal 0.1 under rho=0.25 realizes 0.1, not the specified a0=0.25. The
    registry stores the REALIZED value and refuses to file it as the a0
    candidate, so a silently weakened edit cannot be reported as the edit the
    design named.
    """
    from earthdelta.registry import CandidateRegistry

    registry = CandidateRegistry(name="probe", num_experts=K, rho=0.25)
    registry.register_reference(K)
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register_single_expert_plans(K, coefficient=0.1, expected_coefficient=0.25)
    assert excinfo.value.code == "REGISTRY_REALIZED_COEFFICIENT_MISMATCH"


def test_expert_plan_is_read_out_of_the_registry():
    registry = bt.build_bank_registry(K)
    plan = bt.expert_plan(registry, 2, hold_steps=bt.DESIGN_HOLD_STEPS)
    assert plan.plan_id == "expert_2"
    assert plan.coefficients == (0.0, 0.0, pytest.approx(FIXED_A0), 0.0)
    assert plan.hold_steps == 4
    assert plan.continuation == "reference_after_hold"


def test_max_active_guard_rejects_a_pair_candidate():
    from earthdelta.contracts import EditPlan
    from earthdelta.registry import CandidateRegistry

    registry = CandidateRegistry(name="pairs", num_experts=K, rho=0.25)
    registry.register_reference(K)
    registry.register(
        EditPlan("pair_01", K, (0.25, 0.25, 0.0, 0.0), rho=0.25), source="test"
    )
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.assert_max_active(registry, max_active=1)
    assert excinfo.value.code == "BANK_MAX_ACTIVE_EXCEEDED"


# =============================================================================
# 4. Eligibility rules are pre-declared and cannot be extended after the fact
# =============================================================================

@pytest.mark.parametrize("reason", bt.FORBIDDEN_EXCLUSION_REASONS)
def test_excluding_an_expert_for_its_gain_is_refused(reason):
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.assert_eligibility_rule_is_pre_declared(reason)
    assert excinfo.value.code == "BANK_EXCLUSION_REASON_FORBIDDEN"


def test_an_undeclared_exclusion_reason_is_refused():
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.assert_eligibility_rule_is_pre_declared("DID_NOT_LIKE_IT")
    assert excinfo.value.code == "BANK_EXCLUSION_REASON_UNKNOWN"


@pytest.mark.parametrize(
    "losses, grads, updates, expected",
    [
        ([1.0, 0.5], [0.1, 0.1], 2, None),
        ([1.0, float("nan")], [0.1, 0.1], 2, "NON_FINITE_LOSS"),
        ([1.0, 0.5], [0.1, float("inf")], 2, "NON_FINITE_GRADIENT"),
        ([], [], 0, "NO_UPDATES_RUN"),
    ],
)
def test_eligibility_rules_are_exactly_the_declared_ones(losses, grads, updates, expected):
    eligible, reason = bt.evaluate_eligibility(losses, grads, updates)
    assert reason == expected
    assert eligible is (expected is None)
    assert bt.assert_eligibility_rule_is_pre_declared(reason) == expected


@pytest.mark.parametrize(
    "mode, requested, cap", [("gradient_check", 33, 32), ("formal", 501, 500)]
)
def test_bank_update_caps_are_enforced(mode, requested, cap):
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.BankTrainConfig(mode=mode, max_updates=requested, train_steps=1,
                           hold_steps=1, lead_steps=(1,))
    assert excinfo.value.code == "BANK_UPDATE_CAP_EXCEEDED"
    assert excinfo.value.detail["cap"] == cap


# =============================================================================
# 5. Single-training-step profiler and the K/rank decision
# =============================================================================

def test_profiler_runs_and_returns_a_decision_object(fs_bridge, variables, objective):
    bank = _bank()
    sample = _training_samples(fs_bridge)[0]
    config = bt.BankTrainConfig(
        mode="gradient_check", max_updates=1, train_steps=1, hold_steps=1,
        lead_steps=(1,), num_experts=K, rank_per_expert=RANK,
        target_blocks=TARGET_BLOCKS,
    )
    profile = bt.profile_training_step(
        fs_bridge, bank, sample, objective, expert_index=0, config=config,
        variables=variables, n_measured_steps=1, synthetic=True,
    )
    assert profile.error is None
    assert profile.total_seconds > 0
    assert profile.trainable_parameters > 0
    assert profile.target_blocks == TARGET_BLOCKS
    assert profile.num_experts == K and profile.rank_per_expert == RANK
    assert profile.is_real_gpu_profile is False

    budget = bt.CapacityBudget(
        memory_budget_bytes=80 * 1024 ** 3, max_seconds_per_update=30.0
    )
    decision = bt.decide_bank_capacity(profile, budget)
    assert isinstance(decision, bt.BankCapacityDecision)
    payload = decision.to_dict()
    assert payload["basis"]["device"] == profile.device
    assert payload["rule"] == bt.CAPACITY_DECISION_RULE
    # A synthetic CPU profile may never be reported as the capacity finding.
    assert decision.provisional is True
    assert "real CUDA profile" in decision.provisional_reason


def test_capacity_decision_shrinks_only_when_the_measurement_says_so(
    fs_bridge, variables, objective
):
    """The decision is driven by the measured numbers, not by preference."""
    config = bt.BankTrainConfig(
        mode="gradient_check", max_updates=1, train_steps=1, hold_steps=1,
        lead_steps=(1,), num_experts=K, rank_per_expert=RANK,
        target_blocks=TARGET_BLOCKS,
    )
    profile = bt.profile_training_step(
        fs_bridge, _bank(), _training_samples(fs_bridge)[0], objective,
        expert_index=0, config=config, variables=variables, synthetic=True,
    )

    roomy = bt.decide_bank_capacity(
        profile,
        bt.CapacityBudget(memory_budget_bytes=1024 ** 5, max_seconds_per_update=1e6),
    )
    assert roomy.shrink_required is False
    assert roomy.decided_num_experts == bt.DESIGN_NUM_EXPERTS == 4
    assert roomy.decided_rank_per_expert == bt.DESIGN_RANK_PER_EXPERT == 4

    cramped = bt.decide_bank_capacity(
        profile,
        bt.CapacityBudget(memory_budget_bytes=1024, max_seconds_per_update=1e-9),
    )
    assert cramped.shrink_required is True
    assert cramped.decided_num_experts == bt.FALLBACK_NUM_EXPERTS == 2
    assert cramped.decided_rank_per_expert == bt.FALLBACK_RANK_PER_EXPERT == 2
    assert len(cramped.reasons) >= 2
    assert any("peak memory" in r for r in cramped.reasons)
    assert any("s/update" in r for r in cramped.reasons)


# =============================================================================
# 6. Horizon feasibility and the written fallback spec
# =============================================================================

def _horizon_sample(fs_bridge, max_step: int = 4) -> sa.TrainingSample:
    base = _training_samples(fs_bridge)[0]
    targets = {step: base.targets_raw[1] for step in range(1, max_step + 1)}
    return sa.TrainingSample(
        x_norm=base.x_norm, targets_raw=targets, issue_id=base.issue_id,
        issue_time=base.issue_time, valid_time=base.valid_time,
        history_time=base.history_time, split_id=base.split_id,
        data_role=base.data_role, source=base.source, time_utc=base.time_utc,
    )


def test_horizon_check_reports_feasible_when_the_full_hold_window_backprops(
    fs_bridge, variables, objective
):
    report = bt.check_horizon_feasibility(
        fs_bridge, _bank(), _horizon_sample(fs_bridge), objective,
        expert_index=0, plan_hold_steps=2, target_blocks=TARGET_BLOCKS,
        variables=variables,
    )
    assert report.feasible is True
    assert report.max_feasible_differentiable_steps == 2
    assert report.fallback is None
    assert all(m["ok"] for m in report.measurements)
    assert report.activation_checkpointing_enabled is False


def test_horizon_check_emits_a_fallback_spec_when_forced_to_a_tiny_step_limit(
    fs_bridge, variables, objective
):
    """Forced infeasibility must produce a WRITTEN spec, not a silent shrink."""
    report = bt.check_horizon_feasibility(
        fs_bridge, _bank(), _horizon_sample(fs_bridge), objective,
        expert_index=0, plan_hold_steps=4, target_blocks=TARGET_BLOCKS,
        variables=variables, max_differentiable_steps=1,
        updates_per_expert=500, num_experts_for_cost=4, n_eval_issues=128,
    )

    assert report.feasible is False
    assert report.max_feasible_differentiable_steps == 1
    assert report.requested_hold_steps == 4 and report.requested_hours == 24

    spec = report.fallback
    assert spec is not None
    payload = spec.to_dict()
    assert payload["train_steps"] == 1 and payload["train_hours"] == 6
    assert payload["designed_train_hours"] == 24
    assert payload["eval_hours"] == [24, 72]
    assert payload["eval_differentiable"] is False
    assert payload["activation_checkpointing_used"] is False

    # A changed training horizon is a changed experiment identity.
    assert payload["experiment_identity"] == bt.FALLBACK_EXPERIMENT_IDENTITY
    assert payload["superseded_identity"] == bt.DESIGNED_EXPERIMENT_IDENTITY
    assert payload["experiment_identity"] != payload["superseded_identity"]
    assert "must not be reported under" in payload["note"]

    # The cost estimate separates what was measured from what was assumed.
    cost = payload["cost_estimate"]
    assert set(cost) == {"measured", "assumed", "derived"}
    assert cost["measured"]["train_seconds_per_update"] is not None
    assert cost["assumed"]["updates_per_expert"] == 500
    assert cost["assumed"]["eval_is_non_differentiable"] is True
    assert cost["derived"]["total_gpu_hours"] is not None
    assert "not a hardware measurement" in cost["derived"]["caveat"]


def test_horizon_check_never_enables_activation_checkpointing(
    fs_bridge, variables, objective
):
    """The forbidden shortcut is asserted off before and after the measurement."""
    report = bt.check_horizon_feasibility(
        fs_bridge, _bank(), _horizon_sample(fs_bridge), objective,
        expert_index=0, plan_hold_steps=2, target_blocks=TARGET_BLOCKS,
        variables=variables,
    )
    assert report.activation_checkpointing_enabled is False
    assert report.activation_checkpoint_flags_seen == []
    assert report.supported_modes["activation_checkpoint_replay"] is False
    for _, module in fs_bridge.model.named_modules():
        for flag in ("use_checkpoint", "gradient_checkpointing", "grad_checkpointing"):
            assert getattr(module, flag, False) is False


def test_activation_checkpointing_assertion_actually_fires(fs_bridge):
    """The assertion is real: setting the flag makes it raise."""
    assert bt.assert_activation_checkpointing_disabled(fs_bridge.model) == []
    fs_bridge.model.blocks[0].use_checkpoint = True
    try:
        with pytest.raises(bt.BankTrainingViolation) as excinfo:
            bt.assert_activation_checkpointing_disabled(fs_bridge.model)
        assert excinfo.value.code == "HORIZON_ACTIVATION_CHECKPOINTING_ENABLED"
    finally:
        del fs_bridge.model.blocks[0].use_checkpoint
    assert bt.assert_activation_checkpointing_disabled(fs_bridge.model) == []
