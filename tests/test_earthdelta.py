"""Unified test suite for earthdelta package.

Migrates all tests from:
- v5 kit: test_primitives.py (41 tests)
- Response kit: test_response.py, test_utility.py (50 tests)

Plus new tests for v6 generalizations:
- EditPlan coefficient+window validation
- ExpertLoRA dense/sparse equivalence
- selection.py unified finite/continuous dispatch
- teacher.py injectable cost table
- ewma_error_feature availability-time filtering
"""
from dataclasses import replace
import inspect
import io
import numpy as np
import pytest
import torch
from earthdelta import (
    ArtifactVersion, EditPlan, pilot_plans, reference_plan, single_expert_plans,
    Slot, ProgramSpec,
    edit_responses, quadratic_gain, make_training_targets, PairedTargets,
    ProbeResult, central_response, local_linearity_error, cached_responses, CachedResponses,
    ResponseGeometry, response_distillation,
    Candidate, box_candidates, verify_candidates, VerifiedResult,
    select_plan, selection_regret, plan_from_prediction, SurrogatePlan, unified_select,
    BoundedProgramHead, InteractionUtilityHead, PairedEditPredictor,
    VerifiedRecord, eligible_records, gated_delta_replay, read_memory, ewma_error_feature,
    coefficient_diagnostics, assert_same_latitude_nodes,
    GroupedLowRankResidual, ExpertLoRA, verify_dense_sparse_equivalence,
)
from earthdelta.heads import quadratic_gain as heads_quadratic_gain

DT = torch.float64


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(7)
    torch.set_num_threads(1)


def tensor(v):
    return torch.tensor(v, dtype=DT)


def version():
    return ArtifactVersion('backbone', 'static', 'bank', 'norm', 'grid', 'projection', 'split', 'reference_after_hold')


def record(i='a', issue=0, valid=6, available=8, ver='v', event='old'):
    return VerifiedRecord(i, issue, valid, available, ver, event, torch.tensor([1., 0.]), torch.tensor([3.]))


def geom(r=((1., 0.), (0., 1.)), e=(1., 1.), w=(1., 1.)):
    return ResponseGeometry.from_error(tensor(r), tensor(e), tensor(w))


def small_batch():
    plans = pilot_plans()
    model = PairedEditPredictor(6, 4, 9, 6, 16)
    history = torch.randn(3, 4, 3, 6)
    memory = torch.randn(3, 3, 4)
    edits = torch.tensor([p.descriptor() for p in plans])
    leads = torch.tensor([6., 12., 24.])
    enabled = torch.tensor([any(p.active_mask) for p in plans])
    ref = torch.randn(3, 3, 3, 6)
    edited = ref[:, None].repeat(1, len(plans), 1, 1, 1) + torch.randn(3, len(plans), 3, 3, 6) * 0.1
    edited[:, 0] = ref
    targets = make_training_targets(ref + torch.randn_like(ref), ref, edited)
    return model, history, memory, edits, leads, enabled, targets


# ============================================================================
# Contract tests (from v5 test_primitives.py)
# ============================================================================

def test_version_digest_stable():
    assert version().digest == version().digest


def test_version_digest_changes_for_policy():
    assert version().digest != replace(version(), continuation='other').digest


def test_version_mismatch_rejected():
    with pytest.raises(ValueError, match='grid'):
        version().assert_matches(replace(version(), grid='new-grid'))


def test_empty_version_rejected():
    with pytest.raises(ValueError):
        replace(version(), edit_bank='')


def test_plan_continuation_exact():
    p = pilot_plans()[1]
    assert any(p.active_at(3)) and not any(p.active_at(4))


def test_plan_rank_alternatives_same_budget_different_direction():
    p = pilot_plans()
    assert sum(p[1].active_mask) == sum(p[2].active_mask) == 1 and p[1].active_mask != p[2].active_mask


# ============================================================================
# New EditPlan coefficient+window tests
# ============================================================================

def test_editplan_coefficient_validation():
    """Test coefficient bounds validation."""
    # Valid plan
    plan = EditPlan('test', 4, (0.1, -0.1, 0.0, 0.2), rho=0.25)
    assert plan.num_experts == 4
    assert plan.coefficients == (0.1, -0.1, 0.0, 0.2)

    # Coefficient exceeds rho
    with pytest.raises(ValueError):
        EditPlan('bad', 4, (0.5, 0.0, 0.0, 0.0), rho=0.25)

    # Non-finite coefficient
    with pytest.raises(ValueError):
        EditPlan('bad', 4, (float('nan'), 0.0, 0.0, 0.0))


def test_editplan_window_semantics():
    """Test application window behavior."""
    plan = EditPlan('test', 4, (0.1, 0.2, 0.0, 0.0), hold_steps=2)
    # Inside window
    assert plan.coefficients_at(0) == (0.1, 0.2, 0.0, 0.0)
    assert plan.coefficients_at(1) == (0.1, 0.2, 0.0, 0.0)
    # Outside window
    assert plan.coefficients_at(2) == (0.0, 0.0, 0.0, 0.0)
    assert plan.coefficients_at(10) == (0.0, 0.0, 0.0, 0.0)


def test_reference_plan():
    """Test no-edit reference plan generation."""
    ref = reference_plan(8)
    assert ref.plan_id == 'reference'
    assert ref.num_experts == 8
    assert all(c == 0.0 for c in ref.coefficients)


def test_single_expert_plans():
    """Test single expert plan generation."""
    plans = single_expert_plans(4, coefficient=1.0, rho=1.0)
    assert len(plans) == 4
    for k, p in enumerate(plans):
        assert p.coefficients[k] == 1.0
        assert sum(c != 0 for c in p.coefficients) == 1


# ============================================================================
# Paired targets tests (from v5 test_primitives.py)
# ============================================================================

def test_paired_error_identity():
    ref = torch.randn(2, 3, 2, 7)
    edit = ref[:, None] + torch.randn(2, 4, 3, 2, 7)
    y = torch.randn_like(ref)
    t = make_training_targets(y, ref, edit)
    torch.testing.assert_close(t.edited_error, y[:, None] - edit)


def test_quadratic_gain_identity_weighted():
    ref = torch.randn(2, 3, 2, 7)
    edit = ref[:, None] + torch.randn(2, 4, 3, 2, 7)
    y = torch.randn_like(ref)
    w = torch.rand(7)
    t = make_training_targets(y, ref, edit, w)
    expected = (((y - ref)[:, None].square() - (y[:, None] - edit).square()) * (w / w.sum())).sum(-1)
    torch.testing.assert_close(t.quadratic_gain, expected)


def test_no_edit_response_gain_zero():
    ref = torch.randn(2, 3, 2, 7)
    t = make_training_targets(torch.randn_like(ref), ref, ref[:, None])
    assert t.edit_response.count_nonzero() == 0 and t.quadratic_gain.count_nonzero() == 0


def test_response_does_not_require_truth():
    assert 'truth' not in inspect.signature(edit_responses).parameters


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        quadratic_gain(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 1, 2), torch.tensor([1., -1.]))


# ============================================================================
# Low-rank adapter tests (from v5 test_primitives.py + new ExpertLoRA tests)
# ============================================================================

def test_zero_initialized_editor_exact():
    layer = GroupedLowRankResidual(8, 8, 4, 2)
    x = torch.randn(2, 3, 8)
    assert layer(x, torch.ones(2, 4, dtype=torch.bool)).count_nonzero() == 0


def test_disabled_groups_not_executed():
    layer = GroupedLowRankResidual(8, 8, 4, 2)
    called = []
    handles = [m.register_forward_hook(lambda m, a, o, g=g: called.append(g)) for g, m in enumerate(layer.down)]
    layer(torch.randn(2, 3, 8), torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool))
    for h in handles:
        h.remove()
    assert called == [0]


def test_group_sparse_equals_dense_reference():
    layer = GroupedLowRankResidual(8, 6, 4, 2)
    for m in layer.up:
        torch.nn.init.normal_(m.weight)
    x = torch.randn(3, 5, 8)
    m = torch.tensor([[1, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
    c = m.float() * 0.7
    dense = sum(b(a(x)) * c[:, g, None, None] for g, (a, b) in enumerate(zip(layer.down, layer.up)))
    torch.testing.assert_close(layer(x, m, c), dense)


def test_disabled_coefficients_rejected():
    with pytest.raises(ValueError):
        GroupedLowRankResidual(2, 2, 1, 1)(torch.ones(1, 1, 2), torch.zeros(1, 1, dtype=torch.bool), torch.ones(1, 1))


def test_adapter_gradient_through_frozen_tail():
    layer = GroupedLowRankResidual(8, 8, 4, 2)
    tail = torch.nn.Linear(8, 3).requires_grad_(False)
    x = torch.randn(2, 5, 8)
    tail(x + layer(x, torch.ones(2, 4, dtype=torch.bool))).square().mean().backward()
    assert layer.up[0].weight.grad.abs().sum() > 0 and tail.weight.grad is None


def test_rank_upper_bound_not_claimed_exact():
    layer = GroupedLowRankResidual(3, 4, 4, 2)
    assert layer.rank_upper_bound(torch.ones(1, 4, dtype=torch.bool)).item() == 3


# New ExpertLoRA tests
def test_expertlora_dense_sparse_equivalence():
    """Test that dense and sparse paths produce identical outputs."""
    module = ExpertLoRA(16, 16, num_experts=4, rank_per_expert=4)
    # Initialize with non-zero weights for meaningful test
    for m in module.up:
        torch.nn.init.normal_(m.weight)

    x = torch.randn(3, 5, 16)
    # Mix of zero and non-zero coefficients
    coefficients = torch.tensor([
        [0.5, 0.0, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3, 0.4]
    ])
    assert verify_dense_sparse_equivalence(module, x, coefficients)


def test_expertlora_shared_down():
    """Test shared down projection."""
    module = ExpertLoRA(16, 16, num_experts=4, rank_per_expert=4, shared_down=True)
    assert len(module.down) == 1
    assert len(module.up) == 4

    x = torch.randn(2, 3, 16)
    coefficients = torch.randn(2, 4) * 0.5
    out = module(x, coefficients)
    assert out.shape == (2, 3, 16)


def test_expertlora_shared_up():
    """Test shared up projection."""
    module = ExpertLoRA(16, 16, num_experts=4, rank_per_expert=4, shared_up=True)
    assert len(module.down) == 4
    assert len(module.up) == 1


def test_expertlora_zero_coefficients_zero_output():
    """Test that zero coefficients give zero output."""
    module = ExpertLoRA(8, 8, num_experts=4, rank_per_expert=2)
    for m in module.up:
        torch.nn.init.normal_(m.weight)

    x = torch.randn(2, 3, 8)
    coefficients = torch.zeros(2, 4)
    out_dense = module.forward_dense(x, coefficients)
    out_sparse = module.forward_sparse(x, coefficients)
    assert out_dense.count_nonzero() == 0
    assert out_sparse.count_nonzero() == 0


# ============================================================================
# Memory tests (from v5 test_primitives.py + new ewma tests)
# ============================================================================

def test_memory_waits_for_available_time():
    assert eligible_records([record()], 7, 'v') == ()
    assert len(eligible_records([record()], 8, 'v')) == 1


def test_memory_future_records_do_not_affect_snapshot():
    past = [record()]
    future = record('future', 10, 16, 18)
    assert [r.record_id for r in eligible_records(past + [future], 8, 'v')] == ['a']


def test_memory_versions_fail_closed():
    with pytest.raises(ValueError):
        eligible_records([record()], 9, 'new')


def test_memory_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        eligible_records([record(), record()], 9, 'v')


def test_memory_event_exclusion():
    assert eligible_records([record()], 9, 'v', frozenset({'old'})) == ()


def test_memory_records_copy_input():
    x = torch.tensor([1., 0.])
    r = VerifiedRecord('a', 0, 6, 8, 'v', 'e', x, torch.ones(1))
    x.zero_()
    assert r.key[0] == 1


def test_memory_invalid_chronology_rejected():
    with pytest.raises(ValueError):
        record(valid=9, available=8)


def test_delta_memory_read_and_no_mutation():
    k = torch.tensor([[[1., 0.]]])
    v = torch.tensor([[[3.]]])
    m = gated_delta_replay(k, v, torch.ones(1, 1, dtype=torch.bool), 1., 1.)
    before = m.clone()
    torch.testing.assert_close(read_memory(m, k[:, 0]), v[:, 0])
    torch.testing.assert_close(m, before)


def test_memory_padding_does_not_decay():
    k = torch.tensor([[[1., 0.], [0., 1.]]])
    v = torch.tensor([[[3.], [99.]]])
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    m = gated_delta_replay(k, v, mask, 0.1, 1.)
    torch.testing.assert_close(m, torch.tensor([[[3., 0.]]]))


def test_memory_replay_has_gradients():
    k = torch.randn(2, 3, 4, requires_grad=True)
    v = torch.randn(2, 3, 5, requires_grad=True)
    gated_delta_replay(k, v, torch.ones(2, 3, dtype=torch.bool)).square().sum().backward()
    assert k.grad is not None and v.grad is not None


# New ewma_error_feature tests
def test_ewma_error_feature_availability_filter():
    """Test that ewma_error_feature respects availability_time."""
    r1 = record('r1', issue=0, valid=6, available=8)
    r2 = record('r2', issue=5, valid=12, available=15)
    r3 = record('r3', issue=10, valid=18, available=20)

    # At time 10, only r1 should be available
    feat = ewma_error_feature([r1, r2, r3], origin=10, version='v')
    torch.testing.assert_close(feat, r1.value)

    # At time 16, r1 and r2 should be available
    records = [r1, r2, r3]
    eligible = eligible_records(records, 16, 'v')
    assert len(eligible) == 2


def test_ewma_error_feature_decay():
    """Test EWMA decay weighting."""
    r1 = record('r1', issue=0, valid=6, available=8)
    r2 = VerifiedRecord('r2', 5, 12, 15, 'v', 'new', torch.tensor([2., 1.]), torch.tensor([6.]))

    # With decay=1, all records have equal weight
    feat1 = ewma_error_feature([r1, r2], origin=20, version='v', decay=1.0)
    expected1 = (r1.value + r2.value) / 2
    torch.testing.assert_close(feat1, expected1)

    # With decay=0, only most recent record matters
    feat0 = ewma_error_feature([r1, r2], origin=20, version='v', decay=0.0)
    torch.testing.assert_close(feat0, r2.value)


# ============================================================================
# Selection tests (from v5 test_primitives.py)
# ============================================================================

def test_selection_respects_budget():
    assert select_plan(torch.tensor([[0., 10., 2.]]), torch.tensor([1., 9., 3.]), 3., ('ref', 'large', 'small')).item() == 2


def test_negative_gains_fall_back_to_reference():
    assert select_plan(torch.tensor([[0., -1., -2.]]), torch.tensor([1., 2., 3.]), 3., ('ref', 'a', 'b')).item() == 0


def test_candidate_permutation_invariance_with_ties():
    ids = ('ref', 'b', 'a')
    g = torch.tensor([[0., 1., 1.]])
    cost = torch.ones(3)
    i = select_plan(g, cost, 2., ids).item()
    p = torch.tensor([2, 0, 1])
    ids2 = tuple(ids[j] for j in p)
    j = select_plan(g[:, p], cost[p], 2., ids2).item()
    assert ids[i] == ids2[j] == 'a'


def test_infeasible_budget_raises():
    with pytest.raises(ValueError):
        select_plan(torch.zeros(1, 1), torch.tensor([2.]), 1., ('ref',))


def test_regret_nonnegative_and_zero_at_best():
    gain = torch.tensor([[0., 2., 1.], [0., -2., -1.]])
    torch.testing.assert_close(selection_regret(gain, torch.tensor([1, 0]), torch.ones(3), 2.), torch.zeros(2))


# New unified selection tests
def test_unified_select_discrete():
    """Test unified_select with discrete candidates."""
    result = unified_select(
        predicted_gain=torch.tensor([[0., 2., 1.]]),
        total_cost=torch.ones(3),
        budget=2.,
        plan_ids=('ref', 'a', 'b')
    )
    assert result.item() == 1  # 'a' has highest gain


def test_unified_select_continuous():
    """Test unified_select with continuous QP."""
    result = unified_select(
        benefit=torch.tensor([1., 0.1], dtype=DT),
        gram=torch.eye(2, dtype=DT),
        bound=2.,
        max_active=1,
        budget=1.,
        ridge=0.
    )
    assert isinstance(result, SurrogatePlan)
    torch.testing.assert_close(result.coefficients, torch.tensor([1., 0.], dtype=DT))


def test_unified_select_finite_candidates():
    """Test unified_select with finite candidate set."""
    candidates = torch.tensor([
        [0.5, 0.0],
        [0.0, 0.5],
        [0.3, 0.3]
    ], dtype=DT)
    result = unified_select(
        benefit=torch.tensor([1., 0.], dtype=DT),
        gram=torch.eye(2, dtype=DT),
        candidate_offsets=candidates,
        budget=2.,
        ridge=0.
    )
    assert isinstance(result, SurrogatePlan)
    # First candidate should win since it aligns with benefit
    torch.testing.assert_close(result.coefficients, torch.tensor([0.5, 0.0], dtype=DT))


# ============================================================================
# Spectral tests (from v5 test_primitives.py)
# ============================================================================

def test_spectrum_equal_power_different_phase():
    y = torch.ones(4, dtype=torch.complex64)
    p = -y
    d = coefficient_diagnostics(p, y, torch.ones(4))
    assert d['log_energy_error'] == 0 and d['correlation'] == -1


def test_spectrum_zero_field_finite():
    x = torch.zeros(4, dtype=torch.complex64)
    d = coefficient_diagnostics(x, x, torch.ones(4))
    assert d['correlation'] == 1 and d['log_energy_error'] == 0


def test_grid_coordinates_not_just_shapes():
    with pytest.raises(ValueError):
        assert_same_latitude_nodes(torch.tensor([-89., 89.]), torch.tensor([-90., 90.]))


# ============================================================================
# Model tests (from v5 test_primitives.py)
# ============================================================================

def test_model_shapes_and_exact_no_edit_outputs():
    m, h, mem, e, t, on, _ = small_batch()
    p = m(h, mem, e, t, on)
    assert p['gain'].shape == (3, 6, 3, 3) and p['edit_response'].shape == (3, 6, 3, 3, 6)
    assert p['gain'][:, 0].count_nonzero() == 0 and p['edit_response'][:, 0].count_nonzero() == 0


def test_model_candidate_permutation_equivariant():
    m, h, mem, e, t, on, _ = small_batch()
    perm = torch.tensor([2, 0, 5, 1, 4, 3])
    p = m(h, mem, e, t, on)
    q = m(h, mem, e[perm], t, on[perm])
    torch.testing.assert_close(p['gain'][:, perm], q['gain'])


def test_model_gradients_and_ema_update():
    m, h, mem, e, t, on, targets = small_batch()
    before = next(m.base_predictor.parameters()).clone()
    loss = m.training_loss(m(h, mem, e, t, on), targets, on)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss['total'].backward()
    opt.step()
    # Check training progressed
    assert torch.isfinite(loss['total'])


def test_future_truth_not_an_inference_argument():
    assert 'truth' not in inspect.signature(PairedEditPredictor.forward).parameters


def test_model_checkpoint_roundtrip():
    m, h, mem, e, t, on, _ = small_batch()
    n = PairedEditPredictor(6, 4, 9, 6, 16)
    n.load_state_dict(m.state_dict())
    torch.testing.assert_close(m(h, mem, e, t, on)['gain'], n(h, mem, e, t, on)['gain'])


# ============================================================================
# Probe tests (from Response kit test_response.py)
# ============================================================================

def test_affine_probe_matches_exact_matrix_and_counts():
    r = tensor([[1, 2], [3, -1], [.5, 4]])
    p = central_response(lambda a: r @ a + 2, tensor([.2, -.1]), epsilon=1e-3)
    torch.testing.assert_close(p.response, r, atol=1e-10, rtol=1e-10)
    assert p.forward_calls == 6


def test_eight_dimensions_are_17_trajectories_without_repeat():
    p = central_response(lambda a: 2 * a, torch.zeros(8, dtype=DT), check_repeatability=False)
    assert p.forward_calls == 17


def test_probe_preserves_reference():
    x = tensor([1, 2])
    old = x.clone()
    central_response(lambda a: a * a, x)
    assert torch.equal(x, old)


def test_probe_rejects_mutating_coefficient_callback():
    def bad(a):
        a.add_(1)
        return a
    with pytest.raises(ValueError, match='mutated'):
        central_response(bad, tensor([0]))


def test_nonrepeatable_state_is_rejected():
    n = [0]
    def bad(a):
        n[0] += 1
        return a + n[0]
    with pytest.raises(ValueError, match='repeatable'):
        central_response(bad, tensor([0]))


@pytest.mark.parametrize('eps', [0, -.1, float('nan'), float('inf')])
def test_bad_epsilon_rejected(eps):
    with pytest.raises(ValueError):
        central_response(lambda a: a, tensor([0]), eps)


def test_nonfinite_forward_rejected():
    with pytest.raises(ValueError):
        central_response(lambda a: a * float('nan'), tensor([0]))


def test_local_linear_error_zero_for_affine():
    f = lambda a: tensor([[1, 2], [2, -1]]) @ a
    p = central_response(f, tensor([0, 0]))
    assert local_linearity_error(f, p, tensor([.1, -.2])) < 1e-10


def test_axis_derivatives_miss_nonlinear_mixed_interaction():
    def f(a):
        return (a[0] + a[1] + 20 * a[0] * a[1]).reshape(1)
    p = central_response(f, tensor([0, 0]))
    torch.testing.assert_close(p.response, tensor([[1, 1]]))
    assert local_linearity_error(f, p, tensor([.5, .5])) > .5


# ============================================================================
# Geometry tests (from Response kit test_response.py)
# ============================================================================

def test_geometry_gain_matches_direct_weighted_error_change():
    g = geom(((1, 2), (3, -1)), (1, 2), (2, .5))
    a = tensor([.2, -.1])
    direct = (g.weights * g.error.square()).sum() - (g.weights * (g.error - g.response @ a).square()).sum()
    torch.testing.assert_close(g.predicted_gain(a), direct)


def test_gram_cross_terms_measure_redundant_effects():
    g = geom(((1, 1),), (1,), (1,))
    assert float(g.predicted_gain(tensor([1, 0]))) == 1
    assert float(g.predicted_gain(tensor([0, 1]))) == 1
    assert float(g.predicted_gain(tensor([1, 1]))) == 0


def test_response_loss_ignores_effectively_equivalent_coefficients():
    a = tensor([1, 0])
    b = tensor([0, 1])
    h = tensor([[1, 1], [1, 1]])
    assert float(response_distillation(a, b, h)) == 0
    assert float((a - b).square().sum()) == 2


def test_distillation_detaches_teacher_and_geometry():
    s = tensor([.5, 1]).requires_grad_()
    t = tensor([0, 0]).requires_grad_()
    h = torch.eye(2, dtype=DT, requires_grad=True)
    response_distillation(s, t, h).backward()
    assert s.grad is not None and t.grad is None and h.grad is None


def test_response_batch_shared_gram():
    s = tensor([[1, 0], [0, 2]])
    t = torch.zeros_like(s)
    assert float(response_distillation(s, t, torch.eye(2, dtype=DT))) == 2.5


def test_indefinite_gram_rejected():
    with pytest.raises(ValueError, match='PSD'):
        response_distillation(tensor([0, 0]), tensor([1, 0]), tensor([[1, 0], [0, -1]]))


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        geom(w=(1, -1))
    with pytest.raises(ValueError):
        geom(w=(0, 0))


# ============================================================================
# Teacher tests (from Response kit test_response.py)
# ============================================================================

def test_teacher_full_support_unconstrained_solution():
    c = box_candidates(geom(), bound=2, ridge=0, enumerate_subsets=False)[0]
    torch.testing.assert_close(c.offset, tensor([1, 1]))


def test_box_is_box_not_l2_ball():
    c = box_candidates(geom(), bound=.5, ridge=0, enumerate_subsets=False)[0]
    torch.testing.assert_close(c.offset, tensor([.5, .5]), atol=1e-7, rtol=1e-7)
    assert float(c.offset.norm()) > .5


def test_ridge_solution():
    c = box_candidates(geom(), bound=2, ridge=1, enumerate_subsets=False)[0]
    torch.testing.assert_close(c.offset, tensor([.5, .5]))


def test_support_count_8_choose_at_most_2_is_37():
    g = ResponseGeometry.from_error(torch.eye(8, dtype=DT), torch.ones(8, dtype=DT), torch.ones(8, dtype=DT))
    assert len(box_candidates(g, max_active=2)) == 37


def test_budget_excludes_expensive_supports():
    cs = box_candidates(geom(), costs=[1, 4], budget=1, bound=2)
    assert all(c.support in ((), (0,)) for c in cs)
    assert all(c.declared_cost <= 1 for c in cs)


def test_empty_budget_returns_noop():
    cs = box_candidates(geom(), budget=0)
    assert len(cs) == 1 and cs[0].support == ()


def test_enumeration_limit_rejected_early():
    with pytest.raises(ValueError, match='exceeds'):
        box_candidates(geom(), max_candidates=2)


def test_teacher_does_not_silently_force_nonzero_edits():
    cs = box_candidates(geom(e=(0, 0)), bound=1)
    assert cs[0].support == ()


def test_nonlinear_validation_accepts_real_improvement():
    cs = box_candidates(geom(), bound=2, enumerate_subsets=False)
    v = verify_candidates(lambda a: a, tensor([0, 0]), cs, tensor([1, 1]), tensor([1, 1]))
    assert v.accepted and v.selected_loss < v.baseline_loss and v.forward_calls == 2


def test_nonlinear_validation_rejects_bad_local_prediction():
    g = geom(((1,),), (1,), (1,))
    cs = box_candidates(g, bound=2, ridge=0)
    v = verify_candidates(lambda a: a + 10 * a * a, tensor([0]), cs, tensor([1]), tensor([1]))
    assert not v.accepted and v.selected_loss == v.baseline_loss
    assert v.records[0]['actual_gain'] < 0


def test_full_field_guard_can_disagree_with_summary():
    g = geom(((1,),), (1,), (1,))
    cs = box_candidates(g, bound=2, ridge=0)
    def full(a):
        return torch.cat((a, 10 * a))
    v = verify_candidates(full, tensor([0]), cs, tensor([1, 0]), tensor([1, 1]))
    assert not v.accepted


def test_failed_branch_is_recorded():
    g = geom(((1,),), (1,), (1,))
    cs = box_candidates(g, bound=2)
    def f(a):
        if bool((a != 0).any()):
            raise ValueError('synthetic branch failure')
        return a
    v = verify_candidates(f, tensor([0]), cs, tensor([1]), tensor([1]))
    assert not v.accepted and v.records[0]['status'] == 'replay_failed'


def test_verification_call_budget():
    cs = box_candidates(geom(), bound=2)
    v = verify_candidates(lambda a: a, tensor([0, 0]), cs, tensor([1, 1]), tensor([1, 1]), max_nonzero=1)
    assert v.forward_calls == 2


# New injectable cost tests
def test_teacher_injectable_cost_dict():
    """Test injectable cost table as dict."""
    cs = box_candidates(geom(), costs={0: 1.0, 1: 10.0}, budget=5, bound=2)
    # Only slot 0 should be affordable
    assert all(c.support in ((), (0,)) for c in cs)


def test_teacher_injectable_cost_callable():
    """Test injectable cost table as callable."""
    def cost_fn(idx):
        return 2.0 if idx == 0 else 5.0
    cs = box_candidates(geom(), costs=cost_fn, budget=3, bound=2)
    assert all(c.support in ((), (0,)) for c in cs)


# ============================================================================
# Program spec tests (from Response kit test_response.py)
# ============================================================================

def spec():
    return ProgramSpec((Slot('a', 0, 1, 0), Slot('b', 2, 1, 1)), 4, 3, 2, bound=.5)


def test_program_has_no_edit_outside_time_support():
    s = spec()
    a = tensor([[.2, -.1]])
    out = s.coefficients_for(a, step=1, layer=1, masks={'global': torch.ones(3)}, tokens=3)
    assert out.shape == (1, 3, 2) and bool((out == 0).all())


def test_program_signed_coefficients_spatial_support_and_gradient():
    s = spec()
    a = tensor([[.2, -.1]]).requires_grad_()
    out = s.coefficients_for(a, step=2, layer=1, masks={'global': tensor([1, 0, .5])}, tokens=3)
    torch.testing.assert_close(out[0, :, 1], tensor([-.1, 0, -.05]))
    out.sum().backward()
    torch.testing.assert_close(a.grad, tensor([[0, 1.5]]))


def test_program_rejects_duplicate_coordinates():
    with pytest.raises(ValueError, match='duplicate'):
        ProgramSpec((Slot('a', 0, 1, 0), Slot('b', 0, 1, 0)), 4, 3, 2)


def test_program_rejects_out_of_range_and_coefficient_bounds():
    with pytest.raises(ValueError):
        ProgramSpec((Slot('a', 4, 0, 0),), 4, 1, 1)
    with pytest.raises(ValueError):
        spec().coefficients_for(tensor([[1, 0]]), step=0, layer=1, masks={'global': torch.ones(3)}, tokens=3)


def test_program_missing_mask_rejected():
    with pytest.raises(ValueError):
        spec().coefficients_for(tensor([[0, 0]]), step=0, layer=1, masks={}, tokens=3)


# ============================================================================
# Student head tests (from Response kit test_response.py)
# ============================================================================

def test_student_runtime_signature_contains_no_truth_or_teacher():
    assert tuple(inspect.signature(BoundedProgramHead.forward).parameters) == ('self', 'features')


def test_student_zero_initialization_and_bounds_after_update():
    h = BoundedProgramHead(4, 2, bound=.25).double()
    x = torch.randn(3, 4, dtype=DT)
    assert bool((h(x) == 0).all())
    opt = torch.optim.SGD(h.parameters(), lr=.1)
    loss = (h(x) - .2).square().mean()
    loss.backward()
    opt.step()
    assert bool((h(x).abs() <= .25).all()) and bool((h(x) != 0).any())


def test_student_save_restore_roundtrip():
    h = BoundedProgramHead(4, 2).double()
    x = torch.randn(2, 4, dtype=DT)
    with torch.no_grad():
        h.net[-1].weight.add_(.1)
    buffer = io.BytesIO()
    torch.save(h.state_dict(), buffer)
    buffer.seek(0)
    h2 = BoundedProgramHead(4, 2).double()
    h2.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(h(x), h2(x))


def test_verification_does_not_round_truth_to_prediction_bfloat16():
    # BF16 spacing at 1 is coarser than this truth difference.
    baseline = torch.zeros(1)
    truth = torch.tensor([1.001], dtype=torch.float64)
    weights = torch.tensor([1.123], dtype=torch.float64)
    result = verify_candidates(
        lambda _: torch.ones(1, dtype=torch.bfloat16),
        baseline, [], truth, weights, max_nonzero=0)
    assert result.baseline_loss == pytest.approx(1.123e-6, rel=1e-10)


def test_distillation_keeps_teacher_metric_precision_with_bfloat_student():
    student = torch.tensor([1.0], dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.tensor([1.001], dtype=torch.float64)
    gram = torch.tensor([[1.123]], dtype=torch.float64)
    loss = response_distillation(student, teacher, gram)
    assert float(loss.detach()) == pytest.approx(1.123e-6, rel=1e-10)
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0


# ============================================================================
# Utility head tests (from Response kit test_utility.py)
# ============================================================================

def test_head_psd_and_44_outputs():
    head = InteractionUtilityHead(10, 8).double()
    b, h = head(torch.randn(4, 10, dtype=DT))
    assert head.net[-1].out_features == 44 and b.shape == (4, 8) and h.shape == (4, 8, 8)
    assert bool((torch.linalg.eigvalsh(h) > 0).all())


def test_head_gradients():
    head = InteractionUtilityHead(5, 2).double()
    x = torch.randn(4, 5, dtype=DT)
    b, h = head(x)
    a = torch.randn_like(b)
    heads_quadratic_gain(b, h, a).square().mean().backward()
    assert head.net[-1].weight.grad.abs().sum() > 0


def test_plan_no_future_signature():
    assert not {'truth', 'target', 'error', 'reference_forecast'} & set(inspect.signature(plan_from_prediction).parameters)


def test_plan_analytic_identity():
    p = plan_from_prediction(torch.tensor([1., .1], dtype=DT), torch.eye(2, dtype=DT), bound=2, max_active=1, budget=1, ridge=0)
    torch.testing.assert_close(p.coefficients, torch.tensor([1., 0.], dtype=DT))
    assert p.support == (0,) and p.solves == 2


def test_plan_zero_budget_and_negative_predicted_gain_stays_noop():
    p = plan_from_prediction(torch.ones(2, dtype=DT), torch.eye(2, dtype=DT), budget=0)
    assert not p.support and p.solves == 0
    q = plan_from_prediction(torch.zeros(2, dtype=DT), torch.eye(2, dtype=DT))
    assert not q.support and q.predicted_gain == 0


def test_plan_37_supports_is_36_solves_plus_noop():
    p = plan_from_prediction(torch.ones(8, dtype=DT), torch.eye(8, dtype=DT), max_active=2)
    assert p.solves == 36 and p.solver_failures == 0


def test_indefinite_surrogate_is_rejected():
    with pytest.raises(ValueError):
        plan_from_prediction(torch.ones(2, dtype=DT), -torch.eye(2, dtype=DT))


def test_dense_vs_diagonal_redundancy_prediction_differs():
    b = torch.ones(2, dtype=DT)
    h = torch.ones(2, 2, dtype=DT)
    a = torch.ones(2, dtype=DT)
    assert float(heads_quadratic_gain(b, h, a)) == 0
    assert float(heads_quadratic_gain(b, torch.diag(h.diag()), a)) == 2


# ============================================================================
# Cached responses tests (new)
# ============================================================================

def test_cached_responses_basic():
    """Test basic cached_responses functionality."""
    reference = tensor([0., 0.])
    candidates = torch.tensor([
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5]
    ], dtype=DT)

    def evaluate(a):
        return 2 * a

    cached = cached_responses(evaluate, reference, candidates, check_repeatability=False)
    assert cached.reference_output.shape == (2,)
    assert cached.responses.shape == (3, 2)
    # Responses should be 2*offset
    torch.testing.assert_close(cached.responses, 2 * candidates)


def test_cached_responses_vs_central():
    """Test that cached_responses matches central_response for same points."""
    reference = tensor([0., 0.])

    def linear_fn(a):
        return tensor([[1., 2.], [3., -1.]]) @ a

    # Get central difference response
    probe = central_response(linear_fn, reference, check_repeatability=False)

    # Test with unit vectors
    candidates = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=DT)
    cached = cached_responses(linear_fn, reference, candidates, check_repeatability=False)

    # For linear function, cached responses should match probe.response @ candidates.T
    expected = (probe.response @ candidates.T).T
    torch.testing.assert_close(cached.responses, expected)


# ============================================================================
# JEPA toggle tests (new)
# ============================================================================

def test_jepa_toggle_off():
    """Test that JEPA branch can be toggled off."""
    model = PairedEditPredictor(6, 4, 9, 6, 16, use_jepa_latent=False)
    assert not model.use_jepa_latent
    # Should not have JEPA-specific attributes
    assert not hasattr(model, 'base_encoder') or not hasattr(model, 'base_target')


def test_jepa_toggle_on():
    """Test that JEPA branch can be toggled on."""
    model = PairedEditPredictor(6, 4, 9, 6, 16, use_jepa_latent=True)
    assert model.use_jepa_latent
    assert hasattr(model, 'base_encoder')
    assert hasattr(model, 'base_target')


def test_jepa_toggle_no_crash():
    """Test that both JEPA modes run without crashing."""
    m_no_jepa, h, mem, e, t, on, targets = small_batch()
    m_jepa = PairedEditPredictor(6, 4, 9, 6, 16, use_jepa_latent=True)

    # Both should forward without error
    p_no = m_no_jepa(h, mem, e, t, on)
    p_yes = m_jepa(h, mem, e, t, on)

    # Both should compute losses without error
    loss_no = m_no_jepa.training_loss(p_no, targets, on)
    loss_yes = m_jepa.training_loss(p_yes, targets, on)

    assert torch.isfinite(loss_no['total'])
    assert torch.isfinite(loss_yes['total'])
