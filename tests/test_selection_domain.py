"""Tests for finite-candidate branch hardening in earthdelta.selection.

Covers the new validation checks:
- Finiteness (NaN/Inf rejection)
- Bound enforcement per-candidate
- max_active support-size check
- max_candidates cap enforcement
"""
import pytest
import torch
import numpy as np

from earthdelta.selection import (
    plan_from_prediction,
    SurrogatePlan,
)

DT = torch.float64


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    np.random.seed(42)


def make_valid_inputs(d: int = 4):
    """Create valid benefit/gram for testing."""
    benefit = torch.randn(d, dtype=DT)
    gram = torch.eye(d, dtype=DT)  # Identity is symmetric PSD
    return benefit, gram


# ============================================================================
# Finiteness tests
# ============================================================================

def test_finite_candidates_nan_rejected():
    """Test that candidates containing NaN are rejected."""
    benefit, gram = make_valid_inputs(4)
    candidates = torch.randn(10, 4, dtype=DT)
    candidates[3, 2] = float('nan')  # Inject NaN

    with pytest.raises(ValueError, match='non-finite'):
        plan_from_prediction(benefit, gram, candidate_offsets=candidates)


def test_finite_candidates_inf_rejected():
    """Test that candidates containing Inf are rejected."""
    benefit, gram = make_valid_inputs(4)
    candidates = torch.randn(10, 4, dtype=DT)
    candidates[5, 1] = float('inf')  # Inject Inf

    with pytest.raises(ValueError, match='non-finite'):
        plan_from_prediction(benefit, gram, candidate_offsets=candidates)


def test_finite_candidates_neginf_rejected():
    """Test that candidates containing -Inf are rejected."""
    benefit, gram = make_valid_inputs(4)
    candidates = torch.randn(10, 4, dtype=DT)
    candidates[7, 0] = float('-inf')  # Inject -Inf

    with pytest.raises(ValueError, match='non-finite'):
        plan_from_prediction(benefit, gram, candidate_offsets=candidates)


def test_finite_candidates_valid_passes():
    """Test that valid finite candidates pass."""
    benefit, gram = make_valid_inputs(4)
    candidates = torch.tensor([
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0],
        [0.1, 0.1, 0.0, 0.0],
    ], dtype=DT)

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.5, max_active=2, budget=10.0)
    assert isinstance(result, SurrogatePlan)


# ============================================================================
# Bound enforcement tests
# ============================================================================

def test_bound_enforcement_rejects_over_bound():
    """Test that candidates exceeding bound are rejected (not selected)."""
    d = 4
    # Benefit favors first dimension
    benefit = torch.tensor([10.0, 0.0, 0.0, 0.0], dtype=DT)
    gram = torch.eye(d, dtype=DT)

    # First candidate exceeds bound, second is within bound
    candidates = torch.tensor([
        [0.5, 0.0, 0.0, 0.0],  # Over bound (0.25)
        [0.2, 0.0, 0.0, 0.0],  # Within bound
    ], dtype=DT)

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.25, max_active=4, budget=10.0)

    # Should select the second candidate (within bound)
    assert result.coefficients[0].item() == pytest.approx(0.2, abs=1e-10)


def test_bound_enforcement_all_over_returns_noop():
    """Test that if all candidates exceed bound, returns no-op."""
    d = 4
    benefit = torch.tensor([10.0, 0.0, 0.0, 0.0], dtype=DT)
    gram = torch.eye(d, dtype=DT)

    # All candidates exceed bound
    candidates = torch.tensor([
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
    ], dtype=DT)

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.25, max_active=4, budget=10.0)

    # Should return no-op (all zeros)
    assert result.support == ()
    assert result.predicted_gain == 0.0


def test_bound_enforcement_negative_coefficients():
    """Test bound enforcement for negative coefficients."""
    d = 4
    benefit = torch.tensor([-10.0, 0.0, 0.0, 0.0], dtype=DT)  # Favors negative
    gram = torch.eye(d, dtype=DT)

    candidates = torch.tensor([
        [-0.5, 0.0, 0.0, 0.0],  # Over bound (|a| > 0.25)
        [-0.2, 0.0, 0.0, 0.0],  # Within bound
    ], dtype=DT)

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.25, max_active=4, budget=10.0)

    # Should select the second candidate
    assert result.coefficients[0].item() == pytest.approx(-0.2, abs=1e-10)


# ============================================================================
# max_active support-size tests
# ============================================================================

def test_max_active_rejects_large_support():
    """Test that candidates with too many nonzero coefficients are rejected."""
    d = 4
    benefit = torch.ones(d, dtype=DT)
    gram = torch.eye(d, dtype=DT)

    candidates = torch.tensor([
        [0.1, 0.1, 0.1, 0.0],  # 3 active (too many for max_active=2)
        [0.1, 0.1, 0.0, 0.0],  # 2 active (ok)
    ], dtype=DT)

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.5, max_active=2, budget=10.0)

    # Should select the second candidate (support size <= max_active)
    assert len(result.support) <= 2


def test_max_active_all_over_returns_noop():
    """Test that if all candidates exceed max_active, returns no-op."""
    d = 4
    benefit = torch.ones(d, dtype=DT)
    gram = torch.eye(d, dtype=DT)

    candidates = torch.tensor([
        [0.1, 0.1, 0.1, 0.0],  # 3 active
        [0.1, 0.1, 0.1, 0.1],  # 4 active
    ], dtype=DT)

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.5, max_active=2, budget=10.0)

    # Should return no-op
    assert result.support == ()


def test_max_active_zero_allowed():
    """Test that max_active=0 means no edits allowed (only no-op valid)."""
    d = 4
    benefit = torch.ones(d, dtype=DT)
    gram = torch.eye(d, dtype=DT)

    candidates = torch.tensor([
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
    ], dtype=DT)

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.5, max_active=0, budget=10.0)

    # Should return no-op
    assert result.support == ()


# ============================================================================
# max_candidates cap tests
# ============================================================================

def test_max_candidates_cap_enforced():
    """Test that exceeding max_candidates raises an error."""
    benefit, gram = make_valid_inputs(4)
    candidates = torch.randn(100, 4, dtype=DT) * 0.1  # 100 candidates

    with pytest.raises(ValueError, match='exceeds max_candidates'):
        plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                             max_candidates=50)  # Cap at 50


def test_max_candidates_at_limit_ok():
    """Test that exactly max_candidates is allowed."""
    benefit, gram = make_valid_inputs(4)
    candidates = torch.randn(50, 4, dtype=DT) * 0.1  # Exactly 50 candidates

    # Should not raise
    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.5, max_active=4, max_candidates=50, budget=10.0)
    assert isinstance(result, SurrogatePlan)


def test_max_candidates_under_limit_ok():
    """Test that under max_candidates is allowed."""
    benefit, gram = make_valid_inputs(4)
    candidates = torch.randn(10, 4, dtype=DT) * 0.1  # 10 candidates

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.5, max_active=4, max_candidates=100, budget=10.0)
    assert isinstance(result, SurrogatePlan)


def test_max_candidates_error_message_informative():
    """Test that the max_candidates error message is informative."""
    benefit, gram = make_valid_inputs(4)
    candidates = torch.randn(200, 4, dtype=DT) * 0.1

    with pytest.raises(ValueError) as excinfo:
        plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                             max_candidates=100)

    # Check error message contains both actual and limit
    assert 'K=200' in str(excinfo.value)
    assert '100' in str(excinfo.value)


# ============================================================================
# Combined hardening tests
# ============================================================================

def test_all_hardening_checks_combined():
    """Test that all hardening checks work together correctly."""
    d = 4
    benefit = torch.tensor([1.0, 0.5, 0.2, 0.1], dtype=DT)
    gram = torch.eye(d, dtype=DT)

    # Create candidates with various issues
    candidates = torch.tensor([
        [0.1, 0.0, 0.0, 0.0],        # Valid: 1 active, within bound
        [0.5, 0.0, 0.0, 0.0],        # Invalid: over bound (0.25)
        [0.1, 0.1, 0.1, 0.0],        # Invalid: 3 active > max_active=2
        [0.15, 0.15, 0.0, 0.0],      # Valid: 2 active, within bound
    ], dtype=DT)

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.25, max_active=2, max_candidates=100, budget=10.0)

    # Should select either first or fourth candidate (both valid)
    # Fourth has higher gain due to activating both dimensions
    assert len(result.support) <= 2
    assert all(abs(c) <= 0.25 + 1e-12 for c in result.coefficients.tolist())


def test_solver_failures_count_includes_rejections():
    """Test that solver_failures includes bound/support rejections."""
    d = 4
    benefit = torch.ones(d, dtype=DT)
    gram = torch.eye(d, dtype=DT)

    # All candidates invalid
    candidates = torch.tensor([
        [0.5, 0.0, 0.0, 0.0],        # Over bound
        [0.1, 0.1, 0.1, 0.0],        # Over max_active
    ], dtype=DT)

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.25, max_active=2, max_candidates=100, budget=10.0)

    # All were rejected, so solver_failures should be 2
    assert result.solver_failures == 2


# ============================================================================
# Edge cases
# ============================================================================

def test_empty_support_candidates_skipped():
    """Test that all-zero candidates (empty support) are skipped."""
    d = 4
    benefit = torch.ones(d, dtype=DT)
    gram = torch.eye(d, dtype=DT)

    candidates = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],        # Empty support
        [0.1, 0.0, 0.0, 0.0],        # Valid
    ], dtype=DT)

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.5, max_active=2, budget=10.0)

    # Should select the second candidate
    assert result.support == (0,)
    assert result.coefficients[0].item() == pytest.approx(0.1)


def test_budget_constraint_still_enforced():
    """Test that budget constraint is enforced alongside new checks."""
    d = 4
    benefit = torch.ones(d, dtype=DT)
    gram = torch.eye(d, dtype=DT)

    # Unit costs: activating 3 dimensions costs 3
    candidates = torch.tensor([
        [0.1, 0.1, 0.1, 0.0],        # Cost 3, exceeds budget
        [0.1, 0.1, 0.0, 0.0],        # Cost 2, within budget
    ], dtype=DT)

    result = plan_from_prediction(benefit, gram, candidate_offsets=candidates,
                                  bound=0.5, max_active=4, budget=2.5)

    # Should select second candidate (budget-feasible)
    assert len(result.support) == 2
