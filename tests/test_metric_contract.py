"""Tests for earthdelta.metrics_contract module.

Covers the canonical quadratic_gain function, both weight conventions,
RMSE aggregation order correctness, and MetricSpec validation.
"""
import pytest
import torch
import numpy as np

from earthdelta.metrics_contract import (
    WeightConvention,
    MetricSpec,
    quadratic_gain,
    quadratic_gain_from_benefit_gram,
    quadratic_gain_numpy,
    weighted_mse,
    weighted_rmse,
)

DT = torch.float64


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    np.random.seed(42)


# ============================================================================
# WeightConvention tests
# ============================================================================

def test_weight_convention_enum_values():
    """Test that weight convention enum has expected values."""
    assert WeightConvention.WEIGHTED_SUM.name == 'WEIGHTED_SUM'
    assert WeightConvention.WEIGHTED_MEAN.name == 'WEIGHTED_MEAN'


# ============================================================================
# MetricSpec tests
# ============================================================================

def test_metricspec_valid_construction():
    """Test valid MetricSpec construction."""
    spec = MetricSpec(
        variable_order=('Z500', 'T850'),
        lead_hours=(6, 12, 24),
        q_hsf=torch.ones(3, 2, 2),
        weight_convention=WeightConvention.WEIGHTED_MEAN,
        missing_policy='error',
        utc_time_convention=True,
        units='normalized',
    )
    assert spec.variable_order == ('Z500', 'T850')
    assert spec.lead_hours == (6, 12, 24)
    assert spec.weight_convention == WeightConvention.WEIGHTED_MEAN


def test_metricspec_invalid_missing_policy():
    """Test that invalid missing_policy is rejected."""
    with pytest.raises(ValueError, match='missing_policy'):
        MetricSpec(
            variable_order=('Z500',),
            lead_hours=(6,),
            q_hsf=None,
            weight_convention=WeightConvention.WEIGHTED_SUM,
            missing_policy='invalid',
        )


def test_metricspec_invalid_units():
    """Test that invalid units is rejected."""
    with pytest.raises(ValueError, match='units'):
        MetricSpec(
            variable_order=('Z500',),
            lead_hours=(6,),
            q_hsf=None,
            weight_convention=WeightConvention.WEIGHTED_SUM,
            missing_policy='error',
            units='invalid',
        )


def test_metricspec_negative_weights_rejected():
    """Test that negative weights are rejected."""
    with pytest.raises(ValueError, match='non-negative'):
        MetricSpec(
            variable_order=('Z500',),
            lead_hours=(6,),
            q_hsf=torch.tensor([-1.0, 1.0]),
            weight_convention=WeightConvention.WEIGHTED_SUM,
            missing_policy='error',
        )


def test_metricspec_empty_variable_order_rejected():
    """Test that empty variable_order is rejected."""
    with pytest.raises(ValueError, match='variable_order'):
        MetricSpec(
            variable_order=(),
            lead_hours=(6,),
            q_hsf=None,
            weight_convention=WeightConvention.WEIGHTED_SUM,
            missing_policy='error',
        )


# ============================================================================
# quadratic_gain tests
# ============================================================================

def test_quadratic_gain_weighted_mean_uniform_weights():
    """Test that WEIGHTED_MEAN with uniform weights equals mean reduction."""
    error = torch.randn(2, 3, 2, 7, dtype=DT)
    response = torch.randn(2, 4, 3, 2, 7, dtype=DT)

    # Using WEIGHTED_MEAN with uniform (None) weights
    gain = quadratic_gain(error, response, None, convention=WeightConvention.WEIGHTED_MEAN)

    # Manual calculation with mean
    expected = (2 * error[:, None] * response - response.square()).mean(-1)
    torch.testing.assert_close(gain, expected)


def test_quadratic_gain_weighted_mean_nonuniform_weights():
    """Test WEIGHTED_MEAN with non-uniform weights normalizes correctly."""
    error = torch.randn(2, 3, 2, 7, dtype=DT)
    response = torch.randn(2, 4, 3, 2, 7, dtype=DT)
    weights = torch.rand(7, dtype=DT) + 0.1  # Ensure positive

    gain = quadratic_gain(error, response, weights, convention=WeightConvention.WEIGHTED_MEAN)

    # Manual calculation: normalize weights to sum to 1
    w_normalized = weights / weights.sum()
    w_broadcast = w_normalized.expand_as(error)
    expected = ((2 * error[:, None] * response - response.square()) * w_broadcast[:, None]).sum(-1)
    torch.testing.assert_close(gain, expected)


def test_quadratic_gain_weighted_sum():
    """Test WEIGHTED_SUM does not normalize weights."""
    error = torch.randn(2, 3, 2, 7, dtype=DT)
    response = torch.randn(2, 4, 3, 2, 7, dtype=DT)
    weights = torch.rand(7, dtype=DT) + 0.1

    gain = quadratic_gain(error, response, weights, convention=WeightConvention.WEIGHTED_SUM)

    # Manual calculation: weights NOT normalized
    w_broadcast = weights.expand_as(error)
    expected = ((2 * error[:, None] * response - response.square()) * w_broadcast[:, None]).sum(-1)
    torch.testing.assert_close(gain, expected)


def test_quadratic_gain_convention_required():
    """Test that omitting convention raises an error."""
    error = torch.randn(2, 3, 2, 7, dtype=DT)
    response = torch.randn(2, 4, 3, 2, 7, dtype=DT)

    with pytest.raises(TypeError):
        # convention is keyword-only, so passing without keyword should fail
        quadratic_gain(error, response, None, WeightConvention.WEIGHTED_MEAN)


def test_quadratic_gain_shape_validation():
    """Test shape validation for error and response."""
    error = torch.randn(2, 3, 2, 7, dtype=DT)

    # Wrong response dimensions
    with pytest.raises(ValueError, match='5-D'):
        quadratic_gain(error, torch.randn(2, 3, 2, 7, dtype=DT),
                       convention=WeightConvention.WEIGHTED_MEAN)

    # Mismatched shapes
    with pytest.raises(ValueError, match='inconsistent'):
        quadratic_gain(error, torch.randn(2, 4, 3, 2, 8, dtype=DT),
                       convention=WeightConvention.WEIGHTED_MEAN)


def test_quadratic_gain_nonfinite_rejected():
    """Test that non-finite inputs are rejected."""
    error = torch.randn(2, 3, 2, 7, dtype=DT)
    error[0, 0, 0, 0] = float('nan')
    response = torch.randn(2, 4, 3, 2, 7, dtype=DT)

    with pytest.raises(ValueError, match='non-finite'):
        quadratic_gain(error, response, convention=WeightConvention.WEIGHTED_MEAN)


def test_quadratic_gain_identity_holds():
    """Test the FSO identity: ||e0||^2 - ||e_u||^2 = 2<e0,du> - ||du||^2."""
    error = torch.randn(2, 3, 2, 7, dtype=DT)
    response = torch.randn(2, 4, 3, 2, 7, dtype=DT)
    weights = torch.rand(7, dtype=DT) + 0.1

    # e_u = e_0 - du
    edited_error = error[:, None] - response

    # Gain via identity
    gain = quadratic_gain(error, response, weights, convention=WeightConvention.WEIGHTED_MEAN)

    # Direct computation: (||e0||^2 - ||e_u||^2) in weighted norm
    w_norm = weights / weights.sum()
    # Broadcast weights correctly for each tensor shape
    baseline_sq = (error.square() * w_norm).sum(-1)  # [B, H, S]
    # For edited_error [B, K, H, S, F], broadcast w_norm to [1, 1, 1, 1, F]
    edited_sq = (edited_error.square() * w_norm).sum(-1)  # [B, K, H, S]
    direct_gain = baseline_sq[:, None] - edited_sq  # [B, K, H, S]

    torch.testing.assert_close(gain, direct_gain)


# ============================================================================
# quadratic_gain_from_benefit_gram tests
# ============================================================================

def test_quadratic_gain_from_benefit_gram_unbatched():
    """Test coefficient-space gain computation (unbatched)."""
    d = 4
    benefit = torch.randn(d, dtype=DT)
    gram = torch.eye(d, dtype=DT)
    program = torch.randn(d, dtype=DT)

    gain = quadratic_gain_from_benefit_gram(benefit, gram, program)

    expected = 2 * (benefit * program).sum() - program @ gram @ program
    torch.testing.assert_close(gain, expected)


def test_quadratic_gain_from_benefit_gram_batched():
    """Test coefficient-space gain computation (batched)."""
    b, d = 3, 4
    benefit = torch.randn(b, d, dtype=DT)
    gram = torch.eye(d, dtype=DT)
    program = torch.randn(b, d, dtype=DT)

    gain = quadratic_gain_from_benefit_gram(benefit, gram, program)

    # Manual per-batch computation
    expected = torch.zeros(b, dtype=DT)
    for i in range(b):
        expected[i] = 2 * (benefit[i] * program[i]).sum() - program[i] @ gram @ program[i]
    torch.testing.assert_close(gain, expected)


# ============================================================================
# quadratic_gain_numpy tests
# ============================================================================

def test_quadratic_gain_numpy():
    """Test numpy version of coefficient-space gain."""
    d = 4
    benefit = np.random.randn(d)
    gram = np.eye(d)
    program = np.random.randn(d)

    gain = quadratic_gain_numpy(benefit, gram, program)

    expected = float(2 * benefit @ program - program @ gram @ program)
    assert abs(gain - expected) < 1e-10


def test_quadratic_gain_numpy_nonfinite_rejected():
    """Test that non-finite numpy inputs are rejected."""
    d = 4
    benefit = np.random.randn(d)
    benefit[0] = np.nan
    gram = np.eye(d)
    program = np.random.randn(d)

    with pytest.raises(ValueError, match='non-finite'):
        quadratic_gain_numpy(benefit, gram, program)


# ============================================================================
# RMSE aggregation order tests
# ============================================================================

def test_weighted_mse_weighted_mean():
    """Test weighted MSE with WEIGHTED_MEAN convention."""
    prediction = torch.tensor([1.0, 2.0, 3.0], dtype=DT)
    target = torch.tensor([1.1, 2.2, 3.3], dtype=DT)
    weights = torch.tensor([1.0, 2.0, 1.0], dtype=DT)

    mse = weighted_mse(prediction, target, weights, convention=WeightConvention.WEIGHTED_MEAN)

    # Manual: sum(w * (p-t)^2) / sum(w)
    diff_sq = (prediction - target).square()
    expected = (weights * diff_sq).sum() / weights.sum()
    torch.testing.assert_close(mse, expected)


def test_weighted_mse_weighted_sum():
    """Test weighted MSE with WEIGHTED_SUM convention."""
    prediction = torch.tensor([1.0, 2.0, 3.0], dtype=DT)
    target = torch.tensor([1.1, 2.2, 3.3], dtype=DT)
    weights = torch.tensor([1.0, 2.0, 1.0], dtype=DT)

    mse = weighted_mse(prediction, target, weights, convention=WeightConvention.WEIGHTED_SUM)

    # Manual: sum(w * (p-t)^2) without normalization
    diff_sq = (prediction - target).square()
    expected = (weights * diff_sq).sum()
    torch.testing.assert_close(mse, expected)


def test_weighted_rmse_aggregation_order():
    """Test that RMSE is sqrt(mean(sq_errors)), NOT mean(sqrt(sq_errors)).

    This is the critical test for correct RMSE aggregation order.
    """
    prediction = torch.tensor([1.0, 5.0], dtype=DT)
    target = torch.tensor([0.0, 0.0], dtype=DT)

    # Correct: sqrt(mean([1, 25])) = sqrt(13) ~ 3.606
    rmse_correct = weighted_rmse(prediction, target, convention=WeightConvention.WEIGHTED_MEAN)
    expected_correct = torch.sqrt(torch.tensor(13.0, dtype=DT))
    torch.testing.assert_close(rmse_correct, expected_correct)

    # Wrong order would be: mean([sqrt(1), sqrt(25)]) = mean([1, 5]) = 3.0
    wrong_order = torch.tensor(3.0, dtype=DT)
    assert not torch.allclose(rmse_correct, wrong_order)


def test_weighted_rmse_with_scale():
    """Test RMSE with scale normalization."""
    prediction = torch.tensor([2.0, 4.0], dtype=DT)
    target = torch.tensor([0.0, 0.0], dtype=DT)
    scale = torch.tensor([2.0, 2.0], dtype=DT)  # Divide by scale before squaring

    rmse = weighted_rmse(prediction, target, scale=scale, convention=WeightConvention.WEIGHTED_MEAN)

    # After scaling: errors are [1, 2], sq = [1, 4], mean = 2.5, rmse = sqrt(2.5)
    expected = torch.sqrt(torch.tensor(2.5, dtype=DT))
    torch.testing.assert_close(rmse, expected)


# ============================================================================
# Backward compatibility tests
# ============================================================================

def test_quadratic_gain_matches_paired_py_weighted_mean():
    """Test that canonical gain matches original paired.py implementation.

    This ensures bit-identical behavior for the weighted mean path.
    """
    from earthdelta.paired import quadratic_gain as paired_quadratic_gain

    error = torch.randn(2, 3, 2, 7, dtype=DT)
    response = torch.randn(2, 4, 3, 2, 7, dtype=DT)
    weights = torch.rand(7, dtype=DT) + 0.1

    # Original paired.py (now uses canonical with WEIGHTED_MEAN)
    gain_paired = paired_quadratic_gain(error, response, weights)

    # Direct canonical call
    gain_canonical = quadratic_gain(error, response, weights,
                                     convention=WeightConvention.WEIGHTED_MEAN)

    torch.testing.assert_close(gain_paired, gain_canonical)


def test_quadratic_gain_matches_heads_py_coefficient_space():
    """Test that canonical coefficient-space gain matches heads.py."""
    from earthdelta.heads import quadratic_gain as heads_quadratic_gain

    benefit = torch.randn(3, 4, dtype=DT)
    gram = torch.eye(4, dtype=DT).expand(3, 4, 4)
    program = torch.randn(3, 4, dtype=DT)

    # Original heads.py (now uses canonical)
    gain_heads = heads_quadratic_gain(benefit, gram, program)

    # Direct canonical call
    gain_canonical = quadratic_gain_from_benefit_gram(benefit, gram, program)

    torch.testing.assert_close(gain_heads, gain_canonical)


def test_quadratic_gain_matches_geometry_py():
    """Test that canonical gain matches geometry.py ResponseGeometry."""
    from earthdelta.geometry import ResponseGeometry

    response = torch.randn(10, 4, dtype=DT)
    error = torch.randn(10, dtype=DT)
    weights = torch.rand(10, dtype=DT) + 0.1

    geom = ResponseGeometry.from_error(response, error, weights)
    offset = torch.randn(4, dtype=DT)

    # Geometry's predicted_gain (now uses canonical)
    gain_geom = geom.predicted_gain(offset)

    # Direct canonical call
    gain_canonical = quadratic_gain_from_benefit_gram(geom.benefit, geom.gram, offset.double())

    torch.testing.assert_close(gain_geom, gain_canonical)


def test_quadratic_gain_matches_selection_py():
    """Test that canonical numpy gain matches selection.py."""
    from earthdelta.selection import quadratic_gain_numpy as selection_qg

    d = 4
    benefit = np.random.randn(d)
    gram = np.eye(d)
    program = np.random.randn(d)

    # Selection's quadratic_gain_numpy (now uses canonical)
    gain_selection = selection_qg(benefit, gram, program)

    # Direct canonical call
    gain_canonical = quadratic_gain_numpy(benefit, gram, program)

    assert abs(gain_selection - gain_canonical) < 1e-10
