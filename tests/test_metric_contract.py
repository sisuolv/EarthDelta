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
    effective_quadratic_weight,
    full_objective_gain,
    full_objective_loss,
    gain_analytic,
    gain_calibrated,
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


# ============================================================================
# B05: weighted_mse / weighted_rmse weight and scale validation
#
# Audit repro: weighted_mse(prediction=[1,2], target=0, weights=[2,-1]) used to
# return MSE = -2.0 with no error, because weighted_mse never applied the
# validation quadratic_gain applies via _validate_weights.
# ============================================================================

def test_weighted_mse_negative_weights_rejected():
    """B05: negative weights must raise, not silently yield a negative MSE."""
    prediction = torch.tensor([1.0, 2.0], dtype=DT)
    target = torch.zeros(2, dtype=DT)
    weights = torch.tensor([2.0, -1.0], dtype=DT)

    with pytest.raises(ValueError, match='non-negative'):
        weighted_mse(prediction, target, weights,
                     convention=WeightConvention.WEIGHTED_MEAN)

    # The same weights under WEIGHTED_SUM are equally invalid.
    with pytest.raises(ValueError, match='non-negative'):
        weighted_mse(prediction, target, weights,
                     convention=WeightConvention.WEIGHTED_SUM)

    # And the RMSE wrapper inherits the rejection.
    with pytest.raises(ValueError, match='non-negative'):
        weighted_rmse(prediction, target, weights,
                      convention=WeightConvention.WEIGHTED_MEAN)


def test_weighted_mse_valid_weights_still_correct():
    """B05 regression safety: a valid positive-weight case is unchanged.

    Known value: weights [2,1], prediction [1,2], target 0 gives
    (2*1 + 1*4) / 3 = 2.0 under WEIGHTED_MEAN and 6.0 under WEIGHTED_SUM.
    """
    prediction = torch.tensor([1.0, 2.0], dtype=DT)
    target = torch.zeros(2, dtype=DT)
    weights = torch.tensor([2.0, 1.0], dtype=DT)

    mean_mse = weighted_mse(prediction, target, weights,
                            convention=WeightConvention.WEIGHTED_MEAN)
    sum_mse = weighted_mse(prediction, target, weights,
                           convention=WeightConvention.WEIGHTED_SUM)

    torch.testing.assert_close(mean_mse, torch.tensor(2.0, dtype=DT))
    torch.testing.assert_close(sum_mse, torch.tensor(6.0, dtype=DT))


def test_weighted_mse_all_zero_weights_rejected():
    """B05: all-zero weights over a real target must raise, not NaN/eps-clamp."""
    prediction = torch.tensor([1.0, 2.0], dtype=DT)
    target = torch.zeros(2, dtype=DT)
    zero_weights = torch.zeros(2, dtype=DT)

    with pytest.raises(ValueError, match='positive sum'):
        weighted_mse(prediction, target, zero_weights,
                     convention=WeightConvention.WEIGHTED_MEAN)

    with pytest.raises(ValueError, match='positive sum'):
        weighted_mse(prediction, target, zero_weights,
                     convention=WeightConvention.WEIGHTED_SUM)


def test_weighted_mse_partially_zero_weights_allowed():
    """A legitimate mask (some zeros, not all) is fine and stays finite."""
    prediction = torch.tensor([1.0, 2.0, 3.0], dtype=DT)
    target = torch.zeros(3, dtype=DT)
    weights = torch.tensor([1.0, 0.0, 1.0], dtype=DT)

    mse = weighted_mse(prediction, target, weights,
                       convention=WeightConvention.WEIGHTED_MEAN)

    # (1*1 + 0*4 + 1*9) / 2 = 5.0
    torch.testing.assert_close(mse, torch.tensor(5.0, dtype=DT))


def test_weighted_mse_nonfinite_weights_rejected():
    """B05: NaN/Inf weights must raise."""
    prediction = torch.tensor([1.0, 2.0], dtype=DT)
    target = torch.zeros(2, dtype=DT)

    with pytest.raises(ValueError, match='finite'):
        weighted_mse(prediction, target, torch.tensor([1.0, float('nan')], dtype=DT),
                     convention=WeightConvention.WEIGHTED_MEAN)
    with pytest.raises(ValueError, match='finite'):
        weighted_mse(prediction, target, torch.tensor([1.0, float('inf')], dtype=DT),
                     convention=WeightConvention.WEIGHTED_MEAN)


def test_weighted_mse_nonpositive_scale_rejected():
    """B05: a zero or negative scale must raise (div-by-zero / sign flip)."""
    prediction = torch.tensor([2.0, 4.0], dtype=DT)
    target = torch.zeros(2, dtype=DT)

    with pytest.raises(ValueError, match='strictly positive'):
        weighted_mse(prediction, target, scale=torch.tensor([2.0, 0.0], dtype=DT),
                     convention=WeightConvention.WEIGHTED_MEAN)
    with pytest.raises(ValueError, match='strictly positive'):
        weighted_mse(prediction, target, scale=torch.tensor([2.0, -2.0], dtype=DT),
                     convention=WeightConvention.WEIGHTED_MEAN)


def test_weighted_mse_invalid_convention_rejected():
    """An unrecognized convention object must raise rather than fall through."""
    with pytest.raises(ValueError, match='convention'):
        weighted_mse(torch.ones(2, dtype=DT), torch.zeros(2, dtype=DT),
                     convention='weighted_mean')


# ============================================================================
# B06: full multi-dimensional [B,(K,)H,V,Lat,Lon] objective contract
# ============================================================================

def _two_lead_case():
    """Audit counter-example: 2 leads weighted 100:1, 2 disjoint candidates.

    error is 1 everywhere; candidate 0's response is 1 at lead 0 and 0 at
    lead 1, candidate 1 is the mirror image. Each candidate therefore removes
    the error at exactly one lead, and the correct combined gain must follow
    the 100:1 lead weighting.
    """
    error = torch.ones(1, 2, 1, 1, 1, dtype=DT)                 # [B,H,V,Lat,Lon]
    response = torch.zeros(1, 2, 2, 1, 1, 1, dtype=DT)          # [B,K,H,V,Lat,Lon]
    response[0, 0, 0] = 1.0   # candidate 0 acts on lead 0 (weight 100)
    response[0, 1, 1] = 1.0   # candidate 1 acts on lead 1 (weight 1)
    q = torch.tensor([100.0, 1.0], dtype=DT).reshape(2, 1, 1, 1)  # over (H,V,Lat,Lon)
    return error, response, q


def test_full_objective_gain_two_lead_100_to_1_discrimination():
    """B06: the 100:1 lead weighting must survive into the gain.

    A reducer that normalizes over the flattened feature axis first divides the
    per-lead scalar weight straight back out and reports 0.5 / 0.5. The full
    objective must report 100/101 and 1/101.
    """
    error, response, q = _two_lead_case()

    gain = full_objective_gain(error, response, q)

    expected = torch.tensor([[100.0 / 101.0, 1.0 / 101.0]], dtype=DT)
    assert gain.shape == (1, 2)
    torch.testing.assert_close(gain, expected, rtol=0.0, atol=1e-15)

    # This assertion is what discriminates against the buggy implementation.
    wrong = torch.tensor([[0.5, 0.5]], dtype=DT)
    assert not torch.allclose(gain, wrong, rtol=1e-3, atol=1e-3)

    # And demonstrate that the legacy per-feature reducer really does give 0.5.
    legacy_error = error.reshape(1, 2, 1, 1)                 # [B,H,S,F]
    legacy_response = response.reshape(1, 2, 2, 1, 1)        # [B,K,H,S,F]
    legacy_q = q.reshape(2, 1, 1)
    legacy = quadratic_gain(legacy_error, legacy_response, legacy_q,
                            convention=WeightConvention.WEIGHTED_MEAN).mean((2, 3))
    torch.testing.assert_close(legacy, wrong)


def test_full_objective_loss_two_lead_100_to_1_endpoints():
    """The same 100:1 case, checked through the loss endpoints instead."""
    error, response, q = _two_lead_case()
    target = torch.zeros_like(error)
    forecast = target - error  # so that target - forecast == error

    base = full_objective_loss(forecast, target, q)                     # [B]
    edited = full_objective_loss(forecast.unsqueeze(1) + response, target, q)  # [B,K]

    assert base.shape == (1,)
    assert edited.shape == (1, 2)
    expected = torch.tensor([[100.0 / 101.0, 1.0 / 101.0]], dtype=DT)
    torch.testing.assert_close(base[:, None] - edited, expected, rtol=0.0, atol=1e-15)


def test_full_objective_endpoint_difference_equals_analytic_gain():
    """B06 core identity: L(F) - L(F+u) == full_objective_gain(Y-F, u)."""
    torch.manual_seed(20260921)
    B, K, H, V, LAT, LON = 2, 3, 3, 2, 4, 5

    target = torch.randn(B, H, V, LAT, LON, dtype=DT)
    forecast = torch.randn(B, H, V, LAT, LON, dtype=DT)
    response = 0.3 * torch.randn(B, K, H, V, LAT, LON, dtype=DT)
    q = torch.rand(H, V, LAT, LON, dtype=DT) + 0.05
    scale = torch.rand(H, V, LAT, LON, dtype=DT) + 0.5

    analytic = full_objective_gain(target - forecast, response, q, scale=scale)
    base = full_objective_loss(forecast, target, q, scale=scale)
    edited = full_objective_loss(forecast.unsqueeze(1) + response, target, q, scale=scale)
    endpoint = base[:, None] - edited

    assert analytic.shape == (B, K)
    assert analytic.dtype == torch.float64
    torch.testing.assert_close(analytic, endpoint, rtol=1e-12, atol=1e-12)


def test_full_objective_endpoint_identity_without_candidate_axis():
    """The same identity for the K-free [B,H,V,Lat,Lon] response shape."""
    torch.manual_seed(7)
    B, H, V, LAT, LON = 2, 2, 3, 3, 4

    target = torch.randn(B, H, V, LAT, LON, dtype=DT)
    forecast = torch.randn(B, H, V, LAT, LON, dtype=DT)
    response = 0.5 * torch.randn(B, H, V, LAT, LON, dtype=DT)
    q = torch.rand(H, V, LAT, LON, dtype=DT) + 0.1

    analytic = full_objective_gain(target - forecast, response, q)
    endpoint = (full_objective_loss(forecast, target, q)
                - full_objective_loss(forecast + response, target, q))

    assert analytic.shape == (B,)
    torch.testing.assert_close(analytic, endpoint, rtol=1e-12, atol=1e-12)


def test_full_objective_loss_preserves_batch_and_candidate_axes():
    """The full objective must not collapse to a scalar."""
    torch.manual_seed(3)
    prediction = torch.randn(4, 5, 2, 3, 2, 2, dtype=DT)
    target = torch.randn(4, 2, 3, 2, 2, dtype=DT)

    loss = full_objective_loss(prediction, target)
    assert loss.shape == (4, 5)

    loss_no_k = full_objective_loss(prediction[:, 0], target)
    assert loss_no_k.shape == (4,)
    torch.testing.assert_close(loss[:, 0], loss_no_k)


def test_full_objective_loss_formula_matches_manual():
    """L == sum(q*((Y-F)/s)^2) / sum(q), normalized once over (H,V,Lat,Lon)."""
    torch.manual_seed(11)
    B, H, V, LAT, LON = 2, 2, 2, 3, 3
    prediction = torch.randn(B, H, V, LAT, LON, dtype=DT)
    target = torch.randn(B, H, V, LAT, LON, dtype=DT)
    q = torch.rand(H, V, LAT, LON, dtype=DT) + 0.1
    scale = torch.rand(H, V, LAT, LON, dtype=DT) + 0.5

    loss = full_objective_loss(prediction, target, q, scale=scale)

    manual = (q * ((target - prediction) / scale).square()).sum((1, 2, 3, 4)) / q.sum()
    torch.testing.assert_close(loss, manual, rtol=1e-13, atol=1e-13)


def test_effective_quadratic_weight_is_shared_and_inspectable():
    """Q_eff = diag(q/s^2)/sum(q), and it is the object the loss/gain use."""
    torch.manual_seed(5)
    B, K, H, V, LAT, LON = 2, 3, 2, 2, 3, 3
    q = torch.rand(H, V, LAT, LON, dtype=DT) + 0.1
    scale = torch.rand(H, V, LAT, LON, dtype=DT) + 0.5

    q_eff = effective_quadratic_weight(q, scale, (B, K, H, V, LAT, LON))

    torch.testing.assert_close(q_eff.normalizer, q.sum().expand(B, K).contiguous())
    expected_diag = (q / scale.square() / q.sum()).expand(B, K, H, V, LAT, LON)
    torch.testing.assert_close(q_eff.diagonal, expected_diag.contiguous())
    assert q_eff.objective_dims == (2, 3, 4, 5)
    assert q_eff.leading_shape == (B, K)

    # The gain is exactly the quadratic form of that same Q_eff.
    error = torch.randn(B, H, V, LAT, LON, dtype=DT)
    response = torch.randn(B, K, H, V, LAT, LON, dtype=DT)
    manual = (2 * q_eff.quadratic_form(error.unsqueeze(1), response)
              - q_eff.quadratic_form(response, response))
    torch.testing.assert_close(full_objective_gain(error, response, q, scale=scale), manual)


def test_gain_analytic_alias_matches_full_objective_gain():
    """gain_analytic is the explicitly-named analytic path (no extra terms)."""
    error, response, q = _two_lead_case()
    torch.testing.assert_close(gain_analytic(error, response, q),
                               full_objective_gain(error, response, q))


def test_gain_calibrated_defaults_to_analytic():
    """Calibration is OFF by default: no calibration tensor == analytic gain."""
    error, response, q = _two_lead_case()

    assert torch.equal(gain_calibrated(error, response, q),
                       gain_analytic(error, response, q))


def test_gain_calibrated_adds_caller_supplied_term():
    """A caller-supplied calibration tensor is added; it is never learned here."""
    error, response, q = _two_lead_case()
    calibration = torch.tensor([0.25, -0.5], dtype=DT)

    calibrated = gain_calibrated(error, response, q, calibration=calibration)
    analytic = gain_analytic(error, response, q)

    torch.testing.assert_close(calibrated, analytic + calibration)


def test_gain_calibrated_rejects_bad_calibration():
    """Calibration must be real floating, finite, and shape-compatible."""
    error, response, q = _two_lead_case()

    with pytest.raises(ValueError, match='non-finite'):
        gain_calibrated(error, response, q,
                        calibration=torch.tensor([float('nan'), 0.0], dtype=DT))
    with pytest.raises(ValueError, match='broadcastable'):
        gain_calibrated(error, response, q, calibration=torch.zeros(5, dtype=DT))
    with pytest.raises(ValueError, match='at most'):
        gain_calibrated(error, response, q, calibration=torch.zeros(1, 1, 2, dtype=DT))


def test_full_objective_partial_zero_weight_mask_is_finite():
    """A mask that zeroes SOME grid points must not produce NaN."""
    torch.manual_seed(13)
    B, K, H, V, LAT, LON = 2, 2, 2, 2, 4, 4
    target = torch.randn(B, H, V, LAT, LON, dtype=DT)
    forecast = torch.randn(B, H, V, LAT, LON, dtype=DT)
    response = torch.randn(B, K, H, V, LAT, LON, dtype=DT)

    q = torch.ones(H, V, LAT, LON, dtype=DT)
    q[..., :2, :] = 0.0   # mask out half the latitudes
    q[0, 0] = 0.0         # and one whole (lead, variable) slice

    gain = full_objective_gain(target - forecast, response, q)
    loss = full_objective_loss(forecast.unsqueeze(1) + response, target, q)

    assert torch.isfinite(gain).all()
    assert torch.isfinite(loss).all()

    # Masked points must contribute nothing: the same result is obtained by
    # zeroing the response wherever q is zero.
    masked_response = response * (q > 0).to(DT)
    torch.testing.assert_close(gain, full_objective_gain(target - forecast, masked_response, q))


def test_full_objective_total_zero_weight_for_one_batch_element_raises():
    """A target whose weights are ENTIRELY zero is rejected, not eps-clamped."""
    torch.manual_seed(17)
    B, H, V, LAT, LON = 3, 2, 2, 2, 2
    target = torch.randn(B, H, V, LAT, LON, dtype=DT)
    forecast = torch.randn(B, H, V, LAT, LON, dtype=DT)
    response = torch.randn(B, H, V, LAT, LON, dtype=DT)

    q = torch.ones(B, H, V, LAT, LON, dtype=DT)
    q[1] = 0.0  # batch element 1 has no weight at all

    with pytest.raises(ValueError, match='strictly positive sum'):
        full_objective_loss(forecast, target, q)
    with pytest.raises(ValueError, match='strictly positive sum'):
        full_objective_gain(target - forecast, response, q)

    # The same q with batch element 1 restored is accepted and finite.
    q[1] = 1.0
    assert torch.isfinite(full_objective_loss(forecast, target, q)).all()


def test_full_objective_all_zero_weights_raises():
    """A globally all-zero q is rejected for every target."""
    target = torch.randn(2, 2, 2, 2, 2, dtype=DT)
    forecast = torch.randn(2, 2, 2, 2, 2, dtype=DT)

    with pytest.raises(ValueError, match='strictly positive sum'):
        full_objective_loss(forecast, target, torch.zeros(2, 2, 2, 2, dtype=DT))


def test_full_objective_rejects_nonfinite_fields():
    """B06 entry validation: NaN/Inf in any field is rejected."""
    target = torch.randn(2, 2, 2, 2, 2, dtype=DT)
    forecast = torch.randn(2, 2, 2, 2, 2, dtype=DT)
    response = torch.randn(2, 2, 2, 2, 2, dtype=DT)

    bad = forecast.clone()
    bad[0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        full_objective_loss(bad, target)
    with pytest.raises(ValueError, match='non-finite'):
        full_objective_loss(forecast, bad)

    bad_response = response.clone()
    bad_response[0, 0, 0, 0, 0] = float('inf')
    with pytest.raises(ValueError, match='non-finite'):
        full_objective_gain(target - forecast, bad_response)
    with pytest.raises(ValueError, match='non-finite'):
        full_objective_gain(bad, response)


def test_full_objective_rejects_non_real_floating_fields():
    """Integer and complex fields are rejected (real floating required)."""
    target = torch.randn(2, 2, 2, 2, 2, dtype=DT)

    with pytest.raises(ValueError, match='real floating-point'):
        full_objective_loss(torch.zeros(2, 2, 2, 2, 2, dtype=torch.int64), target)
    with pytest.raises(ValueError, match='real floating-point'):
        full_objective_loss(torch.zeros(2, 2, 2, 2, 2, dtype=torch.complex128), target)


def test_full_objective_rejects_bad_scale():
    """B06 entry validation: scale must be finite and strictly positive."""
    target = torch.randn(2, 2, 2, 2, 2, dtype=DT)
    forecast = torch.randn(2, 2, 2, 2, 2, dtype=DT)
    q = torch.ones(2, 2, 2, 2, dtype=DT)

    with pytest.raises(ValueError, match='strictly positive'):
        full_objective_loss(forecast, target, q, scale=torch.zeros(2, 2, 2, 2, dtype=DT))
    with pytest.raises(ValueError, match='strictly positive'):
        full_objective_loss(forecast, target, q, scale=-torch.ones(2, 2, 2, 2, dtype=DT))
    with pytest.raises(ValueError, match='finite'):
        full_objective_loss(forecast, target, q,
                            scale=torch.full((2, 2, 2, 2), float('inf'), dtype=DT))


def test_full_objective_rejects_negative_q():
    """B06 entry validation: q must be non-negative and finite."""
    target = torch.randn(2, 2, 2, 2, 2, dtype=DT)
    forecast = torch.randn(2, 2, 2, 2, 2, dtype=DT)

    with pytest.raises(ValueError, match='non-negative'):
        full_objective_loss(forecast, target, -torch.ones(2, 2, 2, 2, dtype=DT))
    with pytest.raises(ValueError, match='finite'):
        full_objective_loss(forecast, target,
                            torch.full((2, 2, 2, 2), float('nan'), dtype=DT))


def test_full_objective_rejects_bad_ranks_and_shapes():
    """B06 entry validation: explicit rank/shape checks, no silent broadcast."""
    target = torch.randn(2, 2, 2, 2, 2, dtype=DT)
    forecast = torch.randn(2, 2, 2, 2, 2, dtype=DT)
    response = torch.randn(2, 3, 2, 2, 2, 2, dtype=DT)

    # Wrong rank for prediction.
    with pytest.raises(ValueError, match='prediction must be'):
        full_objective_loss(torch.randn(2, 2, 2, 2, dtype=DT), target)

    # Target neither matching nor the K-free field.
    with pytest.raises(ValueError, match='inconsistent'):
        full_objective_loss(forecast, torch.randn(2, 2, 2, 2, 3, dtype=DT))

    # error must be 5-D.
    with pytest.raises(ValueError, match='error must be'):
        full_objective_gain(torch.randn(2, 3, 2, 2, 2, 2, dtype=DT), response)

    # response rank must be 5 or 6.
    with pytest.raises(ValueError, match='response must be'):
        full_objective_gain(target, torch.randn(2, 2, 2, 2, dtype=DT))

    # response[0] / response[2:] must match error.
    with pytest.raises(ValueError, match='inconsistent'):
        full_objective_gain(target, torch.randn(2, 3, 2, 2, 2, 3, dtype=DT))

    # q that is not broadcastable over (H,V,Lat,Lon).
    with pytest.raises(ValueError, match='not broadcastable'):
        full_objective_loss(forecast, target, torch.ones(2, 2, 2, 5, dtype=DT))


def test_full_objective_rejects_ambiguous_weight_rank():
    """A 5-D q against a 6-D objective is ambiguous (B vs K) and is rejected."""
    target = torch.randn(2, 2, 2, 2, 2, dtype=DT)
    response = torch.randn(2, 2, 2, 2, 2, 2, dtype=DT)  # B == K == 2 on purpose
    ambiguous_q = torch.ones(2, 2, 2, 2, 2, dtype=DT)

    with pytest.raises(ValueError, match='at most 4 dims'):
        full_objective_gain(target, response, ambiguous_q)

    # Making the candidate axis explicit is accepted.
    explicit_q = ambiguous_q.unsqueeze(1)
    assert torch.isfinite(full_objective_gain(target, response, explicit_q)).all()


def test_full_objective_accumulates_in_float64_from_float32_inputs():
    """float32 inputs are accumulated in, and returned as, float64.

    The internal upcast rescues the accumulation, but it cannot rescue
    arithmetic the CALLER already performed in float32: forming ``F + u`` and
    ``Y - F`` in float32 rounds them before this module ever sees them, and that
    rounding alone is ~1e-7 relative. So the identity is only asserted tightly
    when the caller also forms those combinations in float64.
    """
    torch.manual_seed(23)
    target = torch.randn(2, 2, 2, 3, 3)
    forecast = torch.randn(2, 2, 2, 3, 3)
    response = 0.1 * torch.randn(2, 2, 2, 2, 3, 3)
    q = torch.rand(2, 2, 3, 3) + 0.1

    assert target.dtype == torch.float32

    gain = full_objective_gain(target - forecast, response, q)
    loss = full_objective_loss(forecast, target, q)
    assert gain.dtype == torch.float64
    assert loss.dtype == torch.float64

    # float32 caller arithmetic: identity holds only to float32 precision.
    endpoint32 = loss[:, None] - full_objective_loss(
        forecast.unsqueeze(1) + response, target, q)
    torch.testing.assert_close(gain, endpoint32, rtol=1e-5, atol=1e-6)

    # float64 caller arithmetic on the very same float32 values: the identity
    # is recovered to float64 precision.
    t64, f64, r64 = target.double(), forecast.double(), response.double()
    gain64 = full_objective_gain(t64 - f64, r64, q)
    endpoint64 = (full_objective_loss(f64, t64, q)[:, None]
                  - full_objective_loss(f64.unsqueeze(1) + r64, t64, q))
    torch.testing.assert_close(gain64, endpoint64, rtol=1e-12, atol=1e-12)
