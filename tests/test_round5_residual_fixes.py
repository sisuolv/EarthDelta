"""Round-5 regression tests for the remaining audit findings C07/C08/C11."""
from __future__ import annotations

import pytest
import torch

from earthdelta.contracts import EditPlan
from earthdelta.heads import ComposedPredictionHead
from earthdelta.metrics_contract import WeightConvention, quadratic_gain
from earthdelta.registry import CandidateRegistry
from earthdelta.selection import select_from_registry


def test_per_slice_zero_weight_is_rejected_before_division():
    error = torch.ones(1, 2, 1, 2)
    response = torch.ones(1, 1, 2, 1, 2)
    # The first [H,S] slice has positive mass, the second is all zero.  A
    # global sum check would pass and then create NaN in the second slice.
    weights = torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]])
    with pytest.raises(ValueError, match="every reduced feature slice"):
        quadratic_gain(error, response, weights, convention=WeightConvention.WEIGHTED_MEAN)


def test_registry_selection_preserves_tiny_nonzero_candidate_identity():
    registry = CandidateRegistry(name="tiny", num_experts=1)
    registry.register_reference(1)
    registry.register(
        EditPlan("expert_0", 1, (5e-7, )),
        source="fixed tiny candidate",
    )
    selected = select_from_registry(
        registry, torch.tensor([2.0], dtype=torch.float32),
        torch.eye(1, dtype=torch.float32), max_active=1, ridge=0.0,
    )
    assert selected.plan_id == "expert_0"
    assert selected.is_reference is False
    assert selected.surrogate.candidate_index == 1


def _head_inputs():
    torch.manual_seed(7)
    return (
        torch.randn(2, 2, 3, 4),  # history [B,L,S,F]
        torch.randn(2, 3, 3),      # memory [B,S,M]
        torch.randn(2, 2),         # edit descriptors [K,A]
        torch.tensor([6.0, 24.0]),
        torch.tensor([True, True]),
    )


def test_head_exposes_analytic_gain_and_does_not_calibrate_by_default():
    head = ComposedPredictionHead(4, 3, 2, 5, latent_dim=6)
    with torch.no_grad():
        head.gain_calibration.bias.fill_(0.5)
    out = head(*_head_inputs())
    torch.testing.assert_close(out["gain"], out["gain_analytic"])
    assert torch.any((out["gain_calibrated"] - out["gain_analytic"]).abs() > 0)


def test_head_calibration_requires_explicit_opt_in():
    head = ComposedPredictionHead(4, 3, 2, 5, latent_dim=6,
                                  use_gain_calibration=True)
    with torch.no_grad():
        head.gain_calibration.bias.fill_(0.5)
    out = head(*_head_inputs())
    torch.testing.assert_close(out["gain"], out["gain_calibrated"])
    assert torch.any((out["gain"] - out["gain_analytic"]).abs() > 0)
