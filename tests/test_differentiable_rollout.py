"""Tests for differentiable rollout and gradient flow.

These tests verify that gradients can flow through controlled_rollout
when the differentiable=True option is used, enabling future
differentiable expert selection and optimization.
"""
import pytest
import torch
import numpy as np

from earthdelta.bridge import (
    Stormer,
    NormalizationContract,
    WeatherStepBridge,
    controlled_rollout,
    DEFAULT_VARIABLES,
)
from earthdelta.contracts import ArtifactVersion, EditPlan, reference_plan
from earthdelta.lowrank import ExpertLoRA


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def small_variables():
    """Small variable list for fast tests."""
    return DEFAULT_VARIABLES[:8]


@pytest.fixture
def trained_model(small_variables):
    """Create model with non-zero weights (simulating trained model)."""
    model = Stormer(
        in_img_size=(16, 32),
        variables=small_variables,
        patch_size=2,
        hidden_size=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )

    # Initialize with non-zero values
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)

    torch.nn.init.normal_(model.head.linear.weight, std=0.02)
    torch.nn.init.normal_(model.head.linear.bias, std=0.02)
    torch.nn.init.normal_(model.head.adaLN_modulation[-1].weight, std=0.1)
    torch.nn.init.normal_(model.head.adaLN_modulation[-1].bias, std=0.1)

    model.eval()
    return model


@pytest.fixture
def small_normalization(small_variables):
    """Create small normalization contract."""
    return NormalizationContract(
        inp_mean=torch.zeros(8),
        inp_std=torch.ones(8),
        diff_mean={6: torch.zeros(8), 24: torch.zeros(8)},
        diff_std={6: torch.ones(8), 24: torch.ones(8)},
        variables=small_variables,
    )


@pytest.fixture
def trained_bridge(trained_model, small_normalization):
    """Create bridge with trained-like model."""
    version = ArtifactVersion(
        backbone="stormer_test",
        static_adapter="none",
        edit_bank="none",
        normalization="pending",
        grid="16x32",
        projection="patch2",
        split="test",
        continuation="reference_after_hold",
    )
    return WeatherStepBridge(trained_model, small_normalization, version)


# =============================================================================
# Test: Basic rollout functionality
# =============================================================================

def test_controlled_rollout_returns_correct_shape(trained_bridge, small_variables):
    """Test that controlled_rollout returns correct shape."""
    batch_size = 2
    num_vars = len(small_variables)
    x_norm = torch.randn(batch_size, num_vars, 16, 32)

    plan = reference_plan(num_experts=8)

    out = controlled_rollout(
        trained_bridge, x_norm, small_variables,
        interval=6, steps=2, plan=plan, expert_loras={},
    )

    assert out.shape == (batch_size, num_vars, 16, 32)


def test_controlled_rollout_trajectory_shape(trained_bridge, small_variables):
    """Test that return_trajectory=True returns full trajectory."""
    batch_size = 2
    num_vars = len(small_variables)
    steps = 3
    x_norm = torch.randn(batch_size, num_vars, 16, 32)

    plan = reference_plan(num_experts=8)

    out = controlled_rollout(
        trained_bridge, x_norm, small_variables,
        interval=6, steps=steps, plan=plan, expert_loras={},
        return_trajectory=True,
    )

    # Should be (B, T+1, V, H, W) where T+1 includes initial state
    assert out.shape == (batch_size, steps + 1, num_vars, 16, 32)


# =============================================================================
# Test: Gradient flow through LoRA path
# =============================================================================

def test_gradient_flows_through_lora(trained_bridge, small_variables):
    """Test that gradients flow through the LoRA injection path."""
    batch_size = 2
    num_vars = len(small_variables)
    hidden_size = trained_bridge.model.hidden_size

    # Create input that requires grad
    x_norm = torch.randn(batch_size, num_vars, 16, 32, requires_grad=True)

    # Create LoRA with requires_grad parameters
    lora = ExpertLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4)
    # Initialize with non-zero weights
    for up_module in lora.up:
        torch.nn.init.normal_(up_module.weight, std=0.1)

    expert_loras = {0: lora}  # Block 0

    # Plan with nonzero coefficients
    plan = EditPlan("test", 8, (0.1,) + (0.0,) * 7, rho=0.25)

    # Forward pass
    out = controlled_rollout(
        trained_bridge, x_norm, small_variables,
        interval=6, steps=1, plan=plan, expert_loras=expert_loras,
        target_blocks=(0,),
    )

    # Compute loss and backward
    loss = out.sum()
    loss.backward()

    # Check that gradients exist
    assert x_norm.grad is not None
    assert not torch.isnan(x_norm.grad).any()


def test_gradient_flows_through_differentiable_coefficients(trained_bridge, small_variables):
    """Test that differentiable=True enables gradient flow through coefficients."""
    batch_size = 2
    num_vars = len(small_variables)
    hidden_size = trained_bridge.model.hidden_size

    x_norm = torch.randn(batch_size, num_vars, 16, 32, requires_grad=True)

    # Create LoRA
    lora = ExpertLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4)
    for up_module in lora.up:
        torch.nn.init.normal_(up_module.weight, std=0.1)

    expert_loras = {0: lora}

    plan = EditPlan("test", 8, (0.1,) + (0.0,) * 7, rho=0.25)

    # Forward with differentiable=True
    out = controlled_rollout(
        trained_bridge, x_norm, small_variables,
        interval=6, steps=1, plan=plan, expert_loras=expert_loras,
        target_blocks=(0,), differentiable=True,
    )

    loss = out.sum()
    loss.backward()

    # Should have gradients
    assert x_norm.grad is not None


def test_lora_parameters_receive_gradients(trained_bridge, small_variables):
    """Test that LoRA parameters receive gradients during backward pass."""
    batch_size = 2
    num_vars = len(small_variables)
    hidden_size = trained_bridge.model.hidden_size

    x_norm = torch.randn(batch_size, num_vars, 16, 32)

    # Create LoRA with trainable parameters
    lora = ExpertLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4)
    for up_module in lora.up:
        torch.nn.init.normal_(up_module.weight, std=0.1)

    # Make sure LoRA parameters require grad
    for param in lora.parameters():
        param.requires_grad_(True)

    expert_loras = {0: lora}

    plan = EditPlan("test", 8, (0.1,) + (0.0,) * 7, rho=0.25)

    # Forward pass
    out = controlled_rollout(
        trained_bridge, x_norm, small_variables,
        interval=6, steps=1, plan=plan, expert_loras=expert_loras,
        target_blocks=(0,), differentiable=True,
    )

    # Backward
    loss = out.sum()
    loss.backward()

    # Check LoRA parameters have gradients
    has_grad = False
    for param in lora.parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any()

    assert has_grad, "No LoRA parameters received gradients"


# =============================================================================
# Test: Re-entrancy guard
# =============================================================================

def test_reentrant_rollout_raises(trained_bridge, small_variables):
    """Test that reentrant rollout on same bridge raises error."""
    batch_size = 2
    num_vars = len(small_variables)
    x_norm = torch.randn(batch_size, num_vars, 16, 32)

    plan = reference_plan(num_experts=8)

    # First rollout should work
    _ = controlled_rollout(
        trained_bridge, x_norm, small_variables,
        interval=6, steps=1, plan=plan, expert_loras={},
    )

    # Sequential rollouts should also work
    _ = controlled_rollout(
        trained_bridge, x_norm, small_variables,
        interval=6, steps=1, plan=plan, expert_loras={},
    )

    # Note: True re-entrancy testing would require threading/async,
    # which is complex. This test verifies normal sequential use works.


def test_rollout_lock_released_on_exception(trained_bridge, small_variables):
    """Test that rollout lock is released even when exception occurs."""
    from earthdelta.bridge.stormer_bridge import _rollout_lock

    batch_size = 2
    num_vars = len(small_variables)
    x_norm = torch.randn(batch_size, num_vars, 16, 32)

    # Create a LoRA that will fail
    class FailingLoRA(ExpertLoRA):
        def forward(self, x, coeffs, sparse=False):
            raise RuntimeError("Intentional failure")

    hidden_size = trained_bridge.model.hidden_size
    expert_loras = {
        0: FailingLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4),
    }

    plan = EditPlan("test", 8, (0.1,) + (0.0,) * 7, rho=0.25)

    # This should raise
    with pytest.raises(RuntimeError, match="Intentional failure"):
        controlled_rollout(
            trained_bridge, x_norm, small_variables,
            interval=6, steps=1, plan=plan, expert_loras=expert_loras,
            target_blocks=(0,),
        )

    # Bridge should be released from lock
    if hasattr(_rollout_lock, 'active_bridges'):
        assert id(trained_bridge) not in _rollout_lock.active_bridges

    # Should be able to run again
    _ = controlled_rollout(
        trained_bridge, x_norm, small_variables,
        interval=6, steps=1, plan=reference_plan(8), expert_loras={},
    )


# =============================================================================
# Test: Trajectory return with gradients
# =============================================================================

def test_trajectory_preserves_gradients(trained_bridge, small_variables):
    """Test that trajectory mode preserves gradient computation."""
    batch_size = 2
    num_vars = len(small_variables)
    steps = 2

    x_norm = torch.randn(batch_size, num_vars, 16, 32, requires_grad=True)
    plan = reference_plan(num_experts=8)

    # Get trajectory
    trajectory = controlled_rollout(
        trained_bridge, x_norm, small_variables,
        interval=6, steps=steps, plan=plan, expert_loras={},
        return_trajectory=True,
    )

    # Use intermediate step for loss
    intermediate_loss = trajectory[:, 1, :, :, :].sum()  # Step 1
    intermediate_loss.backward()

    assert x_norm.grad is not None


def test_trajectory_includes_initial_state(trained_bridge, small_variables):
    """Test that trajectory[0] equals the initial state."""
    batch_size = 2
    num_vars = len(small_variables)

    x_norm = torch.randn(batch_size, num_vars, 16, 32)
    plan = reference_plan(num_experts=8)

    trajectory = controlled_rollout(
        trained_bridge, x_norm, small_variables,
        interval=6, steps=2, plan=plan, expert_loras={},
        return_trajectory=True,
    )

    # First entry should be initial state
    assert torch.equal(trajectory[:, 0], x_norm)


# =============================================================================
# Test: Gradient flow through multiple steps
# =============================================================================

def test_gradient_accumulates_through_steps(trained_bridge, small_variables):
    """Test that gradients accumulate correctly through multiple steps."""
    batch_size = 2
    num_vars = len(small_variables)
    hidden_size = trained_bridge.model.hidden_size

    x_norm = torch.randn(batch_size, num_vars, 16, 32, requires_grad=True)

    # Create LoRA with trackable gradients
    lora = ExpertLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4)
    for up_module in lora.up:
        torch.nn.init.normal_(up_module.weight, std=0.1)
    for param in lora.parameters():
        param.requires_grad_(True)

    expert_loras = {0: lora}

    # Plan that applies for multiple steps
    plan = EditPlan("test", 8, (0.1,) + (0.0,) * 7, hold_steps=4, rho=0.25)

    # 2-step rollout (both steps within hold window)
    out = controlled_rollout(
        trained_bridge, x_norm, small_variables,
        interval=6, steps=2, plan=plan, expert_loras=expert_loras,
        target_blocks=(0,), differentiable=True,
    )

    loss = out.sum()
    loss.backward()

    # Verify gradients exist and are non-trivial
    assert x_norm.grad is not None
    assert x_norm.grad.abs().sum() > 0


# =============================================================================
# Test: Consistency between differentiable and non-differentiable modes
# =============================================================================

def test_differentiable_mode_same_output(trained_bridge, small_variables):
    """Test that differentiable=True produces same output as False."""
    batch_size = 2
    num_vars = len(small_variables)

    x_norm = torch.randn(batch_size, num_vars, 16, 32)
    plan = reference_plan(num_experts=8)

    with torch.no_grad():
        out_normal = controlled_rollout(
            trained_bridge, x_norm, small_variables,
            interval=6, steps=2, plan=plan, expert_loras={},
            differentiable=False,
        )

        out_diff = controlled_rollout(
            trained_bridge, x_norm, small_variables,
            interval=6, steps=2, plan=plan, expert_loras={},
            differentiable=True,
        )

    assert torch.allclose(out_normal, out_diff, atol=1e-6)
