"""Tests for the earthdelta.bridge package.

Tests verify:
1. SDPA vs explicit-softmax attention parity
2. Zero-edit rollout equals official-equivalent (forward_validation)
3. No state leak from external/future data
4. Normalization round-trip correctness
5. Checkpoint structural load (slow test, requires real checkpoints)
6. ArtifactVersion mismatch rejection
"""
from dataclasses import replace
import torch
import pytest
import numpy as np

from earthdelta.bridge import (
    Stormer,
    MemEffAttention,
    ExplicitAttention,
    NormalizationContract,
    WeatherStepBridge,
    controlled_rollout,
    load_stormer_checkpoint,
    DEFAULT_VARIABLES,
    check_version_match,
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
    torch.set_num_threads(1)


@pytest.fixture
def small_variables():
    """Small variable list for fast tests."""
    return DEFAULT_VARIABLES[:8]


@pytest.fixture
def small_model(small_variables):
    """Small Stormer model for fast tests."""
    return Stormer(
        in_img_size=(16, 32),
        variables=small_variables,
        patch_size=2,
        hidden_size=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )


@pytest.fixture
def small_normalization(small_variables, tmp_path):
    """Create small normalization contract with synthetic data."""
    # Create synthetic normalization files
    mean_dict = {v: np.array([float(i)]) for i, v in enumerate(small_variables)}
    std_dict = {v: np.array([1.0 + 0.1 * i]) for i, v in enumerate(small_variables)}

    np.savez(tmp_path / "normalize_mean.npz", **mean_dict)
    np.savez(tmp_path / "normalize_std.npz", **std_dict)

    for interval in [6, 12, 24]:
        diff_mean = {v: np.array([0.0]) for v in small_variables}
        diff_std = {v: np.array([0.5 + 0.05 * i]) for i, v in enumerate(small_variables)}
        np.savez(tmp_path / f"normalize_diff_mean_{interval}.npz", **diff_mean)
        np.savez(tmp_path / f"normalize_diff_std_{interval}.npz", **diff_std)

    return NormalizationContract.from_npz_dir(str(tmp_path), variables=small_variables)


@pytest.fixture
def small_bridge(small_model, small_normalization):
    """Create small WeatherStepBridge for tests."""
    small_model.eval()
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
    return WeatherStepBridge(small_model, small_normalization, version)


@pytest.fixture
def trained_bridge(small_variables, small_normalization):
    """Create bridge with non-zero initialization (simulating trained model).

    Fresh Stormer models have zero-initialized adaLN gates AND head layer
    (standard DiT init). Real checkpoints have trained non-zero values.
    This fixture simulates that for testing purposes.
    """
    model = Stormer(
        in_img_size=(16, 32),
        variables=small_variables,
        patch_size=2,
        hidden_size=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    # Initialize adaLN gates with non-zero values
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
    # Initialize head layer with non-zero values (default is zero)
    torch.nn.init.normal_(model.head.linear.weight, std=0.02)
    torch.nn.init.normal_(model.head.linear.bias, std=0.02)
    torch.nn.init.normal_(model.head.adaLN_modulation[-1].weight, std=0.1)
    torch.nn.init.normal_(model.head.adaLN_modulation[-1].bias, std=0.1)
    model.eval()

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
    return WeatherStepBridge(model, small_normalization, version)


# =============================================================================
# Test 1: SDPA vs explicit-softmax attention parity
# =============================================================================

def test_sdpa_vs_explicit_attention_parity():
    """Test that SDPA and explicit softmax attention produce equivalent outputs."""
    dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 16

    # Create both attention modules with identical weights
    sdpa_attn = MemEffAttention(dim, num_heads=num_heads, qkv_bias=True)
    explicit_attn = ExplicitAttention(dim, num_heads=num_heads, qkv_bias=True)

    # Copy weights from SDPA to explicit
    explicit_attn.qkv.weight.data.copy_(sdpa_attn.qkv.weight.data)
    explicit_attn.qkv.bias.data.copy_(sdpa_attn.qkv.bias.data)
    explicit_attn.proj.weight.data.copy_(sdpa_attn.proj.weight.data)
    explicit_attn.proj.bias.data.copy_(sdpa_attn.proj.bias.data)

    # Set to eval mode (no dropout)
    sdpa_attn.eval()
    explicit_attn.eval()

    # Random input
    x = torch.randn(batch_size, seq_len, dim)

    # Forward pass
    with torch.no_grad():
        sdpa_out = sdpa_attn(x)
        explicit_out = explicit_attn(x)

    # Check equivalence (within 1e-5 as specified)
    assert torch.allclose(sdpa_out, explicit_out, atol=1e-5, rtol=1e-5), \
        f"Max diff: {(sdpa_out - explicit_out).abs().max().item()}"


def test_sdpa_attention_scaling():
    """Verify SDPA applies correct 1/sqrt(head_dim) scaling."""
    dim = 64
    num_heads = 4
    head_dim = dim // num_heads  # 16

    attn = MemEffAttention(dim, num_heads=num_heads, qkv_bias=False)
    assert attn.scale == pytest.approx(head_dim ** -0.5)


# =============================================================================
# Test 2: Zero-edit equals official-equivalent rollout
# =============================================================================

def test_zero_edit_equals_forward_validation(small_bridge, small_variables):
    """Test that controlled_rollout with zero coefficients equals forward_validation."""
    batch_size = 2
    num_vars = len(small_variables)
    x_norm = torch.randn(batch_size, num_vars, 16, 32)

    # Reference plan (all zeros)
    plan = reference_plan(num_experts=8)

    # Empty expert_loras (no LoRA modules attached)
    expert_loras = {}

    # Run both paths
    with torch.no_grad():
        reference_out = small_bridge.forward_validation(x_norm, small_variables, interval=6, steps=2)
        controlled_out = controlled_rollout(
            small_bridge, x_norm, small_variables, interval=6, steps=2,
            plan=plan, expert_loras=expert_loras
        )

    # Check equivalence (within float32 eps, ~1e-6)
    assert torch.allclose(reference_out, controlled_out, atol=1e-6, rtol=1e-6), \
        f"Max diff: {(reference_out - controlled_out).abs().max().item()}"


def test_zero_coefficients_with_loras_equals_no_loras(small_bridge, small_variables):
    """Test that LoRAs with zero coefficients produce same output as no LoRAs."""
    batch_size = 2
    num_vars = len(small_variables)
    x_norm = torch.randn(batch_size, num_vars, 16, 32)

    # Plan with zero coefficients
    plan = reference_plan(num_experts=8)

    # Create LoRA modules but use zero coefficients
    hidden_size = small_bridge.model.hidden_size
    expert_loras = {
        0: ExpertLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4),
        1: ExpertLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4),
    }

    with torch.no_grad():
        # Without LoRAs
        out_no_lora = controlled_rollout(
            small_bridge, x_norm, small_variables, interval=6, steps=2,
            plan=plan, expert_loras={}
        )
        # With LoRAs but zero coefficients
        out_with_lora = controlled_rollout(
            small_bridge, x_norm, small_variables, interval=6, steps=2,
            plan=plan, expert_loras=expert_loras
        )

    assert torch.allclose(out_no_lora, out_with_lora, atol=1e-6, rtol=1e-6), \
        f"Max diff: {(out_no_lora - out_with_lora).abs().max().item()}"


# =============================================================================
# Test 3: No state leak from external/future data
# =============================================================================

def test_no_state_leak_differential(small_bridge, small_variables):
    """Test that external/future values in scope don't affect controlled_rollout.

    Two calls with identical explicit arguments but different values in scope
    must produce bit-identical output.
    """
    batch_size = 2
    num_vars = len(small_variables)
    x_norm = torch.randn(batch_size, num_vars, 16, 32)

    plan = reference_plan(num_experts=8)
    expert_loras = {}

    # Scope variable that should NOT leak into computation
    future_truth_a = torch.randn(batch_size, num_vars, 16, 32)
    _ = future_truth_a  # Referenced but not passed

    with torch.no_grad():
        out_a = controlled_rollout(
            small_bridge, x_norm, small_variables, interval=6, steps=2,
            plan=plan, expert_loras=expert_loras
        ).clone()

    # Change the "future truth" variable
    future_truth_b = torch.randn(batch_size, num_vars, 16, 32) * 100
    _ = future_truth_b  # Different value in scope

    with torch.no_grad():
        out_b = controlled_rollout(
            small_bridge, x_norm, small_variables, interval=6, steps=2,
            plan=plan, expert_loras=expert_loras
        )

    # Must be bit-identical
    assert torch.equal(out_a, out_b), "Output changed despite identical arguments - state leak detected!"


def test_no_state_leak_function_signature():
    """Verify controlled_rollout signature contains no hidden state paths."""
    import inspect
    sig = inspect.signature(controlled_rollout)
    param_names = list(sig.parameters.keys())

    # All parameters should be explicit
    expected = ['bridge', 'x_norm', 'variables', 'interval', 'steps', 'plan',
                'expert_loras', 'target_blocks', 'sparse', 'return_trajectory',
                'differentiable']
    assert param_names == expected, f"Unexpected parameters: {param_names}"

    # No default mutable arguments that could leak state
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            # Default should be immutable (tuple, bool, etc.)
            assert not isinstance(param.default, (list, dict, set)), \
                f"Mutable default for {name}: {param.default}"


def test_no_closure_leak():
    """Verify controlled_rollout doesn't capture mutable state in closure."""
    import types
    assert not isinstance(controlled_rollout, types.MethodType), \
        "controlled_rollout should be a plain function, not a method"

    # Check closure variables
    if hasattr(controlled_rollout, '__closure__') and controlled_rollout.__closure__:
        for cell in controlled_rollout.__closure__:
            val = cell.cell_contents
            # Should not contain mutable state like tensors
            assert not isinstance(val, torch.Tensor), \
                f"Tensor in closure: {type(val)}"


# =============================================================================
# Test 4: Normalization round-trip
# =============================================================================

def test_normalization_round_trip(small_normalization):
    """Test that denormalize(normalize(x)) == x within tolerance."""
    x_raw = torch.randn(2, 8, 16, 32)

    x_norm = small_normalization.normalize(x_raw)
    x_reconstructed = small_normalization.denormalize(x_norm)

    assert torch.allclose(x_raw, x_reconstructed, atol=1e-6, rtol=1e-6), \
        f"Max diff: {(x_raw - x_reconstructed).abs().max().item()}"


def test_diff_normalization_round_trip(small_normalization):
    """Test diff normalization/denormalization consistency."""
    diff_raw = torch.randn(2, 8, 16, 32)

    for interval in [6, 12, 24]:
        # Since we don't have a normalize_diff method, verify denormalize_diff
        # produces the expected transformation
        std = small_normalization.diff_std[interval].view(1, -1, 1, 1)

        # Simulate normalized diff (assuming zero mean)
        diff_norm = diff_raw / std.expand_as(diff_raw)
        diff_denorm = small_normalization.denormalize_diff(diff_norm, interval)

        assert torch.allclose(diff_raw, diff_denorm, atol=1e-6, rtol=1e-6), \
            f"Interval {interval} max diff: {(diff_raw - diff_denorm).abs().max().item()}"


def test_replace_constant_zeros_correct_channels(small_normalization, small_variables):
    """Test that replace_constant zeros out constant variable channels."""
    # None of our small_variables are in CONSTANTS, but test the mechanism
    from earthdelta.bridge import CONSTANTS

    # Create a fake variable list with a constant
    vars_with_constant = small_variables.copy()
    vars_with_constant[2] = "land_sea_mask"  # This is in CONSTANTS

    yhat = torch.ones(2, len(vars_with_constant), 16, 32)
    yhat_replaced = small_normalization.replace_constant(yhat, vars_with_constant)

    # Channel 2 should be zeroed
    assert yhat_replaced[:, 2].sum() == 0
    # Other channels unchanged
    assert yhat_replaced[:, 0].sum() == yhat[:, 0].sum()
    assert yhat_replaced[:, 1].sum() == yhat[:, 1].sum()
    assert yhat_replaced[:, 3].sum() == yhat[:, 3].sum()


# =============================================================================
# Test 5: Checkpoint structural load (slow test)
# =============================================================================

CHECKPOINT_PATH_PS2 = "/mnt/afs/260010168/EarthDelta/checkpoints/stormer_1.40625_patch_size_2.ckpt"
CHECKPOINT_PATH_PS4 = "/mnt/afs/260010168/EarthDelta/checkpoints/stormer_1.40625_patch_size_4.ckpt"


def _get_memory_limit_gb() -> float:
    """Get cgroup memory limit in GB, or -1 if unlimited."""
    try:
        with open('/sys/fs/cgroup/memory.max') as f:
            limit = f.read().strip()
            if limit == 'max':
                return -1
            return int(limit) / 1e9
    except (FileNotFoundError, OSError):
        pass
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f:
            limit = int(f.read().strip())
            # Very high limit usually means unlimited
            if limit > 1e15:
                return -1
            return limit / 1e9
    except (FileNotFoundError, OSError):
        pass
    return -1  # Unknown, assume unlimited


def _skip_if_insufficient_memory(required_gb: float = 12.0):
    """Skip test if container memory limit is too low for checkpoint loading."""
    limit = _get_memory_limit_gb()
    if limit > 0 and limit < required_gb:
        pytest.skip(f"Insufficient memory: {limit:.1f}GB limit, need ~{required_gb}GB for checkpoint loading")


@pytest.mark.slow
def test_checkpoint_load_patch_size_2():
    """Test loading patch_size=2 checkpoint with strict=True.

    Requires ~12GB memory to load the 5.5GB checkpoint + model + overhead.
    """
    import os
    _skip_if_insufficient_memory(12.0)
    if not os.path.exists(CHECKPOINT_PATH_PS2):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH_PS2}")

    model, version = load_stormer_checkpoint(CHECKPOINT_PATH_PS2, patch_size=2)

    # Verify model loaded correctly
    assert model is not None
    assert version is not None
    assert "stormer" in version.backbone

    # Verify model is frozen
    for param in model.parameters():
        assert not param.requires_grad

    # Test single forward pass
    x = torch.randn(1, 69, 128, 256)
    interval = torch.tensor([0.6])  # 6 hours / 10

    with torch.no_grad():
        output = model(x, DEFAULT_VARIABLES, interval)

    assert output.shape == (1, 69, 128, 256)


@pytest.mark.slow
def test_checkpoint_load_patch_size_4():
    """Test loading patch_size=4 checkpoint with strict=True.

    Requires ~12GB memory to load the 5.5GB checkpoint + model + overhead.
    """
    import os
    _skip_if_insufficient_memory(12.0)
    if not os.path.exists(CHECKPOINT_PATH_PS4):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH_PS4}")

    model, version = load_stormer_checkpoint(CHECKPOINT_PATH_PS4, patch_size=4)

    # Verify model loaded correctly
    assert model is not None
    assert version is not None

    # Verify model is frozen
    for param in model.parameters():
        assert not param.requires_grad

    # Test single forward pass
    x = torch.randn(1, 69, 128, 256)
    interval = torch.tensor([0.6])

    with torch.no_grad():
        output = model(x, DEFAULT_VARIABLES, interval)

    assert output.shape == (1, 69, 128, 256)


@pytest.mark.slow
def test_both_checkpoints_produce_different_outputs():
    """Verify patch_size=2 and patch_size=4 models are distinct.

    Requires ~24GB memory to load both checkpoints simultaneously.
    """
    import os
    _skip_if_insufficient_memory(24.0)
    if not os.path.exists(CHECKPOINT_PATH_PS2) or not os.path.exists(CHECKPOINT_PATH_PS4):
        pytest.skip("Both checkpoints required")

    model_ps2, _ = load_stormer_checkpoint(CHECKPOINT_PATH_PS2, patch_size=2)
    model_ps4, _ = load_stormer_checkpoint(CHECKPOINT_PATH_PS4, patch_size=4)

    x = torch.randn(1, 69, 128, 256)
    interval = torch.tensor([0.6])

    with torch.no_grad():
        out_ps2 = model_ps2(x, DEFAULT_VARIABLES, interval)
        out_ps4 = model_ps4(x, DEFAULT_VARIABLES, interval)

    # Different models should produce different outputs
    assert not torch.allclose(out_ps2, out_ps4, atol=1e-3), \
        "Patch size 2 and 4 models produced suspiciously similar outputs"


# =============================================================================
# Test 6: ArtifactVersion mismatch rejection
# =============================================================================

def test_artifact_version_mismatch_rejection(small_bridge):
    """Test that version mismatch is correctly rejected."""
    bridge_version = small_bridge.version

    # Create a plan version with different backbone
    plan_version = replace(bridge_version, backbone="wrong_backbone")

    with pytest.raises(ValueError, match="backbone"):
        check_version_match(plan_version, bridge_version)


def test_artifact_version_mismatch_grid(small_bridge):
    """Test mismatch rejection for grid field."""
    bridge_version = small_bridge.version
    plan_version = replace(bridge_version, grid="128x256")

    with pytest.raises(ValueError, match="grid"):
        check_version_match(plan_version, bridge_version)


def test_artifact_version_match_succeeds(small_bridge):
    """Test that matching versions don't raise."""
    bridge_version = small_bridge.version
    # Should not raise
    check_version_match(bridge_version, bridge_version)


def test_edit_plan_against_mismatched_version():
    """Test that EditPlan usage can be gated by version checks."""
    # Create two different versions
    version_a = ArtifactVersion(
        backbone="stormer_ps2",
        static_adapter="none",
        edit_bank="bank_v1",
        normalization="norm_a",
        grid="128x256",
        projection="patch2",
        split="train",
        continuation="reference_after_hold",
    )
    version_b = replace(version_a, edit_bank="bank_v2")

    # Simulate: plan was created against version_a
    plan = EditPlan("test", 8, (0.1,) * 8, rho=0.25)
    plan_created_with = version_a

    # Running against version_b should be rejected
    with pytest.raises(ValueError, match="edit_bank"):
        check_version_match(plan_created_with, version_b)


# =============================================================================
# Additional tests for architecture components
# =============================================================================

def test_stormer_forward_shape(small_model, small_variables):
    """Test Stormer forward produces correct output shape."""
    batch_size = 2
    num_vars = len(small_variables)
    x = torch.randn(batch_size, num_vars, 16, 32)
    interval = torch.tensor([0.6, 0.6])

    small_model.eval()
    with torch.no_grad():
        output = small_model(x, small_variables, interval)

    assert output.shape == (batch_size, num_vars, 16, 32)


def test_stormer_deterministic_in_eval(small_model, small_variables):
    """Test that Stormer in eval mode is deterministic."""
    x = torch.randn(2, len(small_variables), 16, 32)
    interval = torch.tensor([0.6, 0.6])

    small_model.eval()
    with torch.no_grad():
        out1 = small_model(x, small_variables, interval)
        out2 = small_model(x, small_variables, interval)

    assert torch.equal(out1, out2)


def test_pad_unpad_identity(small_bridge):
    """Test that pad then unpad preserves content."""
    x = torch.randn(2, 8, 16, 32)

    # Pad
    padded, pad_size = small_bridge.pad(x)

    # For 16 height and patch_size 2, no padding needed
    assert pad_size == 0
    assert torch.equal(x, padded)


def test_pad_when_needed():
    """Test padding when height not divisible by patch_size."""
    # Create bridge with odd height
    variables = DEFAULT_VARIABLES[:4]
    model = Stormer(
        in_img_size=(15, 32),  # 15 not divisible by 4
        variables=variables,
        patch_size=4,
        hidden_size=32,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
    )

    version = ArtifactVersion(
        backbone="test", static_adapter="none", edit_bank="none",
        normalization="none", grid="15x32", projection="patch4",
        split="test", continuation="reference_after_hold"
    )

    # Create minimal normalization
    norm = NormalizationContract(
        inp_mean=torch.zeros(4),
        inp_std=torch.ones(4),
        diff_mean={6: torch.zeros(4)},
        diff_std={6: torch.ones(4)},
        variables=variables,
    )

    bridge = WeatherStepBridge(model, norm, version)

    x = torch.randn(2, 4, 15, 32)
    padded, pad_size = bridge.pad(x)

    # 15 % 4 = 3, need to pad 1 to make it 16
    assert pad_size == 1
    assert padded.shape == (2, 4, 16, 32)
    # Content should be preserved (unpadded region)
    assert torch.equal(padded[:, :, 1:, :], x)


def test_controlled_rollout_with_nonzero_coefficients(trained_bridge, small_variables):
    """Test that nonzero coefficients change the output.

    Uses trained_bridge fixture which has non-zero adaLN gates (simulating trained model).
    Fresh Stormer models have zero-initialized gates which would zero out LoRA contributions.
    """
    batch_size = 2
    num_vars = len(small_variables)
    hidden_size = trained_bridge.model.hidden_size

    x_norm = torch.randn(batch_size, num_vars, 16, 32)

    # Plan with nonzero coefficients
    plan = EditPlan("test", 8, (0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), rho=0.25)

    # Create LoRA modules for block 0 (our small model only has 2 blocks)
    # Initialize with non-zero weights (default LoRA init has zero up projection)
    lora = ExpertLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4)
    # Set non-zero weights in the up projection to ensure LoRA has an effect
    for up_module in lora.up:
        torch.nn.init.normal_(up_module.weight, std=0.1)
    expert_loras = {0: lora}

    with torch.no_grad():
        # Without LoRAs
        out_ref = controlled_rollout(
            trained_bridge, x_norm, small_variables, interval=6, steps=1,
            plan=reference_plan(8), expert_loras={}
        )
        # With nonzero LoRAs
        out_edited = controlled_rollout(
            trained_bridge, x_norm, small_variables, interval=6, steps=1,
            plan=plan, expert_loras=expert_loras, target_blocks=(0,)
        )

    # Outputs should be different (LoRA has effect)
    assert not torch.allclose(out_ref, out_edited, atol=1e-6), \
        "Nonzero LoRA coefficients should change output"


def test_application_window_respected(trained_bridge, small_variables):
    """Test that coefficients are zero outside the application window.

    Uses trained_bridge fixture which has non-zero adaLN gates (simulating trained model).
    """
    batch_size = 2
    num_vars = len(small_variables)
    hidden_size = trained_bridge.model.hidden_size

    x_norm = torch.randn(batch_size, num_vars, 16, 32)

    # Plan with hold_steps=1
    plan = EditPlan("test", 8, (0.2,) + (0.0,) * 7, hold_steps=1, rho=0.25)

    # Initialize with non-zero weights (default LoRA init has zero up projection)
    lora = ExpertLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4)
    for up_module in lora.up:
        torch.nn.init.normal_(up_module.weight, std=0.1)
    expert_loras = {0: lora}

    with torch.no_grad():
        # 2-step rollout with hold_steps=1 means edits only apply at step 0
        out_2step = controlled_rollout(
            trained_bridge, x_norm, small_variables, interval=6, steps=2,
            plan=plan, expert_loras=expert_loras, target_blocks=(0,)
        )

        # Reference: no edits
        out_ref = controlled_rollout(
            trained_bridge, x_norm, small_variables, interval=6, steps=2,
            plan=reference_plan(8), expert_loras={}
        )

    # Should be different because step 0 had edits
    assert not torch.allclose(out_2step, out_ref, atol=1e-6)


# =============================================================================
# Test normalization loading from real files (if available)
# =============================================================================

NORMALIZATION_DIR = "/mnt/afs/260010168/EarthDelta/reference/stormer/normalization_constants"


def test_load_real_normalization():
    """Test loading real normalization constants."""
    import os
    if not os.path.exists(NORMALIZATION_DIR):
        pytest.skip(f"Normalization dir not found: {NORMALIZATION_DIR}")

    norm = NormalizationContract.from_npz_dir(NORMALIZATION_DIR, variables=DEFAULT_VARIABLES)

    # Check shapes
    assert norm.inp_mean.shape == (69,)
    assert norm.inp_std.shape == (69,)
    assert 6 in norm.diff_std
    assert norm.diff_std[6].shape == (69,)

    # Check digest is computed
    assert len(norm.digest) == 16


def test_default_variables_count():
    """Verify DEFAULT_VARIABLES has exactly 69 entries."""
    assert len(DEFAULT_VARIABLES) == 69


# =============================================================================
# Test: NormalizationContract.digest includes diff_mean
# =============================================================================

def test_digest_changes_with_diff_mean(small_variables):
    """Test that changing diff_mean changes the digest."""
    # Create two contracts that differ only in diff_mean
    norm1 = NormalizationContract(
        inp_mean=torch.zeros(8),
        inp_std=torch.ones(8),
        diff_mean={6: torch.zeros(8)},
        diff_std={6: torch.ones(8)},
        variables=small_variables,
    )

    norm2 = NormalizationContract(
        inp_mean=torch.zeros(8),
        inp_std=torch.ones(8),
        diff_mean={6: torch.ones(8)},  # Different diff_mean
        diff_std={6: torch.ones(8)},
        variables=small_variables,
    )

    # Digests should be different
    assert norm1.digest != norm2.digest, "Changing diff_mean should change digest"


def test_digest_stable_with_same_inputs(small_variables):
    """Test that identical contracts produce identical digests."""
    norm1 = NormalizationContract(
        inp_mean=torch.zeros(8),
        inp_std=torch.ones(8),
        diff_mean={6: torch.zeros(8)},
        diff_std={6: torch.ones(8)},
        variables=small_variables,
    )

    norm2 = NormalizationContract(
        inp_mean=torch.zeros(8),
        inp_std=torch.ones(8),
        diff_mean={6: torch.zeros(8)},
        diff_std={6: torch.ones(8)},
        variables=small_variables,
    )

    assert norm1.digest == norm2.digest, "Identical contracts should have identical digests"


# =============================================================================
# Test: Hook cleanup after controlled_rollout
# =============================================================================

def test_hooks_removed_after_normal_completion(small_bridge, small_variables):
    """Test that forward hooks are fully removed after controlled_rollout completes."""
    batch_size = 2
    num_vars = len(small_variables)
    hidden_size = small_bridge.model.hidden_size

    x_norm = torch.randn(batch_size, num_vars, 16, 32)
    plan = EditPlan("test", 8, (0.1,) * 8, rho=0.25)

    # Create LoRA modules for blocks 0 and 1 (our small model has 2 blocks)
    expert_loras = {
        0: ExpertLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4),
        1: ExpertLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4),
    }

    # Check hooks are empty before
    for block_idx in [0, 1]:
        proj = small_bridge.model.blocks[block_idx].attn.proj
        assert len(proj._forward_hooks) == 0, f"Block {block_idx} should have no hooks before"

    # Run controlled_rollout
    with torch.no_grad():
        _ = controlled_rollout(
            small_bridge, x_norm, small_variables, interval=6, steps=1,
            plan=plan, expert_loras=expert_loras, target_blocks=(0, 1)
        )

    # Check hooks are empty after
    for block_idx in [0, 1]:
        proj = small_bridge.model.blocks[block_idx].attn.proj
        assert len(proj._forward_hooks) == 0, f"Block {block_idx} should have no hooks after normal completion"


def test_hooks_removed_after_exception(small_bridge, small_variables):
    """Test that forward hooks are removed even when the forward pass raises."""
    batch_size = 2
    num_vars = len(small_variables)
    hidden_size = small_bridge.model.hidden_size

    x_norm = torch.randn(batch_size, num_vars, 16, 32)
    plan = EditPlan("test", 8, (0.1,) * 8, rho=0.25)

    # Create a LoRA module that will raise an exception
    class FailingLoRA(ExpertLoRA):
        def forward(self, x, coeffs, sparse=False):
            raise RuntimeError("Intentional failure for testing")

    expert_loras = {
        0: FailingLoRA(hidden_size, hidden_size, num_experts=8, rank_per_expert=4),
    }

    # Check hooks are empty before
    proj = small_bridge.model.blocks[0].attn.proj
    assert len(proj._forward_hooks) == 0, "Should have no hooks before"

    # Run controlled_rollout - expect it to raise
    with pytest.raises(RuntimeError, match="Intentional failure"):
        with torch.no_grad():
            controlled_rollout(
                small_bridge, x_norm, small_variables, interval=6, steps=1,
                plan=plan, expert_loras=expert_loras, target_blocks=(0,)
            )

    # Check hooks are empty after the exception
    assert len(proj._forward_hooks) == 0, "Hooks should be removed even after exception"


# =============================================================================
# Test: Variable list consistency
# =============================================================================

def test_variable_list_consistency_bridge_vs_data():
    """Test that DEFAULT_VARIABLES matches CANONICAL_VARIABLES from data module."""
    from earthdelta.data.pull_wb2 import CANONICAL_VARIABLES

    assert DEFAULT_VARIABLES == CANONICAL_VARIABLES, (
        "DEFAULT_VARIABLES (bridge) must exactly match CANONICAL_VARIABLES (data)"
    )


def test_variable_list_consistency_three_way():
    """Test that DEFAULT_VARIABLES, CANONICAL_VARIABLES, and yaml config match.

    This ensures all three sources of truth for the 69-variable list are consistent:
    1. earthdelta.bridge.DEFAULT_VARIABLES (hardcoded in stormer_bridge.py)
    2. earthdelta.data.pull_wb2.CANONICAL_VARIABLES (built programmatically)
    3. reference/stormer/configs/finetune_multi_step.yaml (upstream config)
    """
    import os
    import yaml
    from earthdelta.data.pull_wb2 import CANONICAL_VARIABLES

    yaml_path = "/mnt/afs/260010168/EarthDelta/reference/stormer/configs/finetune_multi_step.yaml"

    if not os.path.exists(yaml_path):
        pytest.skip(f"YAML config not found: {yaml_path}")

    # Load the yaml config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract variable list from yaml (model.net.init_args.list_variables)
    yaml_variables = config['model']['net']['init_args']['list_variables']

    # Check all three sources match
    assert DEFAULT_VARIABLES == CANONICAL_VARIABLES, (
        "DEFAULT_VARIABLES must match CANONICAL_VARIABLES"
    )
    assert DEFAULT_VARIABLES == yaml_variables, (
        "DEFAULT_VARIABLES must match yaml config's list_variables"
    )
    assert len(DEFAULT_VARIABLES) == 69, "Should have exactly 69 variables"
