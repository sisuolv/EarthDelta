"""Tests for NormalizationContract policy handling.

Tests verify:
1. Official zero-diff-mean policy correctly forces diff_mean to zero
2. Legacy policy preserves existing nonzero diff_mean behavior
3. Policy is included in digest computation
4. Backward compatibility: default policy is legacy
"""
import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from earthdelta.bridge import (
    NormalizationContract,
    POLICY_LEGACY,
    POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    DEFAULT_VARIABLES,
)


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
def npz_dir_with_nonzero_diff_mean(small_variables, tmp_path):
    """Create NPZ directory with NONZERO diff_mean values.

    This simulates the real NPZ files which have nonzero diff_mean,
    unlike the official inference.py which uses zeros.
    """
    # Create synthetic normalization files
    mean_dict = {v: np.array([float(i)]) for i, v in enumerate(small_variables)}
    std_dict = {v: np.array([1.0 + 0.1 * i]) for i, v in enumerate(small_variables)}

    np.savez(tmp_path / "normalize_mean.npz", **mean_dict)
    np.savez(tmp_path / "normalize_std.npz", **std_dict)

    for interval in [6, 12, 24]:
        # NONZERO diff_mean to simulate real NPZ files
        diff_mean = {v: np.array([0.01 * (i + 1) * interval]) for i, v in enumerate(small_variables)}
        diff_std = {v: np.array([0.5 + 0.05 * i]) for i, v in enumerate(small_variables)}
        np.savez(tmp_path / f"normalize_diff_mean_{interval}.npz", **diff_mean)
        np.savez(tmp_path / f"normalize_diff_std_{interval}.npz", **diff_std)

    return tmp_path


# =============================================================================
# Test: Official policy forces diff_mean to zero
# =============================================================================

def test_official_policy_forces_zero_diff_mean(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that official_zero_diff_mean policy forces diff_mean to zeros.

    This is the core B01 fix: under the official policy, diff_mean must be
    zeros to match the official inference.py semantics.
    """
    # Load with official policy
    norm_official = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    # All diff_mean values should be zero
    for interval in [6, 24]:
        diff_mean = norm_official.diff_mean[interval]
        assert torch.allclose(diff_mean, torch.zeros_like(diff_mean)), (
            f"Official policy should zero diff_mean for interval {interval}, "
            f"but got nonzero values: {diff_mean}"
        )


def test_legacy_policy_preserves_nonzero_diff_mean(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that legacy policy preserves nonzero diff_mean values.

    This ensures backward compatibility with existing experiments that
    relied on the pre-audit behavior.
    """
    # Load with legacy policy
    norm_legacy = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_LEGACY,
    )

    # diff_mean values should be nonzero (as in the NPZ files)
    for interval in [6, 24]:
        diff_mean = norm_legacy.diff_mean[interval]
        assert not torch.allclose(diff_mean, torch.zeros_like(diff_mean)), (
            f"Legacy policy should preserve nonzero diff_mean for interval {interval}"
        )


# =============================================================================
# Test: Policy affects denormalize_diff behavior
# =============================================================================

def test_official_policy_ignores_nonzero_diff_mean_in_denormalize(
    npz_dir_with_nonzero_diff_mean, small_variables
):
    """Test that denormalize_diff ignores diff_mean under official policy.

    This verifies that even if diff_mean tensors exist in the contract,
    the official policy forces them to be treated as zeros during
    denormalization, matching the official inference.py behavior.
    """
    # Load with official policy
    norm_official = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    # Load with legacy policy
    norm_legacy = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
        policy=POLICY_LEGACY,
    )

    # Create synthetic normalized diff
    batch_size = 2
    num_vars = len(small_variables)
    diff_norm = torch.randn(batch_size, num_vars, 16, 32)

    # Denormalize with both policies
    diff_official = norm_official.denormalize_diff(diff_norm, interval=6)
    diff_legacy = norm_legacy.denormalize_diff(diff_norm, interval=6)

    # They should be DIFFERENT because legacy uses nonzero mean
    assert not torch.allclose(diff_official, diff_legacy), (
        "Official and legacy denormalize_diff should produce different results "
        "when NPZ has nonzero diff_mean"
    )

    # Verify official path uses zero mean: diff_raw = diff_norm * std + 0
    # So diff_official should equal diff_norm * std
    std = norm_official.diff_std[6].view(1, -1, 1, 1)
    expected_official = diff_norm * std
    assert torch.allclose(diff_official, expected_official, atol=1e-6), (
        "Official policy should compute diff_raw = diff_norm * std (zero mean)"
    )


def test_legacy_denormalize_diff_unchanged(npz_dir_with_nonzero_diff_mean, small_variables):
    """Regression test: legacy denormalize_diff behavior is unchanged.

    This proves that existing experiments using the legacy path get
    exactly the same numeric behavior as before.
    """
    norm_legacy = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
        policy=POLICY_LEGACY,
    )

    diff_norm = torch.randn(2, len(small_variables), 16, 32)

    # Legacy: diff_raw = diff_norm * std + mean
    diff_legacy = norm_legacy.denormalize_diff(diff_norm, interval=6)

    std = norm_legacy.diff_std[6].view(1, -1, 1, 1)
    mean = norm_legacy.diff_mean[6].view(1, -1, 1, 1)
    expected_legacy = diff_norm * std + mean

    assert torch.allclose(diff_legacy, expected_legacy, atol=1e-6), (
        "Legacy policy should compute diff_raw = diff_norm * std + mean"
    )


# =============================================================================
# Test: Policy is included in digest
# =============================================================================

def test_policy_changes_digest(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that different policies produce different digests.

    NOTE: this case changes TWO things at once - the policy label and (because
    the official policy zeroes diff_mean) the stored tensors. It therefore
    cannot show that the policy label alone is part of the identity. The
    isolated version of that claim is
    `test_policy_label_alone_changes_digest_with_identical_tensors` below.
    """
    norm_official = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    norm_legacy = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_LEGACY,
    )

    assert norm_official.digest != norm_legacy.digest, (
        "Official and legacy policies should produce different digests"
    )


def test_policy_label_alone_changes_digest_with_identical_tensors(small_variables):
    """ONLY the declared policy differs; every tensor is identical.

    This isolates the claim the contaminated test above cannot make: the policy
    is part of the identity, not metadata riding alongside it. Both contracts
    hold byte-identical constants (diff_mean is zeros in both, so the official
    policy changes nothing numerically) and differ only in their label.
    """
    n = len(small_variables)
    shared = dict(
        inp_mean=torch.arange(n, dtype=torch.float32),
        inp_std=torch.ones(n) + 0.25,
        diff_mean={6: torch.zeros(n), 24: torch.zeros(n)},
        diff_std={6: torch.ones(n) * 0.5, 24: torch.ones(n) * 0.75},
        variables=small_variables,
    )

    norm_legacy = NormalizationContract(**shared, policy=POLICY_LEGACY)
    norm_official = NormalizationContract(**shared, policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN)

    # Prove the tensors really are identical.
    assert torch.equal(norm_legacy.inp_mean, norm_official.inp_mean)
    assert torch.equal(norm_legacy.inp_std, norm_official.inp_std)
    for interval in (6, 24):
        assert torch.equal(norm_legacy.diff_mean[interval], norm_official.diff_mean[interval])
        assert torch.equal(norm_legacy.diff_std[interval], norm_official.diff_std[interval])

    assert norm_legacy.digest != norm_official.digest, (
        "The policy label alone must change the identity digest"
    )
    assert norm_legacy.identity_digest != norm_official.identity_digest


def test_official_label_with_nonzero_diff_mean_is_rejected(small_variables):
    """A contract cannot declare zero diff_mean while holding nonzero values."""
    n = len(small_variables)
    with pytest.raises(ValueError, match="declares zero diff_mean"):
        NormalizationContract(
            inp_mean=torch.zeros(n),
            inp_std=torch.ones(n),
            diff_mean={6: torch.full((n,), 0.3)},
            diff_std={6: torch.ones(n)},
            variables=small_variables,
            policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
        )


def test_mislabelled_policy_identity_reflects_real_content(small_variables):
    """Relabelling a contract after construction does not launder its identity.

    The object is LABELLED official_zero_diff_mean but actually holds a nonzero
    diff_mean. Its identity must differ from an honestly-zero official contract,
    and the inconsistency must be detectable, so nothing downstream can trust
    the label over the contents.
    """
    n = len(small_variables)
    honest = NormalizationContract(
        inp_mean=torch.zeros(n),
        inp_std=torch.ones(n),
        diff_mean={6: torch.zeros(n)},
        diff_std={6: torch.ones(n)},
        variables=small_variables,
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    # Build a legitimate legacy contract, then relabel it (frozen dataclass, so
    # this bypasses __post_init__ exactly as a stale/tampered object would).
    mislabelled = NormalizationContract(
        inp_mean=torch.zeros(n),
        inp_std=torch.ones(n),
        diff_mean={6: torch.full((n,), 0.3)},
        diff_std={6: torch.ones(n)},
        variables=small_variables,
        policy=POLICY_LEGACY,
    )
    object.__setattr__(mislabelled, "policy", POLICY_OFFICIAL_ZERO_DIFF_MEAN)

    assert mislabelled.policy == honest.policy, "both now carry the same label"
    assert not torch.equal(mislabelled.diff_mean[6], honest.diff_mean[6])

    # Identity follows the real content, not the label.
    assert mislabelled.digest != honest.digest
    assert mislabelled.identity_digest != honest.identity_digest

    # And the inconsistency is explicitly detectable.
    assert honest.policy_content_consistent is True
    assert mislabelled.policy_content_consistent is False


def test_gate_rejects_mislabelled_normalization_policy(small_variables, tmp_path):
    """The S0 gate criterion rejects a contract whose label lies about content."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.s0_gate import verify_normalization_parity

    n = len(small_variables)
    mislabelled = NormalizationContract(
        inp_mean=torch.zeros(n),
        inp_std=torch.ones(n),
        diff_mean={6: torch.full((n,), 0.3)},
        diff_std={6: torch.ones(n)},
        variables=small_variables,
        policy=POLICY_LEGACY,
    )
    object.__setattr__(mislabelled, "policy", POLICY_OFFICIAL_ZERO_DIFF_MEAN)

    result = verify_normalization_parity(
        mislabelled, tmp_path, expected_digest=mislabelled.identity_digest,
    )

    assert result["passed"] is False, "gate must not accept a mislabelled contract"
    assert result["policy_content_consistent"] is False
    assert "must match the real contents" in result["error"]


def test_policy_in_digest_deterministic(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that digest is deterministic for same policy."""
    norm1 = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    norm2 = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    assert norm1.digest == norm2.digest, (
        "Same policy should produce identical digests"
    )


# =============================================================================
# Test: Default policy is legacy for backward compatibility
# =============================================================================

def test_default_policy_is_legacy(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that default policy is legacy for backward compatibility.

    Existing callers that don't pass the policy parameter should get
    the legacy behavior to avoid breaking existing experiments.
    """
    # Call without policy parameter
    norm_default = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
    )

    assert norm_default.policy == POLICY_LEGACY, (
        "Default policy should be POLICY_LEGACY for backward compatibility"
    )


def test_direct_construction_default_policy(small_variables):
    """Test that direct construction defaults to legacy policy."""
    norm = NormalizationContract(
        inp_mean=torch.zeros(len(small_variables)),
        inp_std=torch.ones(len(small_variables)),
        diff_mean={6: torch.ones(len(small_variables))},
        diff_std={6: torch.ones(len(small_variables))},
        variables=small_variables,
        # policy not specified
    )

    assert norm.policy == POLICY_LEGACY, (
        "Direct construction should default to POLICY_LEGACY"
    )


# =============================================================================
# Test: Invalid policy rejected
# =============================================================================

def test_invalid_policy_rejected(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that invalid policy values are rejected."""
    with pytest.raises(ValueError, match="Unknown policy"):
        NormalizationContract.from_npz_dir(
            str(npz_dir_with_nonzero_diff_mean),
            variables=small_variables,
            policy="invalid_policy",
        )


# =============================================================================
# Test: Rollout behavior differs by policy
# =============================================================================

def test_rollout_behavior_differs_by_policy(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that full rollout produces different results with different policies.

    This is an end-to-end verification that the policy affects the entire
    prediction pipeline, not just individual operations.
    """
    from earthdelta.bridge import Stormer, WeatherStepBridge
    from earthdelta.contracts import ArtifactVersion

    # Create a small model
    model = Stormer(
        in_img_size=(16, 32),
        variables=small_variables,
        patch_size=2,
        hidden_size=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )

    # Initialize with non-zero weights to ensure non-trivial outputs
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
    torch.nn.init.normal_(model.head.linear.weight, std=0.02)
    torch.nn.init.normal_(model.head.linear.bias, std=0.02)
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

    # Create bridges with different policies
    norm_official = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )
    norm_legacy = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
        policy=POLICY_LEGACY,
    )

    bridge_official = WeatherStepBridge(model, norm_official, version)
    bridge_legacy = WeatherStepBridge(model, norm_legacy, version)

    # Run rollout with both
    x_raw = torch.randn(1, len(small_variables), 16, 32)

    with torch.no_grad():
        x_norm_official = norm_official.normalize(x_raw)
        x_norm_legacy = norm_legacy.normalize(x_raw)

        out_official = bridge_official.forward_validation(
            x_norm_official, small_variables, interval=6, steps=2
        )
        out_legacy = bridge_legacy.forward_validation(
            x_norm_legacy, small_variables, interval=6, steps=2
        )

    # Outputs should differ because of different diff_mean handling
    assert not torch.allclose(out_official, out_legacy, atol=1e-5), (
        "Rollout with official vs legacy policy should produce different outputs "
        "when NPZ has nonzero diff_mean"
    )
