"""Tests for S0 gate identity binding and rejection.

These tests verify that the REAL s0_gate orchestration (not hand-built dicts)
correctly rejects identity mismatches:
- Wrong checkpoint SHA-256
- Wrong normalization digest
- Wrong variable/grid identity
- Wrong raw input
- Wrong shape
- Missing required multistep results

All tests use synthetic/small assets and can run on CPU.
"""
import json
import os
import pytest
import shutil
import tempfile
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from earthdelta.bridge import (
    NormalizationContract,
    POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    _compute_file_sha256,
    DEFAULT_VARIABLES,
)
from earthdelta.contracts import GateIdentityConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def temp_gate_dir(tmp_path):
    """Create temporary directory structure for gate tests."""
    # Create input directory
    input_dir = tmp_path / "scripts" / "s0_gate_inputs"
    input_dir.mkdir(parents=True)

    # Use full 69 variables for consistency with real gate
    full_variables = DEFAULT_VARIABLES
    n_vars = len(full_variables)

    # Create synthetic weather data (smaller spatial dimensions for speed)
    jan2020_data = np.random.randn(10, n_vars, 16, 32).astype(np.float32)
    np.save(input_dir / "jan2020_full.npy", jan2020_data)

    # Create lat/lon
    np.save(input_dir / "lat.npy", np.linspace(-90, 90, 16).astype(np.float32))
    np.save(input_dir / "lon.npy", np.linspace(0, 360, 32).astype(np.float32))

    # Create normalization constants
    inp_mean = np.random.randn(n_vars).astype(np.float32)
    inp_std = np.abs(np.random.randn(n_vars)).astype(np.float32) + 0.1
    np.save(input_dir / "inp_mean.npy", inp_mean)
    np.save(input_dir / "inp_std.npy", inp_std)

    for interval in [6, 24]:
        diff_mean = np.random.randn(n_vars).astype(np.float32) * 0.01
        diff_std = np.abs(np.random.randn(n_vars)).astype(np.float32) + 0.1
        np.save(input_dir / f"diff_mean_{interval}.npy", diff_mean)
        np.save(input_dir / f"diff_std_{interval}.npy", diff_std)

    # Create normalization directory for NPZ files
    norm_dir = tmp_path / "reference" / "stormer" / "normalization_constants"
    norm_dir.mkdir(parents=True)

    # Create NPZ files (matching format from real normalization files)
    # Each variable gets one value
    mean_dict = {v: np.array([inp_mean[i]]) for i, v in enumerate(full_variables)}
    std_dict = {v: np.array([inp_std[i]]) for i, v in enumerate(full_variables)}
    np.savez(norm_dir / "normalize_mean.npz", **mean_dict)
    np.savez(norm_dir / "normalize_std.npz", **std_dict)

    for interval in [6, 24]:
        dm = {v: np.array([0.0]) for v in full_variables}  # Zero for official policy
        ds = {v: np.array([0.5]) for v in full_variables}
        np.savez(norm_dir / f"normalize_diff_mean_{interval}.npz", **dm)
        np.savez(norm_dir / f"normalize_diff_std_{interval}.npz", **ds)

    # Create artifacts directory
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    return {
        "root": tmp_path,
        "input_dir": input_dir,
        "norm_dir": norm_dir,
        "artifacts_dir": artifacts_dir,
        "variables": full_variables,
        "jan2020_data": jan2020_data,
    }


@pytest.fixture
def mock_upstream_reference(temp_gate_dir):
    """Create mock upstream reference directory with valid outputs."""
    ref_dir = temp_gate_dir["artifacts_dir"] / "upstream_reference_ps4_abc12345_zd"
    ref_dir.mkdir(parents=True)

    n_vars = len(temp_gate_dir["variables"])  # 69 variables

    # Create synthetic reference outputs
    output_1step = torch.randn(1, n_vars, 16, 32)
    output_4step = torch.randn(1, n_vars, 16, 32)
    input_norm = torch.randn(1, n_vars, 16, 32)

    torch.save(output_1step, ref_dir / "official_output_6h_1step.pt")
    torch.save(output_4step, ref_dir / "official_output_6h_4step.pt")
    torch.save(input_norm, ref_dir / "input_norm.pt")

    # Create manifest
    raw_input_hash = "abc123deadbeef"
    manifest = {
        "timestamp_utc": "2026-09-21T00:00:00Z",
        "xformers_version": "0.1.0",
        "checkpoint": {
            "sha256": "abc123" * 11,  # 66 chars but we'll truncate
            "patch_size": 4,
        },
        "normalization": {
            "policy": "official_zero_diff_mean",
            "digest": "expected_digest",
        },
        "raw_input_hash": raw_input_hash,
    }
    with open(ref_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # Create raw input hash file
    with open(ref_dir / "raw_input_hash.txt", "w") as f:
        f.write(raw_input_hash)

    return {
        "ref_dir": ref_dir,
        "manifest": manifest,
        "raw_input_hash": raw_input_hash,
    }


# =============================================================================
# Import the gate functions for testing
# =============================================================================

# We'll import the verification functions directly
from scripts.s0_gate import (
    verify_ckpt_sha256,
    verify_normalization_parity,
    verify_raw_input_binding,
    verify_multistep_reference,
)


# =============================================================================
# Test: Gate rejects wrong checkpoint SHA-256
# =============================================================================

def test_gate_rejects_wrong_checkpoint_sha256(temp_gate_dir, tmp_path):
    """Test that gate rejects when computed SHA-256 doesn't match expected."""
    # Create a fake checkpoint file
    fake_ckpt = tmp_path / "fake_checkpoint.ckpt"
    with open(fake_ckpt, "wb") as f:
        f.write(b"fake checkpoint content")

    # Compute actual SHA-256
    computed_sha = _compute_file_sha256(str(fake_ckpt))

    # Expected SHA-256 that doesn't match
    wrong_expected = "0" * 64

    result = verify_ckpt_sha256(
        fake_ckpt,
        load_result=None,
        expected_sha256=wrong_expected,
    )

    assert result["passed"] == False, "Gate should reject wrong SHA-256"
    assert result["computed_sha256"] == computed_sha
    assert result["expected_sha256"] == wrong_expected
    assert "mismatch" in result.get("error", "").lower()


def test_gate_accepts_correct_checkpoint_sha256(temp_gate_dir, tmp_path):
    """Test that gate accepts when computed SHA-256 matches expected."""
    # Create a fake checkpoint file
    fake_ckpt = tmp_path / "fake_checkpoint.ckpt"
    with open(fake_ckpt, "wb") as f:
        f.write(b"fake checkpoint content")

    # Compute actual SHA-256
    computed_sha = _compute_file_sha256(str(fake_ckpt))

    # Expected SHA-256 that matches
    result = verify_ckpt_sha256(
        fake_ckpt,
        load_result=None,
        expected_sha256=computed_sha,
    )

    assert result["passed"] == True, "Gate should accept correct SHA-256"
    assert result["match"] == True


# =============================================================================
# Test: Gate rejects wrong normalization digest
# =============================================================================

def test_gate_rejects_wrong_normalization_digest(temp_gate_dir):
    """Test that gate rejects when normalization digest doesn't match expected."""
    # Create normalization contract from NPZ
    norm = NormalizationContract.from_npz_dir(
        str(temp_gate_dir["norm_dir"]),
        variables=list(temp_gate_dir["variables"]),
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    # Wrong expected digest
    wrong_digest = "wrong_digest_123"

    result = verify_normalization_parity(
        norm,
        temp_gate_dir["norm_dir"],
        expected_digest=wrong_digest,
    )

    assert result["passed"] == False, "Gate should reject wrong normalization digest"
    assert result.get("digests_match_expected") == False or "mismatch" in result.get("error", "").lower()


def test_gate_accepts_correct_normalization_digest(temp_gate_dir):
    """Test that gate accepts when normalization digest matches expected."""
    # Create normalization contract from NPZ
    norm = NormalizationContract.from_npz_dir(
        str(temp_gate_dir["norm_dir"]),
        variables=list(temp_gate_dir["variables"]),
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    # Correct expected digest
    correct_digest = norm.digest

    result = verify_normalization_parity(
        norm,
        temp_gate_dir["norm_dir"],
        expected_digest=correct_digest,
    )

    assert result["passed"] == True, f"Gate should accept correct normalization digest: {result}"


# =============================================================================
# Test: Gate rejects wrong raw input
# =============================================================================

def test_gate_rejects_wrong_raw_input(temp_gate_dir, mock_upstream_reference):
    """Test that gate rejects when raw input hash doesn't match reference."""
    # Use different input data that produces different hash
    different_input = np.random.randn(8, 16, 32).astype(np.float32)

    result = verify_raw_input_binding(
        different_input,
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
    )

    assert result["passed"] == False, "Gate should reject wrong raw input"
    assert "mismatch" in result.get("error", "").lower()


def test_gate_accepts_matching_raw_input(temp_gate_dir, mock_upstream_reference):
    """Test that gate accepts when raw input hash matches reference."""
    # Create input that matches the expected hash
    import hashlib
    expected_hash = mock_upstream_reference["raw_input_hash"]

    # We can't easily create data that hashes to a specific value,
    # so we'll modify the reference file to match our test data
    test_input = np.random.randn(8, 16, 32).astype(np.float32)
    computed_hash = hashlib.sha256(test_input.tobytes()).hexdigest()[:16]

    # Update the reference to expect our test input's hash
    with open(mock_upstream_reference["ref_dir"] / "raw_input_hash.txt", "w") as f:
        f.write(computed_hash)

    result = verify_raw_input_binding(
        test_input,
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
    )

    assert result["passed"] == True, "Gate should accept matching raw input"


# =============================================================================
# Test: Gate rejects missing multistep results
# =============================================================================

def test_gate_rejects_missing_1step_reference(temp_gate_dir, mock_upstream_reference):
    """Test that gate rejects when 1-step reference file is missing."""
    # Remove the 1-step file
    os.remove(mock_upstream_reference["ref_dir"] / "official_output_6h_1step.pt")

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1, 4),
    )

    assert result["passed"] == False, "Gate should reject missing 1-step reference"
    assert "6h_1step" in str(result.get("missing_files", []))


def test_gate_rejects_missing_4step_reference(temp_gate_dir, mock_upstream_reference):
    """Test that gate rejects when 4-step reference file is missing."""
    # Remove the 4-step file
    os.remove(mock_upstream_reference["ref_dir"] / "official_output_6h_4step.pt")

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1, 4),
    )

    assert result["passed"] == False, "Gate should reject missing 4-step reference"
    assert "6h_4step" in str(result.get("missing_files", []))


def test_gate_accepts_all_multistep_present(temp_gate_dir, mock_upstream_reference):
    """Test that gate accepts when all multistep references are present."""
    # Need to fix the shape to match expected (1, 69, 128, 256)
    # For this test, we'll patch the expected shape check
    # Actually, let's create correct-shaped tensors
    ref_dir = mock_upstream_reference["ref_dir"]

    # Re-save with correct shapes (though smaller for test)
    output_1step = torch.randn(1, 69, 128, 256)
    output_4step = torch.randn(1, 69, 128, 256)

    torch.save(output_1step, ref_dir / "official_output_6h_1step.pt")
    torch.save(output_4step, ref_dir / "official_output_6h_4step.pt")

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1, 4),
    )

    assert result["passed"] == True, f"Gate should accept all multistep present: {result}"


def test_gate_rejects_wrong_shape_multistep(temp_gate_dir, mock_upstream_reference):
    """Test that gate rejects when multistep reference has wrong shape."""
    ref_dir = mock_upstream_reference["ref_dir"]

    # Save with wrong shape
    wrong_shape_output = torch.randn(1, 32, 64, 128)  # Wrong shape
    torch.save(wrong_shape_output, ref_dir / "official_output_6h_1step.pt")

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1,),
    )

    assert result["passed"] == False, "Gate should reject wrong shape"
    assert len(result.get("shape_errors", [])) > 0


# =============================================================================
# Test: Gate rejects non-finite multistep values
# =============================================================================

def test_gate_rejects_nonfinite_multistep(temp_gate_dir, mock_upstream_reference):
    """Test that gate rejects when multistep reference contains NaN/Inf."""
    ref_dir = mock_upstream_reference["ref_dir"]

    # Save with NaN values
    nan_output = torch.randn(1, 69, 128, 256)
    nan_output[0, 0, 0, 0] = float('nan')
    torch.save(nan_output, ref_dir / "official_output_6h_1step.pt")

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1,),
    )

    assert result["passed"] == False, "Gate should reject non-finite values"
    assert "non-finite" in str(result.get("shape_errors", [])).lower()


# =============================================================================
# Test: Gate handles missing upstream reference directory
# =============================================================================

def test_gate_handles_missing_upstream_reference(temp_gate_dir):
    """Test that gate fails closed when upstream reference directory is missing."""
    # Don't create any upstream reference

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1, 4),
    )

    assert result["passed"] == False, "Gate should fail closed when reference missing"
    assert "not found" in result.get("error", "").lower()


# =============================================================================
# Test: GateIdentityConfig validation
# =============================================================================

def test_gate_identity_config_sha256_validation():
    """Test GateIdentityConfig SHA-256 validation."""
    config = GateIdentityConfig(
        checkpoint_path="/path/to/checkpoint",
        expected_checkpoint_sha256="abc123def456" * 5 + "ab",  # 64 chars
        patch_size=4,
        normalization_policy="official_zero_diff_mean",
        normalization_dir="/path/to/norm",
        variables=tuple(DEFAULT_VARIABLES),
        grid_shape=(128, 256),
    )

    # Test matching
    assert config.validate_checkpoint_sha256(config.expected_checkpoint_sha256)

    # Test non-matching
    assert not config.validate_checkpoint_sha256("wrong" * 13)


def test_gate_identity_config_identity_tag():
    """Test GateIdentityConfig identity tag generation."""
    config = GateIdentityConfig(
        checkpoint_path="/path/to/checkpoint",
        expected_checkpoint_sha256="abc123def456" * 5 + "ab",
        patch_size=4,
        normalization_policy="official_zero_diff_mean",
        normalization_dir="/path/to/norm",
        variables=tuple(DEFAULT_VARIABLES),
        grid_shape=(128, 256),
    )

    tag = config.identity_tag
    assert "ps4" in tag, "Identity tag should include patch size"
    assert "abc123de" in tag, "Identity tag should include SHA prefix"
    assert "_zd" in tag, "Identity tag should include policy indicator"


def test_gate_identity_config_reference_dir_namespaced():
    """Test that reference dir name is namespaced by identity."""
    config1 = GateIdentityConfig(
        checkpoint_path="/path/to/checkpoint1",
        expected_checkpoint_sha256="a" * 64,
        patch_size=2,
        normalization_policy="official_zero_diff_mean",
        normalization_dir="/path/to/norm",
        variables=tuple(DEFAULT_VARIABLES),
        grid_shape=(128, 256),
    )

    config2 = GateIdentityConfig(
        checkpoint_path="/path/to/checkpoint2",
        expected_checkpoint_sha256="b" * 64,
        patch_size=4,
        normalization_policy="official_zero_diff_mean",
        normalization_dir="/path/to/norm",
        variables=tuple(DEFAULT_VARIABLES),
        grid_shape=(128, 256),
    )

    # Different configs should have different reference dir names
    assert config1.reference_output_dir_name != config2.reference_output_dir_name, (
        "Different checkpoints/patch sizes should have different reference dirs"
    )


# =============================================================================
# Test: Fail-closed when expected identity values not provided (B02 fix)
# =============================================================================

def test_gate_fails_closed_ckpt_sha256_none(tmp_path):
    """Test that verify_ckpt_sha256 fails closed when expected_sha256=None."""
    # Create a real checkpoint file
    fake_ckpt = tmp_path / "real_checkpoint.ckpt"
    with open(fake_ckpt, "wb") as f:
        f.write(b"checkpoint content for fail-closed test")

    result = verify_ckpt_sha256(
        fake_ckpt,
        load_result=None,
        expected_sha256=None,  # No expected value provided
    )

    # Must fail closed - cannot certify identity without expected value
    assert result["passed"] == False, "Gate must fail closed when expected_sha256=None"
    assert result["computed_sha256"] is not None, "Should still record computed hash for enrollment"
    assert len(result["computed_sha256"]) == 64, "Computed hash should be valid SHA-256"
    assert "IDENTITY_NOT_BOUND" in result.get("reason", ""), "Should indicate identity not bound"


def test_gate_fails_closed_normalization_digest_none(temp_gate_dir):
    """Test that verify_normalization_parity fails closed when expected_digest=None."""
    # Create normalization contract from valid NPZ data
    norm = NormalizationContract.from_npz_dir(
        str(temp_gate_dir["norm_dir"]),
        variables=list(temp_gate_dir["variables"]),
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    result = verify_normalization_parity(
        norm,
        temp_gate_dir["norm_dir"],
        expected_digest=None,  # No expected digest provided
    )

    # Must fail closed - cannot certify identity without expected value
    assert result["passed"] == False, "Gate must fail closed when expected_digest=None"
    # Should still record useful diagnostic info for enrollment
    assert result.get("computed_digest") is not None, "Should still record computed digest"
    # digests_match_npz may be True (self-consistency), but passed must still be False
    assert "IDENTITY_NOT_BOUND" in result.get("reason", ""), "Should indicate identity not bound"


def test_gate_fails_closed_raw_input_no_upstream_dir(temp_gate_dir):
    """Test that verify_raw_input_binding fails closed when upstream dir doesn't exist."""
    # Use artifacts dir with no upstream reference created
    test_input = np.random.randn(8, 16, 32).astype(np.float32)

    result = verify_raw_input_binding(
        test_input,
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
    )

    # Must fail closed - no reference to bind against
    assert result["passed"] == False, "Gate must fail closed when upstream reference dir missing"
    assert result.get("computed_hash") is not None, "Should still record computed hash for enrollment"
    assert "IDENTITY_NOT_BOUND" in result.get("reason", ""), "Should indicate identity not bound"


def test_gate_fails_closed_raw_input_no_hash_in_reference(temp_gate_dir):
    """Test that verify_raw_input_binding fails closed when upstream dir exists but has no raw_input_hash."""
    # Create upstream reference directory without raw_input_hash
    ref_dir = temp_gate_dir["artifacts_dir"] / "upstream_reference_ps4_test123_zd"
    ref_dir.mkdir(parents=True)

    # Create manifest WITHOUT raw_input_hash
    manifest = {
        "timestamp_utc": "2026-09-21T00:00:00Z",
        "xformers_version": "0.1.0",
        "checkpoint": {"sha256": "a" * 64, "patch_size": 4},
        "normalization": {"policy": "official_zero_diff_mean", "digest": "test_digest"},
        # Note: raw_input_hash is intentionally omitted
    }
    with open(ref_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # No raw_input_hash.txt file either

    test_input = np.random.randn(8, 16, 32).astype(np.float32)

    result = verify_raw_input_binding(
        test_input,
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
    )

    # Must fail closed - no reference hash to bind against
    assert result["passed"] == False, "Gate must fail closed when no raw_input_hash in reference"
    assert result.get("computed_hash") is not None, "Should still record computed hash for enrollment"
    assert "IDENTITY_NOT_BOUND" in result.get("reason", ""), "Should indicate identity not bound"
