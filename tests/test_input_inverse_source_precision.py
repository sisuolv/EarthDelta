"""Regression tests for preserving official NPZ inverse-transform precision."""

import numpy as np
import torch

from earthdelta.bridge import (
    DEFAULT_VARIABLES,
    NormalizationContract,
    POLICY_OFFICIAL_ZERO_DIFF_MEAN,
)


def _write_npz_constants(directory, variables):
    mean = {
        name: np.array([1.234567890123 + index], dtype=np.float32)
        for index, name in enumerate(variables)
    }
    # Deliberately use float64 for std: this is the mixed-dtype shape present
    # in the official NPZ assets and is where a float32 public projection can
    # change the inverse arithmetic.
    std = {
        name: np.array([0.987654321987 + index], dtype=np.float64)
        for index, name in enumerate(variables)
    }
    np.savez(directory / "normalize_mean.npz", **mean)
    np.savez(directory / "normalize_std.npz", **std)
    for interval in (6, 12, 24):
        np.savez(
            directory / f"normalize_diff_mean_{interval}.npz",
            **{name: np.array([0.0], dtype=np.float32) for name in variables},
        )
        np.savez(
            directory / f"normalize_diff_std_{interval}.npz",
            **{name: np.array([1.0], dtype=np.float64) for name in variables},
        )


def test_from_npz_preserves_private_input_inverse_source(tmp_path):
    variables = DEFAULT_VARIABLES[:2]
    _write_npz_constants(tmp_path, variables)
    contract = NormalizationContract.from_npz_dir(
        str(tmp_path),
        variables=variables,
        intervals=(6,),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    assert contract.inp_std.dtype == torch.float32
    assert contract._reverse_inp_std.dtype == torch.float64

    x_norm = torch.tensor([[[[0.25]], [[-0.75]]]], dtype=torch.float32)
    actual = contract.denormalize(x_norm)
    mean_inverse, std_inverse = contract._reverse_parameters(
        contract._reverse_inp_mean.view(1, -1, 1, 1),
        contract._reverse_inp_std.view(1, -1, 1, 1),
        x_norm.device,
        x_norm.dtype,
    )
    expected = (x_norm - mean_inverse) / std_inverse
    assert torch.equal(actual, expected)

