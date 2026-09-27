"""Negative and positive tests for the probe's full-trajectory convention."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from earthdelta.probe.estimators import per_issue_oracle, ttt_oracle
from earthdelta.probe.rollout import l6_per_issue


def _load_p3():
    requested = os.environ.get("EARTHDELTA_P3_SCRIPT")
    path = Path(requested) if requested else Path(__file__).parents[1] / "scripts" / "probe_p3_directions.py"
    spec = importlib.util.spec_from_file_location("earthdelta_test_p3", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _IdentityBridge:
    def normalize(self, value):
        return value

    def denormalize(self, value):
        return value


def _trajectory(step_values: list[float]) -> torch.Tensor:
    value = torch.zeros((1, 21, 69, 2, 1), dtype=torch.float32)
    for step, number in enumerate(step_values):
        value[:, step, 0] = number
    return value


def _contract_inputs():
    pred = _trajectory(list(range(21)))
    truth = _trajectory([0.0] * 21)
    variables = [{"name": "Z500", "channel_index": 0}]
    leads = [6, 24, 72, 120]
    denominator = {f"Z500@{lead}h": 1.0 for lead in leads}
    return pred, truth, variables, leads, denominator


def test_cell_mse_uses_physical_steps_and_rejects_endpoint_slice():
    p3 = _load_p3()
    pred, truth, variables, leads, _ = _contract_inputs()
    cells = p3.cell_mse(pred, truth, _IdentityBridge(), np.array([-45.0, 45.0]), variables, leads)
    np.testing.assert_allclose(cells[0], [1.0, 16.0, 144.0, 400.0])
    with pytest.raises(ValueError, match="full \[B,21"):
        p3.cell_mse(pred[:, [1, 4, 12, 20]], truth[:, [1, 4, 12, 20]],
                    _IdentityBridge(), np.array([-45.0, 45.0]), variables, leads)


def test_l6_uses_full_truth_and_physical_steps():
    _p3 = _load_p3()
    pred, truth, variables, leads, denominator = _contract_inputs()
    value = l6_per_issue(pred, truth, _IdentityBridge(), np.array([-45.0, 45.0]),
                         variables, leads, denominator)
    assert float(value) == pytest.approx((1.0 + 16.0 + 144.0 + 400.0) / 4.0)
    with pytest.raises(ValueError, match="truth_raw must be a full"):
        l6_per_issue(pred, truth[:, [1, 4, 12, 20]], _IdentityBridge(), np.array([-45.0, 45.0]),
                     variables, leads, denominator)


def test_calibration_effect_uses_24h_position_four():
    p3 = _load_p3()
    edited = _trajectory([0.0, 0.0, 0.0, 99.0, 2.0] + [0.0] * 16)
    f0 = _trajectory([0.0] * 21)
    truth = _trajectory([0.0, 0.0, 0.0, 50.0, 1.0] + [0.0] * 16)
    result = p3.calibration_effect(edited, f0, truth, _IdentityBridge(), np.array([-45.0, 45.0]))
    assert result == pytest.approx(2.0)
    with pytest.raises(ValueError, match="full \[B,21"):
        p3.calibration_effect(edited[:, [1, 4, 12, 20]], f0[:, [1, 4, 12, 20]],
                              truth[:, [1, 4, 12, 20]], _IdentityBridge(), np.array([-45.0, 45.0]))


def test_ttt_and_oracle_keep_one_action_for_all_cells():
    values = np.array([
        [[1.0, 1.0], [0.5, 0.5], [1.3, 1.3]],
        [[1.0, 1.0], [0.6, 0.6], [1.2, 1.2]],
    ])
    oracle = per_issue_oracle(values)
    ttt = ttt_oracle(values)
    assert oracle["selected_arm"] == [1, 1]
    assert ttt["selected_arm"] == oracle["selected_arm"]
    assert ttt["label"].startswith("diagnostic")
