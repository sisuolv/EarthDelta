from __future__ import annotations

import numpy as np
import pytest

from earthdelta.v8.standard_metrics import MetricContract, MetricError, l6_loss, l69_loss, paired_bootstrap


ROOT = "/mnt/afs/260010168/earthdelta_v8_planning_20260926T182037Z_r4"


def test_metric_contract_has_signed_identity():
    c = MetricContract.from_json(f"{ROOT}/METRIC_CONTRACT_R4.json")
    assert c.primary_metrics == ("Z500@72h", "T850@72h")
    assert c.bootstrap_draws >= 10000
    assert c.variables[0]["source_name"] == "geopotential_500"
    assert c.variables[0]["channel_index"] == 11


def test_l69_uses_terminal_std_and_l6_requires_all_cells():
    p = np.zeros((2, 69)); t = np.ones((2, 69)); s = np.ones(69)
    assert l69_loss(p, t, s) == pytest.approx(1.)
    with pytest.raises(MetricError):
        l6_loss({"Z500@6h": np.zeros((2,))}, {}, {"Z500@6h": 1.}, ["Z500"], [6, 24])


def test_bootstrap_refuses_small_draw_count():
    x = np.zeros((2, 1, 1, 1)); lat = np.array([0.])
    with pytest.raises(MetricError):
        paired_bootstrap(x, x + 1, x, lat, ["2021-01-01T00:00:00Z", "2021-01-02T00:00:00Z"], draws=9999)
