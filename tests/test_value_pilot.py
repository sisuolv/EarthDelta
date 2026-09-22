"""Counterexamples at the shared-F0 pilot's artifact and decision boundaries."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from earthdelta.bank_training import build_dynamic_bank
from earthdelta.value_pilot import (
    PilotViolation, action_plan, assert_disjoint_roles, bank_digest, component_gains,
    component_losses, load_dynamic, qualify_losses, require_pass, require_role,
    save_dynamic, validate_cache_rows,
    weighted_projection, legal_features,
)


def test_dynamic_is_persisted_and_loaded_in_another_process(tmp_path):
    bank = build_dynamic_bank(8, [0, 1], seed=19)
    with torch.no_grad():
        bank[1].up[2].weight.normal_()
    identity = {"hidden_size": 8, "target_blocks": [0, 1], "reference_kind": "base_frozen", "checkpoint_sha256": "abc"}
    record = save_dynamic(tmp_path / "dynamic.pt", bank, identity, {"expert_index": 2})
    (tmp_path / "record.json").write_text(json.dumps(record))
    script = "from earthdelta.value_pilot import *; import sys; r=read_json(sys.argv[1]); b=load_dynamic(r,r['identity']); print(bank_digest(b))"
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "record.json")], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == bank_digest(bank)
    with pytest.raises(PilotViolation, match="identity mismatch"):
        load_dynamic(record, {**identity, "checkpoint_sha256": "different"})
    with (tmp_path / "dynamic.pt").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(PilotViolation, match="bytes changed"):
        load_dynamic(record, identity)


def test_native_component_gains_equal_endpoints_and_hand_formula():
    rng = np.random.default_rng(312)
    r, y, u = [torch.from_numpy(rng.normal(size=(3, 5, 7, 8))) for _ in range(3)]
    lat = np.linspace(-85, 85, 7)
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[None, :, None], (5, 7, 8))
    hand = ((2 * (y.numpy() - r.numpy()) * u.numpy() - u.numpy() ** 2) * weights).sum((1, 2, 3)) / weights.sum()
    got = component_gains(r, y, r + u, lat)
    np.testing.assert_allclose(got.numpy(), hand, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(got, component_losses(r, y, lat) - component_losses(r + u, y, lat), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("status", ["FAIL_IMPLEMENTATION", "BLOCKED", None, False])
def test_failed_dependencies_are_not_admitted(status):
    with pytest.raises(PilotViolation):
        require_pass({"status": status}, "predecessor")


def test_quality_gate_rejects_all_pair_degradation_despite_finite_state():
    thresholds = {"min_panel_issues_per_expert": 8, "max_mean_loss24_ratio_to_reference": 1.001,
                  "max_mean_loss72_ratio_to_reference": 1.05}
    ref = np.ones((8, 3))
    assert qualify_losses(ref, ref * 0.9, 0.1, True, thresholds)["status"] == "PASS"
    for candidate, reloaded in [(ref * 5, True), (ref * [1, 0.9, 1.051], True), (ref * 0.9, False)]:
        assert qualify_losses(ref, candidate, 0.1, reloaded, thresholds)["status"] != "PASS"
    assert qualify_losses(ref[:7], ref[:7], 0.1, True, thresholds)["status"] != "PASS"
    assert qualify_losses(ref, ref, 0.0, True, thresholds)["status"] != "PASS"


def test_whole_cache_coverage_not_successful_subset():
    actions = [{"candidate_id": "R"}, {"candidate_id": "e0_a1250"}]
    rows = [{"issue_id": i, "candidate_id": a["candidate_id"], "status": "PASS", "identity": {"R": "same"}}
            for i in ("a", "b") for a in actions]
    assert validate_cache_rows(rows, ["a", "b"], actions, {"R": "same"})
    for bad in [rows[:-1], rows + rows[:1], [{**rows[0], "identity": {"R": "wrong"}}, *rows[1:]]]:
        with pytest.raises(PilotViolation):
            validate_cache_rows(bad, ["a", "b"], actions, {"R": "same"})


def test_role_and_amplitude_access_boundaries():
    with pytest.raises(PilotViolation):
        require_role({"role": "sprint_holdout", "known_action": True}, "policy_fit")
    with pytest.raises(PilotViolation):
        require_role({"role": "policy_fit", "known_action": False}, "policy_fit")
    require_role({"role": "policy_fit", "known_action": True}, "policy_fit")


def test_complete_support_not_just_issue_time_must_be_disjoint():
    rows = [{"role": "fit", "support_start_utc": "2019-01-01T00:00:00Z", "support_end_utc": "2019-01-04T00:00:00Z"},
            {"role": "calibration", "support_start_utc": "2019-01-03T18:00:00Z", "support_end_utc": "2019-01-07T00:00:00Z"}]
    with pytest.raises(PilotViolation, match="crosses roles"):
        assert_disjoint_roles(rows)


def test_candidate_ids_and_hold_window_survive_exactly():
    for a in (0.125, 0.1875, 0.25):
        row = {"candidate_id": f"e2_{a}", "expert": 2, "alpha": a}
        plan = action_plan(row)
        assert plan.plan_id == row["candidate_id"]
        assert tuple(plan.coefficients_at(3)) == (0.0, 0.0, a, 0.0)
        assert tuple(plan.coefficients_at(4)) == (0.0, 0.0, 0.0, 0.0)


def test_projection_is_q_orthogonal_not_an_unweighted_pool():
    torch.manual_seed(413)
    lat = np.linspace(-85, 85, 16)
    x = torch.randn(3, 5, 16, 32, dtype=torch.float64)
    projected = weighted_projection(x, lat, (4, 8))
    native = component_losses(x, torch.zeros_like(x), lat)
    assert torch.all(projected.square().sum(-1) < native)
    boxes = torch.randn(3, 5, 4, 8, dtype=torch.float64)
    piecewise_constant = boxes.repeat_interleave(4, -2).repeat_interleave(4, -1)
    projected = weighted_projection(piecewise_constant, lat, (4, 8))
    native = component_losses(piecewise_constant, torch.zeros_like(x), lat)
    torch.testing.assert_close(projected.square().sum(-1), native, rtol=1e-12, atol=1e-12)


def test_legal_features_depend_on_real_history_and_calendar():
    x = torch.ones(1, 5, 16, 32)
    a = legal_features(x, x.clone(), "2019-03-10T06:00:00Z")
    b = legal_features(x, x - 1, "2019-03-10T06:00:00Z")
    assert a.shape == (2 * 5 * 4 * 8 + 4,)
    assert not torch.equal(a, b)
    assert torch.equal(a[:160], b[:160])
