import json
from pathlib import Path

import pytest

from earthdelta.probe import (
    AccessViolation,
    BudgetBlocked,
    BudgetLedger,
    ProbeAccessController,
    decision_from_values,
    issue_index,
    required_indices,
    shard_issue_ids,
    verify_completed_shard,
)
from earthdelta.probe.contracts import ProbeSpec


def _spec(tmp_path):
    store = tmp_path / "2020.zarr"
    raw = {
        "schema": "earthdelta.probe.headroom.spec.v1",
        "spec_id": "test",
        "bindings": {"data_store": {"path": str(store), "absent_indices": [0]}},
        "metrics": {"variables": [], "leads_hours": [6, 24], "rollout_steps": {"6": 1, "24": 2},
                     "primary_cells": ["Z500@72h", "T850@72h"]},
        "issue_sets": {
            "D": {"n": 1, "issue_times_utc": ["2020-01-02T00:00:00Z"],
                  "list_sha256": __import__("hashlib").sha256(b"2020-01-02T00:00:00Z").hexdigest()},
        },
        "resources": {"h100": {"card_hours_cap_total": 12}, "rtx5090": {"card_hours_cap_total": 64}},
        "decision": {},
    }
    return ProbeSpec(raw, "spec.json", "hash")


def test_issue_index_and_required_window(tmp_path):
    spec = _spec(tmp_path)
    assert issue_index("2020-01-02T00:00:00Z") == 4
    assert required_indices(spec) == {4, 5, 6}


def test_access_guard_rejects_path_and_index(tmp_path):
    spec = _spec(tmp_path)
    guard = ProbeAccessController(spec, tmp_path / "ACCESS_LEDGER.jsonl")
    with pytest.raises(AccessViolation):
        guard.authorize(tmp_path / "2019.zarr", [4], purpose="bad", stage="P0")
    with pytest.raises(AccessViolation):
        guard.authorize(spec.store_path, [999], purpose="bad", stage="P0")
    with pytest.raises(AccessViolation):
        guard.authorize(spec.store_path, [0], purpose="bad", stage="P0")


@pytest.mark.parametrize(
    "values, expected",
    [
        ({"Z500@72h": 2.1, "T850@72h": 2.2}, "GO_V8"),
        ({"Z500@72h": 0.6, "T850@72h": 0.1}, "RELAX_ONCE"),
        ({"Z500@72h": 0.1, "T850@72h": 0.1}, "STOP_PARAM_EDIT"),
    ],
)
def test_frozen_decision_rule(values, expected):
    all_cells = {"Z500@72h": values["Z500@72h"], "T850@72h": values["T850@72h"], "Z500@6h": 0.0}
    assert decision_from_values(
        menu_grad_primary=values,
        static_primary={"Z500@72h": 0.5, "T850@72h": 0.5},
        random_primary={"Z500@72h": 0.5, "T850@72h": 0.5},
        ttt_primary={"Z500@72h": 0.0, "T850@72h": 0.0},
        all_menu_grad=all_cells,
    ) == expected


def test_budget_refuses_overrun_and_records_failures(tmp_path):
    ledger = BudgetLedger(tmp_path / "BUDGET_LEDGER.jsonl", h100_cap=1.0, rtx5090_cap=2.0)
    ledger.record(job_id="x", gpu="H100", card_hours=0.75, stage="P2", status="FAILED")
    with pytest.raises(BudgetBlocked):
        ledger.record(job_id="y", gpu="H100", card_hours=0.30, stage="P3", status="SUBMISSION_BLOCKED")
    ledger.record(job_id="z", gpu="RTX5090", card_hours=1.0, stage="P4", status="PREEMPTED")
    assert ledger.used("H100") == pytest.approx(0.75)


def test_shards_and_resume_payload_hash(tmp_path):
    ids = [str(i) for i in range(17)]
    shards = shard_issue_ids(ids, max_issues=8)
    assert [len(x) for x in shards] == [8, 8, 1]
    payload = tmp_path / "payload.json"
    payload.write_text("{}")
    import hashlib
    row = {
        "status": "OBSERVED", "input_sha256": "in", "spec_sha256": "spec",
        "payload_file": str(payload),
        "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(row))
    assert verify_completed_shard(receipt, expected_input_hash="in", expected_spec_sha256="spec")
    row["payload_sha256"] = "bad"
    receipt.write_text(json.dumps(row))
    assert not verify_completed_shard(receipt, expected_input_hash="in", expected_spec_sha256="spec")
