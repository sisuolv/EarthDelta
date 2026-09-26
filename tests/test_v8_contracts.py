from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from earthdelta.v8.approvals import ApprovalError, ApprovalVerifier
from earthdelta.v8.data_repair import DataRepairError, FailClosedRepair, cross_year_memory_only
from earthdelta.v8.engine_qualification import EngineQualificationError, enforce_same_engine
from earthdelta.v8.receipt_registry import ReceiptError, ReceiptRegistry
from earthdelta.v8.rho_parameterization import epsilon_for_rho, signed_rho_grid
from earthdelta.v8.standard_metrics import MetricError, pooled_rmse
from earthdelta.v8.support_contract import validate_issue_list


ROOT = Path(__file__).resolve().parents[1]
R4 = Path("/mnt/afs/260010168/earthdelta_v8_planning_20260926T182037Z_r4")
APP = Path("/mnt/afs/260010168/earthdelta_v8_approvals/approvals.json")
APP_SHA = APP.with_name("approvals.sha256")


def test_approval_hash_and_scope_are_verified():
    verifier = ApprovalVerifier(APP, APP_SHA,
                                expected_prompt_sha="a9ac5e9c7c5cd06b43c1ccb290f86b349a7acbfa2eaafd61b2f7c20fd5bb358c",
                                expected_amendments_sha="36d1342bd5f147be945253dd1a40fb252e1ed24ac22d424dd86c7e4496918286",
                                expected_r3_manifest_sha="020797ecb13129e111cd02fdd31c311f9b469cb840a1489af4a74fb5267b247f")
    verifier.require(["T01a_code"], available_receipts=["r4_acceptance_receipt"])
    with pytest.raises(ApprovalError):
        verifier.require(["T01_read_2018_fit"])
    with pytest.raises(ApprovalError):
        verifier.require(["unknown_scope"])


def test_approval_sha_mismatch_is_fail_closed(tmp_path):
    app = tmp_path / "approvals.json"; sha = tmp_path / "approvals.sha256"
    app.write_text(APP.read_text()); sha.write_text("0" * 64 + "  approvals.json\n")
    with pytest.raises(ApprovalError):
        ApprovalVerifier(app, sha)


def test_mutually_exclusive_scope_is_rejected(tmp_path):
    payload = {"issued": True, "scopes": [
        {"scope_id": "a", "approved": True, "requires_receipts": [], "budget_cap_card_hours": 0, "mutually_exclusive_with": ["b"]},
        {"scope_id": "b", "approved": True, "requires_receipts": [], "budget_cap_card_hours": 0, "mutually_exclusive_with": ["a"]}],
        "binds": {}}
    app = tmp_path / "a.json"; sha = tmp_path / "a.sha256"; app.write_text(json.dumps(payload));
    sha.write_text(hashlib.sha256(app.read_bytes()).hexdigest() + "  a.json\n")
    v = ApprovalVerifier(app, sha)
    with pytest.raises(ApprovalError):
        v.require(["a", "b"])


def test_registry_is_closed_and_cycle_checked():
    reg = ReceiptRegistry(R4)
    assert reg.spec("contract_receipt").stage_id == "T01a_contract"
    assert reg.hash_manifest()["RECEIPT_REGISTRY.json"]


def test_support_window_cross_year_negative():
    with pytest.raises(ValueError):
        validate_issue_list("bad", ["2021-12-31T18:00:00Z"], role_year=2021)


def test_data_repair_missing_source_does_not_write_marker(tmp_path):
    repair = FailClosedRepair(tmp_path / "v8")
    result = repair.repair(role="fit", year=2019, chunks=[{"chunk_id": "x"}], source_reader=None)
    assert result.status == "BLOCKED"
    assert result.completion_marker is None
    assert not list((tmp_path / "v8").rglob("COMPLETE.json"))


def test_2022_payload_guard_and_memory_exception():
    repair = FailClosedRepair("/tmp/ed-v8-test")
    with pytest.raises(DataRepairError):
        repair.record_access(role="confirm", year=2022, path="x", operation="GET", array_payload=True)
    result = cross_year_memory_only(local_qc_failed=True, year=2021, chunk={"chunk_id": "c"},
                                    source_reader=lambda _: np.ones((2,)))
    assert result["memory_only"] and result["persisted"] is False
    with pytest.raises(DataRepairError):
        cross_year_memory_only(local_qc_failed=False, year=2021, chunk={}, source_reader=lambda _: np.ones((1,)))


def test_metrics_pool_then_root_and_nonfinite_is_error():
    lat = np.array([-45., 45.])
    truth = np.zeros((2, 1, 2, 2)); forecast = np.ones_like(truth)
    assert np.allclose(pooled_rmse(forecast, truth, lat), [1.])
    forecast[0, 0, 0, 0] = np.nan
    with pytest.raises(MetricError):
        pooled_rmse(forecast, truth, lat)


def test_rho_rule_has_no_unfrozen_base_parameter():
    assert epsilon_for_rho(0.25) == pytest.approx(0.0625)
    assert signed_rho_grid()[-1] == 4.0


def test_same_issue_mixed_engine_rejected():
    with pytest.raises(EngineQualificationError):
        enforce_same_engine({"F0": "H100", "edit": "spot_5090"})
