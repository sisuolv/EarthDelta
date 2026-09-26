from __future__ import annotations

import json

from scripts.r5_audit_runner import _load_inputs, fp06_decision


def _fixture(**kwargs):
    comparisons = {
        "oracle_vs_Fs": {"status_vs_delta_min": "ABOVE"},
        "oracle_vs_M1": {"status_vs_delta_min": "ABOVE"},
        "M1_vs_Fs": {"status_vs_delta_min": "ABOVE"},
        "M3_vs_Fs": {"status_vs_delta_min": "ABOVE"},
        "M3_vs_M1": {"status_vs_delta_min": "ABOVE"},
        "M3_vs_Fs_72h": {"status_vs_delta_min": "ABOVE", "harm72_guard_pass": True},
    }
    comparisons.update(kwargs)
    return {"evidence_valid": True, "comparisons": comparisons}


def test_release_gate_rejects_absent_and_non_boolean_evidence(tmp_path):
    for value in (None, "false", 0, 1):
        obj = _fixture()
        if value is None:
            obj.pop("evidence_valid")
        else:
            obj["evidence_valid"] = value
        path = tmp_path / f"input_{repr(value)}.json"
        path.write_text(json.dumps(obj))
        loaded = _load_inputs(path)
        assert loaded["evidence_valid"] is False
        out = tmp_path / f"out_{repr(value)}"
        args = type("Args", (), {"inputs": path, "out": out, "delta_min": 0.0034})()
        assert fp06_decision(args) == 2


def test_release_gate_preserves_negative_science_as_successful_execution(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_fixture(oracle_vs_Fs={"status_vs_delta_min": "BELOW"})))
    out = tmp_path / "out"
    args = type("Args", (), {"inputs": path, "out": out, "delta_min": 0.0034})()
    assert fp06_decision(args) == 0
    receipt = json.loads((out / "decision.json").read_text())
    assert receipt["verdict"] == "STOP_CURRENT_BANK"
