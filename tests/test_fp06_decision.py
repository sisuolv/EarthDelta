from earthdelta.fp06_decision import decide_fp06
from scripts.r5_audit_runner import _load_inputs, fp06_decision


def comp(status, **extra):
    return {"status_vs_delta_min": status, **extra}


def base(**overrides):
    c = {
        "oracle_vs_Fs": comp("ABOVE"),
        "oracle_vs_M1": comp("ABOVE"),
        "M1_vs_Fs": comp("ABOVE"),
        "M3_vs_Fs": comp("ABOVE"),
        "M3_vs_M1": comp("ABOVE"),
        "M3_vs_Fs_72h": comp("ABOVE", harm72_guard_pass=True),
    }
    c.update(overrides)
    return {"evidence_valid": True, "comparisons": c}


def test_invalid_evidence_has_priority():
    assert decide_fp06({"evidence_valid": False})["verdict"] == "INVALID"


def test_rule_order_bank_then_static_then_selector():
    assert decide_fp06(base(oracle_vs_Fs=comp("BELOW")))["verdict"] == "STOP_CURRENT_BANK"
    assert decide_fp06(base(oracle_vs_M1=comp("BELOW")))["verdict"] == "PIVOT_STATIC"
    assert decide_fp06(base(M3_vs_M1=comp("BELOW")))["verdict"] == "STOP_CURRENT_SELECTOR"


def test_continue_requires_two_above_and_harm_guard():
    assert decide_fp06(base())["verdict"] == "CONTINUE_NEXT_ITERATION"
    assert decide_fp06(base(M3_vs_Fs_72h=comp("ABOVE", harm72_guard_pass=False)))["verdict"] == "INCONCLUSIVE_RULE_COVERAGE"


def test_straddle_is_inconclusive_after_earlier_rules():
    assert decide_fp06(base(M3_vs_M1=comp("STRADDLE")))["verdict"] == "INCONCLUSIVE"
    assert decide_fp06(base(M3_vs_M1={})) ["verdict"] == "INCONCLUSIVE_RULE_COVERAGE"


def test_fp06_receipt_flattens_machine_verdict(tmp_path, monkeypatch):
    """A written receipt exposes the verdict without nested-schema guessing."""
    inputs = tmp_path / "inputs.json"
    inputs.write_text(__import__("json").dumps(base(oracle_vs_M1=comp("BELOW"))))
    out = tmp_path / "out"
    args = type("Args", (), {"inputs": inputs, "out": out, "delta_min": 0.0034})()
    assert fp06_decision(args) == 0
    payload = __import__("json").loads((out / "decision.json").read_text())
    assert payload["verdict"] == "PIVOT_STATIC"
    assert payload["status"] == payload["decision"]["verdict"]


def test_runner_fail_closed_for_missing_and_non_boolean_evidence(tmp_path):
    for value, expected_type in ((None, "NoneType"), ("false", "str"), (0, "int"), (1, "int")):
        obj = base()
        if value is None:
            obj.pop("evidence_valid")
        else:
            obj["evidence_valid"] = value
        path = tmp_path / f"input_{expected_type}_{value}.json"
        path.write_text(__import__("json").dumps(obj))
        loaded = _load_inputs(path)
        assert loaded["evidence_valid"] is False
        assert loaded["evidence_valid_type"] == expected_type
        out = tmp_path / f"out_{expected_type}_{value}"
        args = type("Args", (), {"inputs": path, "out": out, "delta_min": 0.0034})()
        assert fp06_decision(args) == 2


def test_runner_keeps_normal_negative_verdict_zero(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(__import__("json").dumps(base(oracle_vs_Fs=comp("BELOW"))))
    out = tmp_path / "out"
    args = type("Args", (), {"inputs": path, "out": out, "delta_min": 0.0034})()
    assert fp06_decision(args) == 0
