"""FP-04 BANK-QUAL-v1 / BANK-SELECT-v1 / assembly rules, as pure functions (CPU).

Each criterion is exercised with a synthetic record that passes everything
except the one property under test, so a PASS elsewhere cannot mask it:

  * E0 validity (INVALID), E1 numerics, E2 stability, E3 own-group direction
    (hard, strict < 1.00), E4 nonzero response, E5 reload, E6 isolation;
  * report-only 6h / 72h ratios never change the verdict;
  * BANK-SELECT-v1: one FAIL stops the whole batch -- the passing three are
    never kept, a three-expert set is INVALID, batches never mix, and gain-
    based exclusion reasons stay forbidden;
  * the assembly rule A1-A5 has no tolerance at all.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta import bank_training as bt  # noqa: E402

RULE = {
    "rule": "BANK-QUAL-v1",
    "horizon": 8,
    "e2": {"clip_events_max": 0, "per_visit_ratio_max": 1.25, "epoch_onset_factor": 1.02,
           "epoch_size": "group_size"},
    "e3": {"own_group_mean_ratio_24h_lt": 1.0},
}
ISSUES = ["iss_a", "iss_b"]
FS_24 = {"iss_a": 0.50, "iss_b": 0.40}


def _panel(scale24: float = 1.0, scale6: float = 1.0, scale72: float = 1.0):
    return {i: {"6": 0.1 * scale6, "24": FS_24[i] * scale24, "72": 2.0 * scale72} for i in ISSUES}


def good_record(**overrides):
    losses = []
    ids = []
    for t in range(8):  # 4 epochs of the 2-issue group, gently decreasing
        issue = ISSUES[t % 2]
        losses.append(FS_24[issue] * (1.0 - 0.002 * t))
        ids.append(issue)
    record = {
        "expert_index": 1,
        "n_updates": 8,
        "n_samples": 2,
        "losses": losses,
        "grad_norms": [1e-3 * (t + 1) for t in range(8)],
        "grad_norms_A": [0.0] + [1e-4] * 7,
        "grad_norms_B": [2e-4] * 8,
        "training_issue_ids": ids,
        "config": {"grad_clip": 1.0, "mode": "formal"},
        "clip_events": 0,
        "eligible": True,
        "ineligible_reason": None,
        "initial_expert_digest": "init",
        "final_expert_digest": "final",
        "other_experts_unchanged": True,
        "other_experts_grad_free": True,
    }
    record.update(overrides)
    return record


def good_panels():
    fs = _panel()
    return {"Fs": fs, "L0": copy.deepcopy(fs), "L1": _panel(scale24=0.98, scale6=0.99, scale72=1.0)}


def good_diag():
    return {
        "nonzero_response": {"nonzero": True, "zero_initialized_untrained": False,
                             "max_abs_response": 0.01},
        "reload": {"file_sha256_matches": True, "expert_digest_matches": True, "probe_exact": True,
                   "probe_max_abs_diff_by_step": {"1": 0.0, "4": 0.0, "12": 0.0}},
        "isolation": {"backbone_digest_unchanged": True, "no_leftover_hooks": True},
    }


VALID = {"official_backend": True, "s0_certificate_consumed": True,
         "certified_fs_digests_are_pinned": True, "initial_bank_digest_is_pinned": True}


def qualify(record=None, panels=None, diag=None, rule=None, validity=VALID):
    return bt.evaluate_bank_expert_qualification(
        record if record is not None else good_record(),
        panels if panels is not None else good_panels(),
        diag if diag is not None else good_diag(),
        rule if rule is not None else RULE,
        validity=validity)


def test_the_good_synthetic_record_passes_every_criterion():
    result = qualify()
    assert result["verdict"] == "PASS", result["status"]
    assert result["status"] == {k: "PASS" for k in ("E0", "E1", "E2", "E3", "E4", "E5", "E6")}
    assert result["e2"]["epoch_size"] == 2 and result["e2"]["n_full_epochs"] == 4


# ----------------------------------------------------------------------------- E0
@pytest.mark.parametrize("mutate", [
    lambda r, p, d, v: v.update(certified_fs_digests_are_pinned=False),
    lambda r, p, d, v: v.update(initial_bank_digest_is_pinned=False),
    lambda r, p, d, v: p["L0"]["iss_a"].update({"24": p["L0"]["iss_a"]["24"] * (1 + 1e-15)}),
    lambda r, p, d, v: p["L1"].pop("iss_b"),
])
def test_e0_validity_failures_make_the_expert_invalid(mutate):
    record, panels, diag, validity = good_record(), good_panels(), good_diag(), dict(VALID)
    mutate(record, panels, diag, validity)
    result = qualify(record, panels, diag, validity=validity)
    assert result["verdict"] == "INVALID" and result["status"]["E0"] == "FAIL"


def test_no_validity_facts_is_invalid_not_pass():
    assert qualify(validity=None)["verdict"] == "INVALID"


def test_a_threshold_that_was_not_preregistered_is_invalid():
    rule = copy.deepcopy(RULE)
    rule["e3"] = {}
    result = qualify(rule=rule)
    assert result["verdict"] == "INVALID"
    assert "e3.own_group_mean_ratio_24h_lt" in result["not_preregistered"]


# ----------------------------------------------------------------------------- E1
@pytest.mark.parametrize("overrides", [
    {"n_updates": 7},
    {"losses": good_record()["losses"][:-1] + [float("nan")]},
    {"grad_norms_A": [0.0] + [float("inf")] + [1e-4] * 6},
    {"grad_norms_B": [2e-4] * 7},
    {"eligible": False, "ineligible_reason": "NON_FINITE_LOSS"},
])
def test_e1_numerics_failures(overrides):
    result = qualify(good_record(**overrides))
    assert result["verdict"] == "FAIL" and result["status"]["E1"] == "FAIL", result["status"]


# ----------------------------------------------------------------------------- E2
def test_e2_a_single_clip_event_fails():
    result = qualify(good_record(clip_events=1))
    assert result["verdict"] == "FAIL" and result["status"]["E2"] == "FAIL"


def test_e2_a_training_loss_above_1_25x_its_fs_24h_fails():
    record = good_record()
    record["losses"][5] = FS_24[record["training_issue_ids"][5]] * 1.26
    result = qualify(record)
    assert result["status"]["E2"] == "FAIL"
    assert result["e2"]["max_per_visit_ratio"] > 1.25


def test_e2_an_epoch_mean_spike_fails():
    record = good_record()
    record["losses"][4] *= 1.05  # epoch 2 mean rises > 2% over the best earlier epoch
    record["losses"][5] *= 1.05
    result = qualify(record)
    assert result["status"]["E2"] == "FAIL" and result["e2"]["onset_epoch"] == 2


# ----------------------------------------------------------------------------- E3
@pytest.mark.parametrize("scale, verdict", [(1.0, "FAIL"), (1.001, "FAIL"), (0.9999, "PASS")])
def test_e3_own_group_direction_is_a_strict_hard_gate(scale, verdict):
    panels = good_panels()
    panels["L1"] = _panel(scale24=scale)
    result = qualify(panels=panels)
    assert result["status"]["E3"] == verdict and result["verdict"] == verdict


# ----------------------------------------------------------------------------- E4
@pytest.mark.parametrize("mutate", [
    lambda r, p, d: d["nonzero_response"].update(nonzero=False),
    lambda r, p, d: d["nonzero_response"].update(zero_initialized_untrained=True),
    lambda r, p, d: r.update(final_expert_digest="init"),
    lambda r, p, d: p.update(L1=copy.deepcopy(p["L0"])),
])
def test_e4_no_response_fails(mutate):
    record, panels, diag = good_record(), good_panels(), good_diag()
    mutate(record, panels, diag)
    result = qualify(record, panels, diag)
    assert result["status"]["E4"] == "FAIL" and result["verdict"] in ("FAIL",), result["status"]


# ----------------------------------------------------------------------------- E5
@pytest.mark.parametrize("mutate", [
    lambda d: d["reload"].update(file_sha256_matches=False),
    lambda d: d["reload"].update(expert_digest_matches=False),
    lambda d: d["reload"]["probe_max_abs_diff_by_step"].update({"12": 1e-12}),
    lambda d: d["reload"].update(probe_max_abs_diff_by_step={}),
])
def test_e5_reload_is_exact_or_fails(mutate):
    diag = good_diag()
    mutate(diag)
    result = qualify(diag=diag)
    assert result["status"]["E5"] == "FAIL" and result["verdict"] == "FAIL"


# ----------------------------------------------------------------------------- E6
@pytest.mark.parametrize("record_over, iso_over", [
    ({"other_experts_unchanged": False}, {}),
    ({"other_experts_grad_free": False}, {}),
    ({}, {"backbone_digest_unchanged": False}),
    ({}, {"no_leftover_hooks": False}),
])
def test_e6_isolation_failures(record_over, iso_over):
    diag = good_diag()
    diag["isolation"].update(iso_over)
    result = qualify(good_record(**record_over), diag=diag)
    assert result["status"]["E6"] == "FAIL" and result["verdict"] == "FAIL"


def test_out_of_sample_style_numbers_are_report_only():
    """A bad 6h or 72h own-group ratio does not change the verdict (FP-05's job)."""
    panels = good_panels()
    panels["L1"] = _panel(scale24=0.98, scale6=1.50, scale72=1.50)
    result = qualify(panels=panels)
    assert result["verdict"] == "PASS"
    assert result["report_only"]["own_group_72h"]["mean_ratio"] == pytest.approx(1.5)
    assert result["report_only"]["own_group_6h"]["mean_ratio"] == pytest.approx(1.5)


# =============================================================================
# BANK-SELECT-v1
# =============================================================================

def _experts(verdicts, *, batch="B-J2", digests=None):
    digests = digests or [f"digest{k}" for k in range(len(verdicts))]
    return [{"expert_index": k, "qualification": {"verdict": v}, "expert_digest": digests[k],
             "expert_file": {"path": f"/x/expert_{k}.pt", "sha256": f"sha{k}{digests[k]}"},
             "probe_file": {"path": f"/x/probe_{k}.pt", "sha256": f"p{k}"},
             "batch_id": batch} for k, v in enumerate(verdicts)]


SELECT = {"rule": "BANK-SELECT-v1", "num_experts": 4}


def test_all_four_pass_selects_the_whole_bank():
    decision = bt.decide_bank_formal(_experts(["PASS"] * 4), SELECT)
    assert decision["verdict"] == "BANK_QUAL_PASS_PENDING_ASSEMBLY"
    assert [e["expert_index"] for e in decision["experts_for_assembly"]] == [0, 1, 2, 3]


def test_one_fail_stops_the_whole_batch_and_keeps_nothing():
    decision = bt.decide_bank_formal(_experts(["PASS", "PASS", "FAIL", "PASS"]), SELECT)
    assert decision["verdict"] == "STOP_CURRENT_BANK"
    assert decision["failed_experts"] == [2]
    assert decision["bank_selected_for_assembly"] is False
    assert decision["experts_for_assembly"] is None
    assert decision["substitution"] == "FORBIDDEN" and decision["partial_bank"] == "FORBIDDEN"


def test_keeping_the_other_three_is_not_a_k4_bank():
    three = [e for e in _experts(["PASS", "PASS", "FAIL", "PASS"]) if e["expert_index"] != 2]
    decision = bt.decide_bank_formal(three, SELECT)
    assert decision["verdict"] == "INVALID"
    assert decision["reason"] == "EXPERT_SET_INCOMPLETE_OR_DUPLICATED"
    # re-indexing the three as 0..2 of a K=3 bank is also not the declared K=4 bank
    reindexed = [dict(e, expert_index=i) for i, e in enumerate(three)]
    assert bt.decide_bank_formal(reindexed, SELECT)["verdict"] == "INVALID"


def test_experts_from_different_batches_never_mix():
    mixed = _experts(["PASS"] * 4)
    mixed[3]["batch_id"] = "B-J2F"
    decision = bt.decide_bank_formal(mixed, SELECT)
    assert decision["verdict"] == "INVALID" and decision["reason"] == "CROSS_BATCH_MIXING_FORBIDDEN"


def test_duplicate_expert_digests_are_not_four_experts():
    decision = bt.decide_bank_formal(
        _experts(["PASS"] * 4, digests=["d0", "d1", "d1", "d3"]), SELECT)
    assert decision["verdict"] == "INVALID"


def test_an_invalid_expert_makes_the_batch_invalid():
    decision = bt.decide_bank_formal(_experts(["PASS", "INVALID", "PASS", "PASS"]), SELECT)
    assert decision["verdict"] == "INVALID"


@pytest.mark.parametrize("reason", bt.FORBIDDEN_EXCLUSION_REASONS)
def test_gain_based_exclusion_reasons_are_still_refused(reason):
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.assert_eligibility_rule_is_pre_declared(reason)
    assert excinfo.value.code == "BANK_EXCLUSION_REASON_FORBIDDEN"


@pytest.mark.parametrize("reason", ["E3", "BANK_QUAL_FAIL", "OWN_GROUP_NOT_IMPROVED"])
def test_a_bank_qual_failure_is_not_an_exclusion_reason(reason):
    """A BANK-QUAL-v1 failure stops the bank; it can never be used to drop one expert."""
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.assert_eligibility_rule_is_pre_declared(reason)
    assert excinfo.value.code == "BANK_EXCLUSION_REASON_UNKNOWN"


# =============================================================================
# Assembly A1-A5: exact, no tolerance
# =============================================================================

def good_verify_record():
    steps = {"1": 0.0, "4": 0.0, "12": 0.0}
    equal = {"1": True, "4": True, "12": True}
    experts = [str(k) for k in range(4)]
    return {
        "A1_source_bank": {k: {"max_abs_diff_by_step": dict(steps), "torch_equal_by_step": dict(equal)}
                           for k in experts},
        "A2_training_probe": {k: {"max_abs_diff_by_step": dict(steps),
                                  "torch_equal_by_step": dict(equal),
                                  "probe_file_sha256": f"p{k}", "expected_probe_file_sha256": f"p{k}"}
                              for k in experts},
        "A3_zero_edit": {"max_abs_diff_vs_fs": 0.0, "max_abs_diff_vs_f0": 0.5},
        "A4_continuation": {k: {"max_abs_diff_vs_fs_continuation_by_step": [0.0] * 8,
                                "max_abs_diff_vs_f0_continuation_by_step": [0.1] * 8,
                                "edit_effect_at_hold_max_abs": 0.01} for k in experts},
        "A5_reload": {"file_sha256": "s", "expected_file_sha256": "s", "bank_digest": "b",
                      "expected_bank_digest": "b", "expert_digests": ["0", "1", "2", "3"],
                      "expected_expert_digests": ["0", "1", "2", "3"]},
    }


def test_exact_zeros_pass_the_assembly_rule():
    assert bt.evaluate_bank_assembly(good_verify_record())["passed"] is True


@pytest.mark.parametrize("check, mutate", [
    ("A1", lambda r: r["A1_source_bank"]["2"]["max_abs_diff_by_step"].update({"1": 1e-38})),
    ("A1", lambda r: r["A1_source_bank"].pop("3")),
    ("A2", lambda r: r["A2_training_probe"]["0"].update(probe_file_sha256="other")),
    ("A2", lambda r: r["A2_training_probe"]["1"]["torch_equal_by_step"].update({"4": False})),
    ("A3", lambda r: r["A3_zero_edit"].update(max_abs_diff_vs_fs=1e-40)),
    ("A3", lambda r: r["A3_zero_edit"].update(max_abs_diff_vs_f0=0.0)),
    ("A4", lambda r: r["A4_continuation"]["1"]["max_abs_diff_vs_fs_continuation_by_step"].__setitem__(3, 1e-9)),
    ("A4", lambda r: r["A4_continuation"]["1"].update(edit_effect_at_hold_max_abs=0.0)),
    ("A5", lambda r: r["A5_reload"].update(expert_digests=["0", "1", "2"])),
])
def test_any_nonzero_difference_fails_its_check(check, mutate):
    record = good_verify_record()
    mutate(record)
    evaluation = bt.evaluate_bank_assembly(record)
    assert evaluation["passed"] is False and evaluation["failed"] == [check]


# =============================================================================
# BANK-BUDGET-v1 fallback: only an all-E2 STOP authorizes the halved-lr batch
# =============================================================================

FALLBACK_SELECT = {"rule": "BANK-SELECT-v1", "num_experts": 4,
                   "fallback": {"job": "B-J2F", "after_job": "B-J2",
                                "requires_failed_criterion": "E2"}}


def _with_status(verdicts, failing):
    experts = _experts(verdicts)
    for entry in experts:
        k = entry["expert_index"]
        status = {c: "PASS" for c in ("E0", "E1", "E2", "E3", "E4", "E5", "E6")}
        for criterion in failing.get(k, ()):
            status[criterion] = "FAIL"
        entry["qualification"]["status"] = status
    return experts


@pytest.mark.parametrize("failing, authorized", [
    ({2: ["E2"]}, True),                      # instability: the case the fallback is for
    ({1: ["E2", "E3"], 3: ["E2"]}, True),     # instability that also sank direction
    ({2: ["E3"]}, False),                     # too weak: halving lr would weaken it further
    ({1: ["E2"], 2: ["E3"]}, False),          # mixed: not every failure is E2
    ({0: ["E4"]}, False),
    ({0: ["E5"], 1: ["E2"]}, False),
])
def test_fallback_is_authorized_only_if_every_failed_expert_failed_e2(failing, authorized):
    verdicts = ["FAIL" if k in failing else "PASS" for k in range(4)]
    decision = bt.decide_bank_formal(_with_status(verdicts, failing), FALLBACK_SELECT)
    assert decision["verdict"] == "STOP_CURRENT_BANK"
    assert decision["fallback_authorized"] is authorized
    assert decision["failed_criteria_by_expert"] == {
        str(k): sorted(v) for k, v in failing.items()}
    assert decision["experts_for_assembly"] is None


def test_no_fallback_without_a_declared_rule_or_after_invalid():
    stop = bt.decide_bank_formal(_with_status(["PASS", "FAIL", "PASS", "PASS"], {1: ["E2"]}), SELECT)
    assert stop["verdict"] == "STOP_CURRENT_BANK" and stop["fallback_authorized"] is False
    invalid = bt.decide_bank_formal(_with_status(["PASS", "INVALID", "FAIL", "PASS"], {2: ["E2"]}),
                                    FALLBACK_SELECT)
    assert invalid["verdict"] == "INVALID" and invalid["fallback_authorized"] is False
    ok = bt.decide_bank_formal(_with_status(["PASS"] * 4, {}), FALLBACK_SELECT)
    assert ok["fallback_authorized"] is False


def test_the_fallback_decision_is_refused_without_an_authorizing_prior(tmp_path):
    import json

    from earthdelta import fs_protocol as fp
    import scripts.r4_bank_decide as decide

    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema_version": fp.PROTOCOL_SCHEMA, "protocol_kind": "bank",
                                    "qualification_rule": RULE, "selection_rule": FALLBACK_SELECT,
                                    "configs": {}}))
    sha = fp.file_sha256(protocol)

    def prior(**fields):
        path = tmp_path / f"prior_{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps({"kind": "formal", "job": "B-J2", "verdict": "STOP_CURRENT_BANK",
                                    "fallback_authorized": True, "protocol_sha256": sha, **fields}))
        return path

    def run(prior_path):
        argv = ["--kind", "formal", "--job", "B-J2F", "--protocol", str(protocol),
                "--protocol-sha256", sha, "--job-dir", str(tmp_path / "none"),
                "--out", str(tmp_path / "out.json")]
        if prior_path is not None:
            argv += ["--prior-formal-decision", str(prior_path)]
        return decide.main(argv)

    assert run(None) == 2
    assert run(prior(fallback_authorized=False)) == 2          # an E3 (or mixed) STOP
    assert run(prior(verdict="INVALID")) == 2
    assert run(prior(protocol_sha256="0" * 64)) == 2
    assert run(prior(job="B-J1")) == 2
    assert not (tmp_path / "out.json").exists()
    # Positive control: an authorizing prior lets the decider run (the refusals are not blanket).
    assert run(prior()) == 0 and (tmp_path / "out.json").exists()
    assert json.loads((tmp_path / "out.json").read_text())["inputs"]["prior_formal_decision"]["sha256"]
