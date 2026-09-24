"""FP-03 Fs quality gate (TEST_PLAN B.5) and its real entry points (CPU).

What is pinned here:

  * a clearly worse but FINITE panel is a quality FAIL in its own right
    (Q4 PANEL degradation), distinct from any non-finite failure;
  * the initial panel of the zero-initialized Fs is bitwise F0;
  * `fs_freeze` refuses anything but a PASS decision for exactly this adapter
    from a full-horizon formal fit -- and when it refuses, the merge and the
    expert trainer are never called;
  * a blow-up-then-recover trace passes the historical per-visit check but
    fails the new Q2 stability criteria;
  * FS-SCREEN-v1 / FS-SELECT-v1 decisions, including "replica0 fails, replica2
    has the best numbers -> still STOP, no substitution";
  * a protocol SHA-256 / config-id mismatch exits 2 before the fit is called;
  * the official-backend guard refuses the SDPA fallback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List
from unittest import mock

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta import fs_protocol as fp  # noqa: E402
from earthdelta import static_adapter as sa  # noqa: E402
from earthdelta import bank_training as bt  # noqa: E402

import scripts.r2_fs_bank_train as r2_cli  # noqa: E402

RULE = {
    "horizon": 16,
    "q2": {"clip_events_max": 0, "per_visit_ratio_max": 1.25, "epoch_onset_factor": 1.02},
    "q3": {"required_issues": 2},
    "q4": {"pass_max_ratio_lt": 1.0, "pass_mean_ratio_le": 0.99,
           "fail_mean_ratio_gt": 1.00, "fail_max_ratio_gt": 1.02},
    "q5": {"mean_ratio_72h_le": 1.01, "report_only": False},
    "q6": {"holdout_mean_ratio_24h_le": 1.02, "report_only": False},
}
ISSUES = ["iss_a", "iss_b"]
L0 = {"iss_a": 1.0, "iss_b": 2.0}


def _record(losses: List[float], clip=None, n=None) -> Dict:
    ids = [ISSUES[t % 2] for t in range(len(losses))]
    grads = [0.01] * len(losses)
    if clip:
        for t in clip:
            grads[t] = 5.0
    return {"mode": "formal", "n_updates": len(losses) if n is None else n, "losses": losses,
            "grad_norms": grads, "training_issue_ids": ids, "config": {"grad_clip": 1.0},
            "clip_events": sum(1 for g in grads if g > 1.0), "ineligible_reason": None}


def _steady(n=16, rate=0.004) -> List[float]:
    return [L0[ISSUES[t % 2]] * (1 - rate * t) for t in range(n)]


def _panel(train_scale: float, hold_scale: float = 0.98, s72: float = None) -> Dict:
    s72 = train_scale if s72 is None else s72
    return {
        "train": {i: {"6": L0[i] * train_scale, "24": L0[i] * train_scale, "72": L0[i] * s72}
                  for i in ISSUES},
        "holdout": {"iss_h": {"6": 1.0 * hold_scale, "24": 1.0 * hold_scale, "72": hold_scale}},
    }


def _panel0() -> Dict:
    return {"train": {i: {"6": L0[i], "24": L0[i], "72": L0[i]} for i in ISSUES},
            "holdout": {"iss_h": {"6": 1.0, "24": 1.0, "72": 1.0}}}


VALID = {"official_backend": True, "torch_version": True, "tf32_off": True}


# =============================================================================
# Qualification
# =============================================================================

def test_healthy_fit_passes_every_criterion():
    q = sa.evaluate_fs_qualification(_record(_steady()), _panel0(), _panel(0.97), RULE,
                                     validity=VALID)
    assert q["verdict"] == "PASS", q["status"]
    assert q["quality_pass"] is True


def test_clearly_worse_but_finite_panel_is_a_panel_degradation_fail():
    """Finite everywhere, stable training, but the final checkpoint is 5% worse
    on the fixed panel: Q4 FAIL -- not a non-finite failure in disguise."""
    q = sa.evaluate_fs_qualification(_record(_steady()), _panel0(), _panel(1.05), RULE,
                                     validity=VALID)
    assert q["verdict"] == "FAIL"
    assert q["status"]["Q4"] == "FAIL"
    assert q["status"]["Q1"] == "PASS" and q["status"]["Q2"] == "PASS"
    assert q["stability"]["q1"]["all_losses_finite"] is True
    assert q["q4_train_24h"]["mean_ratio"] == pytest.approx(1.05)


def test_q4_between_bands_is_inconclusive_not_pass():
    q = sa.evaluate_fs_qualification(_record(_steady()), _panel0(), _panel(0.995), RULE,
                                     validity=VALID)
    assert q["status"]["Q4"] == "INCONCLUSIVE_QUALIFICATION"
    assert q["verdict"] == "INCONCLUSIVE_QUALIFICATION"


def test_72h_guard_and_holdout_guard_fail_independently():
    q72 = sa.evaluate_fs_qualification(_record(_steady()), _panel0(), _panel(0.97, s72=1.02),
                                       RULE, validity=VALID)
    assert q72["status"]["Q5"] == "FAIL" and q72["verdict"] == "FAIL"
    qh = sa.evaluate_fs_qualification(_record(_steady()), _panel0(), _panel(0.97, hold_scale=1.03),
                                      RULE, validity=VALID)
    assert qh["status"]["Q6"] == "FAIL" and qh["verdict"] == "FAIL"


def test_null_threshold_is_not_preregistered_and_blocks_pass_unless_report_only():
    rule = json.loads(json.dumps(RULE))
    rule["q5"]["mean_ratio_72h_le"] = None
    q = sa.evaluate_fs_qualification(_record(_steady()), _panel0(), _panel(0.97), rule,
                                     validity=VALID)
    assert q["status"]["Q5"] == "NOT_PREREGISTERED"
    assert q["verdict"] == "INCONCLUSIVE_QUALIFICATION"
    rule["q5"]["report_only"] = True
    q = sa.evaluate_fs_qualification(_record(_steady()), _panel0(), _panel(0.97), rule,
                                     validity=VALID)
    assert q["status"]["Q5"] == "REPORT_ONLY" and q["verdict"] == "PASS"


def test_q0_validity_failure_is_invalid_whatever_the_numbers():
    q = sa.evaluate_fs_qualification(_record(_steady()), _panel0(), _panel(0.97), RULE,
                                     validity={**VALID, "official_backend": False})
    assert q["verdict"] == "INVALID" and q["substantive_verdict"] == "PASS"
    q = sa.evaluate_fs_qualification(_record(_steady()), _panel0(), _panel(0.97), RULE)
    assert q["verdict"] == "INVALID" and q["q0"]["status"] == "NOT_EVALUATED"


def test_fewer_updates_than_the_horizon_is_never_success_evidence():
    q = sa.evaluate_fs_qualification(_record(_steady(8)), _panel0(), _panel(0.97), RULE,
                                     validity=VALID)
    assert q["status"]["Q1"] == "FAIL" and q["verdict"] == "FAIL"


def test_blow_up_then_recover_passes_per_visit_but_fails_q2():
    """The pt-e8y7ib01 shape: steady descent, a blow-up with clipping, then a
    recovery that ends just below each issue's first visit. The historical
    per-visit gate says 2/2 improved; FS-QUAL-v1 Q2 says unstable."""
    losses = _steady()
    for t in (8, 9, 10):
        losses[t] = L0[ISSUES[t % 2]] * 3.0
    losses[14] = L0["iss_a"] * 0.99
    losses[15] = L0["iss_b"] * 0.99
    record = _record(losses, clip=[8, 9])
    per_visit = sa.evaluate_fs_quality(sa.FsFitRecord(
        mode="formal", losses=losses, training_issue_ids=record["training_issue_ids"],
        n_updates=16))
    assert per_visit["passed"] is True
    stab = sa.training_stability(record, L0, horizon=16)
    assert stab["stable"] is False
    assert stab["q2"]["clip_ok"] is False
    assert stab["q2"]["ratio_ok"] is False
    assert stab["q2"]["onset_epoch"] == 1
    q = sa.evaluate_fs_qualification(record, _panel0(), _panel(0.97), RULE, validity=VALID)
    assert q["status"]["Q2"] == "FAIL" and q["verdict"] == "FAIL"


def test_epoch_onset_rule_uses_the_running_minimum():
    flat = [1.0] * 32
    for t in range(24, 32):
        flat[t] = 1.03
    stab = sa.training_stability({"n_updates": 32, "losses": flat, "grad_norms": [0.1] * 32,
                                  "training_issue_ids": ["x"] * 32, "config": {"grad_clip": 1.0}},
                                 {"x": 1.0}, horizon=32)
    assert stab["q2"]["onset_epoch"] == 3 and stab["q2"]["onset_update"] == 24


# =============================================================================
# FS-SCREEN-v1 / FS-SELECT-v1
# =============================================================================

SCREEN_RULE = {"horizon": 16, "control_arm": "A0", "candidate_arms": ["A1", "A2", "A3"],
               "progressing_mean_ratio_24h_le": 0.99}


def _arm(arm_id, lr, stable, m, valid=True, onset=None):
    stab = {"stable": stable, "q1": {"passed": True},
            "q2": {"passed": stable, "clip_events": 0 if stable else 3, "first_clip_update": None,
                   "max_per_visit_ratio": 1.0 if stable else 4.0, "onset_epoch": None,
                   "onset_update": onset}}
    return {"arm_id": arm_id, "lr": lr, "valid": valid, "stability": stab,
            "train_panel_mean_ratio_24h": m}


def test_screen_stable_positive_control_is_not_testable():
    d = sa.decide_lr_screen([_arm("A0", 1e-2, True, 0.98), _arm("A1", 5e-3, True, 0.98),
                             _arm("A2", 2.5e-3, True, 0.98), _arm("A3", 1.25e-3, True, 0.98)],
                            SCREEN_RULE)
    assert d["verdict"] == "NOT_TESTABLE" and d["stop"] is True


def test_screen_empty_s_refutes_the_lr_hypothesis():
    d = sa.decide_lr_screen([_arm("A0", 1e-2, False, 1.5, onset=80),
                             _arm("A1", 5e-3, False, 1.2, onset=160),
                             _arm("A2", 2.5e-3, True, 0.995), _arm("A3", 1.25e-3, True, 1.0)],
                            SCREEN_RULE)
    assert d["verdict"] == "REFUTED" and d["S"] == [] and d["selected_lr"] is None
    assert {r["arm_id"]: r["lr_x_onset_update"] for r in d["arms"]}["A1"] == pytest.approx(0.8)


def test_screen_selects_highest_lr_in_s_and_next_lower_fallback():
    d = sa.decide_lr_screen([_arm("A0", 1e-2, False, 1.5), _arm("A1", 5e-3, False, 0.9),
                             _arm("A2", 2.5e-3, True, 0.97), _arm("A3", 1.25e-3, True, 0.985)],
                            SCREEN_RULE)
    assert d["verdict"] == "CONFIRMED" and d["S"] == ["A2", "A3"]
    assert (d["selected_arm"], d["selected_lr"]) == ("A2", 2.5e-3)
    assert (d["fallback_arm"], d["fallback_lr"]) == ("A3", 1.25e-3)


def test_screen_missing_or_invalid_arm_is_invalid():
    arms = [_arm("A0", 1e-2, False, 1.5), _arm("A1", 5e-3, True, 0.97),
            _arm("A2", 2.5e-3, True, 0.97)]
    assert sa.decide_lr_screen(arms, SCREEN_RULE)["verdict"] == "INVALID"
    arms.append(_arm("A3", 1.25e-3, True, 0.97, valid=False))
    assert sa.decide_lr_screen(arms, SCREEN_RULE)["verdict"] == "INVALID"


SELECT_RULE = {"n_replicas": 4, "designated_device_index": 0}


def _rep(i, verdict, sha=None, mean_ratio=0.98):
    return {"device_index": i, "adapter_sha256": sha or f"sha{i}",
            "qualification": {"verdict": verdict, "q4_train_24h": {"mean_ratio": mean_ratio}}}


def test_select_replica0_fails_and_replica2_has_best_numbers_still_stops():
    reps = [_rep(0, "FAIL", mean_ratio=1.01), _rep(1, "PASS", mean_ratio=0.98),
            _rep(2, "PASS", mean_ratio=0.90), _rep(3, "PASS", mean_ratio=0.97)]
    d = sa.decide_formal_fs(reps, SELECT_RULE)
    assert d["verdict"] == "STOP_FITTED_FS_QUALITY"
    assert d["designated_adapter_sha256"] is None and d["fs_selected"] is False
    assert d["substitution"] == "FORBIDDEN"


def test_select_any_witness_failing_stops_even_if_designated_passes():
    d = sa.decide_formal_fs([_rep(0, "PASS"), _rep(1, "PASS"), _rep(2, "PASS"), _rep(3, "FAIL")],
                            SELECT_RULE)
    assert d["verdict"] == "STOP_FITTED_FS_QUALITY" and d["failed_replicas"] == [3]


def test_select_all_pass_designates_replica0_pending_its_gates():
    reps = [_rep(i, "PASS") for i in range(4)]
    d = sa.decide_formal_fs(reps, SELECT_RULE)
    assert d["verdict"] == "QUALITY_PASS_PENDING_DESIGNATED_GATES"
    assert d["designated_adapter_sha256"] == "sha0" and d["fs_selected"] is False
    ok = sa.decide_formal_fs(reps, SELECT_RULE, designated_gates={
        "numerical_merge_pass": True, "reload_pass": True, "identity_pass": True})
    assert ok["verdict"] == "FS_SELECTED" and ok["fs_selected"] is True
    bad = sa.decide_formal_fs(reps, SELECT_RULE, designated_gates={
        "numerical_merge_pass": True, "reload_pass": False, "identity_pass": True})
    assert bad["verdict"] == "STOP_DESIGNATED_GATE_FAILED" and bad["fs_selected"] is False


def test_select_inconclusive_and_invalid_and_incomplete():
    d = sa.decide_formal_fs([_rep(0, "PASS"), _rep(1, "INCONCLUSIVE_QUALIFICATION"),
                             _rep(2, "PASS"), _rep(3, "PASS")], SELECT_RULE)
    assert d["verdict"] == "INCONCLUSIVE_QUALIFICATION"
    d = sa.decide_formal_fs([_rep(0, "PASS"), _rep(1, "INVALID"), _rep(2, "FAIL"),
                             _rep(3, "PASS")], SELECT_RULE)
    assert d["verdict"] == "INVALID"
    d = sa.decide_formal_fs([_rep(0, "PASS"), _rep(1, "PASS"), _rep(2, "PASS")], SELECT_RULE)
    assert d["verdict"] == "INVALID"


# =============================================================================
# Panels on the synthetic model: L0 == F0 bitwise, nonzero response after fit
# =============================================================================

def _synthetic(tmp_leads=(1, 4, 12)):
    bridge, variables, lat = r2_cli.build_synthetic_bridge(torch.device("cpu"))
    samples = r2_cli.build_synthetic_samples(bridge, tmp_leads, n_samples=2)
    spec = sa.build_objective_spec(bridge, lat, lead_steps=(4,), space="raw")
    return bridge, variables, samples, spec


def test_initial_zero_init_fs_panel_equals_f0_bitwise():
    torch.set_num_threads(1)
    bridge, variables, samples, spec = _synthetic()
    initial = sa.build_fs_adapter(bridge.model.blocks[0].attn.proj.in_features, (0, 1),
                                  rank_per_expert=4, seed=20260921)
    f0 = sa.fs_panel_losses(bridge, samples, spec, lead_steps=(1, 4, 12), target_blocks=(0, 1))
    l0 = sa.fs_panel_losses(bridge, samples, spec, lead_steps=(1, 4, 12), fs_adapters=initial,
                            target_blocks=(0, 1))
    assert l0 == f0
    assert set(next(iter(f0.values()))) == {"6", "24", "72"}
    # A nonzero B makes the same panel differ: the comparison is discriminating.
    with torch.no_grad():
        for lora in initial.values():
            lora.up[0].weight.fill_(0.05)
    moved = sa.fs_panel_losses(bridge, samples, spec, lead_steps=(1, 4, 12),
                               fs_adapters=initial, target_blocks=(0, 1))
    assert moved != f0


# =============================================================================
# fs_freeze refuses non-PASS / non-formal / short records; merge never called
# =============================================================================

def _fit_and_save(tmp_path: Path, mode="formal", updates=6) -> Path:
    out = tmp_path / "fit"
    code = r2_cli.main(["--stage", "fs_fit", "--synthetic", "--device", "cpu", "--mode", mode,
                        "--max-updates", str(updates), "--output-dir", str(out)])
    assert code in (0, 1)
    return out / "fs_adapter.pt"


def _decision(tmp_path: Path, adapter: Path, *, verdict="QUALITY_PASS_PENDING_DESIGNATED_GATES",
              mode="formal", n_updates=6, horizon=6, sha=None) -> Path:
    path = tmp_path / f"decision_{verdict}_{mode}_{n_updates}.json"
    path.write_text(json.dumps({
        "kind": "formal", "rule": "FS-SELECT-v1", "verdict": verdict, "horizon": horizon,
        "designated_adapter_sha256": sha or fp.file_sha256(adapter),
        "designated_record": {"mode": mode, "n_updates": n_updates},
    }))
    return path


def _freeze(adapter: Path, decision: Path, sha: str):
    with mock.patch.object(r2_cli.sa, "merge_static_adapter",
                           wraps=sa.merge_static_adapter) as merge, \
            mock.patch.object(r2_cli.bt, "train_expert", wraps=bt.train_expert) as train:
        code = r2_cli.main(["--stage", "fs_freeze", "--synthetic", "--device", "cpu",
                            "--fs-adapter", str(adapter), "--fs-adapter-sha256", sha,
                            "--fs-decision", str(decision)])
    return code, merge.call_count, train.call_count


@pytest.mark.parametrize("case", [
    dict(verdict="STOP_FITTED_FS_QUALITY"),
    dict(verdict="INCONCLUSIVE_QUALIFICATION"),
    dict(mode="gradient_check"),
    dict(mode="stability_screen"),
    dict(n_updates=5),
    dict(sha="0" * 64),
])
def test_fs_freeze_refuses_and_never_merges(tmp_path, case):
    adapter = _fit_and_save(tmp_path)
    decision = _decision(tmp_path, adapter, **case)
    code, merges, trains = _freeze(adapter, decision, fp.file_sha256(adapter))
    assert code != 0
    assert merges == 0 and trains == 0


def test_fs_freeze_refuses_when_the_adapter_bytes_are_not_the_decided_ones(tmp_path):
    adapter = _fit_and_save(tmp_path)
    decision = _decision(tmp_path, adapter)
    other = _fit_and_save(tmp_path / "other", updates=4)
    code, merges, _ = _freeze(other, decision, fp.file_sha256(adapter))
    assert code != 0 and merges == 0


def test_fs_freeze_with_a_pass_decision_does_merge(tmp_path):
    """Positive control: the refusal tests above are not refusing everything."""
    adapter = _fit_and_save(tmp_path)
    decision = _decision(tmp_path, adapter)
    code, merges, trains = _freeze(adapter, decision, fp.file_sha256(adapter))
    assert merges == 1 and trains == 0
    assert code == 0


# =============================================================================
# Protocol binding, S0 consumption order, official backend guard
# =============================================================================

def _real_args(tmp_path: Path, protocol: Path, sha: str, config_id="J2_A1", lr="0.005") -> List[str]:
    return ["--stage", "fs_fit", "--mode", "stability_screen", "--max-updates", "256",
            "--learning-rate", lr, "--config", str(tmp_path / "gate_config.json"),
            "--admission", str(tmp_path / "admission.json"), "--protocol", str(protocol),
            "--protocol-sha256", sha, "--config-id", config_id,
            "--s0-certificate", str(tmp_path / "s0.json"), "--device", "cpu"]


def _protocol(tmp_path: Path, lr=0.005) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({
        "schema_version": fp.PROTOCOL_SCHEMA,
        "qualification_rule": {"q0": {"torch_version": "2.3.1+cu121", "xformers_version": "0.0.27"}},
        "inputs": {"s0_certificate": {"sha256": "0" * 64}},
        "configs": {"J2_A1": {"job": "J2", "stage": "fs_fit", "mode": "stability_screen",
                              "max_updates": 256, "learning_rate": lr, "device": "cpu"}},
    }))
    return path


def test_protocol_sha_mismatch_exits_2_before_fit(tmp_path):
    protocol = _protocol(tmp_path)
    with mock.patch.object(r2_cli.sa, "fit_static_adapter") as fit, \
            mock.patch.object(r2_cli, "build_real_bridge") as build:
        code = r2_cli.main(_real_args(tmp_path, protocol, "f" * 64))
    assert code == 2 and fit.call_count == 0 and build.call_count == 0


def test_config_mismatch_exits_2_before_fit(tmp_path):
    protocol = _protocol(tmp_path)
    sha = fp.file_sha256(protocol)
    with mock.patch.object(r2_cli.sa, "fit_static_adapter") as fit, \
            mock.patch.object(r2_cli, "build_real_bridge") as build, \
            mock.patch.object(r2_cli, "REQUIRED_DECLARED", {}):
        wrong_lr = r2_cli.main(_real_args(tmp_path, protocol, sha, lr="0.01"))
        wrong_id = r2_cli.main(_real_args(tmp_path, protocol, sha, config_id="J9_ZZ"))
    assert wrong_lr == 2 and wrong_id == 2
    assert fit.call_count == 0 and build.call_count == 0


def test_underdeclared_config_exits_2(tmp_path):
    protocol = _protocol(tmp_path)
    with mock.patch.object(r2_cli.sa, "fit_static_adapter") as fit:
        code = r2_cli.main(_real_args(tmp_path, protocol, fp.file_sha256(protocol)))
    assert code == 2 and fit.call_count == 0


def test_real_run_without_protocol_is_refused(tmp_path):
    with mock.patch.object(r2_cli.sa, "fit_static_adapter") as fit:
        code = r2_cli.main(["--stage", "fs_fit", "--config", str(tmp_path / "c.json"),
                            "--admission", str(tmp_path / "a.json"), "--device", "cpu"])
    assert code == 2 and fit.call_count == 0


def test_backend_precondition_failure_exits_2_after_s0_before_bridge(tmp_path):
    protocol = _protocol(tmp_path)
    sha = fp.file_sha256(protocol)
    calls = []
    with mock.patch.object(r2_cli, "REQUIRED_DECLARED", {}), \
            mock.patch.object(r2_cli.GateIdentityConfig, "load_json", return_value=object()), \
            mock.patch.object(r2_cli.fp, "consume_s0_certificate",
                              side_effect=lambda *a, **k: calls.append("s0") or {"passed": True}), \
            mock.patch.object(r2_cli.fp, "official_backend_precondition",
                              side_effect=fp.ProtocolViolation("OFFICIAL_BACKEND_UNAVAILABLE", "no")), \
            mock.patch.object(r2_cli, "build_real_bridge") as build, \
            mock.patch.object(r2_cli.sa, "fit_static_adapter") as fit:
        code = r2_cli.main(_real_args(tmp_path, protocol, sha))
    assert code == 2 and calls == ["s0"]
    assert build.call_count == 0 and fit.call_count == 0


def test_official_backend_guard_rejects_the_sdpa_fallback_model():
    bridge, _, _ = r2_cli.build_synthetic_bridge(torch.device("cpu"))
    with pytest.raises(fp.ProtocolViolation) as excinfo:
        fp.official_backend_postcondition(bridge.model)
    assert excinfo.value.code == "OFFICIAL_BACKEND_NOT_BUILT"
    assert excinfo.value.detail["model_class"]["module"] != fp.OFFICIAL_MODEL_MODULE


def test_tf32_fact_uses_the_effective_state_not_the_image_env_default():
    """J1 (pt-56auspj8) regression: the ACP image exports
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which in torch 2.3.1 only initializes
    float32_matmul_precision; the process's explicit allow_tf32=False wins and
    the recorded getters are False. That is TF32 OFF. Any effective TF32 --
    either getter True, or a recorded precision other than "highest" -- is not."""
    j1_like = {"tf32": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                        "env": {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1", "NVIDIA_TF32_OVERRIDE": None}}}
    assert fp.tf32_is_off(j1_like) is True
    assert fp.tf32_is_off({"tf32": {**j1_like["tf32"], "float32_matmul_precision": "highest"}}) is True
    assert fp.tf32_is_off({"tf32": {**j1_like["tf32"], "float32_matmul_precision": "high"}}) is False
    assert fp.tf32_is_off({"tf32": {**j1_like["tf32"], "cuda_matmul_allow_tf32": True}}) is False
    assert fp.tf32_is_off({"tf32": {**j1_like["tf32"], "cudnn_allow_tf32": True}}) is False
    assert fp.tf32_is_off({}) is False
    # This process (the Fs CLI module is imported above) really runs TF32 off.
    prov = fp.collect_provenance(argv=[], device="cpu", source_roots=[])
    assert prov["tf32"]["float32_matmul_precision"] == "highest"
    assert fp.tf32_is_off(prov) is True


@pytest.mark.parametrize("set_flags, expected_off", [(True, True), (False, False)])
def test_tf32_fact_live_with_the_image_env_override(set_flags, expected_off):
    """Fresh interpreter with TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 (as the ACP
    image exports): with the explicit setters the Fs entry points issue at
    import, TF32 is effectively off and the fact says so; without them the env
    initializer leaks through (and cuDNN TF32 is on by default) and the fact
    must say NOT off."""
    import os
    import subprocess

    setter = ("torch.backends.cuda.matmul.allow_tf32 = False; "
              "torch.backends.cudnn.allow_tf32 = False; ") if set_flags else ""
    code = ("import torch; " + setter +
            "from earthdelta import fs_protocol as fp; "
            "p = fp.collect_provenance(argv=[], device='cpu', source_roots=[]); "
            "print('TF32_OFF', fp.tf32_is_off(p), p['tf32']['float32_matmul_precision'])")
    env = dict(os.environ, TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="1",
               PYTHONPATH=os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         timeout=300)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("TF32_OFF")]
    assert line, out.stderr[-2000:]
    assert line[0].split()[1] == str(expected_off)


def test_backend_precondition_refuses_a_wrong_or_missing_xformers():
    with pytest.raises(fp.ProtocolViolation) as excinfo:
        fp.official_backend_precondition(expected_xformers="0.0.27-never-installed")
    assert excinfo.value.code == "OFFICIAL_BACKEND_UNAVAILABLE"


def test_s0_certificate_must_bind_the_loaded_modules(tmp_path):
    cert = tmp_path / "s0.json"
    modules = fp.loaded_module_hashes()
    body = {"status": "ok", "s0_gate_pass": True, "verdict_committed": True,
            "gate_criteria": {"a": True}, "computed_checkpoint_sha256": "c",
            "normalization_identity": "n", "config_digest": "d",
            "torch_version": torch.__version__,
            "source_identity": {"modules": {k: {"sha256": v["sha256"]} for k, v in modules.items()}}}
    cert.write_text(json.dumps(body))
    gate = type("G", (), {"expected_checkpoint_sha256": "c",
                          "expected_normalization_identity": "n", "config_digest": "d"})()
    ok = fp.consume_s0_certificate(cert, expected_sha256=fp.file_sha256(cert), gate_config=gate)
    assert ok["passed"] is True
    body["source_identity"]["modules"]["earthdelta.lowrank"]["sha256"] = "0" * 64
    cert.write_text(json.dumps(body))
    with pytest.raises(fp.ProtocolViolation):
        fp.consume_s0_certificate(cert, expected_sha256=fp.file_sha256(cert), gate_config=gate)
    body["source_identity"]["modules"]["earthdelta.lowrank"]["sha256"] = \
        modules["earthdelta.lowrank"]["sha256"]
    body["verdict_committed"] = False
    cert.write_text(json.dumps(body))
    with pytest.raises(fp.ProtocolViolation):
        fp.consume_s0_certificate(cert, expected_sha256=fp.file_sha256(cert), gate_config=gate)


# =============================================================================
# Adapter file identity and the continuation-after-hold probe
# =============================================================================

def test_load_fs_adapter_checks_bytes_and_blocks(tmp_path):
    adapters = sa.build_fs_adapter(16, (0, 1), rank_per_expert=2, seed=3)
    path = tmp_path / "a.pt"
    torch.save({b: l.state_dict() for b, l in adapters.items()}, path)
    loaded, sha = sa.load_fs_adapter(path, 16, (0, 1), rank_per_expert=2,
                                     expected_sha256=fp.file_sha256(path))
    assert sa.static_adapter_digest(loaded) == sa.static_adapter_digest(adapters)
    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        sa.load_fs_adapter(path, 16, (0, 1), rank_per_expert=2, expected_sha256="0" * 64)
    assert excinfo.value.code == "FS_ADAPTER_SHA256_MISMATCH"
    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        sa.load_fs_adapter(path, 16, (0, 1, 2), rank_per_expert=2)
    assert excinfo.value.code == "FS_ADAPTER_BLOCKS_MISMATCH"


def test_continuation_after_hold_is_fs_from_the_edited_state():
    torch.set_num_threads(1)
    bridge, variables, samples, spec = _synthetic()
    hidden = bridge.model.blocks[0].attn.proj.in_features
    fs = sa.build_fs_adapter(hidden, (0, 1), rank_per_expert=4, seed=7)
    with torch.no_grad():
        for lora in fs.values():
            lora.up[0].weight.normal_(0.0, 0.05)
    merged, artifact = sa.merge_static_adapter(bridge.model, fs, target_blocks=(0, 1))
    fs_bridge = sa.make_fs_bridge(merged, bridge, artifact)
    probe = sa.build_continuation_probe_bank(hidden, (0, 1), num_experts=4, rank_per_expert=2)
    result = sa.verify_continuation_is_fs(fs_bridge, bridge, probe, samples[0].x_norm, variables,
                                          hold=2, total=5, target_blocks=(0, 1))
    assert result["passed"] is True, result
    assert result["exact_fs_continuation"] and result["discriminates_f0"]
    assert result["edit_effect_at_hold_max_abs"] > 0.0
    # The real (zero-B) bank would make the probe vacuous: refused, not passed.
    zero_bank = bt.build_dynamic_bank(hidden, (0, 1), num_experts=4, rank_per_expert=2, seed=1)
    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        sa.verify_continuation_is_fs(fs_bridge, bridge, zero_bank, samples[0].x_norm, variables,
                                     hold=2, total=5, target_blocks=(0, 1))
    assert excinfo.value.code == "CONTINUATION_PROBE_VACUOUS"
    # Continuing as F0 instead of Fs is detected.
    wrong = sa.verify_continuation_is_fs(fs_bridge, fs_bridge, probe, samples[0].x_norm,
                                         variables, hold=2, total=5, target_blocks=(0, 1))
    assert wrong["discriminates_f0"] is False and wrong["passed"] is False
