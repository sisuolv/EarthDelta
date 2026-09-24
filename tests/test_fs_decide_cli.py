"""End-to-end CPU smoke of `scripts/r4_fs_decide.py` on synthetic process outputs.

Real GPU runs cannot happen here, so each synthetic process's run_record.json
is given the Q0 facts a real run records (S0 consumed, official backend,
versions, TF32 off, certified admissions). The decisions must then follow the
rules: the diag decision PASSes only when every negative control is rejected
and flips to STOP when one Q0 fact is broken; screen/formal decisions are
produced from the raw files and bound to the protocol hash.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta import fs_protocol as fp  # noqa: E402
from earthdelta import static_adapter as sa  # noqa: E402

import scripts.r2_fs_bank_train as r2_cli  # noqa: E402
import scripts.r4_fs_decide as decide  # noqa: E402

BASE = ["--synthetic", "--device", "cpu", "--synthetic-samples", "2",
        "--panel-lead-steps", "1", "4", "12"]
FAKE = {"admission": "a" * 64, "panel_admission": "b" * 64, "s0": "c" * 64}


def _inject(proc: Path, protocol_sha: str, config_id: str, *, official=True) -> None:
    run = json.loads((proc / "run_record.json").read_text())
    run["protocol_sha256"] = protocol_sha
    run["config_id"] = config_id
    run["binding"] = {"s0": {"passed": True, "sha256": FAKE["s0"]},
                      "backend_postcondition": {"passed": official},
                      "admission_consumer": {"passed": True},
                      "panel_admission_consumer": {"passed": True}}
    prov = run["provenance"]
    prov["official_backend"] = official
    prov["versions"].update(torch="2.3.1+cu121", xformers="0.0.27")
    prov["tf32"] = {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "env": {}}
    prov["input_files"]["admission"] = {"sha256": FAKE["admission"]}
    prov["input_files"]["panel_admission"] = {"sha256": FAKE["panel_admission"]}
    (proc / "run_record.json").write_text(json.dumps(run))


def _bad_history(path: Path) -> dict:
    losses = [1.0 - 0.001 * t for t in range(64)]
    for t in range(20, 30):
        losses[t] = 5.0
    ids = [f"iss_synthetic_{t % 2:03d}" for t in range(64)]
    grads = [0.01] * 64
    grads[21] = 9.0
    path.write_text(json.dumps({"mode": "formal", "n_updates": 64, "losses": losses,
                                "grad_norms": grads, "training_issue_ids": ids,
                                "config": {"grad_clip": 1.0}, "clip_events": 1}))
    return {"path": str(path), "sha256": fp.file_sha256(path)}


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    bad = tmp_path / "bad_adapter.pt"
    adapters = sa.build_fs_adapter(64, (0, 1), rank_per_expert=4, seed=20260921)
    with torch.no_grad():
        for lora in adapters.values():
            lora.up[0].weight.normal_(0.0, 0.5)
    torch.save({b: l.state_dict() for b, l in adapters.items()}, bad)
    init = sa.static_adapter_digest(sa.build_fs_adapter(64, (0, 1), rank_per_expert=4,
                                                        seed=20260921))
    hist = _bad_history(tmp_path / "hist.json")
    configs = {}
    for i, lr in enumerate((1e-2, 5e-3, 2.5e-3, 1.25e-3)):
        configs[f"J1_A{i}"] = {"job": "J1", "stage": "fs_fit", "mode": "gradient_check",
                               "max_updates": 4, "learning_rate": lr, "device": f"cuda:{i}",
                               "_output_subdir": f"arms/A{i}"}
        configs[f"J1_NC{i}"] = {"job": "J1", "stage": "fs_panel", "device": f"cuda:{i}",
                                "fs_adapter": {"path": str(bad), "sha256": fp.file_sha256(bad)},
                                "_historical_fit_record": hist, "_output_subdir": f"negctl/NC{i}"}
        configs[f"J2_A{i}"] = {"job": "J2", "stage": "fs_fit", "mode": "stability_screen",
                               "max_updates": 16, "learning_rate": lr, "device": f"cuda:{i}",
                               "_output_subdir": f"arms/A{i}"}
    configs["J3"] = {"job": "J3", "stage": "fs_fit", "mode": "formal", "max_updates": 16,
                     "learning_rate": 2.5e-3,
                     "device": {"one_of": [f"cuda:{i}" for i in range(4)]},
                     "_output_subdir_template": "replica{index}"}
    q = {"horizon": 16,
         "q0": {"torch_version": "2.3.1+cu121", "xformers_version": "0.0.27", "f0_panel_rel_tol": 1e-6},
         "q2": {"clip_events_max": 0, "per_visit_ratio_max": 1.25, "epoch_onset_factor": 1.02},
         "q3": {"required_issues": 2},
         "q4": {"pass_max_ratio_lt": 1.0, "pass_mean_ratio_le": 0.99, "fail_mean_ratio_gt": 1.0,
                "fail_max_ratio_gt": 1.02},
         "q5": {"mean_ratio_72h_le": 1.01, "report_only": False},
         "q6": {"holdout_mean_ratio_24h_le": 1.02, "report_only": False}}
    protocol = {
        "schema_version": fp.PROTOCOL_SCHEMA,
        "qualification_rule": q,
        "screen_rule": {"horizon": 16, "control_arm": "A0", "candidate_arms": ["A1", "A2", "A3"],
                        "progressing_mean_ratio_24h_le": 0.99},
        "selection_rule": {"n_replicas": 4, "designated_device_index": 0},
        "inputs": {"initial_adapter_digest": init,
                   "train_issue_ids": ["iss_synthetic_000", "iss_synthetic_001"],
                   "holdout_issue_ids": ["iss_synthetic_holdout_000", "iss_synthetic_holdout_001"],
                   "s0_certificate": {"sha256": FAKE["s0"]},
                   "admission": {"sha256": FAKE["admission"]},
                   "panel_admission": {"sha256": FAKE["panel_admission"]}},
        "configs": configs,
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return tmp_path, path, fp.file_sha256(path), protocol, bad


def _run(job: Path, protocol, sha: str, prefix: str):
    for cid, cfg in protocol["configs"].items():
        if not cid.startswith(prefix):
            continue
        if "_output_subdir_template" in cfg:  # formal: one config, one process per device
            for i, _ in enumerate(cfg["device"]["one_of"]):
                out = job / cfg["_output_subdir_template"].format(index=i)
                code = r2_cli.main(["--stage", "fs_fit", "--mode", cfg["mode"], "--max-updates",
                                    str(cfg["max_updates"]), "--learning-rate",
                                    str(cfg["learning_rate"])] + BASE + ["--output-dir", str(out)])
                assert code in (0, 1)
                _inject(out, sha, cid)
            continue
        out = job / cfg["_output_subdir"]
        if cfg["stage"] == "fs_fit":
            args = ["--stage", "fs_fit", "--mode", cfg["mode"], "--max-updates",
                    str(cfg["max_updates"]), "--learning-rate", str(cfg["learning_rate"])]
        else:
            args = ["--stage", "fs_panel", "--fs-adapter", cfg["fs_adapter"]["path"],
                    "--fs-adapter-sha256", cfg["fs_adapter"]["sha256"]]
        code = r2_cli.main(args + BASE + ["--output-dir", str(out)])
        assert code in (0, 1)
        _inject(out, sha, cid)


def test_decider_diag_screen_formal_end_to_end(setup):
    tmp, protocol_path, sha, protocol, _ = setup
    j1 = tmp / "J1"
    _run(j1, protocol, sha, "J1_")
    diag_out = tmp / "decisions" / "J1.json"
    assert decide.main(["--kind", "diag", "--protocol", str(protocol_path), "--protocol-sha256", sha,
                        "--job-dir", str(j1), "--out", str(diag_out)]) == 0
    diag = json.loads(diag_out.read_text())
    assert diag["protocol_sha256"] == sha
    assert all(nc["rejected"] for nc in diag["negative_controls"]), diag["negative_controls"]
    assert all(arm["checks_pass"] and arm["q0_pass"] for arm in diag["arms"]), diag["arms"]
    assert diag["verdict"] == "PASS"
    # Written once: a second decision into the same file is refused.
    assert decide.main(["--kind", "diag", "--protocol", str(protocol_path), "--protocol-sha256", sha,
                        "--job-dir", str(j1), "--out", str(diag_out)]) == 2

    j2 = tmp / "J2"
    _run(j2, protocol, sha, "J2_")
    screen_out = tmp / "decisions" / "J2.json"
    assert decide.main(["--kind", "screen", "--protocol", str(protocol_path), "--protocol-sha256",
                        sha, "--job-dir", str(j2), "--diag-decision", str(diag_out),
                        "--out", str(screen_out)]) == 0
    screen = json.loads(screen_out.read_text())
    assert screen["verdict"] in ("NOT_TESTABLE", "REFUTED", "CONFIRMED")
    assert [row["arm_id"] for row in screen["arms"]] == ["A0", "A1", "A2", "A3"]
    assert screen["missing_arms"] == [] and screen["invalid_arms"] == []

    j3 = tmp / "J3"
    _run(j3, protocol, sha, "J3")
    formal_out = tmp / "decisions" / "J3.json"
    assert decide.main(["--kind", "formal", "--job", "J3", "--protocol", str(protocol_path),
                        "--protocol-sha256", sha, "--job-dir", str(j3), "--diag-decision",
                        str(diag_out), "--out", str(formal_out)]) == 0
    formal = json.loads(formal_out.read_text())
    assert formal["rule"] == "FS-SELECT-v1" and formal["substitution"] == "FORBIDDEN"
    assert formal["verdict"] in ("QUALITY_PASS_PENDING_DESIGNATED_GATES", "STOP_FITTED_FS_QUALITY",
                                 "INCONCLUSIVE_QUALIFICATION")
    assert formal["designated_record"]["mode"] == "formal"
    assert formal["argv_identical_except_device_output"] is True
    assert [row["device_index"] for row in formal["details"]] == [0, 1, 2, 3]


def test_decider_diag_stops_when_a_q0_fact_breaks(setup):
    tmp, protocol_path, sha, protocol, _ = setup
    j1 = tmp / "J1"
    _run(j1, protocol, sha, "J1_")
    _inject(j1 / "arms" / "A2", sha, "J1_A2", official=False)
    out = tmp / "d.json"
    assert decide.main(["--kind", "diag", "--protocol", str(protocol_path), "--protocol-sha256", sha,
                        "--job-dir", str(j1), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "STOP"


def test_decider_refuses_a_protocol_hash_mismatch(setup):
    tmp, protocol_path, _, _, _ = setup
    assert decide.main(["--kind", "diag", "--protocol", str(protocol_path), "--protocol-sha256",
                        "0" * 64, "--job-dir", str(tmp), "--out", str(tmp / "x.json")]) == 2
