"""FP-03 protocol v2 (CPU, synthetic): the v2 decision path and the full
freeze -> independent reload -> certify chain, end to end.

* A v2 protocol may take its train-F0 reference from an EARLIER protocol's
  PASS diag decision, but only if the bytes are the ones it pinned; a fresh
  holdout has no earlier reference, so its F0 panel must be bitwise identical
  across the job's replicas instead (tampering one replica -> INVALID).
* `--stage fs_freeze` (decision-gated), `--stage fs_verify` (independent
  reload) and `r4_fs_decide.py --kind certify` run in sequence and publish the
  fs/ bundle only on FS_SELECTED; a merge tolerance other than the
  pre-registered one blocks the bundle.
"""
from __future__ import annotations

import copy
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
from tests.test_fs_decide_cli import BASE, _inject, _run, setup  # noqa: E402,F401

TOLERANCE = {"atol_floor": 1e-5, "atol_relative": 1.5e-3, "rtol": 1e-5}


def _v2_protocol(tmp: Path, v1: dict, diag_path: Path, *, pin_sha: str = None) -> tuple:
    q = copy.deepcopy(v1["qualification_rule"])
    q["q0"]["f0_reference_groups"] = ["train"]
    q["q0"]["holdout_f0_cross_replica_bitwise"] = True
    protocol = {
        "schema_version": fp.PROTOCOL_SCHEMA,
        "protocol_id": "TEST-V2",
        "qualification_rule": q,
        "selection_rule": {"n_replicas": 4, "designated_device_index": 0},
        "merge_equivalence_tolerance": TOLERANCE,
        "inputs": {**v1["inputs"],
                   "f0_reference_decision": {"path": str(diag_path),
                                             "sha256": pin_sha or fp.file_sha256(diag_path)}},
        "configs": {"V2_P": {"job": "V2", "stage": "fs_fit", "mode": "formal", "max_updates": 16,
                             "learning_rate": 2.5e-3,
                             "device": {"one_of": [f"cuda:{i}" for i in range(4)]},
                             "_output_subdir_template": "replica{index}"}},
    }
    path = tmp / f"v2_protocol_{'pinned' if pin_sha is None else 'wrongpin'}.json"
    path.write_text(json.dumps(protocol))
    return path, fp.file_sha256(path), protocol


def _diag(tmp: Path, protocol_path: Path, sha: str, protocol: dict) -> Path:
    j1 = tmp / "J1"
    _run(j1, protocol, sha, "J1_")
    out = tmp / "decisions" / "J1.json"
    assert decide.main(["--kind", "diag", "--protocol", str(protocol_path), "--protocol-sha256",
                        sha, "--job-dir", str(j1), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "PASS"
    return out


def _run_v2(job: Path, protocol: dict, sha: str) -> None:
    for i in range(4):
        out = job / f"replica{i}"
        code = r2_cli.main(["--stage", "fs_fit", "--mode", "formal", "--max-updates", "16",
                            "--learning-rate", "0.0025"] + BASE + ["--output-dir", str(out)])
        assert code in (0, 1)
        _inject(out, sha, "V2_P")


def test_v2_formal_decision_uses_pinned_reference_and_cross_replica_holdout(setup):
    tmp, v1_path, v1_sha, v1, _ = setup
    diag_out = _diag(tmp, v1_path, v1_sha, v1)
    v2_path, v2_sha, v2 = _v2_protocol(tmp, v1, diag_out)
    job = tmp / "V2"
    _run_v2(job, v2, v2_sha)

    out = tmp / "decisions" / "V2.json"
    assert decide.main(["--kind", "formal", "--job", "V2_P", "--protocol", str(v2_path),
                        "--protocol-sha256", v2_sha, "--job-dir", str(job), "--diag-decision",
                        str(diag_out), "--out", str(out)]) == 0
    decision = json.loads(out.read_text())
    assert decision["verdict"] != "INVALID"
    for row in decision["details"]:
        facts = row["qualification"]["q0"]["facts"]
        assert facts["f0_panel_matches_reference"] is True
        assert facts["holdout_F0_identical_across_replicas"] is True
        assert row["qualification"]["q0"]["status"] == "PASS"
    assert decision["horizon"] == 16 and decision["argv_identical_except_device_output"] is True

    # One replica's holdout F0 panel differs -> Q0 fails on every replica -> INVALID.
    panels_path = job / "replica2" / "fs_panels.json"
    panels = json.loads(panels_path.read_text())
    first = next(iter(panels["holdout"]["F0"]))
    panels["holdout"]["F0"][first]["24"] *= 1.0 + 1e-9
    panels_path.write_text(json.dumps(panels))
    out2 = tmp / "decisions" / "V2_tampered.json"
    assert decide.main(["--kind", "formal", "--job", "V2_P", "--protocol", str(v2_path),
                        "--protocol-sha256", v2_sha, "--job-dir", str(job), "--diag-decision",
                        str(diag_out), "--out", str(out2)]) == 0
    tampered = json.loads(out2.read_text())
    assert tampered["verdict"] == "INVALID"
    assert all(row["qualification"]["q0"]["facts"]["holdout_F0_identical_across_replicas"] is False
               for row in tampered["details"])


def test_v2_refuses_a_reference_decision_other_than_the_pinned_bytes(setup):
    tmp, v1_path, v1_sha, v1, _ = setup
    diag_out = _diag(tmp, v1_path, v1_sha, v1)
    v2_path, v2_sha, _ = _v2_protocol(tmp, v1, diag_out, pin_sha="0" * 64)
    assert decide.main(["--kind", "formal", "--job", "V2_P", "--protocol", str(v2_path),
                        "--protocol-sha256", v2_sha, "--job-dir", str(tmp), "--diag-decision",
                        str(diag_out), "--out", str(tmp / "never.json")]) == 2
    assert not (tmp / "never.json").exists()


def _certify_protocol(tmp: Path, tolerance: dict) -> tuple:
    protocol = {
        "schema_version": fp.PROTOCOL_SCHEMA,
        "qualification_rule": {"horizon": 16},
        "selection_rule": {"n_replicas": 4, "designated_device_index": 0},
        "merge_equivalence_tolerance": tolerance,
        "inputs": {"checkpoint_sha256": "synthetic", "normalization_identity": "synthetic"},
        "limitations": ["synthetic test"],
        "configs": {},
    }
    path = tmp / f"certify_protocol_{tolerance['rtol']}.json"
    path.write_text(json.dumps(protocol))
    return path, fp.file_sha256(path)


def test_freeze_verify_certify_chain_end_to_end(tmp_path):
    torch.set_num_threads(1)
    replica0 = tmp_path / "V2" / "replica0"
    assert r2_cli.main(["--stage", "fs_fit", "--mode", "formal", "--max-updates", "16",
                        "--learning-rate", "0.0025"] + BASE + ["--output-dir", str(replica0)]) in (0, 1)
    adapter = replica0 / "fs_adapter.pt"
    sha = fp.file_sha256(adapter)
    rows = [{"config_id": "V2_P", "device_index": i, "dir": str(replica0),
             "qualification": {"verdict": "PASS"},
             "files": {"fs_adapter.pt": {"path": str(adapter), "sha256": sha}}} for i in range(4)]
    formal = tmp_path / "decisions" / "V2_formal.json"
    formal.parent.mkdir(parents=True)
    formal.write_text(json.dumps({
        "kind": "formal", "rule": "FS-SELECT-v1", "verdict": "QUALITY_PASS_PENDING_DESIGNATED_GATES",
        "horizon": 16, "designated_adapter_sha256": sha,
        "designated_record": {"mode": "formal", "n_updates": 16}, "details": rows}))

    freeze = tmp_path / "J4" / "freeze"
    verify = tmp_path / "J4" / "verify"
    assert r2_cli.main(["--stage", "fs_freeze", "--synthetic", "--device", "cpu", "--fs-adapter",
                        str(adapter), "--fs-adapter-sha256", sha, "--fs-decision", str(formal),
                        "--output-dir", str(freeze)]) == 0
    assert (freeze / "fs_merged_backbone.pt").is_file()
    assert r2_cli.main(["--stage", "fs_verify", "--synthetic", "--device", "cpu",
                        "--fs-merged-backbone", str(freeze / "fs_merged_backbone.pt"),
                        "--fs-adapter", str(adapter), "--fs-adapter-sha256", sha,
                        "--output-dir", str(verify)]) == 0
    reload = json.loads((verify / "independent_reload.json").read_text())
    assert reload["identity_pass"] and reload["reload_pass"] and reload["numerical_merge_pass"]
    assert reload["continuation"]["exact_fs_continuation"] and reload["continuation"]["discriminates_f0"]

    protocol, protocol_sha = _certify_protocol(tmp_path, TOLERANCE)
    out = tmp_path / "certify" / "V2_certify.json"
    assert decide.main(["--kind", "certify", "--protocol", str(protocol), "--protocol-sha256",
                        protocol_sha, "--formal-decision", str(formal), "--freeze-dir", str(freeze),
                        "--verify-dir", str(verify), "--out", str(out)]) == 0
    final = json.loads(out.read_text())
    assert final["verdict"] == "FS_SELECTED" and final["fs_selected"] is True
    bundle = tmp_path / "certify" / "fs"
    for name in ("fs_adapter.pt", "fs_merged_backbone.pt", "independent_reload.json",
                 "quality_rule.json", "qualification.json", "panel_initial_final.json",
                 "training_records.json", "reference_manifest.json"):
        assert (bundle / name).is_file(), name
    manifest = json.loads((bundle / "reference_manifest.json").read_text())
    assert manifest["fs_adapter_sha256"] == sha
    assert manifest["fs_merged_backbone_sha256"] == fp.file_sha256(freeze / "fs_merged_backbone.pt")

    # A tolerance other than the one actually applied blocks certification.
    wrong, wrong_sha = _certify_protocol(tmp_path, {**TOLERANCE, "rtol": 1e-3})
    out2 = tmp_path / "certify_wrong" / "V2_certify.json"
    assert decide.main(["--kind", "certify", "--protocol", str(wrong), "--protocol-sha256",
                        wrong_sha, "--formal-decision", str(formal), "--freeze-dir", str(freeze),
                        "--verify-dir", str(verify), "--out", str(out2)]) == 0
    blocked = json.loads(out2.read_text())
    assert blocked["verdict"] == "STOP_DESIGNATED_GATE_FAILED" and blocked["fs_selected"] is False
    assert not (tmp_path / "certify_wrong" / "fs").exists()
