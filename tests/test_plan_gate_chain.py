"""FP-02/FP-03 gate chain, Fs subset (TEST_PLAN B.2): the REAL stage dispatch
stops at the first failure and never calls a successor; real-data `--stage all`
is refused outright, and (FP-04, replacing the old blanket "LOCKED until FP-04")
every post-Fs stage is refused on real data unless the certified Fs is bound by
hash -- before any backbone is built.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta import static_adapter as sa  # noqa: E402

import scripts.r2_fs_bank_train as r2_cli  # noqa: E402


def _spies():
    return (
        mock.patch.object(r2_cli.sa, "merge_static_adapter", wraps=sa.merge_static_adapter),
        mock.patch.object(r2_cli.bt, "train_expert", wraps=r2_cli.bt.train_expert),
        mock.patch.object(r2_cli.bt, "profile_training_step",
                          wraps=r2_cli.bt.profile_training_step),
        mock.patch.object(r2_cli.bt, "check_horizon_feasibility",
                          wraps=r2_cli.bt.check_horizon_feasibility),
    )


def _run_all(tmp_path, *extra):
    merge_p, train_p, prof_p, hor_p = _spies()
    with merge_p as merge, train_p as train, prof_p as prof, hor_p as hor:
        code = r2_cli.main(["--stage", "all", "--synthetic", "--device", "cpu",
                            "--output-dir", str(tmp_path), *extra])
    record = json.loads((tmp_path / "run_record.json").read_text())
    return code, record, (merge.call_count, train.call_count, prof.call_count, hor.call_count)


def test_failed_fs_fit_blocks_every_successor(tmp_path):
    real_fit = sa.fit_static_adapter

    def failing_fit(*args, **kwargs):
        adapters, record = real_fit(*args, **kwargs)
        record.eligible = False
        record.ineligible_reason = "FORMAL_LOSS_DIRECTION_REGRESSION"
        return adapters, record

    with mock.patch.object(r2_cli.sa, "fit_static_adapter", side_effect=failing_fit):
        code, record, calls = _run_all(tmp_path, "--mode", "formal", "--max-updates", "6")
    assert code == 1
    assert calls == (0, 0, 0, 0)
    assert record["results"] == {"fs_fit": False}
    assert record["blocked_stages"] == ["fs_freeze", "bank_train", "profile", "horizon_check"]


def test_exception_in_fs_fit_blocks_every_successor(tmp_path):
    with mock.patch.object(r2_cli.sa, "fit_static_adapter", side_effect=RuntimeError("boom")):
        code, record, calls = _run_all(tmp_path, "--mode", "formal", "--max-updates", "6")
    assert code == 1 and calls == (0, 0, 0, 0)
    assert record["errors"][0]["code"] == "RuntimeError"


@pytest.mark.parametrize("mode", ["gradient_check", "stability_screen"])
def test_non_formal_fit_can_never_be_frozen_in_the_chain(tmp_path, mode):
    code, record, calls = _run_all(tmp_path, "--mode", mode, "--max-updates", "6")
    assert code == 1
    assert record["results"]["fs_freeze"] is False
    assert calls == (0, 0, 0, 0)
    assert record["blocked_stages"] == ["bank_train", "profile", "horizon_check"]


def test_real_stage_all_is_refused_before_anything_runs(tmp_path):
    with mock.patch.object(r2_cli, "build_real_bridge") as build, \
            mock.patch.object(r2_cli, "bind_real_run") as bind:
        code = r2_cli.main(["--stage", "all", "--config", str(tmp_path / "c.json"),
                            "--admission", str(tmp_path / "a.json")])
    assert code == 2 and build.call_count == 0 and bind.call_count == 0


POST_FS_STAGES = ["bank_train", "profile", "horizon_check", "bank_assemble", "bank_verify"]


@pytest.mark.parametrize("stage", POST_FS_STAGES)
def test_post_fs_stages_without_a_protocol_are_refused_on_real_data(tmp_path, stage):
    with mock.patch.object(r2_cli, "build_real_bridge") as build:
        code = r2_cli.main(["--stage", stage, "--config", str(tmp_path / "c.json"),
                            "--admission", str(tmp_path / "a.json")])
    assert code == 2 and build.call_count == 0


@pytest.mark.parametrize("stage", POST_FS_STAGES)
def test_post_fs_stages_need_the_certified_fs_on_real_data(tmp_path, stage):
    """Even with the protocol binding satisfied, a real post-Fs stage with no
    certified-Fs flags (or a manifest that is not the pinned bytes) exits 2
    before the backbone is built: FS_REFERENCE_NOT_AUTHORIZED."""
    protocol = {"protocol_kind": "bank", "inputs": {"fs_reference": {}}}
    bound = {"protocol": protocol, "binding": {}}
    with mock.patch.object(r2_cli, "bind_real_run", return_value=bound), \
            mock.patch.object(r2_cli, "build_real_bridge") as build:
        code = r2_cli.main(["--stage", stage, "--config", str(tmp_path / "c.json"),
                            "--admission", str(tmp_path / "a.json"),
                            "--output-dir", str(tmp_path / "out")])
    assert code == 2 and build.call_count == 0
    refused = json.loads((tmp_path / "out" / "binding_refused.json").read_text())
    assert refused["code"] == "FS_REFERENCE_NOT_AUTHORIZED"

    manifest = tmp_path / "reference_manifest.json"
    manifest.write_text(json.dumps({"schema_version": "ed-fs-reference-manifest/1"}))
    for name in ("decision.json", "adapter.pt", "merged.pt"):
        (tmp_path / name).write_text("{}")
    bound = {"protocol": protocol, "binding": {}}
    with mock.patch.object(r2_cli, "bind_real_run", return_value=bound), \
            mock.patch.object(r2_cli, "build_real_bridge") as build:
        code = r2_cli.main(["--stage", stage, "--config", str(tmp_path / "c.json"),
                            "--admission", str(tmp_path / "a.json"),
                            "--fs-reference-manifest", str(manifest),
                            "--fs-reference-manifest-sha256", "0" * 64,
                            "--fs-certify-decision", str(tmp_path / "decision.json"),
                            "--fs-adapter", str(tmp_path / "adapter.pt"),
                            "--fs-merged-backbone", str(tmp_path / "merged.pt"),
                            "--output-dir", str(tmp_path / "out2")])
    assert code == 2 and build.call_count == 0
    refused = json.loads((tmp_path / "out2" / "binding_refused.json").read_text())
    assert refused["code"] == "FS_REFERENCE_NOT_AUTHORIZED"
    assert any("reference manifest hashes to" in f for f in refused["detail"]["failures"])
