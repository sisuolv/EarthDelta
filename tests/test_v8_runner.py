from __future__ import annotations

import json
from pathlib import Path

import pytest

from earthdelta.v8.receipt_registry import ReceiptError
from scripts.v8_research import Autorun


APP = Path("/mnt/afs/260010168/earthdelta_v8_approvals/approvals.json")
APP_SHA = APP.with_name("approvals.sha256")


def test_runner_dry_run_does_not_access_weather(tmp_path):
    run = tmp_path / "run"
    result = Autorun(run, dry_run=True, approvals=APP, approvals_sha=APP_SHA).run_until("T01a_contract")
    assert result["status"] == "OBSERVED"
    assert json.loads((run / "STATUS.json").read_text())["state"] == "TO_BE_RUN"


def test_runner_stops_at_missing_signed_source_after_formal_precheck(tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setenv("EARTHDELTA_V8_DATA_ROOT", str(tmp_path / "v8-data"))
    monkeypatch.setenv("EARTHDELTA_V8_SOURCE_ROOT", str(tmp_path / "missing-source"))
    result = Autorun(run, approvals=APP, approvals_sha=APP_SHA).run_until("T01a_data_repair")
    # The exposed endpoint table now produces a formal precheck; the next
    # gate remains fail-closed because no signed source adapter is configured.
    assert result["status"] == "BLOCKED"
    assert (run / "precision_precheck.json").is_file()
    assert json.loads((run / "STATUS.json").read_text())["state"] == "BLOCKED"


def test_gpu_budget_negative_path(tmp_path):
    from scripts import v8_submit_gpu
    with pytest.raises(Exception):
        v8_submit_gpu.main(["--scope", "T02_gate_gpu_max2", "--run", str(tmp_path), "--engine", "H100",
                            "--card-hours", "3", "--dry-run"])
