"""FP-03 Fs-consumer subset of FP-02: the admission record is certified by the
CONSUMER, not trusted from its own PASS flag (TEST_PLAN B.3).

A small zarr store with real channel names is certified with the real
`verify_content_subset`; each negative case changes exactly one property of
an otherwise-valid admission record. The real CLI entry (`r2_fs_bank_train`)
is also driven end to end with the bridge swapped for a synthetic one: a
tampered admission exits before the trainer is ever called.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

xr = pytest.importorskip("xarray")

from earthdelta import static_adapter as sa  # noqa: E402
from earthdelta.bridge import DEFAULT_VARIABLES  # noqa: E402
from earthdelta.pilot_contract import verify_content_subset  # noqa: E402

import scripts.r2_fs_bank_train as r2_cli  # noqa: E402
from tests.test_content_verification_b08 import NORM_DIR, build_store  # noqa: E402

CHANNELS = list(DEFAULT_VARIABLES[:8])
N_TIME = 20
GRID_HASH = "contentgridhash01"
VAR_HASH = "contentvarhash01"
PGID = "pg_test_fs_consumer"


@pytest.fixture
def bridge():
    b, _, _ = r2_cli.build_synthetic_bridge(torch.device("cpu"))
    return b


@pytest.fixture
def store(tmp_path) -> Path:
    return build_store(tmp_path / "2020.zarr", n_time=N_TIME, channels=CHANNELS)


def _epoch(store: Path, index: int) -> int:
    ds = xr.open_zarr(store)
    try:
        return int(np.datetime64(ds["time"].values[index], "s").astype("int64"))
    finally:
        ds.close()


def _row(store: Path, index: int, norm: str, *, history=2, lead_hours=72) -> dict:
    return {
        "admitted": True, "event_id": f"e{index}", "issue_id": f"iss_{index:03d}",
        "process_group_id": PGID, "split_id": "test", "data_role": "bank_fit",
        "issue_time": _epoch(store, index), "lead_hours": float(lead_hours),
        "normalization_hash": norm, "grid_hash": GRID_HASH, "formal": True,
        "interval_hours": 6, "history_steps": history,
        "history_store": str(store), "history_index": index - history,
        "issue_store": str(store), "issue_index": index,
        "target_store": str(store), "target_index": index + lead_hours // 6,
        "content_certificate": None,
    }


def _cert(store: Path, row: dict) -> dict:
    indices = list(range(row["history_index"], row["target_index"] + 1))
    return verify_content_subset(
        store, indices, batch_size=4, expected_channels=len(CHANNELS),
        normalization_dir=NORM_DIR, data_role="bank_fit", issue_id=row["issue_id"],
        process_group_id=PGID, strict=True,
    ).to_dict()


def _record(store: Path, bridge, *, history=2, lead_hours=72) -> dict:
    rows = [_row(store, i, bridge.normalization.digest, history=history, lead_hours=lead_hours)
            for i in (2, 5)]
    return {
        "gate": "r2_admission_gate", "formal": True, "data_role": "bank_fit",
        "passed": True, "process_group_id": PGID,
        "results": {"b09_real_sample_admission": True, "b08_content_verification": True},
        "admission": {"passed": True, "formal": True, "data_role": "bank_fit",
                      "admitted": rows, "rejected": []},
        "content_certificates": [_cert(store, r) for r in rows],
    }


def _certify(record, bridge, leads=(1, 4, 12)):
    return sa.certify_admission_for_fs(
        record, lead_steps=leads, required_history_steps=2,
        expected_normalization_digest=bridge.normalization.digest,
        expected_grid_hash=GRID_HASH, expected_variable_order_hash=VAR_HASH)


def _codes(excinfo) -> set:
    return {f["code"] for f in excinfo.value.detail["failures"]}


def test_valid_record_certifies_and_loads(store, bridge):
    record = _record(store, bridge)
    report = _certify(record, bridge)
    assert report["passed"] and report["n_rows"] == 2 and report["content_reverified"]
    assert all(r["fresh_content_sha256"] == r["content_sha256"] for r in report["rows"])
    out = {}
    samples = sa.load_admitted_samples(
        record, bridge, lead_steps=(1, 4, 12), require_certified=True,
        expected_grid_hash=GRID_HASH, expected_variable_order_hash=VAR_HASH,
        certification_out=out)
    assert [s.issue_id for s in samples] == ["iss_002", "iss_005"]
    assert sorted(samples[0].targets_raw) == [1, 4, 12] and out["passed"] is True


@pytest.mark.parametrize("mutate, code", [
    (lambda r: r.update(passed=False), "NOT_PASSED"),
    (lambda r: r["admission"].update(passed=False), "NOT_PASSED"),
    (lambda r: r["results"].update(b08_content_verification=False), "GATE_RESULT_NOT_TRUE"),
    (lambda r: r.update(formal=False), "NOT_FORMAL"),
    (lambda r: r["content_certificates"].pop(), "ROW_CERTIFICATE_COUNT"),
    (lambda r: r["content_certificates"][0].update(content_sha256="0" * 64), "CONTENT_SHA256_MISMATCH"),
    (lambda r: r["content_certificates"][0]["times_utc"].__setitem__(0, "2019-12-31T00:00:00"),
     "CERT_UTC_MISMATCH"),
    (lambda r: r["admission"]["admitted"][0].update(issue_time=r["admission"]["admitted"][0]["issue_time"] + 21600),
     "ROW_ISSUE_TIME_MISMATCH"),
    (lambda r: r["admission"]["admitted"][0].update(normalization_hash="ffffffffffffffff"),
     "ROW_NORMALIZATION_MISMATCH"),
    (lambda r: r["admission"]["admitted"][1].update(grid_hash="othergrid"), "ROW_GRID_MISMATCH"),
    (lambda r: r["content_certificates"][1].update(data_role="policy_dev"), "CERT_NOT_BANK_FIT"),
    (lambda r: r["content_certificates"][1].update(issue_id="iss_002"), "ROW_CERTIFICATE_COUNT"),
])
def test_each_single_defect_is_refused(store, bridge, mutate, code):
    record = _record(store, bridge)
    mutate(record)
    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        _certify(record, bridge)
    assert excinfo.value.code == "ADMISSION_NOT_CERTIFIED"
    assert code in _codes(excinfo)


def test_a_6h_certificate_cannot_feed_24h_or_72h_targets(store, bridge):
    record = _record(store, bridge, history=1, lead_hours=6)
    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        _certify(record, bridge, leads=(4,))
    codes = _codes(excinfo)
    assert {"ROW_LEAD_TOO_SHORT", "CERT_DOES_NOT_COVER_WINDOW", "ROW_HISTORY_TOO_SHORT"} <= codes


def test_the_old_uncertified_admission_is_refused(bridge):
    old = REPO_ROOT / "artifacts/round2_next/20260921T170142Z_r3p001_10h/j3_fs_bank/admission_record.json"
    if not old.is_file():
        pytest.skip("historical admission not present on this machine")
    record = json.loads(old.read_text())
    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        sa.certify_admission_for_fs(record, lead_steps=(4,), reverify_content=False,
                                    expected_normalization_digest="3e0b216bfbf34ab0")
    codes = _codes(excinfo)
    assert {"ROW_LEAD_TOO_SHORT", "ROW_HISTORY_TOO_SHORT", "ROW_CERTIFICATE_COUNT"} <= codes


def test_real_cli_entry_refuses_a_tampered_admission_before_training(tmp_path, store, bridge):
    """Drive `main` down the REAL (non-synthetic) path with only the protocol
    binding and the checkpoint load stubbed: the certified loader runs, refuses,
    and the fit is never called."""
    record = _record(store, bridge)
    record["content_certificates"][0]["content_sha256"] = "0" * 64
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps(record))
    lat = torch.linspace(-88.0, 88.0, 16)
    fake_gate = type("G", (), {"config_digest": "d", "identity_tag": "t"})()
    with mock.patch.object(r2_cli, "bind_real_run",
                           return_value={"protocol": None, "binding": {}}), \
            mock.patch.object(r2_cli, "build_real_bridge",
                              return_value=(bridge, list(bridge.variables), lat, fake_gate)), \
            mock.patch.object(r2_cli.sa, "load_admitted_samples",
                              wraps=sa.load_admitted_samples) as loader, \
            mock.patch.object(r2_cli.sa, "fit_static_adapter") as fit:
        code = r2_cli.main(["--stage", "fs_fit", "--mode", "gradient_check", "--config",
                            str(tmp_path / "gate.json"), "--admission", str(admission),
                            "--device", "cpu", "--no-require-official-backend",
                            "--output-dir", str(tmp_path / "out")])
    assert code == 2
    assert fit.call_count == 0
    assert loader.call_args.kwargs["require_certified"] is True
    refused = json.loads((tmp_path / "out" / "binding_refused.json").read_text())
    assert refused["code"] == "ADMISSION_NOT_CERTIFIED"
