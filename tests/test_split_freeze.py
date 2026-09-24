"""FP-05b split freeze: tag never authorises, boundaries exact to the hour, fail closed.

Uses the pinned FP-05a exposure ledger (a local JSON file, read-only) to build a
test freeze in tmp_path; no zarr store is opened (read_slab runs against a fake
store object injected through `split_freeze._open_store`).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from earthdelta import split_freeze as sf

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "plans/plan_v4_0923/run_20260924T093810Z_fp05a/entry/exposure_ledger.json"
FP04_ADMISSION = REPO / "plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/admission/bank_fit_admission.json"
FP03_V1_HOLDOUT = REPO / "plans/plan_v4_0923/run_20260923T192759Z_fp03_fs/admission/holdout_panel_admission.json"
RUN = REPO / "plans/plan_v4_0923/run_20260924T104725Z_fp05b"
PINNED = RUN / "splits/split_freeze_v1.json"
DATA = REPO / "data/era5_1p40625"
DEBUG_IDS = ["iss_292517fb189ee1bd", "iss_039add2d41f5c08a", "iss_4f2b54d7e8953451",
             "iss_6361e4cdd8050ae3", "iss_32d48efebee12be2", "iss_bbd4e6e2ec71d9ca",
             "iss_04eb8e8d56c68faa", "iss_b42c91ed1b555939"]

pytestmark = pytest.mark.skipif(not LEDGER.is_file() or not FP04_ADMISSION.is_file(),
                                reason="pinned FP-05a/FP-04 records not present")


def freeze_payload() -> dict:
    derived = sf.rebuild_from_ledger(LEDGER)
    pool = derived["policy_dev_pool"]
    admission_sha = sf.file_sha256(FP04_ADMISSION)
    return {
        "schema": sf.SCHEMA, "buffer_hours": 24, "derived": derived,
        "derived_digest": sf.canonical_digest(derived),
        "stores_allowed": {
            "2019.zarr": {"path": str(DATA / "2019.zarr"), "time0_utc": "2019-01-01T00Z",
                          "n_timesteps": 1460},
            "2020.zarr": {"path": str(DATA / "2020.zarr"), "time0_utc": "2020-01-01T00Z",
                          "n_timesteps": 1464}},
        "roles": {
            "policy_dev": {"store": "2019.zarr",
                           "support_window_utc": [
                               sf.iso(sf.parse_utc(pool["first_issue_utc"]) - 12 * 3600),
                               sf.iso(sf.parse_utc(pool["last_issue_utc"]) + 72 * 3600)],
                           "issue_guard_window_utc": ["2019-01-02T00Z", "2019-12-29T00Z"]},
            "debug": {"store": "2020.zarr", "allowed_issue_ids": list(DEBUG_IDS),
                      "source_admission": {"path": str(FP04_ADMISSION), "sha256": admission_sha},
                      "support_window_utc": ["2020-01-01T12Z", "2020-07-04T18Z"],
                      "issue_guard_window_utc": ["2020-01-02T00Z", "2020-12-29T00Z"],
                      "excluded_store_indices": [712]},
            "confirm": {"status": sf.CONFIRM_STATUS, "window": None},
            "bank_fit": {"consumable_by_fp05": False}},
    }


def write_freeze(tmp_path: Path, payload: dict | None = None):
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(payload or freeze_payload(), indent=1))
    return sf.load_freeze(path, sf.file_sha256(path)), path


def make_row(issue_utc: str, *, year: int | None = None, tag: str | None = "policy_dev",
             issue_id: str | None = None, store: str | None = None) -> dict:
    t = sf.parse_utc(issue_utc)
    year = year or int(issue_utc[:4])
    t0 = sf.parse_utc(f"{year}-01-01T00")
    i = (t - t0) // (6 * 3600)
    path = str(DATA / (store or f"{year}.zarr"))
    return {"issue_id": issue_id or f"iss_test_{issue_utc}", "issue_time": t,
            "valid_time": t + 72 * 3600, "history_time": t - 12 * 3600, "lead_hours": 72.0,
            "interval_hours": 6, "history_steps": 2, "data_role": tag,
            "history_store": path, "issue_store": path, "target_store": path,
            "issue_index": i, "history_index": i - 2, "target_index": i + 12}


@pytest.fixture()
def freeze(tmp_path):
    return write_freeze(tmp_path)[0]


def fp04_rows():
    return json.loads(FP04_ADMISSION.read_text())["admission"]["admitted"]


def test_rebuild_reproduces_fp05a_pool_exactly():
    d = sf.rebuild_from_ledger(LEDGER)
    pool = d["policy_dev_pool"]
    assert (pool["n_slots"], pool["first_index"], pool["last_index"]) == (698, 746, 1443)
    assert (pool["first_issue_utc"], pool["last_issue_utc"]) == ("2019-07-06T12Z", "2019-12-27T18Z")
    alt = d["alternative_rules_for_the_record"]
    assert alt["full_24h_buffer_certified_rows_only"]["n_slots"] == 700
    assert alt["no_cross_year_buffer"]["n_slots"] == 702
    assert d["exposed_runs_unbuffered_utc"][-3][0] == "2020-01-01T00Z"   # S0 start, 00Z not 12Z
    led = json.loads(LEDGER.read_text())
    theirs = [r[:2] for y in ("2019", "2020")
              for r in led["runs_by_year"][y]["gradient_or_result_runs_unbuffered"]]
    assert [r[:2] for r in d["exposed_runs_unbuffered_utc"]] == theirs


@pytest.mark.parametrize("issue,clear", [
    ("2019-12-27T18Z", True), ("2019-12-28T00Z", False),   # end 12-31T00Z touches 2020 start - 24h
    ("2019-07-06T12Z", True), ("2019-07-06T06Z", False),   # start 07-05T18Z touches 07-04T18Z + 24h
    ("2019-10-01T00Z", True),
])
def test_boundary_exact_to_the_hour(freeze, issue, clear):
    v = sf.classify_row(freeze, make_row(issue), "policy_dev", pre_admission=True)
    assert v["clear"] is clear, v["reasons"]


def test_support_not_issue_time_decides(freeze):
    # issue 07-06T06Z is itself 36h after the last 2019 exposure run, but its SUPPORT
    # start (07-05T18Z) touches the 24h-widened run: refused.
    v = sf.classify_row(freeze, make_row("2019-07-06T06Z"), "policy_dev", pre_admission=True)
    assert "TOUCHES_EXPOSED_RUN_WITHIN_BUFFER" in v["reasons"]
    assert v["touches_exposed_buffered"] == [["2019-07-01T06Z", "2019-07-04T18Z"]]


def test_tag_policy_dev_does_not_authorise_exposed_rows(freeze):
    for issue in ("2019-03-06T00Z", "2019-07-02T00Z", "2019-12-30T00Z"):
        row = make_row(issue, tag="policy_dev")
        v = sf.classify_row(freeze, row, "policy_dev", pre_admission=True)
        assert not v["clear"] and v["tag"] == "policy_dev"


def test_fp04_ninth_row_same_tag_refused_as_debug(freeze):
    rows = fp04_rows()
    sha = sf.file_sha256(FP04_ADMISSION)
    allowed = [r for r in rows if r["issue_id"] in DEBUG_IDS]
    assert len(allowed) == 8 and {r["data_role"] for r in rows} == {"bank_fit"}
    report = sf.assert_rows_clear(freeze, allowed, "debug", source_admission_sha256=sha)
    assert report["passed"] and report["n_clear"] == 8
    ninth = rows[8]
    assert ninth["issue_id"] not in DEBUG_IDS and ninth["data_role"] == "bank_fit"
    with pytest.raises(sf.SplitFreezeViolation) as exc:
        sf.assert_rows_clear(freeze, [ninth], "debug", source_admission_sha256=sha)
    assert exc.value.code == "ROW_NOT_CLEAR"
    assert "NOT_ON_ALLOW_LIST" in exc.value.detail["rows"][0]["reasons"]


def test_debug_needs_the_pinned_source_admission(freeze):
    allowed = [r for r in fp04_rows() if r["issue_id"] in DEBUG_IDS]
    with pytest.raises(sf.SplitFreezeViolation):
        sf.assert_rows_clear(freeze, allowed, "debug", source_admission_sha256="0" * 64)


def test_retagging_never_changes_a_verdict(freeze):
    sha = sf.file_sha256(FP04_ADMISSION)
    rows = fp04_rows()
    for row in (rows[0], rows[8]):
        base = sf.classify_row(freeze, row, "debug", source_admission_sha256=sha)
        for tag in ("policy_dev", "confirm", "bank_fit", None):
            again = sf.classify_row(freeze, {**row, "data_role": tag}, "debug",
                                    source_admission_sha256=sha)
            assert (again["clear"], again["reasons"]) == (base["clear"], base["reasons"])


def test_fp03_v1_holdout_row_refused_under_every_role(freeze):
    row = json.loads(FP03_V1_HOLDOUT.read_text())["admission"]["admitted"][0]
    assert row["data_role"] == "bank_fit"
    assert not sf.classify_row(freeze, row, "policy_dev", pre_admission=True)["clear"]
    assert not sf.classify_row(freeze, row, "debug",
                               source_admission_sha256=sf.file_sha256(FP04_ADMISSION))["clear"]
    for role, code in (("confirm", "CONFIRM_UNASSIGNED_NO_ACCESS"),
                       ("bank_fit", "ROLE_NOT_CONSUMABLE_BY_FP05")):
        with pytest.raises(sf.SplitFreezeViolation) as exc:
            sf.classify_row(freeze, row, role)
        assert exc.value.code == code


def test_2020H2_refused_for_every_role_even_tagged_confirm(freeze):
    for issue in ("2020-07-08T00Z", "2020-09-15T12Z", "2020-12-20T00Z"):
        row = make_row(issue, tag="confirm")
        assert not sf.classify_row(freeze, row, "policy_dev", pre_admission=True)["clear"]
        v = sf.classify_row(freeze, row, "debug",
                            source_admission_sha256=sf.file_sha256(FP04_ADMISSION))
        assert not v["clear"] and "OUTSIDE_ROLE_WINDOW" in v["reasons"]
        for role in ("confirm", "bank_fit"):
            with pytest.raises(sf.SplitFreezeViolation):
                sf.classify_row(freeze, row, role)


@pytest.mark.parametrize("store,year", [("2018.zarr", 2018), ("2021.zarr", 2021),
                                        ("2019_intermediate.zarr", 2019), ("2020_jan.zarr", 2020)])
def test_forbidden_stores_refused(freeze, store, year):
    row = make_row(f"{year}-10-01T00Z", year=year, store=store)
    v = sf.classify_row(freeze, row, "policy_dev", pre_admission=True)
    assert not v["clear"] and ("STORE_NOT_ALLOWED" in v["reasons"]
                               or "STORE_PATH_NOT_THE_FROZEN_ONE" in v["reasons"])


def test_policy_dev_consumption_needs_allow_list(freeze):
    row = make_row("2019-10-01T00Z")
    assert sf.classify_row(freeze, row, "policy_dev", pre_admission=True)["clear"]
    v = sf.classify_row(freeze, row, "policy_dev")
    assert not v["clear"] and "NO_ALLOW_LIST" in v["reasons"]
    allow = sf.AllowList("policy_dev", "a" * 64, frozenset({row["issue_id"]}))
    assert sf.classify_row(freeze, row, "policy_dev", allow_list=allow,
                           source_admission_sha256="a" * 64)["clear"]
    assert not sf.classify_row(freeze, row, "policy_dev", allow_list=allow,
                               source_admission_sha256="b" * 64)["clear"]
    other = sf.AllowList("policy_dev", "a" * 64, frozenset({"iss_other"}))
    assert not sf.classify_row(freeze, row, "policy_dev", allow_list=other,
                               source_admission_sha256="a" * 64)["clear"]
    with pytest.raises(sf.SplitFreezeViolation):
        sf.classify_row(freeze, fp04_rows()[0], "debug", pre_admission=True)


def test_index_time_inconsistency_refused(freeze):
    row = make_row("2019-10-01T00Z")
    row["issue_index"] += 1
    assert "INDEX_TIME_INCONSISTENT" in sf.classify_row(freeze, row, "policy_dev",
                                                        pre_admission=True)["reasons"]


def test_load_freeze_refuses_forged_hash_and_tampering(tmp_path):
    fr, path = write_freeze(tmp_path)
    with pytest.raises(sf.SplitFreezeViolation) as exc:
        sf.load_freeze(path, "0" * 64)
    assert exc.value.code == "FREEZE_SHA256_MISMATCH"
    with pytest.raises(sf.SplitFreezeViolation):
        sf.load_freeze(path, "")
    # one run deleted from the derived core -> digest inconsistent
    bad = freeze_payload()
    del bad["derived"]["exposed_runs_unbuffered_utc"][3]
    p2 = tmp_path / "bad.json"
    p2.write_text(json.dumps(bad))
    with pytest.raises(sf.SplitFreezeViolation) as exc:
        sf.load_freeze(p2, sf.file_sha256(p2))
    assert exc.value.code == "FREEZE_INCONSISTENT"
    # ... and even with a recomputed digest it disagrees with the ledger
    bad["derived_digest"] = sf.canonical_digest(bad["derived"])
    p2.write_text(json.dumps(bad))
    sf.load_freeze(p2, sf.file_sha256(p2))
    with pytest.raises(sf.SplitFreezeViolation) as exc:
        sf.load_freeze(p2, sf.file_sha256(p2), ledger=LEDGER)
    assert exc.value.code == "FREEZE_INCONSISTENT"
    # a confirm window can never be named
    conf = freeze_payload()
    conf["roles"]["confirm"]["window"] = ["2020-07-06T00Z", "2020-07-12T00Z"]
    p3 = tmp_path / "conf.json"
    p3.write_text(json.dumps(conf))
    with pytest.raises(sf.SplitFreezeViolation) as exc:
        sf.load_freeze(p3, sf.file_sha256(p3))
    assert exc.value.code == "FREEZE_CONFIRM_ASSIGNED"
    fr2 = sf.load_freeze(path, sf.file_sha256(path), ledger=LEDGER)
    assert fr2.payload["derived"] == sf.rebuild_from_ledger(LEDGER)


class _FakeArr:
    def __init__(self, values):
        self.values = values


class _FakeData:
    def __init__(self, n):
        self.n = n

    def isel(self, time):
        return _FakeArr(np.full((2, 2, 2), float(time), dtype=np.float32))


class _FakeStore:
    def __init__(self, year, n):
        t0 = np.datetime64(f"{year}-01-01T00", "h")
        self._vars = {"time": _FakeArr(t0 + np.arange(n) * np.timedelta64(6, "h")),
                      "data": _FakeData(n)}

    def __getitem__(self, k):
        return self._vars[k]

    def close(self):
        pass


@pytest.fixture()
def fake_store(monkeypatch):
    opened = []

    def opener(path):
        opened.append(path)
        return _FakeStore(2020 if "2020" in Path(path).name else 2019, 1464)

    monkeypatch.setattr(sf, "_open_store", opener)
    sf.reset_read_log()
    yield opened
    sf.reset_read_log()


def _debug_row_and_cert():
    rec = json.loads(FP04_ADMISSION.read_text())
    row = next(r for r in rec["admission"]["admitted"] if r["issue_id"] == DEBUG_IDS[0])
    cert = next(c for c in rec["content_certificates"] if c["issue_id"] == DEBUG_IDS[0])
    return row, cert, sf.file_sha256(FP04_ADMISSION)


def test_read_slab_inside_slice_logged_and_one_step_outside_refused(freeze, fake_store):
    row, cert, sha = _debug_row_and_cert()
    x = sf.read_slab(freeze, "debug", row, cert, row["issue_store"], row["issue_index"],
                     source_admission_sha256=sha)
    assert x.dtype == np.float32 and float(x.flat[0]) == float(row["issue_index"])
    log = sf.read_log()
    assert log == [{"store": "2020.zarr", "index": row["issue_index"], "role": "debug",
                    "issue_id": row["issue_id"]}]
    for idx in (row["history_index"] - 1, row["target_index"] + 1):
        with pytest.raises(sf.SplitFreezeViolation) as exc:
            sf.read_slab(freeze, "debug", row, cert, row["issue_store"], idx,
                         source_admission_sha256=sha)
        assert exc.value.code == "READ_OUTSIDE_CERTIFIED_SLICE"
    with pytest.raises(sf.SplitFreezeViolation) as exc:
        sf.read_slab(freeze, "debug", row, cert, str(DATA / "2020_jan.zarr"), row["issue_index"],
                     source_admission_sha256=sha)
    assert exc.value.code == "READ_STORE_NOT_ALLOWED"
    assert len(sf.read_log()) == 1
    check = sf.read_log_within(freeze, sf.read_log(), {
        row["issue_id"]: ("debug", "2020.zarr", row["history_index"], row["target_index"])})
    assert check["passed"]
    forged = [{**sf.read_log()[0], "index": row["target_index"] + 1}]
    assert not sf.read_log_within(freeze, forged, {
        row["issue_id"]: ("debug", "2020.zarr", row["history_index"], row["target_index"])})["passed"]


def test_read_slab_refuses_before_opening_anything(freeze, fake_store):
    rows = fp04_rows()
    rec = json.loads(FP04_ADMISSION.read_text())
    ninth = rows[8]
    cert = next(c for c in rec["content_certificates"] if c["issue_id"] == ninth["issue_id"])
    with pytest.raises(sf.SplitFreezeViolation):
        sf.read_slab(freeze, "debug", ninth, cert, ninth["issue_store"], ninth["issue_index"],
                     source_admission_sha256=sf.file_sha256(FP04_ADMISSION))
    with pytest.raises(sf.SplitFreezeViolation) as exc:
        sf.read_slab(freeze, "confirm", ninth, cert, ninth["issue_store"], ninth["issue_index"])
    assert exc.value.code == "CONFIRM_UNASSIGNED_NO_ACCESS"
    assert fake_store == [] and sf.read_log() == []


def test_excluded_index_712_refused_for_debug(freeze):
    row = make_row("2020-06-26T06Z", year=2020, tag="bank_fit", issue_id=DEBUG_IDS[0])
    v = sf.classify_row(freeze, row, "debug", source_admission_sha256=sf.file_sha256(FP04_ADMISSION))
    assert "CONTAINS_EXCLUDED_STORE_INDEX" in v["reasons"]


@pytest.mark.skipif(not PINNED.is_file(), reason="split_freeze_v1.json not built yet")
def test_pinned_freeze_equals_ledger_rebuild():
    sha = (RUN / "splits/split_freeze_v1.sha256").read_text().split()[0]
    fr = sf.load_freeze(PINNED, sha, ledger=LEDGER)
    assert fr.payload["roles"]["debug"]["allowed_issue_ids"] == DEBUG_IDS
    assert fr.payload["roles"]["policy_dev"]["n_slots"] == 698
    assert fr.payload["roles"]["confirm"] == {"status": "UNASSIGNED_NO_ACCESS", "window": None,
                                             "rule": fr.payload["roles"]["confirm"]["rule"]}
    assert set(fr.stores) == {"2019.zarr", "2020.zarr"}
