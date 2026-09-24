#!/usr/bin/env python3
"""FP-05b: build splits/split_freeze_v1.json (+ .sha256) from the FP-05a ledger.

Everything is RECOMPUTED here (via earthdelta.split_freeze.rebuild_from_ledger,
from the ledger's per-row supports and the S0 aggregate-result exposure), then
compared -- never copied -- against the ledger's own run summary and the FP-05a
reconstruction report. The debug allow-list is recomputed from FP-04's pinned
grouping (first and last issue, in time order, of each expert group).

Reads JSON files and zarr METADATA (.zarray / .zattrs) only; no data value of
any store is read. Write-once: refuses to overwrite an existing freeze.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from earthdelta import split_freeze as sf
from earthdelta.data.make_splits import compute_guard_boundaries

REPO = Path("/mnt/afs/260010168/EarthDelta")
RUN = REPO / "plans/plan_v4_0923/run_20260924T104725Z_fp05b"
FP05A = REPO / "plans/plan_v4_0923/run_20260924T093810Z_fp05a"
LEDGER = FP05A / "entry/exposure_ledger.json"
RECON = FP05A / "entry/exposure_reconstruction_report.json"
FP04 = REPO / "plans/plan_v4_0923/run_20260924T033627Z_fp04_bank"
FP04_ADMISSION = FP04 / "admission/bank_fit_admission.json"
FP04_GROUPING = FP04 / "admission/bank_fit_grouping.json"
DATA = REPO / "data/era5_1p40625"
OUT = RUN / "splits/split_freeze_v1.json"
EXPECTED = {"ledger": "d8ee631fca47c79859a01bc4ac4b74e6b962c0347c00c4cf30632b16cf3446bb",
            "fp04_admission": "01d344b8e2bb2cd66766fae290204322a6cf4939c353bc632ed5824ee7a7bdea",
            "fp04_grouping": "ede23fb9"}
DEBUG_EXCLUDED_INDEX = 712  # 2020-06-27T00Z specific_humidity_100 slab at 102 sigma (FP-04)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def store_meta(year: int) -> dict:
    root = DATA / f"{year}.zarr"
    time_attrs = json.loads((root / "time/.zattrs").read_text())
    data_arr = json.loads((root / "data/.zarray").read_text())
    time_arr = json.loads((root / "time/.zarray").read_text())
    units = time_attrs["units"]
    assert units == f"hours since {year}-01-01 00:00:00", units
    n = int(data_arr["shape"][0])
    assert int(time_arr["shape"][0]) == n
    t0, n_grid = sf._store_grid(year)
    assert n == n_grid, (year, n, n_grid)
    return {"path": str(root), "year": year, "time0_utc": sf.iso(t0), "n_timesteps": n,
            "data_shape": data_arr["shape"], "dtype": data_arr["dtype"],
            "time_units": units, "metadata_only": True,
            "metadata_sha256": {rel: sha(root / rel) for rel in
                                ("time/.zattrs", "time/.zarray", "data/.zarray", "data/.zattrs")}}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the freeze is write-once")
    assert sha(LEDGER) == EXPECTED["ledger"]
    assert sha(FP04_ADMISSION) == EXPECTED["fp04_admission"]
    assert sha(FP04_GROUPING).startswith(EXPECTED["fp04_grouping"])
    ledger = json.loads(LEDGER.read_text())
    recon = json.loads(RECON.read_text())

    derived = sf.rebuild_from_ledger(ledger, boundary_rule=sf.CONSERVATIVE_RULE, buffer_hours=24)
    pool = derived["policy_dev_pool"]

    # ---- cross-checks against FP-05a's own summaries (compared, not copied)
    theirs = [r[:2] for y in ("2019", "2020")
              for r in ledger["runs_by_year"][y]["gradient_or_result_runs_unbuffered"]]
    mine = [r[:2] for r in derived["exposed_runs_unbuffered_utc"]]
    fp05a_pool = recon["2019H2_policy_dev_pool"]["with_S0_aggregate_result_and_full_cross_year_buffer"]
    alt = derived["alternative_rules_for_the_record"]
    cross = {
        "runs_equal_ledger_runs_by_year_unbuffered": mine == theirs,
        "pool_equals_fp05a_report": (pool["n_slots"], pool["first_issue_utc"], pool["last_issue_utc"])
        == (fp05a_pool["count"], fp05a_pool["first"], fp05a_pool["last"]),
        "certified_rows_only_equals_fp05a_report":
            alt["full_24h_buffer_certified_rows_only"]["n_slots"]
            == recon["2019H2_policy_dev_pool"]["without_S0_aggregate_result_(only_certified_row_exposure)_full_buffer"]["count"],
        "no_cross_year_equals_fp05a_report":
            alt["no_cross_year_buffer"]["n_slots"]
            == recon["2019H2_policy_dev_pool"]["ignoring_cross_year_buffer_(2019_exposure_only)"]["count"],
    }
    assert all(cross.values()), cross

    # ---- guard windows from make_splits (pure calendar)
    bounds = compute_guard_boundaries(max_history_hours=24, max_lead_hours=72)
    val = [b.strftime("%Y-%m-%dT%HZ") for b in bounds["val"]]
    test = [b.strftime("%Y-%m-%dT%HZ") for b in bounds["test"]]
    step = 6 * 3600
    t0_2019 = sf.parse_utc("2019-01-01T00")
    assert all(sf.parse_utc(val[0]) <= t0_2019 + i * step < sf.parse_utc(val[1])
               for i in range(pool["first_index"], pool["last_index"] + 1))

    # ---- debug allow-list: first/last in time order of each FP-04 group
    admission = json.loads(FP04_ADMISSION.read_text())
    rows = {r["issue_id"]: r for r in admission["admission"]["admitted"]}
    grouping = json.loads(FP04_GROUPING.read_text())
    debug = []
    for g in grouping["groups"]:
        ordered = sorted(g["issue_ids"], key=lambda i: rows[i]["issue_time"])
        for pos, iid in (("first", ordered[0]), ("last", ordered[-1])):
            r = rows[iid]
            debug.append({"expert_group": int(g["expert_index"]), "position": pos, "issue_id": iid,
                          "issue_utc": sf.iso(r["issue_time"]), "issue_index": r["issue_index"],
                          "certificate_slice": [r["history_index"], r["target_index"]]})
    assert len({d["issue_id"] for d in debug}) == 8
    assert all(not (d["certificate_slice"][0] <= DEBUG_EXCLUDED_INDEX <= d["certificate_slice"][1])
               for d in debug)

    pool_support = [sf.iso(sf.parse_utc(pool["first_issue_utc"]) - 12 * 3600),
                    sf.iso(sf.parse_utc(pool["last_issue_utc"]) + 72 * 3600)]
    shared = recon["2019H2_policy_dev_pool"]["sharedF0_sprint_disclosure"]
    payload = {
        "schema": sf.SCHEMA,
        "freeze_id": "FP05B-SPLIT-FREEZE-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve()),
                    "module": {"path": "earthdelta/split_freeze.py",
                               "sha256": sha(REPO / "earthdelta/split_freeze.py")}},
        "principle": ("A row is consumable iff this frozen split (derived from the pinned FP-05a "
                      "ledger) AND an explicit allow-list clear it. The row's data_role tag is "
                      "recorded but never authorises anything."),
        "exposure_ledger": {"path": str(LEDGER), "sha256": sha(LEDGER)},
        "exposure_reconstruction_report": {"path": str(RECON), "sha256": sha(RECON)},
        "support_convention": {"history_hours": 12, "lead_hours": 72, "interval_hours": 6,
                               "interval": "closed", "touching": "sharing one instant = touching"},
        "buffer_hours": 24,
        "derived": derived,
        "derived_digest": sf.canonical_digest(derived),
        "crosschecks_vs_fp05a": cross,
        "stores_allowed": {"2019.zarr": store_meta(2019), "2020.zarr": store_meta(2020)},
        "stores_refused": "every other path (2015/2016/2017/2018/2021/2022, *_intermediate, "
                          "2020_jan, ...) for every role",
        "roles": {
            "policy_dev": {
                "store": "2019.zarr",
                "boundary_rule": sf.CONSERVATIVE_RULE,
                "issue_window_utc": [pool["first_issue_utc"], pool["last_issue_utc"]],
                "support_window_utc": pool_support,
                "store_index_window": [pool["first_index"], pool["last_index"]],
                "n_slots": pool["n_slots"],
                "issue_guard_window_utc": val,
                "issue_guard_window_source": "make_splits.compute_guard_boundaries(24, 72)['val'] "
                                             "[start, end)",
                "must_not_touch": "any exposed run widened by buffer_hours (closed intervals)",
                "allow_list": ("NONE AT FREEZE TIME. Bound later by the FP-05b cache protocol "
                               "(split.allow_lists.policy_dev = the policy_dev admission sha256 + "
                               "its issue ids). Before that, only the manifest builder and "
                               "certify-admission may classify rows (pre_admission=True)."),
                "label": "RETROSPECTIVE_HELD_PERIOD_DEV_NOT_HISTORICAL_DEPLOYMENT",
                "disclosure_sharedF0_integrity_reads": shared,
            },
            "debug": {
                "store": "2020.zarr",
                "selection_rule": "DEBUG-SELECT-v1: first and last issue, in time order, of each "
                                  "FP-04 expert group G_k (bank_fit_grouping.json)",
                "source_admission": {"path": str(FP04_ADMISSION), "sha256": sha(FP04_ADMISSION),
                                     "tag_on_every_row": "bank_fit (tag != clearance)"},
                "grouping": {"path": str(FP04_GROUPING), "sha256": sha(FP04_GROUPING)},
                "allowed_issue_ids": [d["issue_id"] for d in debug],
                "issues": debug,
                "support_window_utc": ["2020-01-01T12Z", "2020-07-04T18Z"],
                "issue_guard_window_utc": test,
                "excluded_store_indices": [DEBUG_EXCLUDED_INDEX],
                "exposure_note": "already GRADIENT_AND_RESULT exposed (FP-04 training rows); "
                                 "used only to validate the cache machinery; reveals nothing new",
            },
            "confirm": {"status": sf.CONFIRM_STATUS, "window": None,
                        "rule": "every FP-05 access raises CONFIRM_UNASSIGNED_NO_ACCESS; naming a "
                                "confirm window is outside FP-05b"},
            "bank_fit": {"consumable_by_fp05": False},
        },
        "labels": {"2020H2": "EXPOSED_NOT_CONFIRM",
                   "policy_dev": "RETROSPECTIVE_HELD_PERIOD_DEV_NOT_HISTORICAL_DEPLOYMENT",
                   "confirm": sf.CONFIRM_STATUS},
        "integrity_read_only_disclosed_not_enforced": {
            "sharedF0_sprint_2019H2": shared,
            "fp03_finiteness_scans": [s["path"] for s in ledger["finiteness_scans"]],
            "fp05b_2019H2_scan": "appended afterwards in splits/exposure_ledger_addendum_fp05b.json",
        },
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    digest = sha(OUT)
    (OUT.parent / "split_freeze_v1.sha256").write_text(f"{digest}  split_freeze_v1.json\n")
    loaded = sf.load_freeze(OUT, digest, ledger=LEDGER)
    print(json.dumps({"freeze": str(OUT), "sha256": digest, "pool": pool,
                      "debug_ids": [d["issue_id"] for d in debug], "crosschecks": cross,
                      "reloaded_with_ledger_consistency": loaded.sha256 == digest}, indent=1))


if __name__ == "__main__":
    main()
