#!/usr/bin/env python3
"""FP-05a: independent exposure-ledger reconstruction from raw admission certificates.

READ-ONLY, CPU-ONLY, STDLIB-ONLY. This script never imports numpy/zarr/torch, never
opens a .zarr store and never runs a model. It reads only JSON certificates that
are already on disk plus the 128-byte header of the S0 .npy input (parsed by hand,
no data bytes are read).

Every timestamp is recomputed from the raw certificate fields (epoch issue_time /
history_time / valid_time, store indices, content-certificate times_utc). Nothing
is copied from FP05A_PLAN.md or from either review package.

Outputs (refuses to overwrite):
  entry/exposure_ledger.json
  entry/exposure_reconstruction_report.json
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
P4 = "plans/plan_v4_0923"
H6 = dt.timedelta(hours=6)
UTC = dt.timezone.utc
SUPPORT_BEFORE_H = 12   # history t-12h (t-6h for history_steps=1 rows is covered)
SUPPORT_AFTER_H = 72    # longest lead
BUFFER_H = 24


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def iso(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ").replace(":00Z", "Z")


def ep(x: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(int(x), UTC)


def parse_naive(s: str) -> dt.datetime:
    s = s.replace("Z", "")
    s = s.split(".")[0]
    return dt.datetime.fromisoformat(s).replace(tzinfo=UTC)


def store_year(store: str) -> int:
    return int(re.search(r"(\d{4})\.zarr", store).group(1))


def idx_time(store: str, idx: int) -> dt.datetime:
    return dt.datetime(store_year(store), 1, 1, tzinfo=UTC) + idx * H6


def grid(a: dt.datetime, b: dt.datetime) -> list[dt.datetime]:
    out, t = [], a
    while t <= b:
        out.append(t)
        t += H6
    return out


def runs(ts: set[dt.datetime]) -> list[tuple[dt.datetime, dt.datetime]]:
    """Maximal runs of consecutive 6h grid timesteps."""
    s = sorted(ts)
    out = []
    for t in s:
        if out and t - out[-1][1] == H6:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out]


def load(rel: str):
    return json.loads((REPO / rel).read_text())


# --------------------------------------------------------------------------------------
# 1. admission certificates (r2_admission_gate format)
# --------------------------------------------------------------------------------------
ADMISSIONS = [
    # id, path, exposure_kind, consumers, note
    ("fp03_v1_train8", f"{P4}/run_20260923T192759Z_fp03_fs/admission/bank_fit_admission.json",
     "GRADIENT_AND_RESULT",
     ["FP03-v1 J1 pt-56auspj8", "J2 pt-z07uway9", "J3 pt-sz6iv529", "J3b pt-n6b5uf0v",
      "X1_N32 pt-jbhf3z73 (first 8)", "X1_N64 pt-t3bfl3e9 (first 8)",
      "FP03-v2 V2-J3 pt-fuiqqeau", "V2-J4 pt-gr3rrmdg"],
     "Fs training panel (certified Fs was fit on these 8)"),
    ("fp03_v1_holdout8", f"{P4}/run_20260923T192759Z_fp03_fs/admission/holdout_panel_admission.json",
     "RESULT", ["J1", "J2", "J3", "J3b", "X1_N32", "X1_N64"],
     "v1 holdout panel; model outcomes observed (Q6 of v1, X1 primary comparison)"),
    ("x1_n32", f"{P4}/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/admission/x1_n32_admission.json",
     "GRADIENT_AND_RESULT", ["X1_N32 pt-jbhf3z73"],
     "EXPLORATORY: adapter trained on all 32 rows; train-ratio outcomes observed"),
    ("x1_n64", f"{P4}/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/admission/x1_n64_admission.json",
     "GRADIENT_AND_RESULT", ["X1_N64 pt-t3bfl3e9"],
     "EXPLORATORY: adapter trained on all 64 rows; train-ratio outcomes observed"),
    ("fp03_v2_holdout_attempt1", f"{P4}/run_20260924T013959Z_fp03_v2/holdout/v2_holdout_admission.json",
     "INTEGRITY_READ_ONLY", [],
     "failed content verification (outer passed=false); never consumed by any model job"),
    ("fp03_v2_holdout_A1", f"{P4}/run_20260924T013959Z_fp03_v2/holdout/v2_holdout_admission_A1.json",
     "RESULT", ["V2-J3 pt-fuiqqeau", "V2-J4 pt-gr3rrmdg"],
     "fresh 2019 holdout; Q6 outcomes observed (Fs certification)"),
    ("fp04_bank_fit48", f"{P4}/run_20260924T033627Z_fp04_bank/admission/bank_fit_admission.json",
     "GRADIENT_AND_RESULT",
     ["B-J1 pt-ycpvyb6n", "B-J2 pt-vt003lhm", "B-J3 pt-m52qwunf (failed, assembly only)",
      "B-J3r pt-831c01g4"],
     "bank experts trained + own-group qualification observed"),
    ("fp04_bank_fit_attempt1", f"{P4}/run_20260924T033627Z_fp04_bank/admission/bank_fit_admission_attempt1.json",
     "INTEGRITY_READ_ONLY", [], "49-slot attempt, failed on slot 47; not consumed by a model job"),
    ("fp04_slot47_replacements", f"{P4}/run_20260924T033627Z_fp04_bank/admission/replacement_candidates_slot47_admission.json",
     "INTEGRITY_READ_ONLY", [], "4 same-month replacement candidates, all failed; not consumed"),
    ("hist_r3_r4_8issue", "artifacts/round2_next/20260921T170142Z_r3p001_10h/j3_fs_bank/admission_record.json",
     "GRADIENT_AND_RESULT",
     ["r3/r4 historical Fs/bank jobs pt-w79f1y3t pt-cdj1s2le pt-3x63g0c6 pt-5ma8r7m2 pt-2v1z1p4i "
      "pt-sgc0joda pt-2r6tbwu7 pt-e8y7ib01 (+ r3 profile jobs pt-y52q562s pt-idopy57q pt-rvonop4m)"],
     "historical 8-issue admission (lead-6h rows; training rolled to 24h per ledger config)"),
]


def admission_rows(src_id, rel, kind):
    d = load(rel)
    a = d["admission"]
    certs = d.get("content_certificates", [])
    rows, checks = [], []
    for r in a["admitted"] + a.get("rejected", []):
        issue = ep(r["issue_time"])
        hist = ep(r["history_time"])
        valid = ep(r["valid_time"])
        # index -> time cross-check (store axis assumption checked against epoch fields)
        ok_idx = (idx_time(r["issue_store"], r["issue_index"]) == issue and
                  idx_time(r["history_store"], r["history_index"]) == hist and
                  idx_time(r["target_store"], r["target_index"]) == valid)
        # matching content certificate: same store, index range covers history..target
        # (exact window first: neighbouring rows' windows can overlap, so containment
        #  alone could pick a neighbour's certificate)
        cert = None
        for c in certs:
            if c["store_path"] == r["issue_store"] and c["indices"] and \
                    c["indices"][0] == r["history_index"] and c["indices"][-1] == r["target_index"]:
                cert = c
                break
        if cert is None:
            for c in certs:
                if c["store_path"] == r["issue_store"] and c["indices"] and \
                        c["indices"][0] == r["history_index"]:
                    cert = c
                    break
        cert_first = cert_last = None
        cert_idx_time_ok = None
        if cert:
            ct = [parse_naive(x) for x in cert["times_utc"]]
            cert_first, cert_last = ct[0], ct[-1]
            cert_idx_time_ok = all(idx_time(cert["store_path"], i) == t
                                   for i, t in zip(cert["indices"], ct))
        # support actually recorded by the row: history..valid; the certificate may span more
        rec_start = min(hist, cert_first) if cert_first else hist
        rec_end = max(valid, cert_last) if cert_last else valid
        # conservative uniform support t-12h..t+72h (all later consumers roll to 72h)
        u_start = issue - dt.timedelta(hours=SUPPORT_BEFORE_H)
        u_end = issue + dt.timedelta(hours=SUPPORT_AFTER_H)
        rows.append({
            "source": src_id, "issue_id": r["issue_id"], "event_id": r.get("event_id"),
            "store": r["issue_store"].split("/")[-1], "issue_index": r["issue_index"],
            "issue_utc": iso(issue), "history_utc": iso(hist), "valid_utc": iso(valid),
            "lead_hours": r.get("lead_hours"), "admitted": r.get("admitted"),
            "cert_matched_exact_window": cert is not None,
            "cert_passed": cert["passed"] if cert else None,
            "cert_times_first": iso(cert_first) if cert_first else None,
            "cert_times_last": iso(cert_last) if cert_last else None,
            "index_epoch_consistent": ok_idx, "cert_index_time_consistent": cert_idx_time_ok,
            "recorded_support": [iso(rec_start), iso(rec_end)],
            "uniform_support_t-12h_t+72h": [iso(u_start), iso(u_end)],
            "exposure_kind": kind,
        })
        checks.append(ok_idx and (cert_idx_time_ok is not False))
    meta = {"path": rel, "sha256": sha256(REPO / rel), "data_role": d["data_role"],
            "outer_passed": d["passed"], "requested": a.get("requested"),
            "n_admitted": a.get("n_admitted"), "n_rejected": a.get("n_rejected"),
            "n_rows": len(rows), "n_content_certificates": len(certs),
            "n_content_certificates_passed": sum(bool(c.get("passed")) for c in certs),
            "n_rows_with_matched_certificate": sum(r["cert_matched_exact_window"] for r in rows),
            "all_index_time_checks_pass": all(checks)}
    return rows, meta


# --------------------------------------------------------------------------------------
# 2. shared-F0 sprint admission (different schema)
# --------------------------------------------------------------------------------------
SPRINT = "artifacts/ed-sprint8h-20260922T035313Z-rerun3/admission.json"


def sprint_rows():
    d = load(SPRINT)
    rows = []
    bad_axis = 0
    for r in d["rows"]:
        issue = parse_naive(r["issue_utc"])
        s0, s1 = parse_naive(r["support_start_utc"]), parse_naive(r["support_end_utc"])
        ok = idx_time(r["store"], r["index"]) == issue and \
            idx_time(r["store"], r["history_index"]) == s0 and \
            idx_time(r["store"], max(r["target_indices"])) == s1
        bad_axis += (not ok)
        rows.append({"source": "sharedF0_sprint_admitted", "issue_id": r["issue_id"],
                     "role": r["role"], "year": r["year"], "issue_utc": iso(issue),
                     "recorded_support": [iso(s0), iso(s1)],
                     "content_timesteps_read_utc": sorted({iso(idx_time(r["store"], int(k)))
                                                           for k in r["content_sha256"]}),
                     "content_verified": r.get("content_verified"),
                     "exposure_kind": "INTEGRITY_READ_ONLY", "index_time_consistent": ok})
    exc = []
    for r in d["excluded"]:
        m = re.search(r"_(\d{8})_(\d{2})$", r["issue_id"])
        t = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=UTC)
        exc.append({"source": "sharedF0_sprint_excluded", "issue_id": r["issue_id"],
                    "issue_utc": iso(t), "reason": r["reason"],
                    "exposure_kind": "INTEGRITY_READ_ONLY_ATTEMPTED",
                    "time_source": "parsed from issue_id (excluded rows carry no index fields)"})
    rows_sha = hashlib.sha256(json.dumps(d["rows"], sort_keys=True, separators=(",", ":"))
                              .encode()).hexdigest()
    meta = {"path": SPRINT, "sha256": sha256(REPO / SPRINT), "status": d["status"],
            "n_rows_listed": len(d["rows"]), "n_excluded_listed": len(d["excluded"]),
            "coverage_recorded": d["coverage"], "rows_sha256_recorded": d["rows_sha256"],
            "rows_sha256_recomputed_sorted_compact": rows_sha,
            "rows_sha256_recompute_note": "recomputed with sort_keys/compact separators; a mismatch only means the recorder used a different serialisation, not tampering",
            "n_rows_index_time_inconsistent": bad_axis,
            "content_checked_recorded": d.get("content_checked"),
            "completed_utc": d.get("completed_utc")}
    dec = load("artifacts/ed-sprint8h-20260922T035313Z-rerun3/decision.json")
    meta["sprint_decision_status"] = dec.get("status")
    meta["sprint_downstream"] = dec.get("downstream")
    return rows, exc, meta


# --------------------------------------------------------------------------------------
# 3. finiteness scans and S0 input
# --------------------------------------------------------------------------------------
def finiteness():
    out = []
    for y in (2015, 2018, 2019):
        rel = f"{P4}/run_20260924T013959Z_fp03_v2/holdout/finiteness_scan_{y}.json"
        d = load(rel)
        a, b = d["steps_scanned"]
        t0 = dt.datetime(y, 1, 1, tzinfo=UTC)
        out.append({"source": f"finiteness_scan_{y}", "path": rel, "sha256": sha256(REPO / rel),
                    "steps_scanned": [a, b], "utc_range": [iso(t0 + a * H6), iso(t0 + b * H6)],
                    "n_steps_with_nonfinite": d["n_steps_with_nonfinite"],
                    "exposure_kind": "INTEGRITY_READ_ONLY (finiteness only; no values/stats recorded)"})
    return out


def npy_header(p: Path):
    with p.open("rb") as f:
        magic = f.read(8)
        assert magic[:6] == b"\x93NUMPY", magic
        major = magic[6]
        hlen = int.from_bytes(f.read(2 if major == 1 else 4), "little")
        hdr = f.read(hlen).decode("latin1")
    return ast.literal_eval(hdr)


def s0_source():
    p = REPO / "scripts/s0_gate_inputs/jan2020_full.npy"
    h = npy_header(p)
    T = h["shape"][0]
    t0 = dt.datetime(2020, 1, 1, tzinfo=UTC)
    return {
        "source": "S0_gate_input_jan2020_full.npy",
        "path": "scripts/s0_gate_inputs/jan2020_full.npy",
        "npy_header": {"shape": list(h["shape"]), "descr": h["descr"], "fortran_order": h["fortran_order"]},
        "size_bytes": p.stat().st_size,
        "sha256": "NOT_HASHED_1.1GB_FILE_NOT_REQUIRED_HEADER_ONLY_READ",
        "time_axis": "ASSUMED index 0 = 2020-01-01T00Z, 6h step (filename 'jan2020_full' and T=124 = 31 days x 4); not verifiable without reading data bytes, which FP-05a does not do",
        "file_load_range_utc": [iso(t0), iso(t0 + (T - 1) * H6)],
        "whole_file_np_load": "scripts/s0_gate.py:225 np.load(input_dir/input_file) loads all T steps",
        "model_forward_input_index": 0,
        "result_exposure": {
            "what": "compute_rmse_sanity (scripts/s0_gate.py ~L1889-1918): z500 RMSE of model vs truth for i=0..4 at 1 and 4 steps -> truth indices 0..8; aggregate only, 'NOT used for gate pass/fail'",
            "utc_range": [iso(t0), iso(t0 + 8 * H6)],
        },
        "consumers": "every S0 gate/diagnose job (r3-j1-s0gate, ed8-s0-*, s0-rerun-*, r4gate0923a-e, r4trace0923*, s0-diagnose/postdiag)",
        "exposure_kind": "INPUT_READ (whole Jan-2020) + AGGREGATE_RESULT (2020-01-01T00Z..01-03T00Z)",
    }


# --------------------------------------------------------------------------------------
# 4. historical Fs ledger cross-check
# --------------------------------------------------------------------------------------
def hist_ledger_check(hist_rows):
    d = load(f"{P4}/run_20260923T192759Z_fp03_fs/ledger/fs_attempts_ledger.json")
    ids = {r["issue_id"] for r in hist_rows}
    per = []
    for e in d["entries"]:
        ex = e.get("exposure", {})
        per.append({"job_id": e["job_id"],
                    "admission_record": ex.get("admission_record"),
                    "issue_ids_equal_hist_admission": set(ex.get("training_issue_ids_unique", [])) == ids,
                    "train_hours": e["config"].get("train_hours")})
    return {"n_entries": len(per), "all_use_same_8_issue_ids": all(p["issue_ids_equal_hist_admission"] for p in per),
            "job_ids": sorted({p["job_id"] for p in per}),
            "max_train_hours": max(p["train_hours"] for p in per)}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> int:
    ledger_out = RUN / "entry/exposure_ledger.json"
    report_out = RUN / "entry/exposure_reconstruction_report.json"
    for o in (ledger_out, report_out):
        if o.exists():
            print("refusing to overwrite", o)
            return 2

    all_rows, metas = [], {}
    by_src = {}
    for sid, rel, kind, consumers, note in ADMISSIONS:
        rows, meta = admission_rows(sid, rel, kind)
        meta.update(consumers=consumers, note=note, exposure_kind=kind)
        metas[sid] = meta
        by_src[sid] = rows
        all_rows += rows

    # --- nesting / derived sets -----------------------------------------------------
    ids = {k: [r["issue_id"] for r in v] for k, v in by_src.items()}
    t8 = set(ids["fp03_v1_train8"])
    n32, n64 = ids["x1_n32"], ids["x1_n64"]
    x1_extra = [r for r in by_src["x1_n64"] if r["issue_id"] not in t8]
    x1_extra_h2 = [r for r in x1_extra if parse_naive(r["issue_utc"]) >= dt.datetime(2020, 7, 1, tzinfo=UTC)]
    nesting = {
        "x1_n32_first8_equal_train8_in_order": n32[:8] == ids["fp03_v1_train8"],
        "x1_n64_first8_equal_train8_in_order": n64[:8] == ids["fp03_v1_train8"],
        "x1_n32_subset_of_n64": set(n32) <= set(n64),
        "x1_n64_extra_count": len(x1_extra),
        "x1_n64_extra_in_2020H2_count": len(x1_extra_h2),
        "x1_n32_extra_count": len(set(n32) - t8),
        "hist_8_equal_train8_issue_ids": set(ids["hist_r3_r4_8issue"]) == t8,
        "fp04_48_subset_of_attempt1_49": set(ids["fp04_bank_fit48"]) <= set(ids["fp04_bank_fit_attempt1"]),
        "fp04_attempt1_minus_48": sorted(set(ids["fp04_bank_fit_attempt1"]) - set(ids["fp04_bank_fit48"])),
        "v2_attempt1_overlap_A1_count": len(set(ids["fp03_v2_holdout_attempt1"]) & set(ids["fp03_v2_holdout_A1"])),
    }
    x1_idx = sorted(r["issue_index"] for r in x1_extra)
    gaps = [(b - a) * 6 for a, b in zip(x1_idx, x1_idx[1:])]
    x1_detail = {
        "extra_issue_indices_2020_store": x1_idx,
        "first_extra_issue_utc": x1_extra[0]["issue_utc"] if x1_extra else None,
        "last_extra_issue_utc": x1_extra[-1]["issue_utc"] if x1_extra else None,
        "min_spacing_hours": min(gaps) if gaps else None,
        "max_spacing_hours": max(gaps) if gaps else None,
        "support_length_hours": SUPPORT_BEFORE_H + SUPPORT_AFTER_H,
        "support_union_contiguous": all(g <= SUPPORT_BEFORE_H + SUPPORT_AFTER_H + 6 for g in gaps),
        "all_extra_cert_passed": all(r["cert_passed"] for r in x1_extra),
        "all_extra_recorded_support_equals_uniform": all(r["recorded_support"] == r["uniform_support_t-12h_t+72h"] for r in x1_extra),
    }

    sp_rows, sp_exc, sp_meta = sprint_rows()
    fin = finiteness()
    s0 = s0_source()
    hist_chk = hist_ledger_check(by_src["hist_r3_r4_8issue"])

    # --- timestep sets --------------------------------------------------------------
    def support_set(rows, key="uniform_support_t-12h_t+72h"):
        s = set()
        for r in rows:
            a, b = (parse_naive(x) for x in r[key])
            s.update(grid(a, b))
        return s

    grad_result_rows = [r for r in all_rows if r["exposure_kind"] in ("GRADIENT_AND_RESULT", "RESULT")]
    integ_rows = [r for r in all_rows if r["exposure_kind"] == "INTEGRITY_READ_ONLY"]
    GR = support_set(grad_result_rows)
    GR_recorded = support_set(grad_result_rows, "recorded_support")
    s0_res = set(grid(*(parse_naive(x) for x in s0["result_exposure"]["utc_range"])))
    GR_all = GR | s0_res
    INTEG = support_set(integ_rows, "recorded_support") | support_set(sp_rows, "recorded_support")
    for f in fin:
        INTEG |= set(grid(*(parse_naive(x) for x in f["utc_range"])))
    INTEG |= set(grid(*(parse_naive(x) for x in s0["file_load_range_utc"])))

    def buffered(ts):
        b = set()
        for a, z in runs(ts):
            b.update(grid(a - dt.timedelta(hours=BUFFER_H), z + dt.timedelta(hours=BUFFER_H)))
        return b

    GRb = buffered(GR_all)

    def fmt_runs(ts, lo=None, hi=None):
        out = []
        for a, b in runs(ts):
            if lo and b < lo:
                continue
            if hi and a > hi:
                continue
            out.append([iso(a), iso(b), int((b - a).total_seconds() // 3600)])
        return out

    # --- 2020H2 residual --------------------------------------------------------------
    H2a, H2b = dt.datetime(2020, 7, 1, tzinfo=UTC), dt.datetime(2020, 12, 31, 18, tzinfo=UTC)
    h2_grid = set(grid(H2a, H2b))
    h2_free_buffered = h2_grid - GRb
    h2_free_unbuffered = h2_grid - GR_all

    def candidates(free: set, within: set):
        """issue times t on the grid whose full t-12h..t+72h support is inside `free`
        (support may not leave the 2020 store: `within`)."""
        c = []
        for t in sorted(within):
            sup = grid(t - dt.timedelta(hours=SUPPORT_BEFORE_H), t + dt.timedelta(hours=SUPPORT_AFTER_H))
            if all(x in free for x in sup):
                c.append(t)
        return c

    def max_disjoint(c, strict_gap_h):
        """greedy max count with issue spacing >= strict_gap_h."""
        chosen = []
        for t in c:
            if not chosen or (t - chosen[-1]).total_seconds() / 3600 >= strict_gap_h:
                chosen.append(t)
        return chosen

    L = SUPPORT_BEFORE_H + SUPPORT_AFTER_H
    cand_b = candidates(h2_free_buffered, h2_grid)
    cand_u = candidates(h2_free_unbuffered, h2_grid)
    h2 = {
        "window": [iso(H2a), iso(H2b)],
        "gradient_or_result_runs_touching_H2_unbuffered": fmt_runs(GR_all, H2a, H2b),
        "free_runs_in_H2_after_24h_buffer": fmt_runs(h2_free_buffered),
        "free_runs_in_H2_no_buffer": fmt_runs(h2_free_unbuffered),
        "candidate_issue_times_full_support_in_buffered_gap": {
            "count": len(cand_b), "first": iso(cand_b[0]) if cand_b else None,
            "last": iso(cand_b[-1]) if cand_b else None},
        "max_candidates_support_disjoint_buffered_(spacing>=%dh)" % (L + 6): len(max_disjoint(cand_b, L + 6)),
        "max_candidates_support_touching_allowed_buffered_(spacing>=%dh)" % L: len(max_disjoint(cand_b, L)),
        "max_candidates_support_disjoint_NO_buffer": len(max_disjoint(cand_u, L + 6)),
        "max_candidates_support_disjoint_with_24h_mutual_buffer_(spacing>=%dh)" % (L + 6 + BUFFER_H): len(max_disjoint(cand_b, L + 6 + BUFFER_H)),
        "candidate_issue_times_full_support_no_buffer": {"count": len(cand_u)},
        "verdict": "EXPOSED_NOT_CONFIRM",
    }

    # --- 2019H2 policy_dev candidate pool --------------------------------------------
    Y19a, Y19b = dt.datetime(2019, 7, 1, tzinfo=UTC), dt.datetime(2019, 12, 31, 18, tzinfo=UTC)
    y19_grid = set(grid(Y19a, Y19b))
    pool_free_store_bounded = (y19_grid - GRb)
    cand_19 = candidates(pool_free_store_bounded, y19_grid)
    # variant: support may not cross into 2020 at all (store-bounded) AND 24h buffer to the
    # 2020-01-01 exposure is honoured (GRb already contains 2019-12-31 buffer points if any)
    buf_into_2019 = sorted(t for t in GRb if dt.datetime(2019, 12, 1, tzinfo=UTC) <= t < dt.datetime(2020, 1, 1, tzinfo=UTC))
    # variant without the S0 aggregate-result exposure
    GRb_noS0 = buffered(GR)
    cand_19_noS0 = candidates(y19_grid - GRb_noS0, y19_grid)
    # variant: ignore any cross-year buffer (store boundary only)
    GRb_2019only = buffered({t for t in GR_all if t.year == 2019})
    cand_19_storeonly = candidates(y19_grid - GRb_2019only, y19_grid)

    sp19 = [r for r in sp_rows if Y19a <= parse_naive(r["issue_utc"]) <= Y19b]
    sp19_support = [r for r in sp_rows if any(Y19a <= parse_naive(t) <= Y19b for t in r["content_timesteps_read_utc"])]
    sp19_exc = [r for r in sp_exc if Y19a <= parse_naive(r["issue_utc"]) <= Y19b]
    fin19 = [f for f in fin if f["source"] == "finiteness_scan_2019"][0]
    pool = {
        "definition": "issue t on 6h grid, full t-12h..t+72h support inside 2019.zarr (2019-01-01T00Z..2019-12-31T18Z) and outside every gradient/result exposure run expanded by 24h",
        "with_S0_aggregate_result_and_full_cross_year_buffer": {
            "count": len(cand_19), "first": iso(cand_19[0]) if cand_19 else None,
            "last": iso(cand_19[-1]) if cand_19 else None},
        "2019_timesteps_inside_buffer_of_2020_exposure": [iso(t) for t in buf_into_2019],
        "without_S0_aggregate_result_(only_certified_row_exposure)_full_buffer": {
            "count": len(cand_19_noS0), "first": iso(cand_19_noS0[0]) if cand_19_noS0 else None,
            "last": iso(cand_19_noS0[-1]) if cand_19_noS0 else None},
        "ignoring_cross_year_buffer_(2019_exposure_only)": {
            "count": len(cand_19_storeonly), "first": iso(cand_19_storeonly[0]) if cand_19_storeonly else None,
            "last": iso(cand_19_storeonly[-1]) if cand_19_storeonly else None},
        "integrity_scanned": False,
        "finiteness_scan_2019_last_step_utc": fin19["utc_range"][1],
        "status": "CANDIDATE_POOL_INTEGRITY_NOT_SCANNED (scan belongs to FP-05b)",
        "sharedF0_sprint_disclosure": {
            "admitted_rows_with_issue_in_2019H2": len(sp19),
            "admitted_rows_with_any_read_timestep_in_2019H2": len(sp19_support),
            "roles_of_2019H2_admitted_rows": {k: sum(1 for r in sp19 if r["role"] == k) for k in sorted({r["role"] for r in sp19})},
            "excluded_rows_with_issue_in_2019H2_(read_attempted,_failed_nonfinite)": len(sp19_exc),
            "exposure_kind": "INTEGRITY_READ_ONLY (no gradient, no model result: sprint stopped at S0, downstream NOT_STARTED)",
            "treatment": "DISCLOSED_NOT_EXCLUDED (FP05A accepted decision 3); retrospective-transfer evaluation label",
        },
    }

    # --- overall runs by year -----------------------------------------------------------
    by_year = {}
    for y in (2015, 2018, 2019, 2020):
        lo, hi = dt.datetime(y, 1, 1, tzinfo=UTC), dt.datetime(y, 12, 31, 18, tzinfo=UTC)
        by_year[str(y)] = {
            "gradient_or_result_runs_unbuffered": fmt_runs(GR_all, lo, hi),
            "gradient_or_result_runs_24h_buffered": fmt_runs(GRb, lo, hi),
            "certified_rows_only_runs_unbuffered": fmt_runs(GR, lo, hi),
            "integrity_read_runs": fmt_runs(INTEG, lo, hi),
            "integrity_read_timesteps": len([t for t in INTEG if t.year == y]),
        }

    # sanity: recorded vs uniform support for grad/result rows
    rec_vs_uniform = {sid: sum(r["recorded_support"] != r["uniform_support_t-12h_t+72h"] for r in by_src[sid])
                      for sid in by_src}

    ledger = {
        "schema": "ed-fp05a-exposure-ledger/1",
        "created_utc": dt.datetime.now(UTC).isoformat(),
        "builder": str(Path(__file__).relative_to(REPO)),
        "builder_sha256": sha256(Path(__file__)),
        "method": "Every time recomputed from raw certificate epoch fields and store indices; index->time checked against content-certificate times_utc. No .zarr opened, no model run.",
        "support_convention": {"uniform": "t-12h..t+72h (15 x 6h steps)", "buffer_hours": BUFFER_H,
                               "recorded": "history_time..valid_time widened by content-certificate times_utc"},
        "exposure_kinds": {
            "GRADIENT_AND_RESULT": "row entered an optimizer step and its outcomes were observed",
            "RESULT": "model outputs vs truth observed (evaluation panel)",
            "INTEGRITY_READ_ONLY": "raw values read for finiteness/content hashing only; no model ran on them",
            "INPUT_READ/AGGREGATE_RESULT": "S0 gate input file"},
        "sources": metas,
        "sharedF0_sprint": sp_meta,
        "finiteness_scans": fin,
        "s0_input": s0,
        "historical_fs_ledger_crosscheck": hist_chk,
        "nesting_checks": nesting,
        "recorded_support_differs_from_uniform_count": rec_vs_uniform,
        "rows": all_rows,
        "sharedF0_sprint_rows": sp_rows,
        "sharedF0_sprint_excluded_rows": sp_exc,
        "runs_by_year": by_year,
    }
    with ledger_out.open("x") as f:
        json.dump(ledger, f, indent=1)
        f.write("\n")

    report = {
        "schema": "ed-fp05a-exposure-reconstruction-report/1",
        "created_utc": ledger["created_utc"],
        "ledger": {"path": str(ledger_out.relative_to(REPO)) if REPO in ledger_out.parents else str(ledger_out), "sha256": sha256(ledger_out)},
        "x1": {**x1_detail, **{k: v for k, v in nesting.items() if k.startswith("x1")}},
        "2020H2": h2,
        "2019H2_policy_dev_pool": pool,
        "runs_by_year_gradient_or_result_24h_buffered": {y: v["gradient_or_result_runs_24h_buffered"] for y, v in by_year.items()},
        "runs_by_year_gradient_or_result_unbuffered": {y: v["gradient_or_result_runs_unbuffered"] for y, v in by_year.items()},
        "sharedF0_sprint_counts": {"admitted_rows": sp_meta["n_rows_listed"], "excluded_rows": sp_meta["n_excluded_listed"],
                                   "admitted_by_year": {str(y): sum(1 for r in sp_rows if r["year"] == y) for y in (2018, 2019, 2020)},
                                   "admitted_by_role": sp_meta["coverage_recorded"]},
    }
    with report_out.open("x") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps(report, indent=1)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
