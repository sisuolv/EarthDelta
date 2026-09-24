#!/usr/bin/env python3
"""FP-05a: raw platform receipt index for every r4-session GPU job.

READ-ONLY, CPU-ONLY, STDLIB-ONLY. Reads artifacts/round2_cci/<run_dir>/{job_result.json,
job-id.txt, submission.json, invocation.json, source_manifest.json} and the tracked
submission records / decisions / findings / task results. Hashes every receipt and
cross-checks the returncode / status / elapsed_seconds each tracked file claims.
Missing receipts are recorded as MISSING (never fabricated).

Output (refuses to overwrite): entry/job_receipts_index.json
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
P4 = "plans/plan_v4_0923"
CCI = "artifacts/round2_cci"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(rel):
    return json.loads((REPO / rel).read_text())


def txt(rel):
    return (REPO / rel).read_text()


# --- the full r4-session job list ---------------------------------------------------------
# derived from: every artifacts/round2_cci/ed-r4* run dir (ls), every plans/plan_v4_0923/run_*/jobs/
# *.submission_record.json, and every decision/finding that names a job_dir. See main() for the
# completeness assertion that ties the three together.
JOBS = [
    # key, stage, run_dir, core(bool: named in the FP-05a dispatch), submission_record or None
    ("FP01-trace-a", "FP-01 four-arm trace attempt 1", "ed-r4trace0923-0923171440-894e7d", False, None),
    ("FP01-trace-b", "FP-01 four-arm trace (S0 gate trace)", "ed-r4trace0923b-0923171903-9d0b5d", True, None),
    ("FP01-gate-a", "FP-01 S0 gate attempt a", "ed-r4gate0923-0923172324-303cae", False, None),
    ("FP01-gate-b", "FP-01 S0 gate attempt b", "ed-r4gate0923b-0923172525-bb0400", False, None),
    ("FP01-gate-c", "FP-01 S0 gate attempt c", "ed-r4gate0923c-0923172842-5fe203", False, None),
    ("FP01-gate-d", "FP-01 S0 gate attempt d", "ed-r4gate0923d-0923173532-685f1b", False, None),
    ("FP01-gate-e", "FP-01 S0 gate (committed PASS)", "ed-r4gate0923e-0923174024-32e8c7", True, None),
    ("HIST-fsformal0923", "pre-FP03 historical Fs formal (ledger-only, INELIGIBLE_AS_FS)",
     "ed-r4fsformal0923-0923180756-7b1776", False, None),
    ("FP03v1-J1", "FP-03 v1 J1 diagnostic", "ed-r4fs03diag-0923203433-4127c2", True,
     f"{P4}/run_20260923T192759Z_fp03_fs/jobs/J1.submission_record.json"),
    ("FP03v1-J2", "FP-03 v1 J2 screen", "ed-r4fs03scrn-0923210334-9fca4b", True,
     f"{P4}/run_20260923T192759Z_fp03_fs/jobs/J2.submission_record.json"),
    ("FP03v1-J3", "FP-03 v1 J3 formal", "ed-r4fs03formal-0923212136-800630", True,
     f"{P4}/run_20260923T192759Z_fp03_fs/jobs/J3.submission_record.json"),
    ("FP03v1-J3b", "FP-03 v1 J3b formal fallback", "ed-r4fs03formalb-0923213547-b7d556", True,
     f"{P4}/run_20260923T192759Z_fp03_fs/jobs/J3b.submission_record.json"),
    ("X1-N32", "EXPLORATORY X1 N=32", "ed-r4fs03x1n32-0924005248-503e5b", True,
     f"{P4}/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/jobs/X1_N32.submission_record.json"),
    ("X1-N64", "EXPLORATORY X1 N=64", "ed-r4fs03x1n64-0924010930-404a65", True,
     f"{P4}/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/jobs/X1_N64.submission_record.json"),
    ("FP03v2-J3", "FP-03 v2 V2-J3 formal fit", "ed-r4fs03v2p-0924020931-71716b", True,
     f"{P4}/run_20260924T013959Z_fp03_v2/jobs/V2_J3.submission_record.json"),
    ("FP03v2-J4", "FP-03 v2 V2-J4 freeze+verify", "ed-r4fs03v2j4-0924021935-be0786", True,
     f"{P4}/run_20260924T013959Z_fp03_v2/jobs/V2_J4.submission_record.json"),
    ("FP04-BJ1", "FP-04 B-J1 shortstep/profile", "ed-r4bankbj1-0924051352-a9732c", True,
     f"{P4}/run_20260924T033627Z_fp04_bank/jobs/B-J1.submission_record.json"),
    ("FP04-BJ2", "FP-04 B-J2 formal", "ed-r4bankbj2-0924052624-dde08d", True,
     f"{P4}/run_20260924T033627Z_fp04_bank/jobs/B-J2.submission_record.json"),
    ("FP04-BJ3", "FP-04 B-J3 assemble (failed)", "ed-r4bankbj3-0924053507-ae0123", True,
     f"{P4}/run_20260924T033627Z_fp04_bank/jobs/B-J3.submission_record.json"),
    ("FP04-BJ3r", "FP-04 B-J3 infra retry", "ed-r4bankbj3r-0924054117-185e82", True,
     f"{P4}/run_20260924T033627Z_fp04_bank/jobs/B-J3_retry.submission_record.json"),
]

# never-submitted dirs (no job-id.txt): recorded for completeness, not GPU jobs
NOT_SUBMITTED = ["ed-r4trace0923-0923171427-377787",
                 "DRYRUN_NOT_SUBMITTED_ed-r4bankbj1-0924051336-ee3c78",
                 "DRYRUN_NOT_SUBMITTED_ed-r4bankbj2-0924052615-4a9651",
                 "DRYRUN_NOT_SUBMITTED_ed-r4bankbj3-0924053458-e089e8",
                 "DRYRUN_NOT_SUBMITTED_ed-r4bankbj3r-0924054107-6ad262"]

RC_RE = re.compile(r"(SUCCEEDED|FAILED)\s*(?:,\s*)?(?:rc\s*=\s*|returncode\s*)(\d+)[\s,;]*([0-9.]+)\s*s\b")


def claims_for(key: str) -> list[dict]:
    """Every tracked statement about this job's rc/status/elapsed (verbatim excerpt + parse)."""
    out = []

    def add(src, excerpt, status=None, rc=None, elapsed=None, precision=None, kind="summary_string"):
        out.append({"source": src, "excerpt": excerpt, "claimed_status": status, "claimed_rc": rc,
                    "claimed_elapsed_s": elapsed, "elapsed_precision_s": precision, "kind": kind})

    def from_string(src, s):
        m = RC_RE.search(s)
        if m:
            dec = len(m.group(3).split(".")[1]) if "." in m.group(3) else 0
            add(src, s, m.group(1), int(m.group(2)), float(m.group(3)), 0.5 * 10 ** (-dec))
        else:
            add(src, s)

    t3 = f"{P4}/run_20260924T013959Z_fp03_v2/task_result_fp03_v2.json"
    t4 = f"{P4}/run_20260924T033627Z_fp04_bank/task_result_fp04.json"
    prm = f"{P4}/CLAUDE_OPUSPLAN_EXECUTION_PROMPT.md"
    if key == "FP03v2-J3":
        from_string(t3 + "#jobs.V2-J3.result", jl(t3)["jobs"]["V2-J3"]["result"])
    if key == "FP03v2-J4":
        from_string(t3 + "#jobs.V2-J4.result", jl(t3)["jobs"]["V2-J4"]["result"])
    if key in ("FP04-BJ1", "FP04-BJ2", "FP04-BJ3", "FP04-BJ3r"):
        name = {"FP04-BJ1": "B-J1", "FP04-BJ2": "B-J2", "FP04-BJ3": "B-J3", "FP04-BJ3r": "B-J3_retry"}[key]
        from_string(f"{t4}#jobs.{name}.result", jl(t4)["jobs"][name]["result"])
    if key == "FP04-BJ3":
        dv = f"{P4}/run_20260924T033627Z_fp04_bank/deviations/DEV-FP04-001_relative_decision_paths.json"
        s = json.dumps(jl(dv))
        m = re.search(r'"result": "([^"]*)"', s)
        from_string(dv + "#...result", m.group(1))
        sr = f"{P4}/run_20260924T033627Z_fp04_bank/jobs/B-J3_retry.submission_record.json"
        add(sr + "#retry_of.result", jl(sr)["retry_of"]["result"], "FAILED", 1, None)
    if key in ("X1-N32", "X1-N64"):
        n = key[-3:]
        fnd = f"{P4}/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/results/x1_{n.lower()}_findings.json"
        s = jl(fnd)["job"]["job_result"]
        m = re.search(r"(SUCCEEDED), returncode (\d+), ([0-9.]+) s", s)
        add(fnd + "#job.job_result", s, m.group(1), int(m.group(2)), float(m.group(3)), 0.0005)
        res = f"{P4}/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/results/x1_results_X1_{n.upper()}.json"
        emb = jl(res).get("job_result", {})
        add(res + "#job_result (embedded copy)", {k: emb.get(k) for k in
            ("run_id", "status", "returncode", "elapsed_seconds", "started_utc", "finished_utc", "hostname")},
            emb.get("status"), emb.get("returncode"), emb.get("elapsed_seconds"), 0.0, kind="embedded_job_result")
    if key in ("FP03v1-J1",):
        sr = f"{P4}/run_20260923T192759Z_fp03_fs/jobs/J2.submission_record.json"
        add(sr + "#preconditions[0]", jl(sr)["preconditions"][0], "SUCCEEDED", None, None)
    if key in ("FP03v1-J2",):
        sr = f"{P4}/run_20260923T192759Z_fp03_fs/jobs/J3.submission_record.json"
        add(sr + "#preconditions[0]", jl(sr)["preconditions"][0], "SUCCEEDED", None, None)
    if key == "HIST-fsformal0923":
        lg = f"{P4}/run_20260923T192759Z_fp03_fs/ledger/fs_attempts_ledger.json"
        e = [x for x in jl(lg)["entries"] if x["job_id"] == "pt-e8y7ib01"][0]
        add(lg + "#entries[pt-e8y7ib01]", {"job_status": e["job_status"], "job_returncode": e["job_returncode"]},
            e["job_status"], e["job_returncode"], None)
    if key.startswith("FP01"):
        p = txt(prm)
        jid = (REPO / CCI / dict((k, rd) for k, _, rd, _, _ in JOBS)[key] / "job-id.txt").read_text().strip()
        for line in p.splitlines():
            if jid in line:
                st = "SUCCEEDED" if ("SUCCEEDED" in line or "已成功" in line) else ("FAILED" if "失败" in line else None)
                add(prm, line.strip(), st, None, None, kind="execution_prompt_statement")
    return out


def main() -> int:
    out = RUN / "entry/job_receipts_index.json"
    if out.exists():
        print("refusing to overwrite", out)
        return 2

    # completeness: every r4 run dir with a job-id must be in JOBS, every submission record too
    r4_dirs = sorted(Path(p).name for p in glob.glob(str(REPO / CCI / "*r4*")))
    listed = {rd for _, _, rd, _, _ in JOBS} | set(NOT_SUBMITTED)
    unlisted = [d for d in r4_dirs if d not in listed]
    subrecs = sorted(glob.glob(str(REPO / P4 / "run_*/jobs/*.submission_record.json")))
    subrec_rel = [str(Path(s).relative_to(REPO)) for s in subrecs]
    unlinked_subrecs = [s for s in subrec_rel if s not in {sr for *_, sr in JOBS}]
    decision_dirs = {}
    for f in glob.glob(str(REPO / P4 / "run_*/decisions/*.json")) + glob.glob(str(REPO / P4 / "run_*/certify/*decision*.json")):
        d = json.loads(Path(f).read_text())
        jd = d.get("job_dir")
        decision_dirs[str(Path(f).relative_to(REPO))] = Path(jd).name if jd else None

    rows = []
    for key, stage, rd, core, srel in JOBS:
        base = REPO / CCI / rd
        rec = {"key": key, "stage": stage, "core_named_in_dispatch": core, "run_dir": f"{CCI}/{rd}"}
        files = {}
        for fn in ("job_result.json", "job-id.txt", "submission.json", "invocation.json",
                   "source_manifest.json", "create_argv.json", "worker.log"):
            p = base / fn
            files[fn] = {"sha256": sha256(p), "bytes": p.stat().st_size} if p.is_file() else "MISSING"
        rec["files"] = files
        jr = json.loads((base / "job_result.json").read_text()) if (base / "job_result.json").is_file() else None
        jid = (base / "job-id.txt").read_text().strip() if (base / "job-id.txt").is_file() else None
        sub = json.loads((base / "submission.json").read_text()) if (base / "submission.json").is_file() else None
        rec["job_id_from_job-id.txt"] = jid
        rec["job_id_in_submission_stdout"] = bool(sub and jid and jid in sub.get("stdout", ""))
        errs = []
        if jr is None:
            rec["receipt_state"] = "MISSING"
            errs.append("job_result.json missing")
        else:
            rec["receipt"] = {k: jr.get(k) for k in ("run_id", "hostname", "status", "returncode",
                                                    "elapsed_seconds", "started_utc", "finished_utc")}
            if jr.get("run_id") != rd:
                errs.append("run_id != run dir name")
            try:
                t0 = dt.datetime.fromisoformat(jr["started_utc"])
                t1 = dt.datetime.fromisoformat(jr["finished_utc"])
                span = (t1 - t0).total_seconds()
                rec["finished_minus_started_s"] = round(span, 3)
                if abs(span - jr["elapsed_seconds"]) > 1.0:
                    errs.append(f"elapsed_seconds {jr['elapsed_seconds']} vs finished-started {span:.3f}")
            except Exception as e:  # noqa: BLE001
                errs.append(f"timestamp parse: {e}")
            if (jr.get("status") == "SUCCEEDED") != (jr.get("returncode") == 0):
                errs.append("status/returncode inconsistent")
            rec["receipt_state"] = "OBSERVED_RAW_RECEIPT"
        # submission record identity join
        if srel:
            sr = jl(srel)
            rec["submission_record"] = {"path": srel, "sha256": sha256(REPO / srel)}
            join = {
                "job_id_match": sr.get("job_id") == jid,
                "run_dir_match": sr.get("run_dir") == f"{CCI}/{rd}",
                "invocation_sha256_match": files["invocation.json"] != "MISSING" and
                sr.get("invocation_sha256") == files["invocation.json"]["sha256"],
                "source_manifest_sha256_match": files["source_manifest.json"] != "MISSING" and
                sr.get("source_manifest_sha256") == files["source_manifest.json"]["sha256"],
            }
            argvf = sr.get("argv_file")
            if argvf and (REPO / argvf).is_file():
                join["argv_file_sha256_match"] = sha256(REPO / argvf) == sr.get("argv_file_sha256")
            rec["identity_join"] = join
            if not all(join.values()):
                errs.append(f"identity join failure {join}")
        else:
            rec["submission_record"] = None
            rec["identity_join"] = "NO_TRACKED_SUBMISSION_RECORD (FP-01/historical jobs predate the per-job record convention)"
        # decisions naming this job dir
        rec["decisions_naming_run_dir"] = sorted(k for k, v in decision_dirs.items() if v == rd)
        # S0 gate cross-link
        if key.startswith("FP01-gate"):
            links = []
            for g in glob.glob(str(REPO / P4 / "run_20260923T_fp01_trace/gate_output/*/s0_gate_result.json")):
                gr = json.loads(Path(g).read_text())
                if jr and gr.get("hostname") == jr.get("hostname"):
                    links.append({"gate_result": str(Path(g).relative_to(REPO)), "sha256": sha256(Path(g)),
                                  "status": gr.get("status"), "s0_gate_pass": gr.get("s0_gate_pass"),
                                  "verdict_committed": gr.get("verdict_committed"),
                                  "gate_started_within_job": jr["started_utc"] <= gr.get("started_utc", "") <= jr["finished_utc"],
                                  "source_root_is_this_run_dir": rd in str(gr.get("source_root"))})
            rec["s0_gate_result_linked_by_hostname"] = links
        # tracked claims
        cl = claims_for(key)
        for c in cl:
            chk = {}
            if jr is not None:
                if c["claimed_status"] is not None:
                    chk["status_match"] = c["claimed_status"] == jr.get("status")
                if c["claimed_rc"] is not None:
                    chk["rc_match"] = c["claimed_rc"] == jr.get("returncode")
                if c["claimed_elapsed_s"] is not None:
                    diff = abs(c["claimed_elapsed_s"] - jr["elapsed_seconds"])
                    chk["elapsed_abs_diff_s"] = round(diff, 6)
                    chk["elapsed_match_within_rounding"] = diff <= (c["elapsed_precision_s"] or 0) + 1e-9
                if c["kind"] == "embedded_job_result":
                    emb = c["excerpt"]
                    chk["embedded_equals_raw_all_fields"] = all(emb.get(k) == jr.get(k) for k in emb)
            c["check"] = chk
            if any(v is False for v in chk.values()):
                errs.append(f"claim mismatch from {c['source']}")
        rec["tracked_claims"] = cl
        rec["tracked_rc_or_elapsed_claim_present"] = any(c["claimed_rc"] is not None or c["claimed_elapsed_s"] is not None for c in cl)
        rec["errors"] = errs
        rec["verdict"] = "RECEIPT_VERIFIED" if not errs else ("MISSING" if jr is None else "MISMATCH")
        rows.append(rec)

    not_sub = []
    for rd in NOT_SUBMITTED:
        base = REPO / CCI / rd
        not_sub.append({"run_dir": f"{CCI}/{rd}", "has_job-id.txt": (base / "job-id.txt").is_file(),
                        "has_job_result.json": (base / "job_result.json").is_file(),
                        "classification": "PREPARED_NOT_SUBMITTED (no job id, no GPU execution)"})

    res = {
        "schema": "ed-fp05a-job-receipts-index/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder": str(Path(__file__).relative_to(REPO)),
        "builder_sha256": sha256(Path(__file__)),
        "scope": "every GPU job of the 2026-09-23/24 r4 session (FP-01, historical Fs formal, FP-03 v1, X1, FP-03 v2, FP-04). Older r3/sprint jobs are ledger-only (exposure ledger), not receipt-audited here.",
        "completeness": {
            "r4_run_dirs_on_disk": len(r4_dirs),
            "r4_run_dirs_not_in_index": unlisted,
            "tracked_submission_records": len(subrec_rel),
            "submission_records_not_linked": unlinked_subrecs,
            "decision_files_job_dir": decision_dirs,
            "decision_job_dirs_not_in_index": sorted({v for v in decision_dirs.values() if v and v not in listed}),
        },
        "n_jobs": len(rows),
        "n_core_named_in_dispatch": sum(r["core_named_in_dispatch"] for r in rows),
        "n_receipt_verified": sum(r["verdict"] == "RECEIPT_VERIFIED" for r in rows),
        "n_missing": sum(r["verdict"] == "MISSING" for r in rows),
        "n_mismatch": sum(r["verdict"] == "MISMATCH" for r in rows),
        "jobs": rows,
        "prepared_not_submitted": not_sub,
    }
    res["overall"] = ("ALL_RECEIPTS_VERIFIED" if res["n_receipt_verified"] == len(rows) and not unlisted
                      and not unlinked_subrecs and not res["completeness"]["decision_job_dirs_not_in_index"]
                      else "BLOCKED")
    with out.open("x") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
        f.write("\n")
    for r in rows:
        print(f'{r["key"]:18s} {r["job_id_from_job-id.txt"]} {r.get("receipt", {}).get("status")} '
              f'rc={r.get("receipt", {}).get("returncode")} el={r.get("receipt", {}).get("elapsed_seconds")} '
              f'claims={len(r["tracked_claims"])} verdict={r["verdict"]} errs={r["errors"]}')
    print(json.dumps({k: res[k] for k in ("n_jobs", "n_core_named_in_dispatch", "n_receipt_verified",
                                          "n_missing", "n_mismatch", "overall")}))
    print(json.dumps(res["completeness"]["r4_run_dirs_not_in_index"]), res["completeness"]["submission_records_not_linked"],
          res["completeness"]["decision_job_dirs_not_in_index"])
    return 0 if res["overall"] == "ALL_RECEIPTS_VERIFIED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
