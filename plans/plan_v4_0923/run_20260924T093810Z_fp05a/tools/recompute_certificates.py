#!/usr/bin/env python3
"""FP-05a: independent recompute of the Fs and bank certification numbers.

READ-ONLY, CPU-ONLY. The arithmetic part is stdlib only (json/math/hashlib): it
recomputes every per-issue ratio, panel mean and mean-ratio from the RAW float64
per-issue loss panels and compares them to what the certificates record. The
optional registry check runs earthdelta.registry.verify_bank_bundle(deep=True,
rehash_external=True) in a child process with CUDA hidden (weights_only tensor
reloads; no model is constructed and no forward pass is run); the child reports
whether zarr was imported.

Output (refuses to overwrite): entry/certificate_recompute.json
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
P4 = "plans/plan_v4_0923"
FSDIR = f"{P4}/run_20260924T013959Z_fp03_v2/certify/fs"
BKDIR = f"{P4}/run_20260924T033627Z_fp04_bank/certify/bank"
TOL = 1e-12


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(rel):
    return json.loads((REPO / rel).read_text())


def summary(L0: dict, L1: dict, ids: list, lead: str):
    """mirror of the recorded definition: per-issue L1/L0, mean_initial=mean(L0),
    mean_final=mean(L1), mean_ratio=mean_final/mean_initial. Two summation orders
    are reported (naive left-to-right in issue order and math.fsum)."""
    a = [float(L0[i][lead]) for i in ids]
    b = [float(L1[i][lead]) for i in ids]
    n = len(ids)
    per = {i: float(L1[i][lead]) / float(L0[i][lead]) for i in ids}
    m0, m1 = sum(a) / n, sum(b) / n
    f0, f1 = math.fsum(a) / n, math.fsum(b) / n
    return {"n_issues": n, "per_issue_ratio": per, "max_ratio": max(per.values()),
            "min_ratio": min(per.values()), "mean_initial": m0, "mean_final": m1,
            "mean_ratio": m1 / m0, "mean_ratio_fsum": f1 / f0,
            "n_worse_than_initial": sum(1 for v in per.values() if v > 1.0)}


def compare(rec: dict, mine: dict) -> dict:
    diffs = {}
    for k in ("max_ratio", "min_ratio", "mean_initial", "mean_final", "mean_ratio"):
        diffs[k] = abs(float(rec[k]) - mine[k])
    diffs["mean_ratio_vs_fsum"] = abs(float(rec["mean_ratio"]) - mine["mean_ratio_fsum"])
    pi = rec.get("per_issue_ratio", {})
    diffs["per_issue_max_abs"] = max(abs(float(pi[i]) - mine["per_issue_ratio"][i]) for i in mine["per_issue_ratio"]) \
        if set(pi) == set(mine["per_issue_ratio"]) else None
    ok = all(v is not None and v <= TOL for v in diffs.values())
    return {"abs_diff": diffs, "issue_sets_equal": set(pi) == set(mine["per_issue_ratio"]),
            "n_issues_equal": int(rec.get("n_issues", -1)) == mine["n_issues"], "within_1e-12": ok}


# ------------------------------------------------------------------------------------------
def fs_block():
    panel_rel = f"{FSDIR}/panel_initial_final.json"
    P = jl(panel_rel)
    Q = jl(f"{FSDIR}/qualification.json")
    rule = jl(f"{FSDIR}/quality_rule.json")
    tr, ho = P["train"], P["holdout"]
    l0_eq_f0 = all(tr["L0"][i] == tr["F0"][i] for i in tr["issue_ids"]) and \
        all(ho["L0"][i] == ho["F0"][i] for i in ho["issue_ids"])
    mine = {
        "q4_train_24h": summary(tr["L0"], tr["L1"], tr["issue_ids"], "24"),
        "q5_train_72h": summary(tr["L0"], tr["L1"], tr["issue_ids"], "72"),
        "q6_holdout_24h": summary(ho["L0"], ho["L1"], ho["issue_ids"], "24"),
        "report_only_train_6h": summary(tr["L0"], tr["L1"], tr["issue_ids"], "6"),
        "report_only_holdout_6h": summary(ho["L0"], ho["L1"], ho["issue_ids"], "6"),
        "report_only_holdout_72h": summary(ho["L0"], ho["L1"], ho["issue_ids"], "72"),
    }
    per_replica = []
    all_ok = True
    for d in Q["formal"]["details"]:
        q = d["qualification"]
        cmp = {k: compare(q[k], mine[k]) for k in ("q4_train_24h", "q5_train_72h", "q6_holdout_24h", "report_only_train_6h")}
        all_ok &= all(c["within_1e-12"] and c["issue_sets_equal"] for c in cmp.values())
        per_replica.append({"device_index": d["device_index"], "verdict": q["verdict"], "status": q["status"],
                            "comparisons": cmp})
    q4, q5, q6 = mine["q4_train_24h"], mine["q5_train_72h"], mine["q6_holdout_24h"]
    r4, r5, r6 = rule["q4"], rule["q5"], rule["q6"]
    thresholds = {
        "Q4_pass": q4["max_ratio"] < r4["pass_max_ratio_lt"] and q4["mean_ratio"] <= r4["pass_mean_ratio_le"],
        "Q5_pass": q5["mean_ratio"] <= r5["mean_ratio_72h_le"],
        "Q6_pass": q6["mean_ratio"] <= r6["holdout_mean_ratio_24h_le"],
    }
    harm_F0 = q6["mean_ratio"] - 1.0
    harm_Fs = (q6["mean_final"] - q6["mean_initial"]) / q6["mean_final"]
    final = Q["final"]
    return {
        "panel": {"path": panel_rel, "sha256": sha256(REPO / panel_rel)},
        "qualification": {"path": f"{FSDIR}/qualification.json", "sha256": sha256(REPO / f"{FSDIR}/qualification.json")},
        "quality_rule": {"path": f"{FSDIR}/quality_rule.json", "rule": rule["rule"]},
        "L0_equals_F0_bitwise_recomputed": l0_eq_f0,
        "L0_equals_F0_bitwise_recorded": P.get("L0_equals_F0_bitwise"),
        "recomputed": {k: {kk: vv for kk, vv in v.items() if kk != "per_issue_ratio"} for k, v in mine.items()},
        "per_replica_comparison": per_replica,
        "all_replicas_within_1e-12": all_ok,
        "thresholds_recomputed": thresholds,
        "final_verdict_recorded": final["verdict"], "designated_gates_recorded": final["designated_gates"],
        "delta_min_recompute": {
            "holdout24_harm_F0_denominator": harm_F0,
            "holdout24_harm_Fs_denominator": harm_Fs,
            "note": "Both measured on the 16-issue 2019H1 Fs-certification holdout. Retained only as a pre-declared engineering screen scale; must not be re-derived from new DEV outcomes.",
        },
    }


# ------------------------------------------------------------------------------------------
def bank_block():
    Q = jl(f"{BKDIR}/qualification.json")
    V = jl(f"{BKDIR}/bank_verify.json")
    experts = []
    ok_all = True
    for d in Q["formal"]["details"]:
        e = d["expert_index"]
        pf = d["files"][f"bank_panels_expert{e}.json"]
        p = Path(pf["path"])
        rehash = sha256(p) if p.is_file() else "MISSING"
        panels = json.loads(p.read_text())["experts"][str(e)]
        ids = panels["issue_ids"]
        l0_eq_fs = all(panels["L0"][i] == panels["Fs"][i] for i in ids)
        mine24 = summary(panels["L0"], panels["L1"], ids, "24")
        rec = d["qualification"]["e3"]["own_group_24h"]
        cmp = compare(rec, mine24)
        thr = d["qualification"]["e3"]["threshold_lt"]
        ok =cmp["within_1e-12"] and rehash == pf["sha256"] and l0_eq_fs
        ok_all &= ok
        experts.append({
            "expert_index": e, "panels_file": str(p.relative_to(REPO)), "panels_sha256_recorded": pf["sha256"],
            "panels_sha256_rehash": rehash, "panels_rehash_match": rehash == pf["sha256"],
            "L0_equals_Fs_bitwise_recomputed": l0_eq_fs, "L0_equals_Fs_bitwise_recorded": panels.get("L0_equals_Fs_bitwise"),
            "n_issues": mine24["n_issues"],
            "own_group24_mean_ratio_recomputed": mine24["mean_ratio"],
            "own_group24_mean_ratio_recorded": rec["mean_ratio"],
            "own_group24_max_ratio_recomputed": mine24["max_ratio"],
            "own_group24_n_worse": mine24["n_worse_than_initial"],
            "threshold_lt_recorded": thr,
            "E3_mean_ratio_lt_threshold_recomputed": mine24["mean_ratio"] < thr,
            "comparison": cmp, "verdict_recorded": d["qualification"]["verdict"],
            "own_group72_mean_ratio_recomputed_report_only": summary(panels["L0"], panels["L1"], ids, "72")["mean_ratio"],
            "own_group6_mean_ratio_recomputed_report_only": summary(panels["L0"], panels["L1"], ids, "6")["mean_ratio"],
        })

    # ---- A1..A5 exactness -------------------------------------------------------------
    a = {}
    viol = []
    steps_req = {"1", "4", "12"}
    for tag in ("A1_source_bank", "A2_training_probe"):
        blk = V[tag]
        a[tag] = {"experts_present": sorted(blk.keys()), "all_steps_1_4_12": True, "all_zero": True, "all_torch_equal": True}
        if sorted(blk.keys()) != ["0", "1", "2", "3"]:
            viol.append(f"{tag} expert set {sorted(blk.keys())}")
        for k, v in blk.items():
            m = v["max_abs_diff_by_step"]
            if set(m) != steps_req:
                a[tag]["all_steps_1_4_12"] = False
                viol.append(f"{tag}[{k}] steps {sorted(m)}")
            for s, x in m.items():
                if not (x == 0.0):
                    a[tag]["all_zero"] = False
                    viol.append(f"{tag}[{k}] step {s} = {x}")
            if not all(v["torch_equal_by_step"].values()) or v.get("exact") is not True:
                a[tag]["all_torch_equal"] = False
                viol.append(f"{tag}[{k}] torch_equal/exact false")
            if tag == "A2_training_probe" and v["probe_file_sha256"] != v["expected_probe_file_sha256"]:
                viol.append(f"A2[{k}] probe sha mismatch")
    a3 = V["A3_zero_edit"]
    a["A3_zero_edit"] = {"atol": a3["atol"], "max_abs_diff_vs_fs": a3["max_abs_diff_vs_fs"],
                         "max_abs_diff_vs_f0": a3["max_abs_diff_vs_f0"], "discriminating": a3["discriminating"],
                         "exact_zero": a3["max_abs_diff_vs_fs"] == 0.0 and a3["atol"] == 0.0}
    if not a["A3_zero_edit"]["exact_zero"]:
        viol.append("A3 not exactly zero")
    a4 = V["A4_continuation"]
    a["A4_continuation"] = {}
    for k, v in a4.items():
        byst = v["max_abs_diff_vs_fs_continuation_by_step"]
        vals = list(byst.values()) if isinstance(byst, dict) else list(byst)
        z = all(x == 0.0 for x in vals)
        n_post_hold = len(vals)
        a["A4_continuation"][k] = {"n_post_hold_steps": n_post_hold, "expected_post_hold_steps": v["total"] - v["hold"],
                                   "all_zero": z, "hold": v["hold"], "coefficient": v["coefficient"],
                                   "discriminates_f0": v["discriminates_f0"], "edit_effect_at_hold_max_abs": v["edit_effect_at_hold_max_abs"]}
        if not z or n_post_hold != v["total"] - v["hold"]:
            viol.append(f"A4[{k}] continuation not exact or wrong length")
    if sorted(a4.keys()) != ["0", "1", "2", "3"]:
        viol.append("A4 expert set")
    a5 = V["A5_reload"]
    a5ok = (a5["file_sha256"] == a5["expected_file_sha256"] and a5["bank_digest"] == a5["expected_bank_digest"]
            and a5["expert_digests"] == a5["expected_expert_digests"] == a5["decided_expert_digests"]
            and a5["assembly_decision_sha256_matches"] is True)
    bank_pt = REPO / BKDIR / "bank.pt"
    bank_rehash = sha256(bank_pt)
    a["A5_reload"] = {"digests_equal": a5ok, "bank_pt_rehash": bank_rehash,
                      "bank_pt_rehash_matches_recorded": bank_rehash == a5["file_sha256"]}
    if not a5ok or bank_rehash != a5["file_sha256"]:
        viol.append("A5 digest/rehash mismatch")
    # expert decided digests vs qualification's experts_for_assembly
    efa = {x["expert_index"]: x["expert_digest"] for x in Q["formal"]["experts_for_assembly"]}
    a["A5_reload"]["decided_digests_equal_qualification_experts_for_assembly"] = \
        [efa[i] for i in range(4)] == a5["decided_expert_digests"]
    return {
        "qualification": {"path": f"{BKDIR}/qualification.json", "sha256": sha256(REPO / f"{BKDIR}/qualification.json")},
        "bank_verify": {"path": f"{BKDIR}/bank_verify.json", "sha256": sha256(REPO / f"{BKDIR}/bank_verify.json")},
        "experts": experts,
        "own_group24_mean_ratios_recomputed": [x["own_group24_mean_ratio_recomputed"] for x in experts],
        "group_sizes_recomputed": [x["n_issues"] for x in experts],
        "all_experts_within_1e-12": ok_all,
        "assembly_equivalence": a,
        "assembly_exactness_violations": viol,
        "assembly_all_exact_zero": not viol,
        "certify_verdict_recorded": jl(f"{P4}/run_20260924T033627Z_fp04_bank/certify/BJ3_certify_decision.json").get("verdict"),
    }


# ------------------------------------------------------------------------------------------
CHILD = r'''
import json, sys, os
sys.path.insert(0, os.environ["ED_REPO"])
from pathlib import Path
from earthdelta.registry import verify_bank_bundle
proto = json.loads(Path(os.environ["ED_PROTO"]).read_text())
res = verify_bank_bundle(os.environ["ED_BUNDLE"], deep=True, rehash_external=True,
                         expected_protocol_sha256=os.environ["ED_PROTO_SHA"],
                         expected_source=proto["source_at_preregistration"])
import torch
out = {"result": res, "zarr_imported": "zarr" in sys.modules,
       "xarray_imported": "xarray" in sys.modules,
       "cuda_initialized": bool(torch.cuda.is_initialized()),
       "torch_version": torch.__version__}
print("@@JSON@@" + json.dumps(out, default=str))
'''


def registry_block():
    proto = REPO / f"{P4}/run_20260924T033627Z_fp04_bank/protocol/bank_protocol_v1.json"
    # .pydeps supplies timm (imported by earthdelta.bridge at module level); zarr is also
    # present there, so the child asserts afterwards that zarr was never imported.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", ED_REPO=str(REPO), ED_BUNDLE=str(REPO / BKDIR),
               ED_PROTO=str(proto), ED_PROTO_SHA=sha256(proto),
               PYTHONPATH=f"{REPO}:{REPO / '.pydeps'}",
               PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")  # same setting the GPU jobs used
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, env=env, timeout=1800)
    log = RUN / "logs/verify_bank_bundle.child.log"
    with log.open("x") as f:
        f.write(f"rc={p.returncode}\n--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}\n")
    out = {"argv": "verify_bank_bundle(<certify/bank>, deep=True, rehash_external=True, expected_protocol_sha256=<bank_protocol_v1 sha>, expected_source=<17 pins>)",
           "child_returncode": p.returncode, "log": str(log.relative_to(REPO)) if REPO in log.parents else str(log), "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": env["PYTHONPATH"]}
    if p.returncode == 0 and "@@JSON@@" in p.stdout:
        j = json.loads(p.stdout.split("@@JSON@@", 1)[1])
        out.update(j)
        out["passed"] = True
    else:
        out["passed"] = False
        out["stderr_tail"] = p.stderr[-4000:]
    return out


def main() -> int:
    outp = RUN / "entry/certificate_recompute.json"
    if outp.exists():
        print("refusing to overwrite", outp)
        return 2
    fs = fs_block()
    bank = bank_block()
    reg = registry_block()
    res = {
        "schema": "ed-fp05a-certificate-recompute/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder": str(Path(__file__).relative_to(REPO)), "builder_sha256": sha256(Path(__file__)),
        "tolerance_abs": TOL,
        "fs": fs, "bank": bank, "bank_bundle_registry_verify": reg,
    }
    res["pass"] = bool(fs["all_replicas_within_1e-12"] and all(fs["thresholds_recomputed"].values())
                       and fs["L0_equals_F0_bitwise_recomputed"] and bank["all_experts_within_1e-12"]
                       and bank["assembly_all_exact_zero"] and all(e["E3_mean_ratio_lt_threshold_recomputed"] for e in bank["experts"])
                       and reg.get("passed") and not reg.get("zarr_imported"))
    with outp.open("x") as f:
        json.dump(res, f, indent=1, default=str)
        f.write("\n")
    short = {"fs_q4": fs["recomputed"]["q4_train_24h"]["mean_ratio"], "fs_q5": fs["recomputed"]["q5_train_72h"]["mean_ratio"],
             "fs_q6": fs["recomputed"]["q6_holdout_24h"]["mean_ratio"], "fs_q6_worse": fs["recomputed"]["q6_holdout_24h"]["n_worse_than_initial"],
             "fs_within": fs["all_replicas_within_1e-12"], "thresholds": fs["thresholds_recomputed"],
             "delta": fs["delta_min_recompute"], "bank_ratios": bank["own_group24_mean_ratios_recomputed"],
             "bank_sizes": bank["group_sizes_recomputed"], "bank_within": bank["all_experts_within_1e-12"],
             "A1_A5_exact": bank["assembly_all_exact_zero"], "violations": bank["assembly_exactness_violations"],
             "registry_passed": reg.get("passed"), "zarr_imported": reg.get("zarr_imported"),
             "cuda_initialized": reg.get("cuda_initialized"), "pass": res["pass"]}
    print(json.dumps(short, indent=1))
    return 0 if res["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
