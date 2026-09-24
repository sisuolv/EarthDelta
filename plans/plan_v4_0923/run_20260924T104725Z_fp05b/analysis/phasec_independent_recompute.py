#!/usr/bin/env python3
"""Independent re-computation of the EVAL-v1 point estimates (and a separate-RNG
paired block bootstrap as a cross-check of the CI widths) from the canonical
candidate_results.json, frozen oof_predictions.json and background_f0.json.
numpy/json only; does not import earthdelta.policy_oof. Run AFTER score; it
never feeds back into any frozen artifact."""
import json
import sys
from pathlib import Path

import numpy as np

RUN = Path("/mnt/afs/260010168/EarthDelta/plans/plan_v4_0923/run_20260924T104725Z_fp05b")
out = Path(sys.argv[1])
rows = json.loads((RUN / "cache/dev/candidate_results.json").read_text())["rows"]
preds = json.loads((RUN / "policies/oof_predictions.json").read_text())["predictions"]
bg = json.loads((RUN / "cache/dev/background_f0.json").read_text())["issues"]
official = json.loads((RUN / "evaluation/paired_block_bootstrap.json").read_text())
ids = sorted(preds)
L = {}
for r in rows:
    L[(r["issue_id"], r["candidate_id"], r["lead_hours"])] = r["cpu_loss"]
times = {r["issue_id"]: r["issue_time"] for r in rows}
C = ["reference", "expert_0", "expert_1", "expert_2", "expert_3"]
leads = (6, 24, 72)
M5 = min(C, key=lambda c: sum(L[(i, c, 24)] for i in ids))
act = {m: {i: preds[i][m] for i in ids} for m in ("M0", "M1", "M2", "M3")}
act["M4"] = {i: min(C, key=lambda c: L[(i, c, 24)]) for i in ids}  # oracle on 24h
act["M5"] = {i: M5 for i in ids}
loss = {m: {h: np.array([L[(i, act[m][i], h)] for i in ids]) for h in leads} for m in act}
loss["F0"] = {h: np.array([bg[i]["f0_cpu"][str(h)] for i in ids]) for h in leads}
Fs = {h: loss["M0"][h] for h in leads}
comps = {"oracle_vs_Fs": ("M4", "M0", 24), "oracle_vs_M1": ("M4", "M1", 24), "M5_vs_Fs": ("M5", "M0", 24),
         "M1_vs_Fs": ("M1", "M0", 24), "M2_vs_Fs": ("M2", "M0", 24), "M2_vs_M1": ("M2", "M1", 24),
         "M3_vs_Fs": ("M3", "M0", 24), "M3_vs_M1": ("M3", "M1", 24),
         "M1_vs_Fs_72h": ("M1", "M0", 72), "M2_vs_Fs_72h": ("M2", "M0", 72), "M3_vs_Fs_72h": ("M3", "M0", 72),
         "M1_vs_Fs_6h": ("M1", "M0", 6), "M2_vs_Fs_6h": ("M2", "M0", 6), "M3_vs_Fs_6h": ("M3", "M0", 6),
         "F0_vs_Fs_background": ("F0", "M0", 24)}
origin = 1562414400
blk = np.array([(times[i] - origin) // (7 * 86400) for i in ids])
ub = np.unique(blk)
rng = np.random.default_rng(987654321)  # deliberately NOT the protocol seed: independent cross-check
draws = rng.integers(0, len(ub), size=(10000, len(ub)))
idx_of = [np.where(blk == b)[0] for b in ub]
res = {"n_issues": len(ids), "n_blocks": int(len(ub)), "M5_choice": M5, "comparisons": {}}
H = (Fs[24].sum() - loss["F0"][24].sum()) / Fs[24].sum()
res["H_Fs_24h_point"] = float(H)
maxdiff = abs(H - official["primary_7day_blocks"]["H_Fs_24h"]["point"])
for n, (m, b, h) in comps.items():
    pt = (loss[b][h].sum() - loss[m][h].sum()) / Fs[h].sum()
    bs = np.empty(10000)
    for k in range(10000):
        sel = np.concatenate([idx_of[j] for j in draws[k]])
        bs[k] = (loss[b][h][sel].sum() - loss[m][h][sel].sum()) / Fs[h][sel].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    st = "ABOVE" if lo >= 0.0034 else ("BELOW" if hi < 0.0034 else "STRADDLE")
    o = official["primary_7day_blocks"]["comparisons"][n]
    d = abs(pt - o["point"])
    maxdiff = max(maxdiff, d)
    res["comparisons"][n] = {"point": float(pt), "ci_low_alt_rng": float(lo), "ci_high_alt_rng": float(hi),
                             "status_alt_rng": st, "official_status": o["status_vs_delta_min"],
                             "point_abs_diff_vs_official": float(d)}
res["max_point_abs_diff_vs_official"] = float(maxdiff)
res["status_agreement"] = all(v["status_alt_rng"] == v["official_status"] for v in res["comparisons"].values())
with open(out, "x") as f:
    json.dump(res, f, indent=1)
print(json.dumps({"max_point_abs_diff_vs_official": maxdiff, "status_agreement": res["status_agreement"],
                  "M5": M5, "disagree": {k: (v["status_alt_rng"], v["official_status"])
                                         for k, v in res["comparisons"].items()
                                         if v["status_alt_rng"] != v["official_status"]}}))
