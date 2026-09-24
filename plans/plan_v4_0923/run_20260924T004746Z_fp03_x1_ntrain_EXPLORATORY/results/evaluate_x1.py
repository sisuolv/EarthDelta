#!/usr/bin/env python3
"""X1 (EXPLORATORY) evaluation: train/holdout panel ratios at N vs the N=8 baseline (J3).

CPU only; reads the jobs' raw files; writes results/x1_results_<run>.json.
Labelled EXPLORATORY: nothing here is an FS-QUAL-v1 verdict.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

REPO = Path("/mnt/afs/260010168/EarthDelta")
sys.path.insert(0, str(REPO))
from earthdelta import static_adapter as sa  # noqa: E402

X = REPO / "plans/plan_v4_0923/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY"
BASELINE = REPO / "artifacts/round2_cci/ed-r4fs03formal-0923212136-800630/replica0"
LABEL = "EXPLORATORY_X1 - not FS-QUAL-v1, not pre-registered, cannot select/freeze/certify an Fs"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(proc: Path):
    return (json.loads((proc / "fs_fit_record.json").read_text()),
            json.loads((proc / "fs_panels.json").read_text()))


def ratios(panels, group, lead, subset=None):
    p0, p1 = panels[group]["L0"], panels[group]["L1"]
    if subset is not None:
        p0 = {k: p0[k] for k in subset}
        p1 = {k: p1[k] for k in subset}
    s = sa.panel_ratio_summary(p0, p1, lead)
    return {"mean_ratio": s["mean_ratio"], "max_ratio": s["max_ratio"], "min_ratio": s["min_ratio"],
            "n_issues": s["n_issues"], "n_worse": sum(1 for v in s["per_issue_ratio"].values() if v > 1),
            "per_issue_ratio": s["per_issue_ratio"]}


def onset(losses, epoch_size, factor=1.02):
    e = [sum(losses[k:k + epoch_size]) / epoch_size
         for k in range(0, len(losses) - len(losses) % epoch_size, epoch_size)]
    for k in range(1, len(e)):
        if e[k] > factor * min(e[:k]):
            return {"epoch_size": epoch_size, "epoch_means": e, "onset_epoch": k,
                    "onset_update": k * epoch_size}
    return {"epoch_size": epoch_size, "epoch_means": e, "onset_epoch": None, "onset_update": None}


def per_issue_trend(rec):
    """For each issue: loss at each visit relative to its first visit (no cross-issue mixing)."""
    by = {}
    for loss, iid in zip(rec["losses"], rec["training_issue_ids"]):
        by.setdefault(iid, []).append(loss)
    worst_rise = max(max(v[i] / min(v[:i]) for i in range(1, len(v))) for v in by.values())
    return {"n_issues": len(by), "visits_per_issue": sorted({len(v) for v in by.values()}),
            "max_visit_over_running_min": worst_rise,
            "n_last_below_first": sum(1 for v in by.values() if v[-1] < v[0])}


def main(run_dir: str) -> None:
    run = REPO / run_dir
    cid = json.loads((run / "invocation.json").read_text())["argv"][2].split("--config-id ")[1].split()[0]
    proc = run / cid
    rec, panels = load(proc)
    base_rec, base_panels = load(BASELINE)
    original8 = list(base_panels["train"]["L0"])
    cfg = json.loads((X / "x1_exploratory_config.json").read_text())
    rule = cfg["decision_rule_for_N64"]

    stab_declared = sa.training_stability(rec, sa._panel_column(panels["train"]["L0"], 24), horizon=256)
    n = len(panels["train"]["L0"])
    out = {
        "status": LABEL,
        "run_dir": run_dir, "config_id": cid,
        "job_id": (run / "job-id.txt").read_text().strip(),
        "job_result": json.loads((run / "job_result.json").read_text()),
        "files_sha256": {f: sha(proc / f) for f in ("fs_fit_record.json", "fs_panels.json",
                                                  "fs_adapter.pt", "run_record.json")},
        "n_train": n,
        "continuity_first8_losses_equal_J3": rec["losses"][:8] == base_rec["losses"][:8],
        "F0_holdout_panel_equal_J3": panels["holdout"]["F0"] == base_panels["holdout"]["F0"],
        "F0_original8_train_panel_equal_J3": all(panels["train"]["F0"][k] == base_panels["train"]["F0"][k]
                                                 for k in original8),
        "this_run": {
            "train_24h_all": ratios(panels, "train", 24),
            "train_24h_original8": ratios(panels, "train", 24, original8),
            "train_6h_all": ratios(panels, "train", 6),
            "train_72h_all": ratios(panels, "train", 72),
            "holdout_6h": ratios(panels, "holdout", 6),
            "holdout_24h": ratios(panels, "holdout", 24),
            "holdout_72h": ratios(panels, "holdout", 72),
        },
        "baseline_N8_J3": {
            "train_24h_all": ratios(base_panels, "train", 24),
            "train_6h_all": ratios(base_panels, "train", 6),
            "train_72h_all": ratios(base_panels, "train", 72),
            "holdout_6h": ratios(base_panels, "holdout", 6),
            "holdout_24h": ratios(base_panels, "holdout", 24),
            "holdout_72h": ratios(base_panels, "holdout", 72),
        },
        "stability": {
            "declared_rule_epoch_size_8": {
                "stable": stab_declared["stable"], "clip_events": stab_declared["q2"]["clip_events"],
                "max_per_visit_ratio": stab_declared["q2"]["max_per_visit_ratio"],
                "onset_update": stab_declared["q2"]["onset_update"]},
            "epoch_size_N_one_full_pass_POST_HOC_DESCRIPTIVE": onset(rec["losses"], n),
            "per_issue_visit_trend": per_issue_trend(rec),
            "note": ("The declared epoch-mean rule (epoch = 8 updates) was defined for N=8, where "
                     "each epoch is one full pass over the same 8 issues. For N>8 consecutive "
                     "8-update epochs contain DIFFERENT issues, so the rule compares issues, not "
                     "training progress. The epoch_size=N variant (one full pass) is computed "
                     "post hoc for description only."),
        },
    }
    h = out["this_run"]["holdout_24h"]["mean_ratio"]
    lit = {"stop_unstable": not stab_declared["stable"], "stop_resolved": h <= 1.0,
           "stop_no_help": h >= rule["definitions"]["no_help_bound"]}
    lit["run_N64_band"] = (not lit["stop_resolved"]) and (not lit["stop_no_help"])
    out["decision_rule_for_N64_literal"] = lit
    out["evaluated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out["evaluator_sha256"] = sha(Path(__file__))
    target = X / "results" / f"x1_results_{cid}.json"
    with target.open("x") as stream:
        json.dump(out, stream, indent=1)
        stream.write("\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main(sys.argv[1])
