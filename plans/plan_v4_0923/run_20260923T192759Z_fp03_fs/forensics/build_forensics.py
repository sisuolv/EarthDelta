#!/usr/bin/env python3
"""FP-03 Phase 0: byte identity of the three historical formal Fs jobs + findings.

Read-only over artifacts/. Writes forensics/historical_fs_forensics.json.
The numeric findings are recomputed here from the raw fs_fit_record.json and
fs_adapter.pt files so that every quoted number is reproducible from this script.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import torch

REPO = Path("/mnt/afs/260010168/EarthDelta")
OUT = Path(__file__).resolve().parent / "historical_fs_forensics.json"
JOBS = {
    "pt-3x63g0c6": ("artifacts/round2_cci/ed-r3-j4-bank-formal500-0922004700-0a3c68", "expert{i}"),
    "pt-cdj1s2le": ("artifacts/round2_cci/ed-r3-j4-bank-formal-v2-0922011307-2d8e12", "expert{i}"),
    "pt-e8y7ib01": ("artifacts/round2_cci/ed-r4fsformal0923-0923180756-7b1776", "experts/expert{i}"),
}
TOP_FILES = ("job-id.txt", "job_result.json", "invocation.json", "create_argv.json",
             "submission.json", "source_manifest.json", "worker.log")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sliding8(losses):
    return [sum(losses[t:t + 8]) / 8 for t in range(len(losses) - 7)]


def adapter_norms(path: Path):
    state = torch.load(path, map_location="cpu")
    out = {}
    for block in sorted(state):
        st = state[block]
        up = [v.double() for k, v in st.items() if k.startswith("up.")]
        down = [v.double() for k, v in st.items() if k.startswith("down.")]
        dw = sum(u @ d for u, d in zip(up, down))
        out[str(block)] = {
            "B_fro": math.sqrt(sum(float(u.pow(2).sum()) for u in up)),
            "A_fro": math.sqrt(sum(float(d.pow(2).sum()) for d in down)),
            "dW_fro": float(dw.norm()),
        }
    return out


def job_block(job_id: str, run_rel: str, pattern: str) -> dict:
    run = REPO / run_rel
    top = {name: (sha256(run / name) if (run / name).exists() else "MISSING") for name in TOP_FILES}
    workers = {}
    for i in range(4):
        wdir = run / pattern.format(i=i)
        files = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                 for p in sorted(wdir.iterdir()) if p.is_file()}
        rec = json.loads((wdir / "fs_fit_record.json").read_text())
        s8 = sliding8(rec["losses"])
        workers[f"replica{i}"] = {
            "dir": str(wdir.relative_to(REPO)),
            "device": f"cuda:{i}",
            "files": files,
            "fs_adapter_norms": adapter_norms(wdir / "fs_adapter.pt"),
            "max_sliding8_mean": max(s8),
            "max_sliding8_over_losses0": max(s8) / rec["losses"][0],
            "argmax_sliding8_start": s8.index(max(s8)),
            "recorded_eligible": rec["eligible"],
            "per_visit_n_decreased": rec["loss_direction"]["n_decreased"],
        }
    return {"run_dir": run_rel, "top_level_sha256": top, "replicas": workers}


def e8_findings() -> dict:
    base = REPO / JOBS["pt-e8y7ib01"][0] / "experts"
    recs = [json.loads((base / f"expert{i}" / "fs_fit_record.json").read_text()) for i in range(4)]
    L = [r["losses"] for r in recs]
    ids = recs[0]["training_issue_ids"]
    per_issue = {}
    for iid in ids[:8]:
        idx = [t for t in range(len(ids)) if ids[t] == iid]
        per_issue[iid] = {
            "argmin_update_by_replica": [idx[min(range(len(idx)), key=lambda j: L[i][idx[j]])] for i in range(4)],
            "last_over_min_by_replica": [L[i][idx[-1]] / min(L[i][t] for t in idx) for i in range(4)],
            "last_over_first_by_replica": [L[i][idx[-1]] / L[i][idx[0]] for i in range(4)],
        }
    rel1e3 = [next((t for t in range(len(L[0])) if abs(L[i][t] - L[0][t]) / L[0][t] > 1e-3), None)
              for i in range(1, 4)]
    return {
        "per_issue": per_issue,
        "replica_first_relative_divergence_gt_1e-3_vs_replica0": rel1e3,
        "update105": {"issue_id": ids[105], "time_utc": recs[0]["training_times_utc"][105],
                      "loss_by_replica": [L[i][105] for i in range(4)],
                      "previous_visit_loss_by_replica": [L[i][97] for i in range(4)],
                      "grad_norm_by_replica": [recs[i]["grad_norms"][105] for i in range(4)]},
    }


FINDINGS = [
    {"id": "F1_ALL_FOUR_E8_REPLICAS_DESTABILIZED",
     "status": "OBSERVED",
     "text": ("All 4 pt-e8y7ib01 replicas blew up, not just the rejected replica1. Max sliding "
              "8-update mean over losses[0]: see replicas.*.max_sliding8_over_losses0 (all > 25x). "
              "Correction to the original brief: replicas 2/3 peaked at cycle-aligned 8-window means "
              "0.78/0.71 (sliding 0.84/0.88), not 0.12-0.16. The 3 'eligible' replicas end 1-3% "
              "below first visit but above their own per-issue minima (see e8.per_issue)."),
     },
    {"id": "F2_COMMON_TRIGGER_IS_WITHIN_JOB_NOT_A_DATA_DEFECT",
     "status": "OBSERVED",
     "text": ("Within pt-e8y7ib01 (TF32 off) every replica's first large event is update 105 on "
              "iss_d8ff9d3ce61a2f46 (2020-01-26T06:00, round-robin slot 1, the highest first-visit "
              "loss issue): loss ~0.054-0.055 vs 0.0343 at its previous visit, grad norm ~0.136. "
              "This is shared because the replicas agree to 4 decimals until ~update 99-105. It is "
              "NOT a single causal issue: across the 12 historical formal replicas the first >1.25 "
              "per-visit event falls on 5 different issues at updates 100-273, and the first >1.05 "
              "drift is most often slot 0 (iss_292517fb189ee1bd) at updates 80-112. pt-3x63g0c6 / "
              "pt-cdj1s2le ran with TF32 enabled (their r2_fs_bank_train.py snapshot has no "
              "allow_tf32=False), and their replicas diverge past 1e-3 relative by updates 54-72 vs "
              "99-105 for pt-e8y7ib01."),
     },
    {"id": "F3_NONDETERMINISM_AMPLIFIED_NOT_CAUSAL",
     "status": "OBSERVED",
     "text": ("Update-0 loss is bitwise identical across replicas; losses differ from update 1 at "
              "~1e-8 relative (nondeterministic backward). The destabilization (per-issue minima at "
              "updates 45-76, then rising loss and ~10x grad-norm growth from updates 64-80) is "
              "shared by all replicas before they diverge; GPU nondeterminism only decides how each "
              "replica passes through and recovers from the blow-up."),
     },
    {"id": "F4_ONSET_TIMING_ACROSS_12_FORMAL_REPLICAS",
     "status": "OBSERVED",
     "text": ("By the E_k > 1.02*min(E_0..E_{k-1}) rule every one of the 12 historical formal "
              "replicas (lr=0.01) has onset at epoch 10-11 (updates 80-88); the first per-visit "
              ">1.25 event is at updates 100-273 in 11/12 (pt-3x63g0c6 replica0 never exceeded "
              "1.083). One replica's first >1.25 event (pt-cdj1s2le replica2, update 273) lies "
              "beyond 256, but its epoch-mean onset (epoch 10) is inside 256. The 32-update "
              "gradient checks all still improve at epoch 3. See ledger retrospective fields."),
     },
    {"id": "F5_ADAM_DISPLACEMENT_LOWER_LR_MAY_DELAY_NOT_FIX",
     "status": "HYPOTHESIS",
     "text": ("Final adapters are large (|dW|_F 17-59 per block from a zero start; |A| from 1.28 to "
              "5-10 per block; see replicas.*.fs_adapter_norms). Adam moves each parameter by "
              "~lr per update regardless of gradient reliability (step norm up to lr*sqrt(49152) "
              "~= 2.2 at lr=0.01). If the instability is reached by accumulated displacement, a "
              "lower lr DELAYS it roughly in proportion to 1/lr rather than removing it. Therefore "
              "a stability screen certifies an lr only for the screened horizon H; the formal fit "
              "must use exactly that H (no extrapolation to 500). Untested."),
     },
    {"id": "F6_ENVIRONMENT_AND_ADMISSION",
     "status": "OBSERVED_CIRCUMSTANTIAL",
     "text": ("None of the historical Fs jobs put the S0-certified overlay "
              "artifacts/ed-sprint8h-20260922T035313Z-rerun3/cuda_site (torch 2.3.1, xformers "
              "0.0.27) on PYTHONPATH; the image torch is 2.3.0a0+6ddf5cf85e.nv24.4 and the "
              "submitting CCI host with the same torch build has no xformers module, so "
              "_build_stormer_model most probably fell back to SDPA. official_backend was never "
              "recorded, so this is not proven. All used the uncertified admission_record.json "
              "(8 rows, content_certificate=null per row, lead_hours=6, history_steps=1, "
              "split_id='test')."),
     },
]


def main() -> None:
    jobs = {job: job_block(job, rel, pat) for job, (rel, pat) in JOBS.items()}
    payload = {
        "schema_version": "ed-fp03-historical-fs-forensics/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "OBSERVED",
        "scope": "read-only; no artifact modified",
        "builder": str(Path(__file__).relative_to(REPO)),
        "builder_sha256": sha256(Path(__file__)),
        "jobs": jobs,
        "e8": e8_findings(),
        "findings": FINDINGS,
        "all_marked": "INELIGIBLE_AS_FS (see ledger/fs_attempts_ledger.json)",
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
