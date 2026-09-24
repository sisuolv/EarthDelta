#!/usr/bin/env python3
"""FP-03 Phase 0: catalogue every historical fs_fit_record.json (read-only).

Reads only; writes ledger/fs_attempts_ledger.json next to this script. Every
record is marked INELIGIBLE_AS_FS. The retrospective stability numbers are
DESCRIPTIVE: they were computed after the fact and are never a qualification.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import statistics
from pathlib import Path

REPO = Path("/mnt/afs/260010168/EarthDelta")
OUT = Path(__file__).resolve().parent / "fs_attempts_ledger.json"
OLD_ADMISSION = REPO / "artifacts/round2_next/20260921T170142Z_r3p001_10h/j3_fs_bank/admission_record.json"
OVERLAY = "artifacts/ed-sprint8h-20260922T035313Z-rerun3/cuda_site"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir_of(record: Path) -> Path:
    for parent in record.parents:
        if (parent / "invocation.json").exists():
            return parent
    raise RuntimeError(f"no invocation.json above {record}")


def retrospective(rec: dict) -> dict:
    """Descriptive stability numbers from the recorded per-update trace."""
    losses, grads, ids = rec["losses"], rec["grad_norms"], rec["training_issue_ids"]
    n = len(losses)
    first = {}
    for loss, iid in zip(losses, ids):
        first.setdefault(iid, loss)
    clip = rec["config"].get("grad_clip")
    per_visit_ratio = [loss / first[iid] for loss, iid in zip(losses, ids)]
    epochs = [losses[k:k + 8] for k in range(0, n - n % 8, 8)]
    e_means = [sum(e) / 8 for e in epochs]
    onset_epoch = None
    for k in range(1, len(e_means)):
        if e_means[k] > 1.02 * min(e_means[:k]):
            onset_epoch = k
            break
    sliding = [sum(losses[t:t + 8]) / 8 for t in range(0, max(0, n - 7))]
    return {
        "note": ("DESCRIPTIVE ONLY, computed after the fact. L0 per issue is the "
                 "FIRST VISIT loss (only update 0 is exactly zero-init/F0), so "
                 "per-visit ratios are an approximation of FS-QUAL-v1 Q2."),
        "n_updates": n,
        "clip_events_pre_clip_norm_gt_clip": (sum(1 for g in grads if clip is not None and g > clip)),
        "max_grad_norm": max(grads) if grads else None,
        "argmax_grad_norm_update": grads.index(max(grads)) if grads else None,
        "max_per_visit_ratio_vs_first_visit": max(per_visit_ratio) if per_visit_ratio else None,
        "argmax_per_visit_ratio_update": (per_visit_ratio.index(max(per_visit_ratio))
                                          if per_visit_ratio else None),
        "epoch_means": e_means,
        "argmin_epoch_mean": e_means.index(min(e_means)) if e_means else None,
        "onset_epoch_Ek_gt_1p02_min_prev": onset_epoch,
        "onset_update": onset_epoch * 8 if onset_epoch is not None else None,
        "max_sliding8_mean_over_losses0": (max(sliding) / losses[0]) if sliding else None,
        "median_grad_norm_updates_8_39": (statistics.median(grads[8:40]) if len(grads) >= 40 else None),
    }


def main() -> None:
    records = sorted(REPO.glob("artifacts/**/fs_fit_record.json"))
    adm_sha = sha256(OLD_ADMISSION)
    entries = []
    for path in records:
        rec = json.loads(path.read_text())
        run = run_dir_of(path)
        inv = json.loads((run / "invocation.json").read_text())
        cmd = " ".join(inv["argv"])
        job_result = json.loads((run / "job_result.json").read_text()) if (run / "job_result.json").exists() else None
        job_id = (run / "job-id.txt").read_text().strip() if (run / "job-id.txt").exists() else None
        stdout = path.parent / "stdout.log"
        siblings = {p.name: sha256(p) for p in sorted(path.parent.iterdir()) if p.is_file()}
        direction = rec.get("loss_direction", {})
        reasons = [
            "ENV_BACKEND_UNCERTIFIED: invocation PYTHONPATH lacks the S0-certified "
            f"overlay {OVERLAY} (torch 2.3.1 + xformers 0.0.27); official_backend "
            "was never recorded; the image torch is 2.3.0a0 without xformers, so the "
            "bridge's silent SDPA fallback is probable (circumstantial, not proven)",
            "ADMISSION_UNCERTIFIED: consumed admission_record.json sha256 "
            f"{adm_sha[:16]}..: 8 rows, content_certificate=null on every row, 2 "
            "top-level certificates covering only t-6/t/t+6, lead_hours=6 but "
            "consumed at 24h, history_steps=1 (REQUIRED_INPUTS.md: 不可直接拿它训练)",
            "NO_PREREGISTRATION: quality rule, LR, selection and retry budget were "
            "not frozen before the run (STOP_CONDITIONS section 1/3)",
        ]
        if rec["mode"] != "formal":
            reasons.append(f"MODE_NOT_FORMAL: mode={rec['mode']} (diagnostic only)")
        if direction.get("n_samples_with_repeats") and direction.get("n_decreased") != direction.get("n_samples_with_repeats"):
            reasons.append(
                "PER_VISIT_REGRESSION: "
                f"{direction.get('n_decreased')}/{direction.get('n_samples_with_repeats')} "
                "issues ended below their first visit")
        retro = retrospective(rec)
        if retro["onset_epoch_Ek_gt_1p02_min_prev"] is not None or retro["clip_events_pre_clip_norm_gt_clip"]:
            reasons.append(
                "TRAINING_INSTABILITY_OBSERVED (retrospective): onset epoch "
                f"{retro['onset_epoch_Ek_gt_1p02_min_prev']}, clip events "
                f"{retro['clip_events_pre_clip_norm_gt_clip']}")
        entries.append({
            "path": str(path.relative_to(REPO)),
            "sha256": sha256(path),
            "run_dir": str(run.relative_to(REPO)),
            "job_id": job_id,
            "job_status": job_result.get("status") if job_result else "MISSING",
            "job_returncode": job_result.get("returncode") if job_result else None,
            "device_flag": (re.findall(r"--device (\S+)", cmd) or [None])[0],
            "stage_flag": (re.findall(r"--stage (\w+)", cmd) or [None])[0],
            "pythonpath_has_certified_overlay": "cuda_site" in cmd,
            "stdout_sha256": sha256(stdout) if stdout.exists() else "MISSING",
            "sibling_files_sha256": siblings,
            "config": rec["config"],
            "n_updates": rec["n_updates"],
            "recorded_eligible": rec["eligible"],
            "recorded_ineligible_reason": rec["ineligible_reason"],
            "recorded_quality_gate": rec.get("quality_gate", "ABSENT_PRE_GATE_CODE"),
            "per_visit_direction": {k: direction.get(k) for k in
                                    ("n_samples_with_repeats", "n_decreased", "fraction_decreased")},
            "retrospective_stability": retro,
            "exposure": {
                "data_role": rec.get("data_role"),
                "admission_record": str(OLD_ADMISSION.relative_to(REPO)),
                "admission_sha256": adm_sha,
                "training_issue_ids_unique": sorted(set(rec["training_issue_ids"])),
                "training_times_utc_unique": sorted(set(rec["training_times_utc"])),
                "policy_dev_or_confirm_read": False,
                "downstream_artifacts_in_same_dir": sorted(
                    n for n in siblings if n not in {"fs_fit_record.json", "stdout.log", "run_record.json"}),
            },
            "verdict": "INELIGIBLE_AS_FS",
            "ineligibility_reasons": reasons,
        })
    ledger = {
        "schema_version": "ed-fp03-fs-attempts-ledger/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "OBSERVED",
        "n_records": len(entries),
        "rule": ("Every historical Fs fit record is INELIGIBLE_AS_FS. None may be "
                 "substituted for a future FS-SELECT-v1 selection (clause 4)."),
        "builder": str(Path(__file__).relative_to(REPO)),
        "builder_sha256": sha256(Path(__file__)),
        "entries": entries,
    }
    OUT.write_text(json.dumps(ledger, indent=1) + "\n")
    print(f"wrote {OUT} ({len(entries)} records)")


if __name__ == "__main__":
    main()
