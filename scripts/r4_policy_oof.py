#!/usr/bin/env python3
"""FP-05 cheap policies, out of fold: folds -> fit-predict -> score (CPU only).

    folds       --protocol P --protocol-sha256 S --cache-manifest M --out policies/folds.json
    fit-predict --protocol P --protocol-sha256 S --cache-manifest M --folds F --out-dir policies
                [--poison-eval-labels-of-fold f --poison-value nan|1e9]   (CP3 re-check only;
                 poisons that fold's eval and purged labels)
    score       --protocol P --protocol-sha256 S --cache-manifest M
                --prediction-freeze policies/prediction_freeze.json --out-dir evaluation

Every stage: protocol by hash, split freeze by hash, cache manifest PASS and
bound to this protocol, every issue policy_dev-clear AND on the protocol's
allow-list, confirm UNASSIGNED_NO_ACCESS -- all BEFORE any feature or label
file is opened. fit-predict sees labels only through FoldLabels; score refuses
without a matching prediction_freeze.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402

import scripts.r4_candidate_cache as rc  # noqa: E402
from earthdelta import candidate_cache as cc  # noqa: E402
from earthdelta import policy_oof as po  # noqa: E402
from earthdelta import split_freeze as sf  # noqa: E402


# The original FP-05b protocol predates the scorer fixes made after the
# historical run.  A new protocol must pin the scorer itself; otherwise a
# changed scorer could consume an old cache while still presenting the old
# protocol hash as provenance.
SCORER_SOURCE_FILES = ("earthdelta/policy_oof.py", "scripts/r4_policy_oof.py")


def _json(p) -> Dict[str, Any]:
    return json.loads(Path(p).read_text())


def _source_contract(protocol: Dict[str, Any]) -> Dict[str, str]:
    pins = protocol.get("source_at_preregistration") or {}
    missing = [rel for rel in SCORER_SOURCE_FILES if rel not in pins]
    if missing:
        raise rc.Refused(
            "SCORER_SOURCE_PIN_MISSING",
            "protocol does not pin the scorer sources",
            {"missing": missing},
        )
    # Reuse the cache protocol's complete source check for the certified core,
    # then explicitly include the two files that implement OOF and scoring.
    rc.check_sources(protocol)
    drift = {
        rel: {"expected": pins[rel], "actual": cc.file_sha256(SOURCE_ROOT / rel)}
        for rel in SCORER_SOURCE_FILES
        if cc.file_sha256(SOURCE_ROOT / rel) != pins[rel]
    }
    if drift:
        raise rc.Refused("SCORER_SOURCE_DRIFT", "scorer source differs from protocol", {"files": drift})
    return {rel: pins[rel] for rel in SCORER_SOURCE_FILES}


def _check_manifest_binding(manifest: Dict[str, Any], manifest_path: Path,
                            protocol_sha256: str, freeze: sf.SplitFreeze,
                            allow: sf.AllowList) -> Dict[str, Any]:
    if manifest.get("schema") != cc.CACHE_SCHEMA or manifest.get("kind") != "dev":
        raise rc.Refused("CACHE_SCHEMA_MISMATCH", "expected a sealed dev cache manifest")
    if manifest.get("passed") is not True:
        raise rc.Refused("CACHE_NOT_VALID", "cache manifest is not a PASS")
    if manifest.get("protocol_sha256") != protocol_sha256:
        raise rc.Refused("CACHE_PROTOCOL_MISMATCH", "cache manifest is bound to another protocol")
    if manifest.get("split_freeze_sha256") != freeze.sha256:
        raise rc.Refused("CACHE_FREEZE_MISMATCH", "cache built under another split freeze")
    vp = (manifest.get("validation") or {}).get("v_pass") or {}
    if not vp or not all(v is True for v in vp.values()) or not (manifest.get("read_set") or {}).get("passed"):
        raise rc.Refused("CACHE_VALIDATION_INCOMPLETE", "V1-V8 and read-set must all pass")
    ids = list(manifest.get("issue_ids") or [])
    if len(ids) != len(set(ids)) or sorted(ids) != sorted(allow.issue_ids):
        raise rc.Refused("CACHE_ISSUE_ALLOWLIST_MISMATCH", "manifest ids are not exactly policy_dev allowlist")
    if manifest.get("identity", {}).get("protocol_sha256") != protocol_sha256:
        raise rc.Refused("CACHE_IDENTITY_MISMATCH", "manifest identity is not protocol-bound")
    for key in ("dev_protocol", "candidate_results", "background_f0"):
        ref = manifest.get(key)
        if not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256"):
            raise rc.Refused("CACHE_REFERENCE_MISSING", f"manifest reference {key} is incomplete")
        path = Path(ref["path"])
        if not path.is_file() or cc.file_sha256(path) != ref["sha256"]:
            raise rc.Refused("CACHE_REFERENCE_MISMATCH", f"manifest reference {key} changed")
    return {"manifest_sha256": cc.file_sha256(manifest_path), "issue_ids": ids}


def bind(args) -> Dict[str, Any]:
    """Everything checked before any feature/label byte is read."""
    protocol = rc.load_cache_protocol(args.protocol, args.protocol_sha256)
    scorer_sources = _source_contract(protocol)
    freeze = rc.bind_freeze(protocol)
    if freeze.role("confirm").get("status") != sf.CONFIRM_STATUS:
        raise rc.Refused("CONFIRM_ASSIGNED", "confirm must be UNASSIGNED_NO_ACCESS")
    manifest_path = Path(args.cache_manifest)
    manifest = _json(manifest_path)
    allow = rc.allow_list(protocol, freeze, "policy_dev")
    manifest_binding = _check_manifest_binding(manifest, manifest_path, args.protocol_sha256, freeze, allow)
    record, admission_sha = rc.role_admission(protocol, "policy_dev")
    ids = manifest_binding["issue_ids"]
    pairs = cc.rows_and_certificates(record, ids)
    rc.clear_rows(freeze, "policy_dev", pairs, allow, admission_sha)
    times = {r["issue_id"]: int(r["issue_time"]) for r, _ in pairs}
    return {"protocol": protocol, "freeze": freeze, "manifest": manifest,
            "manifest_sha256": manifest_binding["manifest_sha256"], "ids": ids, "times": times,
            "scorer_sources": scorer_sources}


def stage_folds(args, b) -> int:
    folds = po.build_folds(b["times"])
    if not po.check_no_overlap(folds, b["times"]):
        raise rc.Refused("PURGE_FAILED", "support overlap after purge")
    folds.update({"protocol_sha256": args.protocol_sha256, "cache_manifest_sha256": b["manifest_sha256"],
                  "coverage_caveat": b["protocol"]["split"]["coverage_caveat"]})
    cc.write_json_once(args.out, folds)
    print(json.dumps({"n_issues": len(b["ids"]), "fold_sizes": [len(f["eval_ids"]) for f in folds["folds"]],
                      "train_sizes": [len(f["train_ids"]) for f in folds["folds"]],
                      "purged": [len(f["purged_ids"]) for f in folds["folds"]],
                      "empty_blocks": folds["empty_blocks"]}))
    return 0


def _load_features(manifest) -> Dict[str, np.ndarray]:
    out = {}
    for iid, ref in manifest["features"].items():
        if cc.file_sha256(ref["path"]) != ref["sha256"]:
            raise rc.Refused("FEATURE_FILE_MISMATCH", iid)
        out[iid] = np.load(ref["path"], allow_pickle=False)
    return out


def _load_rows(manifest):
    ref = manifest["candidate_results"]
    if cc.file_sha256(ref["path"]) != ref["sha256"]:
        raise rc.Refused("CANDIDATE_RESULTS_MISMATCH", "candidate_results.json changed")
    return _json(ref["path"])["rows"]


def _poison_fold_labels(table, fold, value):
    """Copy ``table`` with both eval and purged labels replaced for CP3."""
    out = {iid: dict(labels) for iid, labels in table.items()}
    ids = list(fold["eval_ids"]) + list(fold["purged_ids"])
    for iid in ids:
        out[iid] = {c: value for c in po.CANDIDATES}
    return out, ids


def stage_fit_predict(args, b) -> int:
    folds_path = Path(args.folds)
    folds = _json(folds_path)
    rebuilt = po.build_folds(b["times"])
    if any(folds.get(k) != v for k, v in rebuilt.items()) or \
            folds.get("cache_manifest_sha256") != b["manifest_sha256"]:
        raise rc.Refused("FOLDS_NOT_REPRODUCIBLE", "folds.json differs from build_folds(issue times)")
    features = _load_features(b["manifest"])
    table = po.realized_gain_table(_load_rows(b["manifest"]))
    poisoned = None
    if args.poison_eval_labels_of_fold is not None:
        f = int(args.poison_eval_labels_of_fold)
        value = float(args.poison_value)
        table, poisoned_ids = _poison_fold_labels(table, folds["folds"][f], value)
        poisoned = {"fold": f, "value": args.poison_value,
                    "scope": "eval_and_purged_ids; only this fold is refit: its own OOF predictions must be byte-identical",
                    "n_eval_ids": len(folds["folds"][f]["eval_ids"]),
                    "n_purged_ids": len(folds["folds"][f]["purged_ids"]),
                    "n_poisoned_ids": len(poisoned_ids)}
    run_folds = folds if poisoned is None else {"folds": [folds["folds"][poisoned["fold"]]]}
    per_fold, access = po.fit_predict(run_folds, features, table)
    for fo in run_folds["folds"]:
        if not set(access[str(fo["fold"])]) <= set(fo["train_ids"]):
            raise rc.Refused("LABEL_ACCESS_OUTSIDE_TRAIN", str(fo["fold"]))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for fr in per_fold:
        p = out / f"oof_predictions_fold{fr['fold']}.json"
        cc.write_json_once(p, fr)
        files[f"fold{fr['fold']}"] = {"path": str(p), "sha256": cc.file_sha256(p)}
    combined = {"schema": "ed-fp05-oof-predictions/1", "protocol_sha256": args.protocol_sha256,
                "methods": {k: po.METHOD_NAMES[k] for k in ("M0", "M1", "M2", "M3")},
                "per_fold_files": files, "poisoned": poisoned,
                "predictions": {iid: {**pred, "fold": fr["fold"]} for fr in per_fold
                                for iid, pred in fr["predictions"].items()}}
    cc.write_json_once(out / "oof_predictions.json", combined)
    cc.write_json_once(out / "label_access_log.json", {"per_fold_train_ids_read": access})
    if poisoned is None:
        freeze_rec = {
            "schema": "ed-fp05-prediction-freeze/1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "protocol_sha256": args.protocol_sha256, "cache_manifest_sha256": b["manifest_sha256"],
            "dev_protocol_sha256": b["manifest"]["dev_protocol"]["sha256"],
            "folds": {"path": str(folds_path.resolve()), "sha256": cc.file_sha256(folds_path)},
            "oof_predictions": {"path": str(out / "oof_predictions.json"),
                                "sha256": cc.file_sha256(out / "oof_predictions.json")},
            "per_fold_files": files,
            "label_access_log": {"path": str(out / "label_access_log.json"),
                                 "sha256": cc.file_sha256(out / "label_access_log.json")},
            "features_manifest_digest": cc.canonical_digest(b["manifest"]["features"]),
            "source": {rel: cc.file_sha256(SOURCE_ROOT / rel) for rel in
                       ("earthdelta/policy_oof.py", "scripts/r4_policy_oof.py")}}
        with (out / "prediction_freeze.json").open("x") as f:
            json.dump(freeze_rec, f, indent=1, sort_keys=True)
            f.write("\n")
    print(json.dumps({"folds": len(per_fold), "poisoned": poisoned,
                      "hpo": {fr["fold"]: [fr["hpo"]["lambda"], fr["hpo"]["k"]] for fr in per_fold}}))
    return 0


def stage_score(args, b) -> int:
    fz_path = Path(args.prediction_freeze)
    if not fz_path.is_file():
        raise rc.Refused("PREDICTION_FREEZE_MISSING", "score runs only on frozen predictions")
    fz = _json(fz_path)
    if fz.get("protocol_sha256") != args.protocol_sha256 or \
            fz.get("cache_manifest_sha256") != b["manifest_sha256"]:
        raise rc.Refused("PREDICTION_FREEZE_MISMATCH", "freeze bound to another protocol/cache")
    for key in ("folds", "oof_predictions", "label_access_log", *[f"per_fold_files.{k}" for k in fz["per_fold_files"]]):
        ref = fz[key] if "." not in key else fz["per_fold_files"][key.split(".")[1]]
        if cc.file_sha256(ref["path"]) != ref["sha256"]:
            raise rc.Refused("PREDICTION_FREEZE_MISMATCH", f"{key} changed after the freeze")
    preds = _json(fz["oof_predictions"]["path"])
    if preds.get("poisoned") is not None or sorted(preds["predictions"]) != sorted(b["ids"]):
        raise rc.Refused("PREDICTIONS_INVALID", "poisoned or incomplete predictions")
    rows = _load_rows(b["manifest"])
    bg = _json(b["manifest"]["background_f0"]["path"])["issues"]
    f0 = {i: v["f0_cpu"] for i, v in bg.items()}
    real = po.realized_losses(rows, preds["predictions"], f0)
    blocks = {i: po.block_of(b["times"][i]) for i in b["ids"]}
    boot7 = po.paired_block_bootstrap(real["losses"], blocks)
    boot14 = po.paired_block_bootstrap(real["losses"], blocks, block_days_factor=2)
    h_by_lead = {int(k): v["point"] for k, v in boot7["H_Fs_by_lead"].items()}
    for c in boot7["comparisons"].values():
        c["G_F0_point"] = (c["point"] - h_by_lead[int(c["lead_hours"])]
                            if c["baseline"] == "M0" else None)
    caveat = b["protocol"]["split"]["coverage_caveat"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cc.write_json_once(out / "paired_block_bootstrap.json",
                       {"primary_7day_blocks": boot7, "sensitivity_14day_blocks_report_only": boot14,
                        "coverage_caveat": caveat, "M5_static_choice": real["M5_static_choice"],
                        "prediction_freeze_sha256": cc.file_sha256(fz_path)})
    with (out / "dev_results.csv").open("x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# coverage caveat: " + caveat["text"]])
        w.writerow(["comparison", "method", "baseline", "lead_hours", "point", "ci_low", "ci_high",
                    "status_vs_delta_min", "harm72_guard_pass", "G_F0_point"])
        for name, c in boot7["comparisons"].items():
            w.writerow([name, c["method"], c["baseline"], c["lead_hours"], repr(c["point"]),
                        repr(c["ci_low"]), repr(c["ci_high"]), c["status_vs_delta_min"],
                        c.get("harm72_guard_pass", ""), c.get("G_F0_point")])
    cc.write_json_once(out / "failures.json", {"failures_24h_issue_count": real["failures_issue_count"],
                                               "rates": po.rates(real["actions"], rows)})
    secs = [r["seconds"] for r in rows if r.get("seconds") is not None and r["lead_hours"] == 24]
    cc.write_json_once(out / "costs.json", {
        "gpu_seconds_per_rollout_mean": float(np.mean(secs)) if secs else None,
        "online_cost": "features (CPU, 3 slabs) + 1 rollout", "offline_cost_note":
        "cache (5 rollouts/issue + plain-Fs + F0 report-only) + CPU fitting; FP-04 job seconds in its ledger"})
    print(json.dumps({"statuses": {k: v["status_vs_delta_min"] for k, v in boot7["comparisons"].items()},
                      "coverage_caveat": caveat["short"]}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("folds", "fit-predict", "score"):
        p = sub.add_parser(name)
        p.add_argument("--protocol", type=Path, required=True)
        p.add_argument("--protocol-sha256", required=True)
        p.add_argument("--cache-manifest", type=Path, required=True)
        if name == "folds":
            p.add_argument("--out", type=Path, required=True)
        if name == "fit-predict":
            p.add_argument("--folds", type=Path, required=True)
            p.add_argument("--out-dir", type=Path, required=True)
            p.add_argument("--poison-eval-labels-of-fold", type=int, default=None)
            p.add_argument("--poison-value", default="nan")
        if name == "score":
            p.add_argument("--prediction-freeze", type=Path, required=True)
            p.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        b = bind(args)
        return {"folds": stage_folds, "fit-predict": stage_fit_predict, "score": stage_score}[args.stage](args, b)
    except (rc.Refused, sf.SplitFreezeViolation, cc.CacheViolation, po.PolicyViolation) as exc:
        print(f"ERROR: refused: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
