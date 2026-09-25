#!/usr/bin/env python3
"""Recompute the FP-05 score as an append-only lead-specific erratum.

This tool consumes the already sealed cache and prediction freeze. It never
rewrites the historical evaluation directory and records both the frozen
protocol pins and the scorer source used for the correction. It exists because
the original score reused the 24h F0 background correction for 6h and 72h.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from earthdelta import candidate_cache as cc
from earthdelta import policy_oof as po
import scripts.r4_candidate_cache as rc


SOURCE_ROOT = Path(__file__).resolve().parent.parent
SCORER_FILES = ("earthdelta/policy_oof.py", "scripts/r4_policy_oof.py")


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _git_head() -> str | None:
    cfg = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        cfg.write("[safe]\n\tdirectory = /mnt/afs/260010168/EarthDelta\n")
        cfg.close()
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = cfg.name
        return subprocess.check_output(
            ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True, env=env
        ).strip()
    except Exception:
        return None
    finally:
        try:
            os.unlink(cfg.name)
        except OSError:
            pass


def _source_record() -> Dict[str, Any]:
    return {
        "head": _git_head(),
        "files": {rel: cc.file_sha256(SOURCE_ROOT / rel) for rel in SCORER_FILES},
    }


def _ref(ref: Dict[str, Any], label: str) -> Path:
    if not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256"):
        raise ValueError(f"{label} reference is incomplete")
    path = Path(ref["path"])
    actual = cc.file_sha256(path) if path.is_file() else None
    if actual != ref["sha256"]:
        raise ValueError(f"{label} hash mismatch: {path}")
    return path


def _bind(args: argparse.Namespace) -> Dict[str, Any]:
    protocol_path = Path(args.protocol)
    protocol = rc.load_cache_protocol(protocol_path, args.protocol_sha256)
    freeze = rc.bind_freeze(protocol)
    if freeze.role("confirm").get("status") != "UNASSIGNED_NO_ACCESS":
        raise ValueError("confirm is assigned")
    manifest_path = Path(args.cache_manifest)
    manifest = _json(manifest_path)
    if manifest.get("passed") is not True:
        raise ValueError("cache manifest is not PASS")
    if manifest.get("protocol_sha256") != args.protocol_sha256:
        raise ValueError("cache manifest is bound to another protocol")
    if manifest.get("split_freeze_sha256") != freeze.sha256:
        raise ValueError("cache manifest is bound to another split freeze")
    allow = rc.allow_list(protocol, freeze, "policy_dev")
    ids = list(manifest.get("issue_ids") or [])
    if len(ids) != len(set(ids)) or sorted(ids) != sorted(allow.issue_ids):
        raise ValueError("cache issue ids do not equal the policy_dev allowlist")
    if not (manifest.get("validation") or {}).get("passed"):
        raise ValueError("cache validation did not pass")
    if not (manifest.get("read_set") or {}).get("passed"):
        raise ValueError("cache read-set did not pass")
    dev_path = _ref(manifest["dev_protocol"], "dev protocol")
    _ref(manifest["candidate_results"], "candidate results")
    bg_path = _ref(manifest["background_f0"], "background F0")
    freeze_rec = _json(Path(args.prediction_freeze))
    for key in ("folds", "oof_predictions", "label_access_log"):
        _ref(freeze_rec[key], key)
    for ref in freeze_rec.get("per_fold_files", {}).values():
        _ref(ref, "per-fold predictions")
    if freeze_rec.get("protocol_sha256") != args.protocol_sha256:
        raise ValueError("prediction freeze protocol mismatch")
    if freeze_rec.get("cache_manifest_sha256") != cc.file_sha256(manifest_path):
        raise ValueError("prediction freeze cache manifest mismatch")
    preds = _json(Path(freeze_rec["oof_predictions"]["path"]))
    if preds.get("poisoned") is not None or sorted(preds.get("predictions", {})) != sorted(ids):
        raise ValueError("prediction freeze is poisoned or incomplete")
    protocol_split = protocol["split"]["coverage_caveat"]
    return {
        "protocol": protocol,
        "freeze": freeze,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "dev_path": dev_path,
        "background": _json(bg_path),
        "prediction_freeze": freeze_rec,
        "predictions": preds["predictions"],
        "ids": ids,
        "coverage_caveat": protocol_split,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    b = _bind(args)
    rows = _json(Path(b["manifest"]["candidate_results"]["path"]))["rows"]
    bg = b["background"]["issues"]
    f0 = {iid: rec["f0_cpu"] for iid, rec in bg.items()}
    real = po.realized_losses(rows, b["predictions"], f0)
    # Issue times are stored in the admission rows, not in the matrix. Reuse
    # the protocol's role admission record to reconstruct the exact mapping.
    admission_ref = b["protocol"]["inputs"]["policy_dev"]["admission"]
    admission = _json(_ref(admission_ref, "policy_dev admission"))
    times = {r["issue_id"]: int(r["issue_time"]) for r in admission["admission"]["admitted"]}
    blocks = {iid: po.block_of(times[iid]) for iid in b["ids"]}
    boot7 = po.paired_block_bootstrap(real["losses"], blocks)
    boot14 = po.paired_block_bootstrap(real["losses"], blocks, block_days_factor=2)
    h_by_lead = {int(k): v["point"] for k, v in boot7["H_Fs_by_lead"].items()}
    for comp in boot7["comparisons"].values():
        comp["G_F0_point"] = (
            comp["point"] - h_by_lead[int(comp["lead_hours"])]
            if comp["baseline"] == "M0" else None
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    old = _json(Path(args.old_evaluation)) if args.old_evaluation else None
    old_comp = ((old or {}).get("primary_7day_blocks") or {}).get("comparisons", {})
    unchanged_24h = {}
    for name, comp in boot7["comparisons"].items():
        if int(comp["lead_hours"]) == 24 and name in old_comp:
            unchanged_24h[name] = {
                "old_status": old_comp[name].get("status_vs_delta_min"),
                "new_status": comp.get("status_vs_delta_min"),
                "same_status": old_comp[name].get("status_vs_delta_min") == comp.get("status_vs_delta_min"),
            }
    result = {
        "schema": "ed-fp05-lead-specific-erratum/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {"path": str(Path(args.protocol).resolve()), "sha256": args.protocol_sha256},
        "cache_manifest": {"path": str(b["manifest_path"].resolve()), "sha256": cc.file_sha256(b["manifest_path"])},
        "prediction_freeze": {"path": str(Path(args.prediction_freeze).resolve()), "sha256": cc.file_sha256(args.prediction_freeze)},
        "source_used": _source_record(),
        "formula": {
            "H_h": "(sum(L_Fs,h) - sum(L_F0,h)) / sum(L_Fs,h)",
            "G_F0_h": "G_Fs_h - H_h",
            "affected_leads": [6, 72],
            "unaffected_primary_lead": 24,
        },
        "coverage_caveat": b["coverage_caveat"],
        "primary_7day_blocks": boot7,
        "sensitivity_14day_blocks_report_only": boot14,
        "unchanged_24h_status": unchanged_24h,
    }
    cc.write_json_once(out / "effects_by_lead.json", result)
    cc.write_json_once(out / "paired_F0_intervals.json", {
        "H_Fs_by_lead": boot7["H_Fs_by_lead"],
        "coverage_caveat": b["coverage_caveat"],
        "source_used": result["source_used"],
    })
    cc.write_json_once(out / "gate.json", {
        "schema": "ed-fp05-erratum-gate/1",
        "passed": all(v["same_status"] for v in unchanged_24h.values()) if unchanged_24h else False,
        "checks": {"exact_allowlist": True, "cache_passed": True, "read_set_passed": True,
                   "prediction_complete": True, "unaffected_24h_status": unchanged_24h},
    })
    (out / "ERRATUM.md").write_text(
        "# FP-05 lead-specific score erratum\n\n"
        "The original score reused the 24h F0 background correction for every lead. "
        "This append-only result recomputes `H_Fs` and `G_F0` separately for 6h, 24h, and 72h. "
        "The historical evaluation directory is untouched. The 24h status is checked against "
        "the supplied historical result; this correction does not change the primary 24h claim.\n"
    )
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--cache-manifest", type=Path, required=True)
    ap.add_argument("--prediction-freeze", type=Path, required=True)
    ap.add_argument("--old-evaluation", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed with no output
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps({"out": str(args.out), "H_Fs_by_lead": result["primary_7day_blocks"]["H_Fs_by_lead"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
