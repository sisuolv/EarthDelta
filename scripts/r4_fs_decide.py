#!/usr/bin/env python3
"""FP-03 decisions from raw Fs job outputs (CPU only; never trains, never submits).

    --kind diag     J1: Q0 facts + code-path checks for the 32-update arms, the
                    reference F0 panel, and the negative controls (known-bad
                    historical adapters must be REJECTED or everything stops)
    --kind screen   J2: FS-SCREEN-v1 over arms A0..A3
    --kind formal   J3/J3b: FS-QUAL-v1 per replica + FS-SELECT-v1
    --kind certify  J4: final FS-SELECT-v1 with the designated candidate's
                    merge/reload/identity gates, and the fs/ evidence bundle

Every decision is recomputed from the per-process files (fs_fit_record.json,
fs_panels.json, fs_diagnostics.json, run_record.json) and the pre-registered
protocol, whose SHA-256 must match. Rules are applied exactly; nothing here
ranks processes by a quality number.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from earthdelta import fs_protocol as fp  # noqa: E402
from earthdelta import static_adapter as sa  # noqa: E402


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _rel_diff(a: float, b: float) -> float:
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def compare_panels(panel: Mapping[str, Mapping[str, float]],
                   reference: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
    """Bitwise equality and max relative difference of two F0 panels."""
    if set(panel) != set(reference):
        return {"same_issues": False, "bitwise_equal": False, "max_rel_diff": None}
    diffs = [_rel_diff(float(panel[i][k]), float(reference[i][k]))
             for i in reference for k in reference[i]]
    return {"same_issues": True, "bitwise_equal": panel == reference,
            "max_rel_diff": max(diffs) if diffs else 0.0}


def process_files(proc_dir: Path) -> Dict[str, Any]:
    files = {}
    for name in ("fs_fit_record.json", "fs_panels.json", "fs_diagnostics.json",
                 "run_record.json", "provenance.json", "fs_panel_record.json",
                 "binding_refused.json", "stdout.log", "fs_adapter.pt"):
        path = proc_dir / name
        files[name] = {"path": str(path), "sha256": fp.file_sha256(path)} if path.is_file() else None
    return files


def validity_facts(proc_dir: Path, protocol: Mapping[str, Any], protocol_sha: str,
                   config_id: str, reference_f0: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """FS-QUAL-v1 Q0 facts for one real process, each a strict boolean."""
    run = _load(proc_dir / "run_record.json")
    prov = run.get("provenance") or {}
    binding = run.get("binding") or {}
    q0 = protocol["qualification_rule"]["q0"]
    inputs = protocol["inputs"]
    facts: Dict[str, bool] = {}
    facts["official_backend"] = bool(prov.get("official_backend") is True and
                                     (binding.get("backend_postcondition") or {}).get("passed") is True)
    versions = prov.get("versions") or {}
    facts["torch_version"] = versions.get("torch") == q0["torch_version"]
    facts["xformers_version"] = versions.get("xformers") == q0["xformers_version"]
    facts["tf32_off"] = fp.tf32_is_off(prov)
    s0 = binding.get("s0") or {}
    facts["s0_certificate_consumed"] = bool(s0.get("passed") is True and
                                            s0.get("sha256") == inputs["s0_certificate"]["sha256"])
    adm = binding.get("admission_consumer") or {}
    panel_adm = binding.get("panel_admission_consumer") or {}
    in_files = prov.get("input_files") or {}
    facts["admission_consumer_pass"] = bool(
        adm.get("passed") is True and panel_adm.get("passed") is True
        and (in_files.get("admission") or {}).get("sha256") == inputs["admission"]["sha256"]
        and (in_files.get("panel_admission") or {}).get("sha256") == inputs["panel_admission"]["sha256"])
    facts["protocol_sha256_match"] = run.get("protocol_sha256") == protocol_sha
    facts["config_id_match"] = run.get("config_id") == config_id
    diag_path = proc_dir / "fs_diagnostics.json"
    if diag_path.is_file():
        diag = _load(diag_path)
        facts["init_digest_match"] = diag.get("initial_adapter_digest") == inputs["initial_adapter_digest"]
    else:
        rec = _load(proc_dir / "fs_panel_record.json")
        facts["init_digest_match"] = rec.get("initial_adapter_digest") == inputs["initial_adapter_digest"]
    panels = _load(proc_dir / "fs_panels.json")
    # FS-QUAL-v1: L0 is the zero-init Fs, bitwise equal to F0.
    facts["L0_equals_F0_bitwise"] = all(panels[g]["L0"] == panels[g]["F0"]
                                        for g in ("train", "holdout"))
    rec_path = proc_dir / "fs_fit_record.json"
    if rec_path.is_file() and inputs.get("train_issue_ids") is not None:
        ids = _load(rec_path)["training_issue_ids"]
        order = list(inputs["train_issue_ids"])
        facts["issue_order_match"] = ids[:len(order)] == order and len(set(ids)) == len(order)
    if inputs.get("holdout_issue_ids") is not None:
        facts["holdout_panel_match"] = sorted(panels["holdout"]["F0"]) == sorted(inputs["holdout_issue_ids"])
    if reference_f0 is None:
        facts["f0_panel_matches_reference"] = True  # this process IS the reference
    else:
        # Only the groups the reference actually carries (protocol v2: "train"
        # only -- a fresh holdout cannot have an earlier reference panel).
        tol = float(q0["f0_panel_rel_tol"])
        ok = bool(reference_f0)
        for group in reference_f0:
            cmp = compare_panels(panels[group]["F0"], reference_f0[group])
            ok = ok and cmp["same_issues"] and (cmp["bitwise_equal"] or cmp["max_rel_diff"] <= tol)
        facts["f0_panel_matches_reference"] = bool(ok)
    return facts


def _panels01(panels: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    p0 = {"train": panels["train"]["L0"], "holdout": panels["holdout"]["L0"]}
    p1 = {"train": panels["train"]["L1"], "holdout": panels["holdout"]["L1"]}
    return p0, p1


def _config_ids(protocol: Mapping[str, Any], job: str, prefix: str) -> List[Tuple[str, Dict[str, Any]]]:
    out = [(cid, cfg) for cid, cfg in protocol["configs"].items()
           if cfg.get("job") == job and cid.startswith(f"{job}_{prefix}")]
    return sorted(out, key=lambda item: item[0])


def _proc_dir(job_dir: Path, cfg: Mapping[str, Any]) -> Path:
    return job_dir / str(cfg["_output_subdir"])


# =============================================================================
# J1: diagnostics + negative controls
# =============================================================================

def decide_diag(protocol, protocol_sha, job_dir: Path) -> Dict[str, Any]:
    arms = _config_ids(protocol, "J1", "A")
    ncs = _config_ids(protocol, "J1", "NC")
    rule = protocol["qualification_rule"]
    ref_cfg_id, ref_cfg = arms[0]
    ref_panels = _load(_proc_dir(job_dir, ref_cfg) / "fs_panels.json")
    reference_f0 = {"train": ref_panels["train"]["F0"], "holdout": ref_panels["holdout"]["F0"]}
    arm_rows = []
    for cid, cfg in arms:
        pdir = _proc_dir(job_dir, cfg)
        rec = _load(pdir / "fs_fit_record.json")
        diag = _load(pdir / "fs_diagnostics.json")
        facts = validity_facts(pdir, protocol, protocol_sha, cid,
                               None if cid == ref_cfg_id else reference_f0)
        grads_a, grads_b = rec["grad_norms_A"], rec["grad_norms_B"]
        checks = {
            "n_updates_as_declared": rec["n_updates"] == cfg["max_updates"],
            "finite": all(v == v and abs(v) != float("inf")
                          for v in rec["losses"] + rec["grad_norms"]),
            "B_grad_positive_at_update0": bool(grads_b and grads_b[0] > 0.0),
            "A_grad_zero_at_update0": bool(grads_a and grads_a[0] == 0.0),
            "A_grad_positive_from_update1": bool(len(grads_a) > 1 and all(g > 0.0 for g in grads_a[1:])),
            "nonzero_response": diag.get("nonzero_response") is True,
            "save_reload_digest_exact": diag.get("reload_digest_matches") is True,
            "backbone_digest_unchanged": diag.get("backbone_digest_unchanged") is True,
            "no_leftover_hooks": diag.get("no_leftover_hooks") is True,
            "L0_equals_F0_bitwise": all((diag.get("L0_equals_F0_bitwise") or {}).get(g) is True
                                        for g in ("train", "holdout")),
        }
        arm_rows.append({"config_id": cid, "dir": str(pdir), "q0_facts": facts,
                         "q0_pass": all(facts.values()), "checks": checks,
                         "checks_pass": all(checks.values()), "files": process_files(pdir)})
    nc_rows = []
    for cid, cfg in ncs:
        pdir = _proc_dir(job_dir, cfg)
        panels = _load(pdir / "fs_panels.json")
        hist = cfg["_historical_fit_record"]
        if fp.file_sha256(hist["path"]) != hist["sha256"]:
            raise SystemExit(f"historical record {hist['path']} changed since pre-registration")
        record = _load(Path(hist["path"]))
        nc_rule = dict(rule)
        nc_rule["horizon"] = int(record["n_updates"])  # Q1 must not reject trivially
        p0, p1 = _panels01(panels)
        qual = sa.evaluate_fs_qualification(record, p0, p1, nc_rule, validity=None)
        facts = validity_facts(pdir, protocol, protocol_sha, cid, reference_f0)
        substantive_fails = [k for k in ("Q2", "Q3", "Q4", "Q5", "Q6") if qual["status"][k] == "FAIL"]
        nc_rows.append({
            "config_id": cid, "adapter": cfg["fs_adapter"], "dir": str(pdir),
            "q0_facts_panel_process": facts, "q0_panel_process_pass": all(facts.values()),
            "status": qual["status"], "substantive_verdict": qual["substantive_verdict"],
            "q4_train_24h_mean_ratio": qual["q4_train_24h"]["mean_ratio"],
            "q4_train_24h_max_ratio": qual["q4_train_24h"]["max_ratio"],
            "q5_train_72h_mean_ratio": qual["q5_train_72h"]["mean_ratio"],
            "q6_holdout_24h_mean_ratio": qual["q6_holdout_24h"]["mean_ratio"],
            "rejected": qual["substantive_verdict"] == "FAIL" and bool(substantive_fails),
            "rejected_by": substantive_fails,
            "qualification": qual,
        })
    passed = (all(r["q0_pass"] and r["checks_pass"] for r in arm_rows)
              and all(r["rejected"] and r["q0_panel_process_pass"] for r in nc_rows)
              and len(arm_rows) == 4 and len(nc_rows) == 4)
    return {"kind": "diag", "job": "J1", "arms": arm_rows, "negative_controls": nc_rows,
            "reference_f0_panel": reference_f0, "reference_config_id": ref_cfg_id,
            "verdict": "PASS" if passed else "STOP",
            "note": ("PASS requires every arm's Q0 facts and code-path checks and every "
                     "negative control REJECTED by FS-QUAL-v1 Q2-Q6 (Q1 horizon set to the "
                     "record's own length). Any unrejected negative control = STOP.")}


# =============================================================================
# J2 / J3: screen and formal
# =============================================================================

def decide_screen(protocol, protocol_sha, job_dir: Path, reference_f0) -> Dict[str, Any]:
    rule = protocol["screen_rule"]
    qrule = protocol["qualification_rule"]
    arms = []
    rows = []
    for cid, cfg in _config_ids(protocol, "J2", "A"):
        pdir = _proc_dir(job_dir, cfg)
        arm_id = cid.split("_", 1)[1]
        entry: Dict[str, Any] = {"config_id": cid, "arm_id": arm_id, "lr": cfg["learning_rate"],
                                 "dir": str(pdir), "files": process_files(pdir)}
        if not (pdir / "fs_fit_record.json").is_file():
            entry["missing"] = True
            rows.append(entry)
            continue
        rec = _load(pdir / "fs_fit_record.json")
        panels = _load(pdir / "fs_panels.json")
        facts = validity_facts(pdir, protocol, protocol_sha, cid, reference_f0)
        facts["record_matches_declared"] = (rec["mode"] == cfg["mode"]
                                            and rec["config"]["learning_rate"] == cfg["learning_rate"]
                                            and rec["config"]["max_updates"] == cfg["max_updates"])
        stab = sa.training_stability(
            rec, sa._panel_column(panels["train"]["L0"], 24), horizon=int(rule["horizon"]),
            clip_events_max=int(qrule["q2"]["clip_events_max"]),
            per_visit_ratio_max=float(qrule["q2"]["per_visit_ratio_max"]),
            epoch_onset_factor=float(qrule["q2"]["epoch_onset_factor"]))
        m = sa.panel_ratio_summary(panels["train"]["L0"], panels["train"]["L1"], 24)
        arms.append({"arm_id": arm_id, "lr": cfg["learning_rate"], "valid": all(facts.values()),
                     "stability": stab, "train_panel_mean_ratio_24h": m["mean_ratio"]})
        entry.update(q0_facts=facts, stability=stab, train_panel_24h=m,
                     train_panel_72h=sa.panel_ratio_summary(panels["train"]["L0"], panels["train"]["L1"], 72),
                     holdout_panel_24h=sa.panel_ratio_summary(panels["holdout"]["L0"], panels["holdout"]["L1"], 24))
        rows.append(entry)
    decision = sa.decide_lr_screen(arms, rule)
    decision.update(kind="screen", job="J2", details=rows)
    return decision


def decide_formal(protocol, protocol_sha, job_dir: Path, job: str, reference_f0) -> Dict[str, Any]:
    """FS-SELECT-v1 (1): ONE configuration id for all replicas (byte-identical
    argv except --device/--output-dir); the replica index is the device index."""
    qrule = protocol["qualification_rule"]
    cid = job
    cfg = protocol["configs"][cid]
    replicas = []
    rows = []
    # Protocol v2: the fresh holdout has no earlier reference, so its F0 panel
    # must instead be bitwise identical across the job's replicas.
    cross_replica_holdout = bool(qrule["q0"].get("holdout_f0_cross_replica_bitwise"))
    holdout_f0 = []
    for device in cfg["device"]["one_of"]:
        path = job_dir / str(cfg["_output_subdir_template"]).format(
            index=int(str(device).split(":")[1])) / "fs_panels.json"
        holdout_f0.append(_load(path)["holdout"]["F0"] if path.is_file() else None)
    holdout_consistent = all(p is not None and p == holdout_f0[0] for p in holdout_f0)
    for device in cfg["device"]["one_of"]:
        device_index = int(str(device).split(":")[1])
        pdir = job_dir / str(cfg["_output_subdir_template"]).format(index=device_index)
        row: Dict[str, Any] = {"config_id": cid, "device_index": device_index, "dir": str(pdir),
                               "files": process_files(pdir)}
        if not (pdir / "fs_fit_record.json").is_file():
            qual = {"verdict": "INVALID", "reason": "MISSING_OUTPUT"}
        else:
            rec = _load(pdir / "fs_fit_record.json")
            panels = _load(pdir / "fs_panels.json")
            facts = validity_facts(pdir, protocol, protocol_sha, cid, reference_f0)
            facts["record_is_formal_full_horizon"] = (rec["mode"] == "formal"
                                                      and rec["n_updates"] == qrule["horizon"])
            if cross_replica_holdout:
                facts["holdout_F0_identical_across_replicas"] = holdout_consistent
            p0, p1 = _panels01(panels)
            qual = sa.evaluate_fs_qualification(rec, p0, p1, qrule, validity=facts)
            row["record"] = {"mode": rec["mode"], "n_updates": rec["n_updates"],
                             "config_id": rec["config"].get("config_id")}
        row["qualification"] = qual
        rows.append(row)
        replicas.append({"device_index": device_index, "qualification": qual,
                         "adapter_sha256": (row["files"]["fs_adapter.pt"] or {}).get("sha256")})
    decision = sa.decide_formal_fs(replicas, protocol["selection_rule"])
    designated = next((r for r in rows if r["device_index"] ==
                       int(protocol["selection_rule"]["designated_device_index"])), None)
    # FS-SELECT-v1 (1), checked from what each process actually recorded.
    stripped = {}
    for row in rows:
        run_path = Path(row["dir"]) / "run_record.json"
        if run_path.is_file():
            argv = list((_load(run_path).get("provenance") or {}).get("argv") or [])
            stripped[row["device_index"]] = _strip_device_output(argv)
    identical = len(stripped) == len(rows) and len({json.dumps(v) for v in stripped.values()}) == 1
    decision.update(kind="formal", job=job, horizon=qrule["horizon"], details=rows,
                    designated_record=(designated or {}).get("record"),
                    argv_identical_except_device_output=identical)
    if not identical and decision["verdict"] != "INVALID":
        decision.update(verdict="INVALID", fs_selected=False, designated_adapter_sha256=None,
                        reason="FS-SELECT-v1(1): argv differ beyond --device/--output-dir")
    return decision


def _strip_device_output(argv: List[str]) -> List[str]:
    out: List[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in ("--device", "--output-dir"):
            skip = True
            continue
        out.append(token)
    return out


# =============================================================================
# J4: certify
# =============================================================================

def certify(protocol, protocol_sha, formal_decision_path: Path, freeze_dir: Path,
            verify_dir: Path, out_dir: Path) -> Dict[str, Any]:
    formal = _load(formal_decision_path)
    reload = _load(verify_dir / "independent_reload.json")
    gates = {"numerical_merge_pass": bool(reload.get("numerical_merge_pass")),
             "reload_pass": bool(reload.get("reload_pass")),
             "identity_pass": bool(reload.get("identity_pass"))}
    tolerance = protocol.get("merge_equivalence_tolerance")
    if tolerance is None:
        gates["numerical_merge_pass"] = False
        gates["numerical_merge_note"] = "NOT_PREREGISTERED merge tolerance: BLOCKED"
    else:
        post = reload.get("post_freeze") or {}
        merge = post.get("merge_equivalence") or {}
        delta = float((post.get("artifact") or {}).get("delta_max_abs", float("nan")))
        want_atol = max(float(tolerance["atol_floor"]), float(tolerance["atol_relative"]) * delta)
        applied_ok = (merge.get("rtol") == float(tolerance["rtol"])
                      and merge.get("atol") is not None
                      and abs(float(merge["atol"]) - want_atol) <= 1e-15 * max(1.0, want_atol))
        if not applied_ok:
            gates["numerical_merge_pass"] = False
            gates["numerical_merge_note"] = (
                f"applied tolerance atol={merge.get('atol')} rtol={merge.get('rtol')} is not the "
                f"pre-registered atol={want_atol} rtol={tolerance['rtol']}")
    replicas = [{"device_index": r["device_index"], "qualification": r["qualification"],
                 "adapter_sha256": (r["files"].get("fs_adapter.pt") or {}).get("sha256")}
                for r in formal["details"]]
    final = sa.decide_formal_fs(replicas, protocol["selection_rule"], designated_gates=gates)
    final.update(kind="certify", formal_decision=str(formal_decision_path),
                 formal_decision_sha256=fp.file_sha256(formal_decision_path))
    if final["verdict"] != "FS_SELECTED":
        return final
    fs_dir = out_dir / "fs"
    fs_dir.mkdir(parents=True, exist_ok=False)
    designated = next(r for r in formal["details"]
                      if r["device_index"] == int(protocol["selection_rule"]["designated_device_index"]))
    shutil.copy2(Path(designated["dir"]) / "fs_adapter.pt", fs_dir / "fs_adapter.pt")
    shutil.copy2(freeze_dir / "fs_merged_backbone.pt", fs_dir / "fs_merged_backbone.pt")
    shutil.copy2(verify_dir / "independent_reload.json", fs_dir / "independent_reload.json")
    (fs_dir / "quality_rule.json").write_text(json.dumps(protocol["qualification_rule"], indent=1))
    (fs_dir / "qualification.json").write_text(json.dumps({"formal": formal, "final": final}, indent=1))
    (fs_dir / "panel_initial_final.json").write_text(
        (Path(designated["dir"]) / "fs_panels.json").read_text())
    (fs_dir / "training_records.json").write_text(json.dumps(
        {r["config_id"]: r["files"] for r in formal["details"]}, indent=1))
    manifest = {
        "schema_version": "ed-fs-reference-manifest/1",
        "protocol_sha256": protocol_sha,
        "fs_adapter_sha256": fp.file_sha256(fs_dir / "fs_adapter.pt"),
        "fs_merged_backbone_sha256": fp.file_sha256(fs_dir / "fs_merged_backbone.pt"),
        "independent_reload": reload.get("identity"),
        "checkpoint_sha256": protocol["inputs"]["checkpoint_sha256"],
        "normalization_identity": protocol["inputs"]["normalization_identity"],
        "designated_config_id": designated["config_id"],
        "limitations": protocol.get("limitations", []),
    }
    (fs_dir / "reference_manifest.json").write_text(json.dumps(manifest, indent=1))
    final["fs_bundle"] = str(fs_dir)
    return final


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("diag", "screen", "formal", "certify"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--job-dir", type=Path, default=None)
    parser.add_argument("--job", default=None, help="J3 or J3b for --kind formal")
    parser.add_argument("--diag-decision", type=Path, default=None,
                        help="J1 decision holding the reference F0 panel")
    parser.add_argument("--formal-decision", type=Path, default=None)
    parser.add_argument("--freeze-dir", type=Path, default=None)
    parser.add_argument("--verify-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        protocol = fp.load_protocol(args.protocol, args.protocol_sha256)
    except fp.ProtocolViolation as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.out.exists():
        print(f"ERROR: {args.out} exists; decisions are written once.", file=sys.stderr)
        return 2
    reference_f0 = None
    inputs: Dict[str, Any] = {"protocol": {"path": str(args.protocol), "sha256": args.protocol_sha256}}
    if args.kind in ("screen", "formal"):
        if args.diag_decision is None:
            print("ERROR: --diag-decision (J1) is required for the F0 panel reference", file=sys.stderr)
            return 2
        diag = _load(args.diag_decision)
        pinned = (protocol.get("inputs") or {}).get("f0_reference_decision")
        if pinned is not None:
            # Protocol v2: the reference is an EARLIER protocol's PASS diag
            # decision, accepted only if its bytes are the ones this protocol pinned.
            if fp.file_sha256(args.diag_decision) != pinned.get("sha256") or diag.get("verdict") != "PASS":
                print("ERROR: --diag-decision is not the PASS decision this protocol pinned",
                      file=sys.stderr)
                return 2
        elif diag.get("verdict") != "PASS" or diag.get("protocol_sha256") != args.protocol_sha256:
            print("ERROR: the J1 decision is not a PASS bound to this protocol", file=sys.stderr)
            return 2
        groups = protocol["qualification_rule"]["q0"].get("f0_reference_groups",
                                                          ["train", "holdout"])
        reference_f0 = {g: diag["reference_f0_panel"][g] for g in groups}
        inputs["diag_decision"] = {"path": str(args.diag_decision),
                                   "sha256": fp.file_sha256(args.diag_decision)}
    if args.kind == "diag":
        decision = decide_diag(protocol, args.protocol_sha256, args.job_dir)
    elif args.kind == "screen":
        decision = decide_screen(protocol, args.protocol_sha256, args.job_dir, reference_f0)
    elif args.kind == "formal":
        decision = decide_formal(protocol, args.protocol_sha256, args.job_dir, args.job or "J3",
                                 reference_f0)
    else:
        decision = certify(protocol, args.protocol_sha256, args.formal_decision, args.freeze_dir,
                           args.verify_dir, args.out.parent)
    decision.update(protocol_sha256=args.protocol_sha256, inputs=inputs,
                    job_dir=str(args.job_dir) if args.job_dir else None, decided_utc=fp.utc_now(),
                    decider_sha256=fp.file_sha256(Path(__file__)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(decision, stream, indent=1, default=str)
        stream.write("\n")
    print(json.dumps({"kind": args.kind, "verdict": decision.get("verdict"),
                      "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
