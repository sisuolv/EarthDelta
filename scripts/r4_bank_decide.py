#!/usr/bin/env python3
"""FP-04 decisions from raw bank job outputs (CPU only; never trains, never submits).

    --kind shortstep  B-J1: machinery only. Per worker: E0 validity, the gradient
                      pattern (A exactly 0 and B > 0 at update 0, A > 0 after),
                      digests, reload, isolation; across workers: ONE shared
                      certified Fs, one initial bank, one grouping; the K=4 profile
                      and the full-hold horizon check with the bank unchanged.
                      No training-effect number (loss trend, L1/Fs ratio) is read.
    --kind formal     B-J2 / B-J2F: BANK-QUAL-v1 per expert, BANK-SELECT-v1 over the
                      batch (all four PASS or the bank stops; nothing substituted).
                      A STOP_CURRENT_BANK records fallback_authorized (every failed
                      expert failed E2); the B-J2F decision is refused without such a
                      prior B-J2 decision (--prior-formal-decision).
    --kind certify    B-J3: assembly A1-A5 recomputed from bank_verify.json's raw
                      numbers, bank.pt / expert files re-hashed and reloaded, the
                      registry bound; on BANK_CERTIFIED only, the certify/bank/
                      bundle is built, re-checked by registry.verify_bank_bundle,
                      and published.

Every decision is recomputed from the per-process files and the pre-registered
protocol, whose SHA-256 must match. Decisions are written once (open "x").
Rules are applied exactly; nothing here ranks experts by a quality number.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from earthdelta import bank_training as bt  # noqa: E402
from earthdelta import fs_protocol as fp  # noqa: E402
from earthdelta import registry as reg  # noqa: E402

BANK_BUNDLE_FILES = ("bank.pt", "bank_assembly.json", "bank_verify.json", "qualification.json",
                     "registry.json", "bank_manifest.json")


def _load(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _finite(values) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def _ref(path: Path | str) -> Dict[str, Any]:
    return {"path": str(path), "sha256": fp.file_sha256(path)}


def process_files(proc_dir: Path, names: List[str]) -> Dict[str, Any]:
    return {name: (_ref(proc_dir / name) if (proc_dir / name).is_file() else None)
            for name in names}


def _worker_names(k: int) -> List[str]:
    return [f"bank_training_expert{k}.json", f"bank_panels_expert{k}.json",
            f"bank_diagnostics_expert{k}.json", f"expert_{k}.pt", f"bank_probe_expert{k}.pt",
            "run_record.json", "provenance.json", "stdout.log", "binding_refused.json"]


def _job_configs(protocol: Mapping[str, Any], job: str, stage: str) -> List[Tuple[str, Dict[str, Any]]]:
    out = [(cid, dict(cfg)) for cid, cfg in protocol["configs"].items()
           if cfg.get("job") == job and cfg.get("stage") == stage]
    return sorted(out, key=lambda item: (item[1].get("expert_index") or 0, item[0]))


def source_drift(provenance: Mapping[str, Any], pinned: Mapping[str, str]) -> List[str]:
    """Imported source files whose SHA-256 differs from the pre-registration pin."""
    imported = provenance.get("imported_source_sha256") or {}
    drift = []
    for rel, want in pinned.items():
        for path, got in imported.items():
            if str(path).endswith("/" + rel) and got != want:
                drift.append(rel)
    return sorted(set(drift))


def fs_reference_facts(run: Mapping[str, Any], protocol: Mapping[str, Any]) -> Dict[str, bool]:
    """The process was bound to, and loaded, exactly the pinned certified Fs."""
    pinned = (protocol.get("inputs") or {}).get("fs_reference") or {}
    fs_ref = (run.get("binding") or {}).get("fs_reference") or {}
    authorized = bool(
        fs_ref.get("passed") is True and fs_ref.get("bound_to_protocol") is True
        and all((fs_ref.get(key) or {}).get("sha256") == (pinned.get(key) or {}).get("sha256")
                and (pinned.get(key) or {}).get("sha256") is not None
                for key in ("reference_manifest", "certify_decision", "fs_adapter",
                            "fs_merged_backbone"))
        and fs_ref.get("fs_protocol_sha256") == pinned.get("fs_protocol_sha256"))
    return {"certified_fs_authorized": authorized}


def fs_identity_facts(identity: Optional[Mapping[str, Any]], protocol: Mapping[str, Any],
                      fs_source: Optional[str]) -> Dict[str, bool]:
    pinned = ((protocol.get("inputs") or {}).get("fs_reference") or {}).get("digests") or {}
    ident = dict(identity or {})
    digests = dict(ident.get("digests") or {})
    return {
        "fs_source_is_certified_bundle": fs_source == "certified_bundle",
        "certified_fs_identity_pass": ident.get("identity_pass") is True,
        "certified_fs_digests_are_pinned": bool(pinned) and all(
            digests.get(key) == pinned.get(key) for key in
            ("merged_backbone_digest", "base_backbone_digest", "static_adapter_digest",
             "artifact_version_digest")),
    }


def environment_facts(proc_dir: Path, protocol: Mapping[str, Any], protocol_sha: str,
                      config_id: str) -> Dict[str, bool]:
    """Backend, versions, TF32, S0, admission, protocol, config id, Fs binding, source."""
    run = _load(proc_dir / "run_record.json")
    prov = run.get("provenance") or {}
    binding = run.get("binding") or {}
    rule = protocol["qualification_rule"]
    e0 = rule.get("e0") or {}
    inputs = protocol["inputs"]
    facts: Dict[str, bool] = {}
    if e0.get("require_official_backend", True):
        facts["official_backend"] = bool(
            prov.get("official_backend") is True
            and (binding.get("backend_postcondition") or {}).get("passed") is True)
    versions = prov.get("versions") or {}
    facts["torch_version"] = versions.get("torch") == e0.get("torch_version")
    facts["xformers_version"] = versions.get("xformers") == e0.get("xformers_version")
    facts["tf32_off"] = fp.tf32_is_off(prov)
    s0 = binding.get("s0") or {}
    facts["s0_certificate_consumed"] = bool(
        s0.get("passed") is True and s0.get("sha256") == inputs["s0_certificate"]["sha256"])
    adm = binding.get("admission_consumer") or {}
    facts["admission_consumer_pass"] = bool(
        adm.get("passed") is True
        and ((prov.get("input_files") or {}).get("admission") or {}).get("sha256")
        == inputs["admission"]["sha256"])
    facts["protocol_sha256_match"] = run.get("protocol_sha256") == protocol_sha
    facts["config_id_match"] = run.get("config_id") == config_id
    facts.update(fs_reference_facts(run, protocol))
    pinned_source = protocol.get("source_at_preregistration") or {}
    if pinned_source:
        facts["no_source_drift"] = not source_drift(prov, pinned_source)
    return facts


# =============================================================================
# B-J1: short-step machinery check
# =============================================================================

def _worker_bundle(pdir: Path, k: int):
    training = _load(pdir / f"bank_training_expert{k}.json")
    panels = _load(pdir / f"bank_panels_expert{k}.json")["experts"][str(k)]
    diagnostics = _load(pdir / f"bank_diagnostics_expert{k}.json")
    return training, panels, diagnostics


def _common_bank_facts(diagnostics: Mapping[str, Any], record: Mapping[str, Any], k: int,
                       protocol: Mapping[str, Any]) -> Dict[str, bool]:
    pins = protocol["inputs"]["bank"]
    group = [str(i) for i in pins["grouping"][str(k)]]
    n = len(group)
    ids = [str(i) for i in record.get("training_issue_ids") or []]
    return {
        "initial_bank_digest_is_pinned":
            diagnostics.get("initial_bank_digest") == pins["initial_bank_digest"],
        "initial_expert_digest_is_pinned":
            record.get("initial_expert_digest") == pins["initial_expert_digests"][k],
        "grouping_is_pinned": diagnostics.get("grouping_by_expert") == pins["grouping"],
        "training_order_is_the_pinned_group": record.get("n_samples") == n and ids[:n] == group,
        "probe_issue_is_pinned": diagnostics.get("probe_issue_id") == pins["probe_issue_id"],
        "backbone_is_certified_fs": ((diagnostics.get("experts") or {}).get(str(k)) or {})
            .get("isolation", {}).get("backbone_is_certified_fs") is True,
    }


def decide_shortstep(protocol, protocol_sha, job_dir: Path, job: str) -> Dict[str, Any]:
    rule = protocol.get("shortstep_rule") or {}
    rows = []
    fs_digests, initial_digests, groupings = set(), set(), set()
    for cid, cfg in _job_configs(protocol, job, "bank_train"):
        k = int(cfg["expert_index"])
        pdir = job_dir / str(cfg["_output_subdir"])
        row: Dict[str, Any] = {"config_id": cid, "expert_index": k, "dir": str(pdir),
                               "files": process_files(pdir, _worker_names(k))}
        if not (pdir / f"bank_training_expert{k}.json").is_file():
            row.update(passed=False, reason="MISSING_OUTPUT")
            rows.append(row)
            continue
        training, panels, diagnostics = _worker_bundle(pdir, k)
        record = training["experts"][0]
        diag = (diagnostics.get("experts") or {}).get(str(k)) or {}
        facts = environment_facts(pdir, protocol, protocol_sha, cid)
        facts.update(fs_identity_facts(diagnostics.get("fs_identity"), protocol,
                                       diagnostics.get("fs_source")))
        facts.update(_common_bank_facts(diagnostics, record, k, protocol))
        grads_a = [float(v) for v in record.get("grad_norms_A") or []]
        grads_b = [float(v) for v in record.get("grad_norms_B") or []]
        reload = diag.get("reload") or {}
        expert_ref = row["files"].get(f"expert_{k}.pt") or {}
        checks = {
            "n_updates_as_declared": record.get("n_updates") == cfg["max_updates"],
            "mode_as_declared": (record.get("config") or {}).get("mode") == cfg["mode"],
            "lr_as_declared": (record.get("config") or {}).get("learning_rate") == cfg["learning_rate"],
            "finite": _finite(list(record.get("losses") or []) + list(record.get("grad_norms") or [])
                              + grads_a + grads_b),
            "B_grad_positive_at_update0": bool(grads_b) and grads_b[0] > 0.0,
            "A_grad_zero_at_update0": bool(grads_a) and grads_a[0] == 0.0,
            "A_grad_positive_from_update1": len(grads_a) > 1 and all(g > 0.0 for g in grads_a[1:]),
            "untrained_L0_equals_Fs_bitwise": bool(panels.get("Fs"))
                and panels.get("L0") == panels.get("Fs"),
            "expert_moved": record.get("initial_expert_digest") != record.get("final_expert_digest"),
            "nonzero_response": (record.get("nonzero_response") or {}).get("nonzero") is True,
            "other_experts_unchanged": record.get("other_experts_unchanged") is True,
            "other_experts_grad_free": record.get("other_experts_grad_free") is True,
            "backbone_digest_unchanged": (diag.get("isolation") or {}).get("backbone_digest_unchanged") is True,
            "no_leftover_hooks": (diag.get("isolation") or {}).get("no_leftover_hooks") is True,
            "reload_exact": bool(reload.get("probe_exact") is True
                                 and reload.get("file_sha256_matches") is True
                                 and reload.get("expert_digest_matches") is True
                                 and bool(reload.get("probe_max_abs_diff_by_step"))
                                 and all(v == 0.0 for v in
                                         reload["probe_max_abs_diff_by_step"].values())),
            "expert_file_rehashes": expert_ref.get("sha256") is not None
                and expert_ref.get("sha256") == (reload.get("expert_file") or {}).get("sha256"),
        }
        fs_digests.add(json.dumps((diagnostics.get("fs_identity") or {}).get("digests"), sort_keys=True))
        initial_digests.add(diagnostics.get("initial_bank_digest"))
        groupings.add(json.dumps(diagnostics.get("grouping_by_expert"), sort_keys=True))
        row.update(validity_facts=facts, checks=checks,
                   passed=all(facts.values()) and all(checks.values()),
                   failed=sorted([k2 for k2, v in facts.items() if v is not True]
                                 + [k2 for k2, v in checks.items() if v is not True]))
        rows.append(row)
    n_workers = int(rule.get("n_workers", bt.DESIGN_NUM_EXPERTS))
    across = {
        "all_workers_present": sorted(r["expert_index"] for r in rows) == list(range(n_workers)),
        "one_initial_bank": len(initial_digests) == 1,
        "one_grouping": len(groupings) == 1,
    }
    measurements = {}
    for stage, name in (("profile", "capacity_profile.json"), ("horizon_check", "horizon_feasibility.json")):
        configs = _job_configs(protocol, job, stage)
        if not configs:
            measurements[stage] = {"passed": False, "reason": "NOT_DECLARED"}
            continue
        cid, cfg = configs[0]
        pdir = job_dir / str(cfg["_output_subdir"])
        if not (pdir / name).is_file():
            measurements[stage] = {"config_id": cid, "passed": False, "reason": "MISSING_OUTPUT",
                                   "files": process_files(pdir, [name, "run_record.json", "stdout.log",
                                                                 "binding_refused.json"])}
            continue
        payload = _load(pdir / name)
        facts = environment_facts(pdir, protocol, protocol_sha, cid)
        facts.update(fs_identity_facts(payload.get("fs_identity"), protocol,
                                       payload.get("fs_source")))
        immut = payload.get("bank_immutability") or {}
        checks = {"bank_unchanged": immut.get("bank_unchanged") is True
                  and immut.get("measured_on_a_deep_copy") is True,
                  "bank_is_the_pinned_initial_bank":
                      immut.get("bank_digest_before") == protocol["inputs"]["bank"]["initial_bank_digest"]}
        if stage == "profile":
            prof = payload.get("profile") or {}
            dec = payload.get("decision") or {}
            checks.update({
                "profile_no_error": prof.get("error") is None,
                "real_cuda_profile": (prof.get("is_real_gpu_profile") is True
                                      if rule.get("require_real_cuda_profile", True) else True),
                "keeps_K4_rank4": (dec.get("decided_num_experts") == bt.DESIGN_NUM_EXPERTS
                                   and dec.get("decided_rank_per_expert") == bt.DESIGN_RANK_PER_EXPERT
                                   and dec.get("shrink_required") is False),
                "profiled_K4_rank4": (prof.get("num_experts") == bt.DESIGN_NUM_EXPERTS
                                      and prof.get("rank_per_expert") == bt.DESIGN_RANK_PER_EXPERT),
            })
        else:
            checks.update({
                "full_hold_feasible": payload.get("feasible") is True
                    and payload.get("max_feasible_differentiable_steps") == payload.get("requested_hold_steps"),
                "no_activation_checkpointing": payload.get("activation_checkpointing_enabled") is False,
                "no_fallback_needed": payload.get("fallback") is None,
            })
        fs_digests.add(json.dumps((payload.get("fs_identity") or {}).get("digests"), sort_keys=True))
        measurements[stage] = {"config_id": cid, "validity_facts": facts, "checks": checks,
                               "passed": all(facts.values()) and all(checks.values()),
                               "failed": sorted([k2 for k2, v in facts.items() if v is not True]
                                                + [k2 for k2, v in checks.items() if v is not True]),
                               "summary": ({"peak_memory_gib": (payload.get("profile") or {}).get("peak_memory_gib"),
                                            "seconds_per_update": (payload.get("profile") or {}).get("seconds_per_update")}
                                           if stage == "profile" else
                                           {"max_feasible_differentiable_steps":
                                                payload.get("max_feasible_differentiable_steps")})}
    across["one_shared_certified_fs"] = len(fs_digests) == 1
    passed = (all(r.get("passed") for r in rows) and all(across.values())
              and all(m.get("passed") for m in measurements.values()))
    return {
        "kind": "shortstep", "rule": "BANK-SHORTSTEP-v1", "job": job,
        "workers": rows, "across_workers": across, "measurements": measurements,
        "verdict": "PASS" if passed else "STOP",
        "reads_no_training_effect_number": True,
        "note": ("Machinery only: validity, gradient pattern, digests, reload, isolation, one "
                 "shared certified Fs, K=4 profile, full-hold horizon, bank unchanged by the "
                 "measurements. No loss trend or L1/Fs ratio was read."),
    }


# =============================================================================
# B-J2 / B-J2F: BANK-QUAL-v1 + BANK-SELECT-v1
# =============================================================================

def _prefix_equal(formal_record: Mapping[str, Any], short_dir: Optional[Path], k: int,
                  n: int) -> Optional[bool]:
    """REPORT-ONLY: the formal run's first n updates replay the short-step run's."""
    if short_dir is None:
        return None
    path = short_dir / f"bank_training_expert{k}.json"
    if not path.is_file():
        return None
    short = _load(path)["experts"][0]
    keys = ("losses", "grad_norms", "grad_norms_A", "grad_norms_B", "training_issue_ids")
    return all(list(formal_record.get(key) or [])[:n] == list(short.get(key) or [])[:n]
               for key in keys) and len(short.get("losses") or []) >= n


def decide_formal(protocol, protocol_sha, job_dir: Path, job: str,
                  shortstep: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    rule = protocol["qualification_rule"]
    experts = []
    rows = []
    short_dirs = {int(w["expert_index"]): Path(w["dir"])
                  for w in (shortstep or {}).get("workers", [])}
    short_n = int((protocol.get("shortstep_rule") or {}).get("horizon", 0))
    for cid, cfg in _job_configs(protocol, job, "bank_train"):
        k = int(cfg["expert_index"])
        pdir = job_dir / str(cfg["_output_subdir"])
        row: Dict[str, Any] = {"config_id": cid, "expert_index": k, "dir": str(pdir),
                               "files": process_files(pdir, _worker_names(k))}
        expert_ref = row["files"].get(f"expert_{k}.pt")
        probe_ref = row["files"].get(f"bank_probe_expert{k}.pt")
        digest = None
        if not (pdir / f"bank_training_expert{k}.json").is_file() or expert_ref is None:
            qual: Dict[str, Any] = {"verdict": "INVALID", "reason": "MISSING_OUTPUT"}
        else:
            training, panels, diagnostics = _worker_bundle(pdir, k)
            record = training["experts"][0]
            diag = (diagnostics.get("experts") or {}).get(str(k)) or {}
            facts = environment_facts(pdir, protocol, protocol_sha, cid)
            facts.update(fs_identity_facts(diagnostics.get("fs_identity"), protocol,
                                           diagnostics.get("fs_source")))
            facts.update(_common_bank_facts(diagnostics, record, k, protocol))
            config = record.get("config") or {}
            facts["record_matches_declared"] = bool(
                record.get("expert_index") == k and config.get("mode") == cfg["mode"]
                and config.get("learning_rate") == cfg["learning_rate"]
                and config.get("max_updates") == cfg["max_updates"])
            facts["record_is_formal_full_horizon"] = (config.get("mode") == "formal"
                                                      and record.get("n_updates") == rule["horizon"])
            try:
                loaded = bt.load_expert(expert_ref["path"], expected_sha256=expert_ref["sha256"],
                                        expected_expert_index=k)
                digest = loaded["expert_digest"]
                facts["expert_file_is_the_trained_expert"] = (
                    digest == record.get("final_expert_digest")
                    and expert_ref["sha256"] == ((diag.get("reload") or {}).get("expert_file") or {}).get("sha256"))
            except bt.BankTrainingViolation:
                facts["expert_file_is_the_trained_expert"] = False
            facts["probe_file_is_recorded"] = probe_ref is not None and probe_ref["sha256"] == \
                ((diag.get("reload") or {}).get("probe_file") or {}).get("sha256")
            qual = bt.evaluate_bank_expert_qualification(
                record, {"Fs": panels["Fs"], "L0": panels["L0"], "L1": panels["L1"]}, diag, rule,
                validity=facts)
            row["record"] = {"mode": config.get("mode"), "n_updates": record.get("n_updates"),
                             "learning_rate": config.get("learning_rate"),
                             "n_samples": record.get("n_samples")}
            row["report_only_prefix_equals_shortstep"] = _prefix_equal(
                record, short_dirs.get(k), k, short_n) if short_n else None
        row["qualification"] = qual
        rows.append(row)
        experts.append({"expert_index": k, "qualification": qual, "expert_digest": digest,
                        "expert_file": expert_ref, "probe_file": probe_ref, "batch_id": job})
    decision = bt.decide_bank_formal(experts, protocol["selection_rule"])
    decision.update(kind="formal", job=job, horizon=rule["horizon"], details=rows)
    return decision


# =============================================================================
# B-J3: certify (A1-A5, reload, registry, bundle)
# =============================================================================

def _stage_facts(proc_dir: Path, protocol, protocol_sha, config_id: str,
                 record_name: str) -> Dict[str, bool]:
    if not (proc_dir / "run_record.json").is_file() or not (proc_dir / record_name).is_file():
        return {"outputs_present": False}
    facts = environment_facts(proc_dir, protocol, protocol_sha, config_id)
    record = _load(proc_dir / record_name)
    facts.update(fs_identity_facts(record.get("fs_identity"), protocol, record.get("fs_source")))
    return facts


def build_bundle(protocol, protocol_sha, protocol_path: Path, formal: Mapping[str, Any],
                 formal_path: Path, assembly_dir: Path, verify_dir: Path,
                 final: Mapping[str, Any], tmp: Path) -> Dict[str, Any]:
    """Copy the certified artifacts and write registry.json + bank_manifest.json."""
    tmp.mkdir(parents=True, exist_ok=False)
    shutil.copy2(assembly_dir / "bank.pt", tmp / "bank.pt")
    shutil.copy2(assembly_dir / "bank_assembly.json", tmp / "bank_assembly.json")
    shutil.copy2(verify_dir / "bank_verify.json", tmp / "bank_verify.json")
    experts = []
    for entry in formal["experts_for_assembly"]:
        k = int(entry["expert_index"])
        shutil.copy2(entry["expert_file"]["path"], tmp / f"expert_{k}.pt")
        shutil.copy2(entry["probe_file"]["path"], tmp / f"probe_expert_{k}.pt")
        experts.append({"expert_index": k, "file": f"expert_{k}.pt",
                        "sha256": fp.file_sha256(tmp / f"expert_{k}.pt"),
                        "expert_digest": entry["expert_digest"],
                        "probe_file": f"probe_expert_{k}.pt",
                        "probe_sha256": fp.file_sha256(tmp / f"probe_expert_{k}.pt"),
                        "source": {"expert_file": entry["expert_file"], "probe_file": entry["probe_file"]}})
    (tmp / "qualification.json").write_text(json.dumps({"formal": formal, "certify": final}, indent=1))
    assembly = _load(assembly_dir / "bank_assembly.json")
    inputs = protocol["inputs"]
    fs_ref = inputs["fs_reference"]
    binding = {
        "num_experts": int(protocol["selection_rule"]["num_experts"]),
        "rank_per_expert": int(protocol["design"]["rank_per_expert"]),
        "a0": float(protocol["design"]["a0"]),
        "rho": float(protocol["design"]["rho"]),
        "hold_steps": int(protocol["design"]["hold_steps"]),
        "target_blocks": list(protocol["design"]["target_blocks"]),
        "bank": {"file": "bank.pt", "sha256": fp.file_sha256(tmp / "bank.pt"),
                 "bank_digest": assembly["bank_digest"],
                 "expert_digests": list(assembly["expert_digests"]),
                 "initial_bank_digest": inputs["bank"]["initial_bank_digest"]},
        "experts": experts,
        "records": {name.split(".")[0]: {"file": name, "sha256": fp.file_sha256(tmp / name)}
                    for name in ("bank_assembly.json", "bank_verify.json", "qualification.json")},
        "fs": {"reference_manifest": dict(fs_ref["reference_manifest"]),
               "certify_decision": dict(fs_ref["certify_decision"]),
               "fs_adapter": dict(fs_ref["fs_adapter"]),
               "fs_merged_backbone": dict(fs_ref["fs_merged_backbone"]),
               "fs_protocol_sha256": fs_ref["fs_protocol_sha256"],
               **{key: fs_ref["digests"][key] for key in
                  ("merged_backbone_digest", "base_backbone_digest", "static_adapter_digest",
                   "artifact_version_digest")}},
        "normalization_identity": inputs["normalization_identity"],
        "checkpoint_sha256": inputs["checkpoint_sha256"],
        "inputs": {key: dict(inputs[key]) for key in ("admission", "grouping", "gate_config",
                                                      "s0_certificate") if key in inputs},
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "decisions": {"formal": _ref(formal_path)},
        "source": dict(protocol.get("source_at_preregistration") or {}),
    }
    registry = bt.build_bank_registry(
        binding["num_experts"], rho=binding["rho"], a0=binding["a0"],
        artifact_refs=reg.bank_entry_artifact_refs(binding),
        metadata={"bank_binding": binding, "certified_by": "r4_bank_decide --kind certify"})
    registry.to_json(tmp / "registry.json")
    manifest = {"schema_version": reg.BANK_BUNDLE_SCHEMA, "binding": binding,
                "registry": {"file": "registry.json", "sha256": fp.file_sha256(tmp / "registry.json")},
                "limitations": list(protocol.get("limitations") or [])}
    (tmp / "bank_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def certify(protocol, protocol_sha, protocol_path: Path, formal_path: Path,
            assemble_dir: Path, verify_dir: Path, out_dir: Path) -> Dict[str, Any]:
    formal = _load(formal_path)
    num_experts = int(protocol["selection_rule"]["num_experts"])
    reasons: List[str] = []
    if formal.get("kind") != "formal" or formal.get("verdict") != "BANK_QUAL_PASS_PENDING_ASSEMBLY":
        reasons.append(f"formal decision verdict is {formal.get('verdict')!r}")
    if formal.get("protocol_sha256") != protocol_sha:
        reasons.append("formal decision is not bound to this protocol")
    assemble_cfg = _job_configs(protocol, "B-J3", "bank_assemble")
    verify_cfg = _job_configs(protocol, "B-J3", "bank_verify")
    facts = {
        "assemble": _stage_facts(assemble_dir, protocol, protocol_sha,
                                 assemble_cfg[0][0] if assemble_cfg else "", "bank_assembly.json"),
        "verify": _stage_facts(verify_dir, protocol, protocol_sha,
                               verify_cfg[0][0] if verify_cfg else "", "bank_verify.json"),
    }
    for stage, stage_facts in facts.items():
        failed = sorted(k for k, v in stage_facts.items() if v is not True)
        if failed:
            reasons.append(f"{stage} validity facts failed: {failed}")
    final: Dict[str, Any] = {"rule": "BANK-CERTIFY-v1", "kind": "certify",
                             "formal_decision": _ref(formal_path), "validity_facts": facts,
                             "bank_certified": False}
    evaluation = None
    checks: Dict[str, Any] = {}
    if not reasons:
        assembly = _load(assemble_dir / "bank_assembly.json")
        verify = _load(verify_dir / "bank_verify.json")
        evaluation = bt.evaluate_bank_assembly(verify, num_experts=num_experts)
        decided = {int(e["expert_index"]): e for e in formal["experts_for_assembly"]}
        bank_path = assemble_dir / "bank.pt"
        checks["bank_file_rehashes"] = fp.file_sha256(bank_path) == assembly["bank_file"]["sha256"]
        checks["assembly_used_the_decided_experts"] = all(
            e["expert_file"]["sha256"] == decided[int(e["expert_index"])]["expert_file"]["sha256"]
            and e["expert_digest"] == decided[int(e["expert_index"])]["expert_digest"]
            for e in assembly["experts"]) and len(assembly["experts"]) == num_experts
        checks["assembly_bound_to_this_decision"] = \
            (assembly.get("decision") or {}).get("sha256") == fp.file_sha256(formal_path)
        checks["verify_read_this_assembly"] = \
            (verify.get("bank_assembly") or {}).get("sha256") == fp.file_sha256(assemble_dir / "bank_assembly.json")
        checks["assembly_self_checks"] = all(v is True for v in (assembly.get("checks") or {}).values())
        try:
            bank, meta = bt.load_bank(bank_path, expected_sha256=assembly["bank_file"]["sha256"])
            checks["bank_reloads_to_recorded_digests"] = (
                meta["bank_digest"] == assembly["bank_digest"]
                and meta["expert_digests"] == assembly["expert_digests"])
            slices_ok = True
            for k in range(num_experts):
                loaded = bt.load_expert(decided[k]["expert_file"]["path"],
                                        expected_sha256=decided[k]["expert_file"]["sha256"],
                                        expected_expert_index=k)
                slices_ok = slices_ok and loaded["expert_digest"] == meta["expert_digests"][k]
            checks["bank_slices_are_the_expert_files"] = slices_ok
        except bt.BankTrainingViolation as exc:
            checks["bank_reloads_to_recorded_digests"] = False
            checks["bank_reload_error"] = exc.code
        if not evaluation["passed"]:
            reasons.append(f"assembly equivalence failed: {evaluation['failed']}")
        failed_checks = sorted(k for k, v in checks.items() if v is not True)
        if failed_checks:
            reasons.append(f"certify checks failed: {failed_checks}")
    final.update(assembly_evaluation=evaluation, checks=checks)
    if reasons:
        final.update(verdict="STOP_ASSEMBLY", reasons=reasons)
        return final
    tmp = out_dir / "bank.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    manifest = build_bundle(protocol, protocol_sha, protocol_path, formal, formal_path,
                            assemble_dir, verify_dir, {**final, "verdict": "PENDING_BUNDLE_CHECK"}, tmp)
    try:
        consumer = reg.verify_bank_bundle(
            tmp,
            expected_fs={key: protocol["inputs"]["fs_reference"][key]["sha256"]
                         for key in ("reference_manifest", "certify_decision", "fs_adapter",
                                     "fs_merged_backbone")},
            expected_protocol_sha256=protocol_sha,
            expected_source=dict(protocol.get("source_at_preregistration") or {}),
            expected_normalization_identity=protocol["inputs"]["normalization_identity"])
    except reg.RegistryViolation as exc:
        shutil.rmtree(tmp)
        final.update(verdict="STOP_ASSEMBLY", reasons=[f"bundle consumer check failed: {exc.message}"],
                     bundle_check=exc.detail)
        return final
    target = out_dir / "bank"
    tmp.rename(target)
    final.update(verdict="BANK_CERTIFIED", bank_certified=True, bank_bundle=str(target),
                 bank_manifest_sha256=fp.file_sha256(target / "bank_manifest.json"),
                 registry_sha256=manifest["registry"]["sha256"],
                 bundle_consumer_check={k: v for k, v in consumer.items()
                                        if k in ("passed", "num_experts", "registry", "external_rehashed")})
    return final


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("shortstep", "formal", "certify"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--job-dir", type=Path, default=None)
    parser.add_argument("--job", default=None, help="B-J1 / B-J2 / B-J2F")
    parser.add_argument("--shortstep-decision", type=Path, default=None,
                        help="the PASS B-J1 decision (formal: report-only prefix replay)")
    parser.add_argument("--prior-formal-decision", type=Path, default=None,
                        help="the fallback job's authorization: the primary job's formal "
                             "decision (STOP_CURRENT_BANK with fallback_authorized)")
    parser.add_argument("--formal-decision", type=Path, default=None)
    parser.add_argument("--assemble-dir", type=Path, default=None)
    parser.add_argument("--verify-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        protocol = fp.load_protocol(args.protocol, args.protocol_sha256)
    except fp.ProtocolViolation as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if protocol.get("protocol_kind") != "bank":
        print("ERROR: not a bank protocol (protocol_kind != 'bank')", file=sys.stderr)
        return 2
    if args.out.exists():
        print(f"ERROR: {args.out} exists; decisions are written once.", file=sys.stderr)
        return 2
    inputs: Dict[str, Any] = {"protocol": {"path": str(args.protocol), "sha256": args.protocol_sha256}}
    if args.kind == "shortstep":
        if args.job_dir is None:
            print("ERROR: --job-dir is required", file=sys.stderr)
            return 2
        decision = decide_shortstep(protocol, args.protocol_sha256, args.job_dir, args.job or "B-J1")
    elif args.kind == "formal":
        if args.job_dir is None or args.job is None:
            print("ERROR: --job-dir and --job are required", file=sys.stderr)
            return 2
        fallback = (protocol.get("selection_rule") or {}).get("fallback") or {}
        if args.job == fallback.get("job"):
            # BANK-BUDGET-v1: the fallback batch is decided only if the primary
            # batch's decision authorized it (every failed expert failed E2).
            prior = _load(args.prior_formal_decision) if args.prior_formal_decision else {}
            if not (prior.get("kind") == "formal" and prior.get("job") == fallback.get("after_job")
                    and prior.get("verdict") == "STOP_CURRENT_BANK"
                    and prior.get("fallback_authorized") is True
                    and prior.get("protocol_sha256") == args.protocol_sha256):
                print(f"ERROR: {args.job} is the lr fallback; it needs --prior-formal-decision: the "
                      f"{fallback.get('after_job')} STOP_CURRENT_BANK decision with "
                      "fallback_authorized (every failed expert failed "
                      f"{fallback.get('requires_failed_criterion')}), bound to this protocol",
                      file=sys.stderr)
                return 2
            inputs["prior_formal_decision"] = _ref(args.prior_formal_decision)
        shortstep = None
        if args.shortstep_decision is not None:
            shortstep = _load(args.shortstep_decision)
            if shortstep.get("verdict") != "PASS" or shortstep.get("protocol_sha256") != args.protocol_sha256:
                print("ERROR: --shortstep-decision is not a PASS bound to this protocol", file=sys.stderr)
                return 2
            inputs["shortstep_decision"] = _ref(args.shortstep_decision)
        decision = decide_formal(protocol, args.protocol_sha256, args.job_dir, args.job, shortstep)
    else:
        if None in (args.formal_decision, args.assemble_dir, args.verify_dir):
            print("ERROR: --formal-decision, --assemble-dir and --verify-dir are required",
                  file=sys.stderr)
            return 2
        inputs["formal_decision"] = _ref(args.formal_decision)
        decision = certify(protocol, args.protocol_sha256, args.protocol, args.formal_decision,
                           args.assemble_dir, args.verify_dir, args.out.parent)
    decision.update(protocol_sha256=args.protocol_sha256, inputs=inputs,
                    job_dir=str(args.job_dir) if args.job_dir else None, decided_utc=fp.utc_now(),
                    decider_sha256=fp.file_sha256(Path(__file__)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(decision, stream, indent=1, default=str)
        stream.write("\n")
    print(json.dumps({"kind": args.kind, "verdict": decision.get("verdict"), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
