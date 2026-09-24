#!/usr/bin/env python3
"""FP-05 candidate cache: certify-admission (CPU), cache (GPU worker), merge (CPU).

    certify-admission --role {policy_dev,debug} --freeze F --freeze-sha256 S
                      --admission A --out <run>/admission/<role>_consumer_check.json
    cache  --protocol P --protocol-sha256 S --config-id CJ1_W0 --device cuda:0
           --output-dir @OUTPUT@/w0 [--dev-protocol D --dev-protocol-sha256 H]
    merge  --protocol P --protocol-sha256 S --dev-protocol D --dev-protocol-sha256 H
           --shard-dirs d0 d1 d2 d3 --out <run>/cache

Every stage runs the split-freeze gate (`earthdelta.split_freeze`) BEFORE any
data read and -- for `cache` -- before any model, bank or Fs byte is loaded.
The row's data_role tag never authorises anything.

THIS SCRIPT NEVER SUBMITS OR POLLS A JOB (see plans/plans_v2_0921/cci/submit_job.py).

Exit codes: 0 PASS; 1 a stage ran and failed a check; 2 refused before running.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402

from earthdelta import candidate_cache as cc  # noqa: E402
from earthdelta import split_freeze as sf  # noqa: E402

SUBMITS_JOBS = False
PROTOCOL_KIND = "cache"
NORMALIZATION_DIGEST = "3e0b216bfbf34ab0"
EXPECTED_TAG = {"policy_dev": "policy_dev", "debug": "bank_fit"}


class Refused(Exception):
    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{code}] {message}")
        self.code, self.message, self.detail = code, message, dict(detail or {})


def _json(path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _check_ref(ref: Mapping[str, Any], label: str) -> Path:
    path = Path(str(ref.get("path")))
    if not path.is_file() or cc.file_sha256(path) != ref.get("sha256"):
        raise Refused("PINNED_FILE_MISMATCH", f"{label} {path} missing or not the pinned sha256")
    return path


# =============================================================================
# protocol / freeze / allow-list binding (no data, no model)
# =============================================================================

def load_cache_protocol(path, sha256) -> Dict[str, Any]:
    from earthdelta import fs_protocol as fp

    if not path or not sha256:
        raise Refused("PREREGISTRATION_MISSING", "--protocol and --protocol-sha256 are required")
    try:
        protocol = fp.load_protocol(path, sha256)
    except fp.ProtocolViolation as exc:
        raise Refused(exc.code, exc.message) from exc
    if protocol.get("protocol_kind") != PROTOCOL_KIND:
        raise Refused("PROTOCOL_KIND_MISMATCH", f"{path} is not a cache protocol")
    if protocol.get("status") != "PREREGISTERED_BEFORE_FP05_JOBS":
        raise Refused("PROTOCOL_NOT_FROZEN", f"protocol status {protocol.get('status')!r}")
    return protocol


def bind_freeze(protocol: Mapping[str, Any]) -> sf.SplitFreeze:
    ref = protocol["inputs"]["split_freeze"]
    try:
        return sf.load_freeze(ref["path"], ref["sha256"])
    except sf.SplitFreezeViolation as exc:
        raise Refused(exc.code, exc.message) from exc


def allow_list(protocol: Mapping[str, Any], freeze: sf.SplitFreeze, role: str) -> sf.AllowList:
    if role == "debug":
        return freeze.debug_allow_list()
    return sf.AllowList.from_mapping(role, protocol["split"]["allow_lists"][role])


def role_admission(protocol: Mapping[str, Any], role: str) -> Tuple[Dict[str, Any], str]:
    ref = protocol["inputs"][role]["admission"]
    path = _check_ref(ref, f"{role} admission")
    return _json(path), ref["sha256"]


def check_sources(protocol: Mapping[str, Any], root: Path = SOURCE_ROOT) -> Dict[str, Any]:
    pins = protocol.get("source_at_preregistration") or {}
    bad = {rel: sha for rel, sha in pins.items()
           if not (root / rel).is_file() or cc.file_sha256(root / rel) != sha}
    if not pins or bad:
        raise Refused("SOURCE_DRIFT", f"{len(bad)} pinned source file(s) differ", {"files": sorted(bad)})
    return {"check": "source_at_preregistration", "n_files": len(pins), "passed": True}


def clear_rows(freeze, role, pairs, allow, admission_sha) -> Dict[str, Any]:
    try:
        return sf.assert_rows_clear(freeze, [r for r, _ in pairs], role, allow_list=allow,
                                    source_admission_sha256=admission_sha)
    except sf.SplitFreezeViolation as exc:
        raise Refused(exc.code, exc.message, {"verdicts": exc.detail.get("rows")}) from exc


def bind_dev_protocol(protocol, protocol_sha256, dev_path, dev_sha) -> Dict[str, Any]:
    """dev_protocol.json must be hash-chained to this protocol and to a PASS debug decision."""
    if not dev_path or not dev_sha:
        raise Refused("DEV_PROTOCOL_MISSING", "the dev cache needs --dev-protocol and its sha256")
    if cc.file_sha256(dev_path) != dev_sha:
        raise Refused("DEV_PROTOCOL_SHA256_MISMATCH", f"{dev_path} is not {dev_sha}")
    dev = _json(dev_path)
    if dev.get("protocol_sha256") != protocol_sha256:
        raise Refused("DEV_PROTOCOL_NOT_CHAINED", "dev_protocol is bound to another protocol")
    dec_ref = dev.get("debug_decision") or {}
    dec_path = _check_ref(dec_ref, "debug decision")
    decision = _json(dec_path)
    if decision.get("verdict") != "PASS" or decision.get("protocol_sha256") != protocol_sha256:
        raise Refused("DEBUG_DECISION_NOT_PASS", f"debug decision verdict {decision.get('verdict')!r}")
    if dev.get("dev_scale", {}).get("verdict") != "CHOSEN" or not dev.get("issue_ids"):
        raise Refused("DEV_SCALE_NOT_CHOSEN", "dev_protocol has no DEV-SCALE-v1 choice")
    return dev


def resolve_worker(protocol, protocol_sha256, config_id, dev_path=None, dev_sha=None):
    """(config, role, issue ids, shard index) for one cache worker, from the protocol only."""
    configs = protocol.get("configs") or {}
    if config_id not in configs:
        raise Refused("CONFIG_UNKNOWN", f"config {config_id!r} is not declared")
    cfg = dict(configs[config_id])
    if cfg.get("stage") != "cache":
        raise Refused("CONFIG_NOT_CACHE", f"{config_id} is not a cache config")
    if cfg["kind"] == "debug":
        return cfg, "debug", list(cfg["issue_ids"]), int(cfg["worker"])
    dev = bind_dev_protocol(protocol, protocol_sha256, dev_path, dev_sha)
    shard = int(cfg["shard"])
    ids = [i for i in dev["issue_ids"] if int(dev["shard_map"][i]) == shard]
    return cfg, "policy_dev", ids, shard


# =============================================================================
# certify-admission (CPU)
# =============================================================================

def stage_certify_admission(args) -> int:
    from earthdelta.data.pull_wb2 import grid_hash, variable_order_hash

    freeze = sf.load_freeze(args.freeze, args.freeze_sha256)
    record = _json(args.admission)
    admission_sha = cc.file_sha256(args.admission)
    rows = record["admission"]["admitted"]
    if args.role == "debug":
        ids = sorted(freeze.debug_allow_list().issue_ids,
                     key=freeze.role("debug")["allowed_issue_ids"].index)
        pairs = cc.rows_and_certificates(record, ids)
        clearance = sf.assert_rows_clear(freeze, [r for r, _ in pairs], "debug",
                                         source_admission_sha256=admission_sha)
        only = ids
    else:
        clearance = sf.assert_rows_clear(freeze, rows, "policy_dev", pre_admission=True)
        only = None
    report = cc.certify_admission_for_role(
        record, expected_tag=EXPECTED_TAG[args.role], lead_steps=cc.LEAD_STEPS,
        required_history_steps=2, expected_normalization_digest=NORMALIZATION_DIGEST,
        expected_grid_hash=grid_hash(), expected_variable_order_hash=variable_order_hash(),
        reverify_content=True, only_issue_ids=only)
    fresh = all(r.get("fresh_content_sha256") == r.get("content_sha256") for r in report["rows"])
    out = {"schema": "ed-fp05-consumer-check/1", "role": args.role,
           "record": {"path": str(Path(args.admission).resolve()), "sha256": admission_sha},
           "split_freeze": {"path": str(Path(args.freeze).resolve()), "sha256": freeze.sha256},
           "clearance": {k: clearance[k] for k in ("passed", "n_rows", "n_clear", "pre_admission")},
           "clearance_rows": clearance["rows"],
           "issue_ids": [r["issue_id"] for r in report["rows"]],
           "passed": bool(report["passed"] and clearance["passed"] and fresh),
           "fresh_content_sha256_equals_stored_for_every_row": fresh,
           "certification": report}
    cc.write_json_once(args.out, out)
    print(json.dumps({"role": args.role, "passed": out["passed"], "n_rows": report["n_rows"],
                      "fresh_equal": fresh}))
    return 0 if out["passed"] else 1


# =============================================================================
# cache (GPU worker)
# =============================================================================

def _fs_args(protocol) -> SimpleNamespace:
    ref = protocol["inputs"]["fs_reference"]
    return SimpleNamespace(
        stage="bank_verify",
        fs_reference_manifest=Path(ref["reference_manifest"]["path"]),
        fs_reference_manifest_sha256=ref["reference_manifest"]["sha256"],
        fs_certify_decision=Path(ref["certify_decision"]["path"]),
        fs_adapter=Path(ref["fs_adapter"]["path"]), fs_adapter_sha256=None,
        fs_merged_backbone=Path(ref["fs_merged_backbone"]["path"]),
        config=Path(protocol["inputs"]["gate_config"]["path"]), rank_per_expert=4)


def verify_bank(protocol) -> Dict[str, Any]:
    from earthdelta import registry as reg

    b = protocol["inputs"]["bank_bundle"]
    report = reg.verify_bank_bundle(
        b["dir"], expected_protocol_sha256=b["fp04_protocol_sha256"],
        expected_normalization_identity=protocol["inputs"]["normalization_identity"],
        rehash_external=True, deep=True)
    registry = reg.CandidateRegistry.from_json(Path(b["dir"]) / "registry.json")
    cc.check_registry_entries(registry.entries, protocol["inputs"]["registry_entries"])
    return {"bundle_report_passed": report["passed"], "registry_entries": "pinned"}, registry


def stage_cache(args) -> int:
    raw_argv = list(sys.argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # ---- E0a: protocol, freeze, clearance, sources -- nothing loaded yet ---------
    try:
        protocol = load_cache_protocol(args.protocol, args.protocol_sha256)
        cfg, role, ids, shard = resolve_worker(protocol, args.protocol_sha256, args.config_id,
                                               args.dev_protocol, args.dev_protocol_sha256)
        if cfg.get("device") != args.device:
            raise Refused("CLI_DIFFERS_FROM_DECLARED_CONFIG", f"device {args.device} != {cfg.get('device')}")
        freeze = bind_freeze(protocol)
        allow = allow_list(protocol, freeze, role)
        record, admission_sha = role_admission(protocol, role)
        pairs = cc.rows_and_certificates(record, ids)
        clearance = clear_rows(freeze, role, pairs, allow, admission_sha)
        sources = check_sources(protocol)
    except (Refused, sf.SplitFreezeViolation, cc.CacheViolation, KeyError, OSError, ValueError) as exc:
        cc.write_json_once(out / "binding_refused.json", {
            "stage": "cache", "error": str(exc), "code": getattr(exc, "code", type(exc).__name__),
            "argv": raw_argv, "phase": "E0a_before_any_model_or_data"})
        print(f"ERROR: refused before loading anything: {exc}", file=sys.stderr, flush=True)
        return 2

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import scripts.r2_fs_bank_train as r2
    from earthdelta import bank_training as bt
    from earthdelta import fs_protocol as fp
    from earthdelta import static_adapter as sa
    from earthdelta.contracts import GateIdentityConfig

    started = fp.utc_now()
    e0: Dict[str, Any] = {"clearance": {k: clearance[k] for k in ("passed", "n_rows", "n_clear")},
                          "sources": sources, "role": role, "config_id": args.config_id}
    try:
        inputs = protocol["inputs"]
        gate = GateIdentityConfig.load_json(inputs["gate_config"]["path"])
        _check_ref(inputs["gate_config"], "gate config")
        e0["s0"] = fp.consume_s0_certificate(inputs["s0_certificate"]["path"],
                                             expected_sha256=inputs["s0_certificate"]["sha256"],
                                             gate_config=gate)["passed"]
        e0["backend_pre"] = fp.official_backend_precondition(
            expected_xformers=protocol["cache_run_rule"]["e0"]["xformers_version"],
            expected_torch=protocol["cache_run_rule"]["e0"]["torch_version"])["passed"]
        fs_args = _fs_args(protocol)
        auth = r2.authorize_certified_fs(fs_args, protocol)
        bank_report, registry = verify_bank(protocol)
        e0["bank_bundle"] = bank_report
    except Exception as exc:  # noqa: BLE001 - every E0b failure refuses before model load
        cc.write_json_once(out / "binding_refused.json", {
            "stage": "cache", "error": str(exc), "code": getattr(exc, "code", type(exc).__name__),
            "argv": raw_argv, "phase": "E0b_before_model_load", "e0": e0})
        print(f"ERROR: E0 refused before model load: {exc}", file=sys.stderr, flush=True)
        return 2

    device = torch.device(args.device)
    t_load = time.perf_counter()
    try:
        bridge, variables, lat, _ = r2.build_real_bridge(Path(inputs["gate_config"]["path"]), device)
        e0["backend_post"] = fp.official_backend_postcondition(bridge.model)["passed"]
        blocks = tuple(int(b) for b in protocol["design"]["target_blocks"])
        ctx = {"record": {"binding": {"fs_reference": auth}}, "bridge": bridge,
               "target_blocks": blocks, "protocol": protocol}
        fs_rep = r2.load_certified_fs(fs_args, ctx)
        fs_bridge = ctx["fs_bridge"]
        b = inputs["bank_bundle"]
        bank, meta = bt.load_bank(Path(b["dir"]) / "bank.pt", expected_sha256=b["bank_sha256"],
                                  device=device)
        if meta["bank_digest"] != b["bank_digest"] or meta["expert_digests"] != b["expert_digests"]:
            raise Refused("BANK_DIGEST_MISMATCH", "loaded bank is not the pinned one")
        bt.check_bank_device(bank, device, context="fp05 cache")
        spec = sa.build_objective_spec(bridge, lat, lead_steps=(4,), space="raw")
        e0["fs_identity_pass"] = fs_rep["identity_pass"]
        e0["bank_digest"] = meta["bank_digest"]
    except Exception as exc:  # noqa: BLE001
        cc.write_json_once(out / "binding_refused.json", {
            "stage": "cache", "error": str(exc), "code": getattr(exc, "code", type(exc).__name__),
            "argv": raw_argv, "phase": "E0c_model_identity", "e0": e0})
        print(f"ERROR: E0 identity failed: {exc}", file=sys.stderr, flush=True)
        return 2
    model_load_seconds = time.perf_counter() - t_load
    provenance = fp.collect_provenance(argv=raw_argv, device=device, source_roots=[SOURCE_ROOT],
                                       model=bridge.model, started_utc=started,
                                       input_files={"protocol": args.protocol,
                                                    "dev_protocol": args.dev_protocol})
    e0["tf32_off"] = fp.tf32_is_off(provenance)
    e0_pass = all(v is True for k, v in e0.items() if k in (
        "s0", "backend_pre", "backend_post", "fs_identity_pass", "tf32_off")) and \
        bank_report["bundle_report_passed"] is True
    merged_before = sa.state_dict_digest(fs_bridge.model)
    hooks_before = cc.count_hooks(fs_bridge.model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    entries = list(registry.entries)
    issues: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    ok = e0_pass
    for row, cert in pairs:
        iid = row["issue_id"]
        t0 = time.perf_counter()
        sf.reset_read_log()
        x_raw, truth = cc.load_issue_raw(freeze, role, row, cert, allow_list=allow,
                                         source_admission_sha256=admission_sha)
        sample = cc.build_sample(bridge, row, x_raw, truth, device)
        res = cc.rollout_issue(fs_bridge, bridge, bank, entries, sample, spec, variables, blocks)
        d = out / f"issue_{iid}"
        d.mkdir(parents=True, exist_ok=False)
        truth_arr = np.stack([truth[s] for s in cc.LEAD_STEPS]).astype(np.float32)
        files = {"endpoints": {"path": str(d / "endpoints.npy"),
                               "sha256": cc.save_npy_once(d / "endpoints.npy", res["endpoints"]),
                               "shape": list(res["endpoints"].shape), "dtype": "float32"},
                 "truth": {"path": str(d / "truth.npy"),
                           "sha256": cc.save_npy_once(d / "truth.npy", truth_arr),
                           "shape": list(truth_arr.shape), "dtype": "float32"},
                 "f0_endpoints": {"path": str(d / "f0_endpoints.npy"),
                                  "sha256": cc.save_npy_once(d / "f0_endpoints.npy",
                                                             res["f0_background"]["endpoints"]),
                                  "shape": list(res["f0_background"]["endpoints"].shape),
                                  "dtype": "float32"}}
        cc.write_json_once(d / "gpu_losses.json", {
            "sealed": True, "candidates": {c: v["losses"] for c, v in res["candidates"].items()},
            "plain_fs": res["plain_fs"]["losses"], "f0_background": res["f0_background"]["losses"]})
        files["gpu_losses"] = {"path": str(d / "gpu_losses.json"),
                               "sha256": cc.file_sha256(d / "gpu_losses.json")}
        hooks = res["hooks_after"]
        issue = {"schema": cc.CACHE_SCHEMA, "issue_id": iid, "role": role, "shard": shard,
                 "config_id": args.config_id, "issue_time": int(row["issue_time"]),
                 "row": {k: row.get(k) for k in ("event_id", "issue_index", "history_index",
                                                 "target_index", "issue_store", "data_role")},
                 "files": files,
                 "status": {c: v["status"] for c, v in res["candidates"].items()},
                 "f0_status": res["f0_background"]["status"],
                 "reference_endpoints_array_sha256": cc.array_sha256(res["endpoints"][0]),
                 "plain_fs_endpoints_array_sha256": cc.array_sha256(res["plain_fs"]["endpoints"]),
                 "reference_equals_plain_fs_bitwise": res["reference_equals_plain_fs_bitwise"],
                 "hooks_before": hooks_before, "hooks_after": hooks,
                 "hooks_clean": all(h == hooks_before for h in hooks.values()),
                 "seconds": res["seconds"], "issue_seconds": time.perf_counter() - t0,
                 "bytes": sum(Path(f["path"]).stat().st_size for f in files.values()),
                 "read_log": sf.read_log()}
        cc.write_json_once(d / "issue.json", issue)
        issues.append({"issue_id": iid, "dir": str(d), "issue_json_sha256": cc.file_sha256(d / "issue.json")})
        ok = ok and issue["hooks_clean"]
        for c in res["candidates"].values():
            for s in c["status"].values():
                status_counts[s] = status_counts.get(s, 0) + 1
        print(f"  {iid}: statuses={sorted(set(s for c in res['candidates'].values() for s in c['status'].values()))} "
              f"ref==plainFs={res['reference_equals_plain_fs_bitwise']} "
              f"seconds={issue['issue_seconds']:.1f}", flush=True)
        del res, sample
    merged_after = sa.state_dict_digest(fs_bridge.model)
    bank_after = bt.bank_digest(bank)
    peak = (torch.cuda.max_memory_allocated(device) / 2 ** 30) if torch.cuda.is_available() else 0.0
    shard_rec = {"schema": cc.CACHE_SCHEMA, "stage": "cache", "config_id": args.config_id,
                 "protocol_sha256": args.protocol_sha256, "role": role, "shard": shard,
                 "e0": e0, "e0_passed": e0_pass, "issues": issues,
                 "merged_fs_digest_before": merged_before, "merged_fs_digest_after": merged_after,
                 "bank_digest_after": bank_after,
                 "digests_unchanged": merged_before == merged_after and bank_after == e0["bank_digest"],
                 "model_load_seconds": model_load_seconds, "peak_gpu_gib": peak,
                 "status_counts": status_counts, "provenance": provenance,
                 "finished_utc": fp.utc_now()}
    shard_rec["passed"] = bool(ok and shard_rec["digests_unchanged"])
    cc.write_json_once(out / "shard.json", shard_rec)
    print(f"RESULT: {'PASS' if shard_rec['passed'] else 'FAIL'} issues={len(issues)} "
          f"statuses={status_counts} load={model_load_seconds:.1f}s peak={peak:.1f}GiB", flush=True)
    return 0 if shard_rec["passed"] else 1


# =============================================================================
# merge (CPU): shared with r4_cache_decide --kind debug
# =============================================================================

def objective_on_cpu(protocol) -> Tuple[Any, Any, Dict[str, Any]]:
    """(q, scale, normalization) from the pinned gate config -- no model."""
    import torch

    import scripts.r2_fs_bank_train as r2
    from earthdelta import static_adapter as sa
    from earthdelta.bridge import NormalizationContract
    from earthdelta.contracts import GateIdentityConfig

    gate = GateIdentityConfig.load_json(protocol["inputs"]["gate_config"]["path"])
    norm = NormalizationContract.from_npz_dir(gate.normalization_dir, variables=list(gate.variables),
                                              intervals=tuple(gate.normalization_intervals),
                                              policy=gate.normalization_policy)
    if norm.digest != NORMALIZATION_DIGEST:
        raise Refused("NORMALIZATION_MISMATCH", f"normalization digest {norm.digest}")
    lat = r2._load_latitude(gate)
    spec = sa.build_objective_spec(SimpleNamespace(normalization=norm, variables=list(gate.variables)),
                                   lat, lead_steps=(4,), space="raw")
    return spec.q, spec.scale, {"inp_mean": norm.inp_mean.detach().cpu().double().numpy(),
                                "inp_std": norm.inp_std.detach().cpu().double().numpy(),
                                "digest": norm.digest}


def identity_block(protocol, protocol_sha256, role) -> Dict[str, Any]:
    i = protocol["inputs"]
    return {"protocol_sha256": protocol_sha256, "role": role,
            "fs_digests": i["fs_reference"]["digests"], "bank_digest": i["bank_bundle"]["bank_digest"],
            "registry_sha256": i["bank_bundle"]["registry_sha256"],
            "admission_sha256": i[role]["admission"]["sha256"],
            "normalization_identity": i["normalization_identity"],
            "hold_steps": 4, "rho": 0.25, "continuation": "reference_after_hold",
            "store": protocol["split"]["stores"][role]}


def process_issue(issue_json: Path, *, q, scale) -> Dict[str, Any]:
    """Re-hash the worker's files (V3) and recompute every loss on CPU (V5/V6).

    Returns the issue record plus CPU losses; nothing is printed.
    """
    issue = _json(issue_json)
    issue["gpu_losses"] = _json(issue["files"]["gpu_losses"]["path"])["candidates"]
    f0_gpu = _json(issue["files"]["gpu_losses"]["path"])["f0_background"]
    rehash = {}
    arrays = {}
    for name, shape in (("endpoints", (5, 3)), ("truth", (3,)), ("f0_endpoints", (3,))):
        ref = issue["files"][name]
        ok = cc.file_sha256(ref["path"]) == ref["sha256"]
        a = np.load(ref["path"], allow_pickle=False)
        arrays[name] = a
        rehash[name] = ok and a.dtype == np.float32 and tuple(a.shape[:len(shape)]) == shape \
            and a.shape[-3:] == (cc.N_CHANNELS, 128, 256)
    gl_ok = cc.file_sha256(issue["files"]["gpu_losses"]["path"]) == issue["files"]["gpu_losses"]["sha256"]
    cpu = cc.cpu_losses(arrays["endpoints"], arrays["truth"], q, scale)
    analytic = cc.analytic_gains(arrays["endpoints"], arrays["truth"], q, scale)
    f0_cpu = cc.cpu_losses(arrays["f0_endpoints"][None], arrays["truth"], q, scale)[0]
    return {"issue": issue, "cpu": cpu, "analytic": analytic, "f0_cpu": f0_cpu, "f0_gpu": f0_gpu,
            "rehash": {"endpoints_ok": rehash["endpoints"], "truth_ok": rehash["truth"],
                       "shape_dtype_ok": all(rehash.values()) and gl_ok,
                       "f0_ok": rehash["f0_endpoints"], "gpu_losses_ok": gl_ok}}


def stage_merge(args) -> int:
    try:
        protocol = load_cache_protocol(args.protocol, args.protocol_sha256)
        dev = bind_dev_protocol(protocol, args.protocol_sha256, args.dev_protocol,
                                args.dev_protocol_sha256)
        freeze = bind_freeze(protocol)
        role = "policy_dev"
        allow = allow_list(protocol, freeze, role)
        record, admission_sha = role_admission(protocol, role)
        ids = list(dev["issue_ids"])
        pairs = cc.rows_and_certificates(record, ids)
        clear_rows(freeze, role, pairs, allow, admission_sha)
    except (Refused, sf.SplitFreezeViolation, cc.CacheViolation, KeyError, OSError) as exc:
        print(f"ERROR: merge refused before reading anything: {exc}", file=sys.stderr, flush=True)
        return 2
    out = Path(args.out)
    (out / "features").mkdir(parents=True, exist_ok=False)
    q, scale, norm = objective_on_cpu(protocol)
    shard_of = {i: int(dev["shard_map"][i]) for i in ids}
    by_issue: Dict[str, Path] = {}
    shards = []
    for sd in args.shard_dirs:
        rec = _json(Path(sd) / "shard.json")
        shards.append({"dir": str(sd), "config_id": rec["config_id"], "passed": rec["passed"],
                       "e0_passed": rec["e0_passed"], "digests_unchanged": rec["digests_unchanged"],
                       "shard_json_sha256": cc.file_sha256(Path(sd) / "shard.json")})
        for it in rec["issues"]:
            p = Path(it["dir"]) / "issue.json"
            if cc.file_sha256(p) != it["issue_json_sha256"] or it["issue_id"] in by_issue:
                raise cc.CacheViolation("INVALID_EVIDENCE", f"issue.json {p} tampered or duplicated")
            by_issue[it["issue_id"]] = p
    identity = identity_block(protocol, args.protocol_sha256, role)
    identity_sha = cc.canonical_digest(identity)
    pins = protocol["inputs"]["registry_entries"]
    rows, rehash, worker_log, bg = [], {}, [], {}
    feature_files = {}
    sf.reset_read_log()
    for row, cert in pairs:
        iid = row["issue_id"]
        if iid not in by_issue:
            continue  # V1 reports the gap; never score a subset
        p = process_issue(by_issue[iid], q=q, scale=scale)
        x_t, x6, x12 = cc.load_issue_history(freeze, role, row, cert, allow_list=allow,
                                             source_admission_sha256=admission_sha)
        feats = cc.legal_features_np(x_t, x6, x12, int(row["issue_time"]), norm["inp_mean"], norm["inp_std"])
        fsha = cc.save_npy_once(out / "features" / f"{iid}.npy", feats)
        feature_files[iid] = {"path": str(out / "features" / f"{iid}.npy"), "sha256": fsha,
                              "array_sha256": cc.array_sha256(feats)}
        rows += cc.matrix_rows_for_issue(p["issue"], cpu=p["cpu"], analytic=p["analytic"],
                                         entries_pin=pins, identity_sha256=identity_sha,
                                         features_sha256=feature_files[iid]["array_sha256"])
        rehash[iid] = p["rehash"]
        worker_log += p["issue"]["read_log"]
        bg[iid] = {"f0_cpu": dict(zip(map(str, cc.LEAD_HOURS), p["f0_cpu"])),
                   "f0_gpu": p["f0_gpu"], "f0_status": p["issue"]["f0_status"]}
    slices = {r["issue_id"]: (role, Path(r["issue_store"]).name, int(r["history_index"]),
                              int(r["target_index"])) for r, _ in pairs}
    v8 = sf.read_log_within(freeze, worker_log + sf.read_log(), slices)
    manifest = {"schema": cc.CACHE_SCHEMA, "kind": "dev", "protocol_sha256": args.protocol_sha256,
                "dev_protocol": {"path": str(Path(args.dev_protocol).resolve()),
                                 "sha256": args.dev_protocol_sha256},
                "split_freeze_sha256": freeze.sha256, "identity": identity,
                "identity_sha256": identity_sha, "issue_ids": ids, "shard_map": shard_of,
                "shards": shards, "features": feature_files, "n_features": cc.N_FEATURES,
                "read_set": v8, "merge_reads": len(sf.read_log())}
    try:
        report = cc.validate_cache_matrix(rows, expected_issue_ids=ids, entries_pin=pins,
                                          expected_identity_sha256=identity_sha, shard_of=shard_of,
                                          rehash=rehash, read_check=v8)
        shards_ok = all(s["passed"] and s["e0_passed"] and s["digests_unchanged"] for s in shards)
        manifest["validation"] = report
        manifest["passed"] = bool(report["passed"] and shards_ok)
    except cc.CacheViolation as exc:
        manifest["validation"] = exc.detail
        manifest["passed"] = False
    cc.write_json_once(out / "candidate_results.json",
                       {"schema": cc.CACHE_SCHEMA, "sealed_labels": True,
                        "rows": sorted(rows, key=lambda r: (r["issue_id"], r["candidate_id"], r["lead_hours"]))})
    cc.write_json_once(out / "background_f0.json", {"report_only": True, "never_a_candidate": True,
                                                    "issues": bg})
    manifest["candidate_results"] = {"path": str(out / "candidate_results.json"),
                                     "sha256": cc.file_sha256(out / "candidate_results.json"),
                                     "canonical_matrix_digest": cc.matrix_digest(rows)}
    manifest["background_f0"] = {"path": str(out / "background_f0.json"),
                                 "sha256": cc.file_sha256(out / "background_f0.json")}
    try:
        import pandas as pd  # convenience copy only; the canonical record is the JSON

        pd.DataFrame(rows).drop(columns=["coefficients", "support"]).to_parquet(
            out / "candidate_results.parquet")
        manifest["parquet_convenience_copy"] = str(out / "candidate_results.parquet")
    except Exception as exc:  # noqa: BLE001 - optional
        manifest["parquet_convenience_copy"] = f"not written: {type(exc).__name__}"
    cc.write_json_once(out / "cache_manifest.json", manifest)
    v = manifest["validation"]
    print(json.dumps({"passed": manifest["passed"], "n_rows": len(rows),
                      "v_pass": v.get("v_pass"), "n_nonfinite": v.get("n_nonfinite"),
                      "read_set_passed": v8["passed"]}))
    return 0 if manifest["passed"] else 1


# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("certify-admission")
    c.add_argument("--role", choices=["policy_dev", "debug"], required=True)
    c.add_argument("--freeze", type=Path, required=True)
    c.add_argument("--freeze-sha256", required=True)
    c.add_argument("--admission", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    k = sub.add_parser("cache")
    k.add_argument("--protocol", type=Path, required=True)
    k.add_argument("--protocol-sha256", required=True)
    k.add_argument("--config-id", required=True)
    k.add_argument("--device", required=True)
    k.add_argument("--output-dir", type=Path, required=True)
    k.add_argument("--dev-protocol", type=Path, default=None)
    k.add_argument("--dev-protocol-sha256", default=None)
    m = sub.add_parser("merge")
    m.add_argument("--protocol", type=Path, required=True)
    m.add_argument("--protocol-sha256", required=True)
    m.add_argument("--dev-protocol", type=Path, required=True)
    m.add_argument("--dev-protocol-sha256", required=True)
    m.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    m.add_argument("--out", type=Path, required=True)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.stage == "certify-admission":
            return stage_certify_admission(args)
        if args.stage == "cache":
            return stage_cache(args)
        return stage_merge(args)
    except (sf.SplitFreezeViolation, cc.CacheViolation, Refused) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
