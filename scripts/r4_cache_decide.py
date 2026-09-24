#!/usr/bin/env python3
"""FP-05 C-J1 debug decision (DEBUG-DECIDE-v1) and dev scale (DEV-SCALE-v1).

    python3 scripts/r4_cache_decide.py --kind debug --protocol P --protocol-sha256 S \
        --job-dir <C-J1 job dir with w0..w3> --out <run>/decisions/CJ1_debug_decision.json \
        [--dev-protocol-out <run>/cache/dev_protocol.json]

DEBUG-DECIDE-v1: E0 on all 4 workers; the 8x5x3 matrix complete and CACHE-VALID
V1-V8; every debug issue's Fs 6/24/72 losses (both the zero-plan-with-bank
reference and FP-04's own plain-Fs call path) BITWISE equal to FP-04's
bank_panels Fs values, and expert_k's losses equal FP-04's L1 for issues of
G_k; duplicates byte-identical; reference endpoints == plain-Fs endpoints.
The debug issue set must be EXACTLY the split freeze's debug allow-list.

DEV-SCALE-v1 (only after PASS) reads admission membership, timing, bytes and
peak memory only -- never a loss.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402

import scripts.r4_candidate_cache as rc  # noqa: E402
from earthdelta import candidate_cache as cc  # noqa: E402
from earthdelta import split_freeze as sf  # noqa: E402


def _json(p) -> Dict[str, Any]:
    return json.loads(Path(p).read_text())


def debug_configs(protocol) -> List[Dict[str, Any]]:
    cfgs = [dict(v, config_id=k) for k, v in (protocol.get("configs") or {}).items()
            if v.get("stage") == "cache" and v.get("kind") == "debug"]
    return sorted(cfgs, key=lambda c: int(c["worker"]))


def mde_table(protocol, ns: Sequence[int], n_blocks: int) -> Dict[str, Any]:
    """REPORT-ONLY planning MDE from EXPOSED data (FP-04 own-group panels)."""
    rel = []
    for k, ref in protocol["inputs"]["debug"]["anchors"].items():
        panel = _json(rc._check_ref(ref, f"anchor {k}"))["experts"][str(k)]
        for iid in panel["issue_ids"]:
            fs, l1 = panel["Fs"][iid]["24"], panel["L1"][iid]["24"]
            rel.append((fs - l1) / fs)
    sd = float(np.std(rel, ddof=1))
    z = 1.959963984540054 + 0.8416212335729143
    return {"report_only": True, "never_feeds_delta_min": True,
            "source": "FP-04 bank_panels own-group 24h relative gains (in-sample, exposed)",
            "per_issue_sd": sd, "n_panel_issues": len(rel),
            "alpha_two_sided": 0.05, "power": 0.8,
            "mde_if_issues_independent": {str(n): z * sd / math.sqrt(n) for n in ns},
            "mde_if_blocks_are_the_unit": {str(n_blocks): z * sd / math.sqrt(n_blocks)}}


def decide(args) -> int:
    try:
        protocol = rc.load_cache_protocol(args.protocol, args.protocol_sha256)
        freeze = rc.bind_freeze(protocol)
        cfgs = debug_configs(protocol)
        allowed = list(freeze.role("debug")["allowed_issue_ids"])
        own = [i for c in cfgs for i in c["own_issue_ids"]]
        if sorted(own) != sorted(allowed) or len(cfgs) != 4:
            raise rc.Refused("DEBUG_SET_NOT_THE_ALLOW_LIST", "debug issues != freeze allow-list")
        record, admission_sha = rc.role_admission(protocol, "debug")
        pairs = cc.rows_and_certificates(record, allowed)
        rc.clear_rows(freeze, "debug", pairs, freeze.debug_allow_list(), admission_sha)
    except (rc.Refused, sf.SplitFreezeViolation, cc.CacheViolation, KeyError, OSError) as exc:
        print(f"ERROR: refused: {exc}", file=sys.stderr, flush=True)
        return 2
    job = Path(args.job_dir)
    shards, e0, copies = {}, {}, {}
    for c in cfgs:
        d = job / c["output_subdir"]
        rec = _json(d / "shard.json") if (d / "shard.json").is_file() else None
        shards[c["config_id"]] = rec
        e0[c["config_id"]] = bool(rec and rec.get("passed") and rec.get("e0_passed")
                                  and rec.get("config_id") == c["config_id"]
                                  and rec.get("protocol_sha256") == args.protocol_sha256)
        for it in (rec or {}).get("issues", []):
            p = Path(it["dir"]) / "issue.json"
            if cc.file_sha256(p) == it["issue_json_sha256"]:
                copies.setdefault(it["issue_id"], {})[c["config_id"]] = p
    q, scale, _ = rc.objective_on_cpu(protocol)
    owner = {i: c["config_id"] for c in cfgs for i in c["own_issue_ids"]}
    shard_of = {i: int(next(c["worker"] for c in cfgs if c["config_id"] == owner[i])) for i in allowed}
    identity = rc.identity_block(protocol, args.protocol_sha256, "debug")
    identity_sha = cc.canonical_digest(identity)
    pins = protocol["inputs"]["registry_entries"]
    rows, rehash, logs = [], {}, []
    dup, refeq, anchors = {}, {}, {}
    panels = {int(k): _json(rc._check_ref(ref, f"anchor {k}"))["experts"][str(k)]
              for k, ref in protocol["inputs"]["debug"]["anchors"].items()}
    group_of = {iid: k for k, p in panels.items() for iid in p["issue_ids"]}
    for iid in allowed:
        cp = copies.get(iid, {})
        if owner[iid] not in cp:
            continue
        p = rc.process_issue(cp[owner[iid]], q=q, scale=scale)
        rows += cc.matrix_rows_for_issue(p["issue"], cpu=p["cpu"], analytic=p["analytic"],
                                         entries_pin=pins, identity_sha256=identity_sha,
                                         features_sha256=None)
        rehash[iid] = p["rehash"]
        issues = {k: _json(v) for k, v in cp.items()}
        for it in issues.values():
            logs += it["read_log"]
        keys = ("endpoints", "truth", "f0_endpoints")
        dup[iid] = len(issues) == 2 and len({tuple(it["files"][k]["sha256"] for k in keys)
                                             for it in issues.values()}) == 1 \
            and len({json.dumps(_json(it["files"]["gpu_losses"]["path"]), sort_keys=True)
                     for it in issues.values()}) == 1
        refeq[iid] = all(it["reference_equals_plain_fs_bitwise"] and
                         it["reference_endpoints_array_sha256"] == it["plain_fs_endpoints_array_sha256"]
                         for it in issues.values())
        gl = _json(p["issue"]["files"]["gpu_losses"]["path"])
        k = group_of[iid]
        want_fs = panels[k]["Fs"][iid]
        want_l1 = panels[k]["L1"][iid]
        anchors[iid] = {
            "group": k,
            "fs": [(gl["candidates"]["reference"][h], want_fs[h]) for h in ("6", "24", "72")]
            + [(gl["plain_fs"][h], want_fs[h]) for h in ("6", "24", "72")],
            "expert": {"k": k, "pairs": [(gl["candidates"][f"expert_{k}"][h], want_l1[h])
                                         for h in ("6", "24", "72")]}}
    slices = {r["issue_id"]: ("debug", Path(r["issue_store"]).name, int(r["history_index"]),
                              int(r["target_index"])) for r, _ in pairs}
    v8 = sf.read_log_within(freeze, logs, slices)
    try:
        matrix = cc.validate_cache_matrix(rows, expected_issue_ids=allowed, entries_pin=pins,
                                          expected_identity_sha256=identity_sha, shard_of=shard_of,
                                          rehash=rehash, read_check=v8)
    except cc.CacheViolation as exc:
        matrix = exc.detail
    verdict = cc.decide_debug(e0_by_worker=e0, matrix_passed=bool(matrix.get("passed")),
                              anchors=anchors, duplicates_identical=dup,
                              reference_equals_plain_fs=refeq)
    decision = {"kind": "debug", "rule": "DEBUG-DECIDE-v1", "protocol_sha256": args.protocol_sha256,
                "job_dir": str(job), "verdict": verdict["verdict"], "decision": verdict,
                "e0_by_worker": e0, "matrix": matrix, "read_set": v8,
                "duplicates_identical": dup, "reference_equals_plain_fs": refeq,
                "anchors": anchors, "debug_issue_ids": allowed}
    cc.write_json_once(args.out, decision)
    print(json.dumps({"verdict": verdict["verdict"], "checks": verdict["checks"],
                      "anchors_exact": verdict["anchors_exact"]}))
    if verdict["verdict"] != "PASS" or args.dev_protocol_out is None:
        return 0 if verdict["verdict"] == "PASS" else 1
    # ---------------- DEV-SCALE-v1 (no loss is read below this line) -------------
    sel = _json(rc._check_ref(protocol["inputs"]["policy_dev"]["selection"], "selection"))
    subsets = {k: v["issue_ids"] for k, v in sel["nested_subsets"].items()}
    adm, _ = rc.role_admission(protocol, "policy_dev")
    admitted = [r["issue_id"] for r in adm["admission"]["admitted"]]
    issue_recs = [_json(p) for cp in copies.values() for p in cp.values()]
    art = Path(protocol["dev_scale_rule"]["artifact_root"])
    scale_in = {"subsets": subsets, "admitted_ids": admitted,
                "model_load_seconds": max(s["model_load_seconds"] for s in shards.values()),
                "issue_seconds": [it["issue_seconds"] for it in issue_recs],
                "bytes_per_issue": max(it["bytes"] for it in issue_recs),
                "free_disk_bytes": shutil.disk_usage(art).free,
                "peak_gpu_gib": max(s["peak_gpu_gib"] for s in shards.values())}
    ds = cc.decide_dev_scale(**scale_in)
    dev: Dict[str, Any] = {"schema": "ed-fp05-dev-protocol/1", "protocol_sha256": args.protocol_sha256,
                           "debug_decision": {"path": str(Path(args.out).resolve()),
                                              "sha256": cc.file_sha256(args.out)},
                           "dev_scale_inputs": {k: v for k, v in scale_in.items()
                                                if k not in ("subsets", "admitted_ids")},
                           "dev_scale": ds}
    if ds["verdict"] == "CHOSEN":
        order = {r["issue_id"]: r["issue_time"] for r in adm["admission"]["admitted"]}
        ids = sorted(subsets[ds["chosen_subset"]], key=lambda i: order[i])
        dev.update({"N": len(ids), "issue_ids": ids, "shard_map": cc.shard_map(ids),
                    "mde_planning": mde_table(protocol, [len(v) for v in subsets.values()], 23),
                    "coverage_caveat": protocol["split"]["coverage_caveat"]})
    cc.write_json_once(args.dev_protocol_out, dev)
    print(json.dumps({"dev_scale": ds["verdict"], "N": dev.get("N")}))
    return 0 if ds["verdict"] == "CHOSEN" else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["debug"], required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--job-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dev-protocol-out", type=Path, default=None)
    return decide(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
