#!/usr/bin/env python3
"""CP0 CPU dry-run of every C-J1 worker's E0a/E0b bindings against the FROZEN protocol.

Runs, on CPU and without loading any backbone or reading any data value:
protocol load by hash, split-freeze load (+ ledger consistency), worker
resolution, split-freeze clearance with the allow-lists, source pins,
authorize_certified_fs, verify_bank_bundle(deep=True), registry pin check, and
the CPU objective's normalization identity. The S0 certificate and the
official-backend checks need the GPU job's torch 2.3.1/xformers and are
exercised there (and refused-before-model in tests/test_fp05_consumption_chain.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/mnt/afs/260010168/EarthDelta")
sys.path.insert(0, str(REPO))
RUN = REPO / "plans/plan_v4_0923/run_20260924T104725Z_fp05b"

import scripts.r2_fs_bank_train as r2  # noqa: E402
import scripts.r4_candidate_cache as rc  # noqa: E402
from earthdelta import candidate_cache as cc  # noqa: E402
from earthdelta import split_freeze as sf  # noqa: E402


def main() -> None:
    proto_path = RUN / "protocol/cache_protocol_v1.json"
    digest = (RUN / "protocol/cache_protocol_v1.sha256").read_text().split()[0]
    p = rc.load_cache_protocol(proto_path, digest)
    freeze = sf.load_freeze(p["inputs"]["split_freeze"]["path"], p["inputs"]["split_freeze"]["sha256"],
                            ledger=p["inputs"]["exposure_ledger"]["path"])
    out = {"protocol_sha256": digest, "freeze_ledger_consistent": True, "workers": {}}
    for w in range(4):
        cfg, role, ids, shard = rc.resolve_worker(p, digest, f"CJ1_W{w}")
        record, adm = rc.role_admission(p, role)
        rep = rc.clear_rows(freeze, role, cc.rows_and_certificates(record, ids),
                            rc.allow_list(p, freeze, role), adm)
        out["workers"][f"CJ1_W{w}"] = {"issue_ids": ids, "clear": rep["passed"], "device": cfg["device"]}
    record, adm = rc.role_admission(p, "policy_dev")
    ids = p["split"]["allow_lists"]["policy_dev"]["issue_ids"]
    out["policy_dev_clear"] = rc.clear_rows(freeze, "policy_dev", cc.rows_and_certificates(record, ids),
                                            rc.allow_list(p, freeze, "policy_dev"), adm)["passed"]
    out["sources"] = rc.check_sources(p)
    auth = r2.authorize_certified_fs(rc._fs_args(p), p)
    out["fs_authorized"] = auth["passed"]
    bank_rep, registry = rc.verify_bank(p)
    out["bank_bundle"] = bank_rep
    _, _, norm = rc.objective_on_cpu(p)
    out["normalization_digest"] = norm["digest"]
    out["passed"] = all(v["clear"] for v in out["workers"].values()) and out["policy_dev_clear"] \
        and out["fs_authorized"] and bank_rep["bundle_report_passed"]
    cc.write_json_once(RUN / "analysis/cpu_e0_dryrun.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "workers"}))


if __name__ == "__main__":
    main()
