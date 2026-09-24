#!/usr/bin/env python3
"""CPU GLUE SMOKE of `r4_candidate_cache.py cache` on ONE debug issue -- NOT EVIDENCE.

Purpose: exercise the worker's end-to-end file/record glue (E0a clearance, E0c
real-model identity, read_slab, 7 rollouts, file writing, issue.json/shard.json)
before spending a GPU job. Differences from C-J1, all deliberate and recorded:
a copy of the frozen protocol with CJ1_W0 restricted to its first debug issue and
device "cpu" (so its sha differs); the S0 certificate / official-backend checks
are stubbed (the local CPU torch is 2.3.0a0 and has no xformers: those checks
are exercised on the GPU and by the chain tests). The issue is an FP-04 training
row (already GRADIENT_AND_RESULT exposed); no policy_dev row is touched. CPU
float results are NOT comparable bitwise to the GPU anchors and are not used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path("/mnt/afs/260010168/EarthDelta")
sys.path.insert(0, str(REPO))
RUN = REPO / "plans/plan_v4_0923/run_20260924T104725Z_fp05b"
OUT = RUN / "analysis/cpu_glue_smoke"

import torch  # noqa: E402

torch.set_num_threads(2)

import scripts.r4_candidate_cache as rc  # noqa: E402
from earthdelta import fs_protocol as fp  # noqa: E402

fp.consume_s0_certificate = lambda *a, **k: {"passed": True, "stubbed": "cpu smoke"}
fp.official_backend_precondition = lambda **k: {"passed": True, "stubbed": "cpu smoke"}
fp.official_backend_postcondition = lambda model: {"passed": True, "stubbed": "cpu smoke"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    p = json.loads((RUN / "protocol/cache_protocol_v1.json").read_text())
    q = copy.deepcopy(p)
    cfg = q["configs"]["CJ1_W0"]
    cfg["issue_ids"] = cfg["own_issue_ids"][:1]
    cfg["device"] = "cpu"
    q["smoke_note"] = "CPU glue smoke copy; not the frozen protocol"
    path = OUT / "smoke_protocol_copy.json"
    path.write_text(json.dumps(q, indent=1))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rc_code = rc.main(["cache", "--protocol", str(path), "--protocol-sha256", digest,
                       "--config-id", "CJ1_W0", "--device", "cpu", "--output-dir", str(OUT / "w0")])
    shard = json.loads((OUT / "w0/shard.json").read_text())
    issue_dir = Path(shard["issues"][0]["dir"])
    qq, scale, _ = rc.objective_on_cpu(p)
    proc = rc.process_issue(issue_dir / "issue.json", q=qq, scale=scale)
    issue = proc["issue"]
    rel = max(abs(proc["cpu"][ci][j] - issue["gpu_losses"][c][str(h)]) / proc["cpu"][ci][j]
              for ci, c in enumerate(["reference", "expert_0", "expert_1", "expert_2", "expert_3"])
              for j, h in enumerate([6, 24, 72]))
    summary = {"NOT_EVIDENCE": True, "rc": rc_code, "shard_passed": shard["passed"],
               "e0": {k: v for k, v in shard["e0"].items() if k != "bank_bundle"},
               "digests_unchanged": shard["digests_unchanged"],
               "reference_equals_plain_fs_bitwise": issue["reference_equals_plain_fs_bitwise"],
               "hooks_clean": issue["hooks_clean"], "rehash": proc["rehash"],
               "v5_max_rel_cpu_vs_worker": rel, "read_log": issue["read_log"],
               "issue_seconds": issue["issue_seconds"], "model_load_seconds": shard["model_load_seconds"],
               "bytes": issue["bytes"]}
    (OUT / "smoke_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "read_log"}))


if __name__ == "__main__":
    main()
