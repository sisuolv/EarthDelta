#!/usr/bin/env python3
"""Fail-closed GPU submission wrapper for v8 stages.

This wrapper performs all local checks before invoking ``sco``.  It never
submits a job for an unapproved scope or a budget overrun and supports a
dry-run used by CPU tests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from earthdelta.v8.approvals import ApprovalVerifier
from earthdelta.v8.io import append_jsonl, read_json
from earthdelta.v8.receipt_registry import ReceiptRegistry

APPROVALS = Path("/mnt/afs/260010168/earthdelta_v8_approvals/approvals.json")
APPROVALS_SHA = APPROVALS.with_name("approvals.sha256")
R4_ROOT = Path(os.environ.get("EARTHDELTA_V8_R4_ROOT", "/mnt/afs/260010168/earthdelta_v8_planning_20260926T182037Z_r4"))
PROMPT_SHA = "a9ac5e9c7c5cd06b43c1ccb290f86b349a7acbfa2eaafd61b2f7c20fd5bb358c"
AMEND_SHA = "36d1342bd5f147be945253dd1a40fb252e1ed24ac22d424dd86c7e4496918286"
R3_MANIFEST_SHA = "020797ecb13129e111cd02fdd31c311f9b469cb840a1489af4a74fb5267b247f"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scope", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--engine", choices=("H100", "spot_H100", "spot_5090"), required=True)
    p.add_argument("--card-hours", type=float, required=True)
    p.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    run = Path(args.run); run.mkdir(parents=True, exist_ok=True)
    verifier = ApprovalVerifier(APPROVALS, APPROVALS_SHA, expected_prompt_sha=PROMPT_SHA,
                                expected_amendments_sha=AMEND_SHA, expected_r3_manifest_sha=R3_MANIFEST_SHA)
    reg = ReceiptRegistry(R4_ROOT)
    available = {rid for rid in reg.specs if reg.observed(rid, run)}
    scope = verifier.require([args.scope], available_receipts=available)[0]
    if args.card_hours < 0 or args.card_hours > scope.budget_cap_card_hours:
        raise SystemExit(f"budget exceeds scope cap {scope.budget_cap_card_hours}")
    if args.engine == "spot_5090":
        if scope.scope_id.startswith("T04") or scope.scope_id == "T02_gate_gpu_max2":
            raise SystemExit("spot 5090 cannot run gradient/G0/confirm scope")
        if "E5090_receipt" not in available:
            raise SystemExit("spot 5090 requires E5090 receipt")
    entry = {"scope": scope.scope_id, "engine": args.engine, "card_hours": args.card_hours,
             "command": args.command, "dry_run": args.dry_run}
    append_jsonl(run / "GPU_SUBMISSIONS.jsonl", entry)
    if args.dry_run:
        print(json.dumps({"status": "OBSERVED", "dry_run": True, **entry}))
        return 0
    if not args.command:
        raise SystemExit("missing executable command")
    proc = subprocess.run(args.command, cwd=Path(__file__).resolve().parents[1], check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
