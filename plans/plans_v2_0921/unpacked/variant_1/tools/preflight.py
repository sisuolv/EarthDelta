#!/usr/bin/env python3
"""Read-only Git preflight. Never imports torch, reads credentials, or loads models."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import platform
import subprocess
import sys

EXPECTED_HEAD = "403b55db65f4c35c1a85d0794ad0de2765b07d96"
AUDITED_HEAD = "fb767f7f6efbc428be39c9ad84f5905331d6e40f"
EXPECTED_BRANCH = "audit/round2-review-20260921"
SCOPES = ["earthdelta", "scripts", "tests", "pyproject.toml"]
REQUIRED = ["earthdelta/metrics_contract.py", "earthdelta/bridge/stormer_bridge.py",
            "scripts/s0_gate.py", "scripts/export_upstream_reference.py",
            "codex_audit_round2/results/evidence/pytest_cpu.log"]

def git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True, timeout=15, check=False)
    if p.returncode:
        raise RuntimeError(f"git command failed ({p.returncode}): {' '.join(args)}")
    return p.stdout.strip()

def inspect(repo: Path) -> dict:
    result = {
        "tool": "package_read_only_preflight", "task_id": "R2I-01",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "expected_delivery_head": EXPECTED_HEAD, "audited_source_head": AUDITED_HEAD,
        "expected_branch": EXPECTED_BRANCH, "requested_repo": str(repo),
        "python": sys.version.split()[0], "platform": platform.system(),
        "status": "BLOCKED", "evidence_grade": "STRUCTURAL_ONLY",
        "repository_tests": "NOT_RUN", "gpu_s0": "NOT_RUN", "training": "NOT_RUN",
        "remote_access": "NOT_PERFORMED", "checkpoint_data_status": "UNVERIFIED",
        "next_task_authorized": False, "reason": None,
    }
    try:
        if not repo.is_dir():
            raise ValueError("Repository path does not exist or is not a directory")
        actual_root = Path(git(repo, "rev-parse", "--show-toplevel")).resolve()
        result["repository_root"] = str(actual_root)
        result["actual_head"] = git(actual_root, "rev-parse", "HEAD")
        result["actual_branch"] = git(actual_root, "rev-parse", "--abbrev-ref", "HEAD")
        # Only names/status, never diffs or remote URL contents that could contain secrets.
        result["working_tree_status"] = git(actual_root, "status", "--porcelain=v1").splitlines()
        result["required_files"] = {p: (actual_root / p).is_file() for p in REQUIRED}
        try:
            result["audited_to_head_scoped_diff_names"] = git(
                actual_root, "diff", "--name-status", AUDITED_HEAD, "HEAD", "--", *SCOPES
            ).splitlines()
        except RuntimeError:
            result["audited_to_head_scoped_diff_names"] = None
            result["audit_object_available"] = False
        else:
            result["audit_object_available"] = True
        blockers = []
        if result["actual_head"] != EXPECTED_HEAD:
            blockers.append("HEAD_DRIFT_REQUIRES_RECONCILIATION")
        if result["actual_branch"] != EXPECTED_BRANCH:
            blockers.append("BRANCH_DIFF_REQUIRES_RECONCILIATION")
        if result["working_tree_status"]:
            blockers.append("DIRTY_TREE_REQUIRES_OWNERSHIP_REVIEW_NO_RESET")
        if not all(result["required_files"].values()):
            blockers.append("MISSING_EXPECTED_FILES")
        if result["audited_to_head_scoped_diff_names"] is None:
            blockers.append("AUDITED_OBJECT_NOT_AVAILABLE_LOCALLY")
        elif result["audited_to_head_scoped_diff_names"]:
            blockers.append("SOURCE_DIFF_REQUIRES_RECONCILIATION")
        result["reason"] = blockers or ["IDENTITY_ONLY_NOT_S0_OR_RESEARCH_PASS"]
        result["status"] = "BLOCKED_RECONCILIATION" if blockers else "IDENTITY_CHECKED"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result["reason"] = [f"{type(exc).__name__}: {exc}"]
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        print("Refusing to overwrite an existing preflight artifact", file=sys.stderr)
        return 3
    result = inspect(args.repo.expanduser().resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
    except FileExistsError:
        print("Output appeared concurrently; refusing overwrite", file=sys.stderr)
        return 3
    print(json.dumps({"status": result["status"], "artifact": str(out),
                      "next_task_authorized": False}, ensure_ascii=False))
    return 0 if result["status"] == "IDENTITY_CHECKED" else 2

if __name__ == "__main__":
    raise SystemExit(main())
