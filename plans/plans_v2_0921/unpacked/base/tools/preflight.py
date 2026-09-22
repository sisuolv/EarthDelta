#!/usr/bin/env python3
"""Read-only local Git/asset inventory; never downloads, imports ML, or loads weights."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

AUDITED = "fb767f7f6efbc428be39c9ad84f5905331d6e40f"
OBSERVED = "403b55db65f4c35c1a85d0794ad0de2765b07d96"
BRANCH = "audit/round2-review-20260921"
EXPECTED = {
    "earthdelta": "867ea7c5ad43ffd7850cda3b6627287032a4db8f",
    "scripts": "17893b917ce06ffd7fed2655e15bb0a129268320",
    "tests": "e3c5d9d8bd207b492554487e7d9b0bd669ab4193",
    "pyproject.toml": "adcad73a0ae57fa8fe3bdaf830f200c20969fa55",
}

def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                            text=True, timeout=10, check=False)
    if result.returncode:
        # Do not emit environment or credential-bearing remote URLs.
        raise RuntimeError("local git command failed: " + " ".join(args[:2]))
    return result.stdout.strip()

def inspect(repo: Path) -> dict:
    receipt = {
        "iteration_id": "round2-next-iteration-20260921",
        "task_id": "P0-01", "scope": "metadata_only",
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "audited_source_head": AUDITED, "observed_remote_head_at_delivery": OBSERVED,
        "status": "BLOCKED", "blockers": [], "warnings": [],
        "tests_executed": False, "weights_loaded": False, "network_used": False,
        "weather_readiness": "BLOCKED_UNVERIFIED_ASSETS",
        "allowed_next_task": None,
    }
    if not repo.is_dir():
        receipt["blockers"].append("REPOSITORY_DIRECTORY_NOT_FOUND")
        return receipt
    try:
        root = Path(git(repo, "rev-parse", "--show-toplevel")).resolve()
        head = git(root, "rev-parse", "HEAD")
        branch = git(root, "symbolic-ref", "--short", "-q", "HEAD") if (
            subprocess.run(["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
                           capture_output=True, timeout=10).returncode == 0
        ) else "DETACHED"
        dirty = git(root, "status", "--short", "--untracked-files=normal").splitlines()
        receipt.update(repo=str(root), local_head=head, local_branch=branch, dirty_paths=dirty)
        if head not in {AUDITED, OBSERVED}:
            receipt["blockers"].append("HEAD_DRIFT_REQUIRES_EXPLICIT_RECONCILIATION")
        if dirty:
            receipt["blockers"].append("DIRTY_WORKTREE_REQUIRES_RECONCILIATION_DO_NOT_RESET")
        if branch != BRANCH:
            receipt["warnings"].append("BRANCH_NAME_DIFFERS_DO_NOT_SWITCH_AUTOMATICALLY")
        actual = {path: git(root, "rev-parse", "HEAD:" + path) for path in EXPECTED}
        receipt["source_identities"] = actual
        receipt["source_identities_match"] = actual == EXPECTED
        if actual != EXPECTED:
            receipt["blockers"].append("AUDITED_SOURCE_TREE_MISMATCH")
        assets = []
        checkpoint_dir = root / "checkpoints"
        if checkpoint_dir.is_dir():
            for p in sorted(checkpoint_dir.glob("*.ckpt")):
                if p.is_file():
                    assets.append({"path":str(p), "size_bytes":p.stat().st_size,
                                   "qualification":"EXISTENCE_ONLY_NOT_LOADED_NOT_VALIDATED"})
        receipt["checkpoint_inventory"] = assets
        receipt["data_directory_exists"] = (root / "data").is_dir()
        receipt["asset_note"] = "No weight deserialization, data-content scan, CUDA test or license attestation performed."
        if not receipt["blockers"]:
            receipt["status"] = "PREFLIGHT_METADATA_PASS"
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        receipt["blockers"].append(type(exc).__name__ + ": " + str(exc))
    return receipt

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        print("Refusing to overwrite an existing receipt.", file=sys.stderr)
        return 2
    receipt = inspect(args.repo.expanduser().resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status":receipt["status"], "receipt":str(out),
                      "allowed_next_task":None}, ensure_ascii=False))
    return 0 if receipt["status"] == "PREFLIGHT_METADATA_PASS" else 2

if __name__ == "__main__":
    raise SystemExit(main())
