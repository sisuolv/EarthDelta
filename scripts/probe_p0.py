#!/usr/bin/env python3
"""Run the CPU-only P0 admission checks for the headroom probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


BASE = "a0e4acf82c690e37248bb6d0410cec4c053493c1"
TAGS = {
    "474a8e1": "v8-hist-474a8e1",
    "c086237": "v8-hist-c086237",
    "747fde6": "v8-hist-747fde6",
    "382356c": "v8-hist-382356c",
    "d5628ea": "v8-hist-d5628ea",
    "a26e85d": "v8-hist-a26e85d",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch != "probe-headroom-20260927" or head != BASE:
        raise SystemExit(f"P0 must run before the P0 commit on the approved branch, got {branch} {head}")
    for short, tag in TAGS.items():
        tagged = subprocess.check_output(["git", "rev-parse", f"{tag}^{{commit}}"], cwd=repo, text=True).strip()
        actual = subprocess.check_output(["git", "rev-parse", f"{short}^{{commit}}"], cwd=repo, text=True).strip()
        if tagged != actual:
            raise SystemExit(f"historical tag mismatch: {tag}")
    quarantine = repo / "data/era5_1p40625_v8/2019.QUARANTINE.json"
    deviations = run / "DEVIATIONS.md"
    if not quarantine.is_file() or not deviations.is_file():
        raise SystemExit("P0 quarantine/deviation receipt is missing")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_probe_contracts.py",
         "tests/test_probe_edits.py", "tests/test_probe_estimators.py"],
        cwd=repo, text=True, capture_output=True,
        env={**__import__("os").environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
    )
    (run / "P0_TEST_OUTPUT.txt").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise SystemExit("P0 synthetic tests failed; see P0_TEST_OUTPUT.txt")
    receipt = {
        "schema": "earthdelta.probe.p0_receipt.v1",
        "status": "OBSERVED",
        "base_commit": BASE,
        "branch": branch,
        "historical_tags": {tag: subprocess.check_output(["git", "rev-parse", f"{tag}^{{commit}}"], cwd=repo, text=True).strip() for tag in TAGS.values()},
        "quarantine_sha256": digest(quarantine),
        "deviations_sha256": digest(deviations),
        "synthetic_tests": "P0_TEST_OUTPUT.txt",
        "synthetic_test_returncode": result.returncode,
        "forbidden": ["v8 runner", "2019/legacy/2021/2022 payload", "network download"],
    }
    (run / "P0_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
