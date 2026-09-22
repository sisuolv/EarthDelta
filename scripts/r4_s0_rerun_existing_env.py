#!/usr/bin/env python3
"""Run an isolated export + S0 gate using an already prepared CUDA overlay."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--overlay-run", type=Path, required=True)
    args = p.parse_args()
    source = Path(__file__).resolve().parents[1]
    config = args.run / "s0/gate_config.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(
        [str(source), str(source / "reference/stormer"),
         str(args.overlay_run / "cuda_site"), str(args.assets / ".pydeps")]
    )
    env["OMP_NUM_THREADS"] = "4"
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    wrapper = source / "scripts/r4_run_strict_fp32.py"
    reference = args.run / "s0/reference/upstream_reference_ps4_fixed"
    stages = {"status": "RUNNING", "run": str(args.run), "stages": []}

    def run_stage(name: str, argv: list[str], timeout: int) -> int:
        started = time.monotonic()
        log_path = args.run / "s0" / f"{name}.log"
        with log_path.open("w") as log:
            result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout)
        stages["stages"].append({"name": name, "returncode": result.returncode,
                                 "seconds": time.monotonic() - started,
                                 "log": str(log_path)})
        (args.run / "s0/rerun_stages.json").write_text(json.dumps(stages, indent=2) + "\n")
        return result.returncode

    export_code = run_stage(
        "export", [sys.executable, str(wrapper), str(source / "scripts/export_upstream_reference.py"),
                    "--config", str(config), "--output-dir", str(reference)], 900)
    if export_code:
        stages["status"] = "BLOCKED_EXPORT"
        (args.run / "s0/rerun_stages.json").write_text(json.dumps(stages, indent=2) + "\n")
        return export_code

    if not reference.is_dir():
        stages["status"] = "BLOCKED_REFERENCE_SELECTION"
        stages["references"] = []
        (args.run / "s0/rerun_stages.json").write_text(json.dumps(stages, indent=2) + "\n")
        return 1
    gate_code = run_stage(
        "gate", [sys.executable, str(wrapper), str(source / "scripts/s0_gate.py"),
                  "--config", str(config), "--reference-dir", str(reference),
                  "--output-dir", str(args.run / "s0/gate_output")], 900)
    stages["reference_dir"] = str(reference)
    stages["status"] = "PASS" if gate_code == 0 else "BLOCKED_GATE"
    stages["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (args.run / "s0/rerun_stages.json").write_text(json.dumps(stages, indent=2) + "\n")
    return gate_code


if __name__ == "__main__":
    raise SystemExit(main())
