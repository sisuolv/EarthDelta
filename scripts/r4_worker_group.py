#!/usr/bin/env python3
"""Run ready tasks on independent GPU processes, recording every exit code."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    a = p.parse_args()
    source = Path(__file__).resolve().parents[1]
    tasks = json.loads(a.tasks.read_text())
    if not 1 <= len(tasks) <= 4 or len({t["gpu"] for t in tasks}) != len(tasks):
        raise ValueError("one ready task sequence per GPU, at most four")
    a.out.mkdir(parents=True, exist_ok=False)
    children = []
    for task in tasks:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(task["gpu"])
        env["PYTHONPATH"] = ":".join(map(str, [source, source / "reference/stormer", a.run / "cuda_site", a.assets / ".pydeps"]))
        env["OMP_NUM_THREADS"] = "4"
        commands = [[word.replace("@SNAPSHOT@", str(source)).replace("@RUN@", str(a.run)) for word in cmd] for cmd in task["commands"]]
        # The child sequence exits immediately on its first failure; dependent
        # qualification can never run after failed training.
        script = "import subprocess,sys,json; commands=json.loads(sys.argv[1]);\nfor c in commands:\n r=subprocess.run(c)\n if r.returncode: sys.exit(r.returncode)\n"
        log = (a.out / f"gpu{task['gpu']}.log").open("x")
        proc = subprocess.Popen([sys.executable, "-c", script, json.dumps(commands)], env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        children.append((task, proc, log, time.monotonic()))
    records = []
    for task, proc, log, start in children:
        code = proc.wait()
        log.close()
        records.append({"gpu": task["gpu"], "returncode": code, "seconds": time.monotonic() - start})
    (a.out / "workers.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records), flush=True)
    return 0 if all(r["returncode"] == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
