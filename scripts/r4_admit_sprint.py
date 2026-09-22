#!/usr/bin/env python3
"""Read every selected history/input/target and publish a bound admission record."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
from pathlib import Path
import time

from earthdelta.value_pilot import (
    assert_disjoint_roles, canonical_digest, file_digest, load_frozen_run,
    read_row_content, verify_store, write_json,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    spec, stores, candidates = load_frozen_run(a.run)
    opened = {year: verify_store(store) for year, store in stores.items()}
    groups = {}
    for row in candidates:
        groups.setdefault((row["role"], row["expert_index"]), []).append(row)
    admitted, excluded = [], []

    def inspect(row):
        try:
            _, hashes = read_row_content(row, opened[str(row["year"])])
            return {**row, "content_verified": True, "content_sha256": hashes}, None
        except Exception as exc:
            return None, {"issue_id": row["issue_id"], "reason": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=4) as pool:
        for (role, expert), rows in groups.items():
            needed = sum(r["primary"] for r in rows)
            accepted = []
            cursor = 0
            while len(accepted) < needed:
                batch = rows[cursor:cursor + (needed - len(accepted))]
                if not batch:
                    raise RuntimeError(f"Insufficient valid content in {role}/{expert}")
                for row, error in pool.map(inspect, batch):
                    if error:
                        excluded.append(error)
                    else:
                        row["admitted_order"] = len(accepted)
                        accepted.append(row)
                cursor += len(batch)
            admitted.extend(accepted)
            print(role, expert, len(accepted), "elapsed", round(time.monotonic() - start, 1), flush=True)
    assert_disjoint_roles(admitted)
    record = {"status": "PASS", "spec_sha256": file_digest(a.run / "frozen_spec.json"),
              "rows": admitted, "rows_sha256": canonical_digest(admitted), "excluded": excluded,
              "coverage": {r: sum(x["role"] == r for x in admitted) for r in sorted({x["role"] for x in admitted})},
              "content_checked": "all five history/input/6h/24h/72h arrays for every admitted issue",
              "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "seconds": time.monotonic() - start}
    write_json(a.run / "admission.json", record)
    print(record["coverage"], "total_seconds", record["seconds"], flush=True)


if __name__ == "__main__":
    main()
