#!/usr/bin/env python3
"""FP-05a: verify the 17 FP-04-pinned source files against bank_protocol_v1.json.

Read-only. Stdlib only (hashlib/json). Writes a new file; refuses to overwrite.
Usage: pin_check.py <label>   (label = before | after)
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
PROTO = REPO / "plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/protocol/bank_protocol_v1.json"
PROTO_SHA_FILE = PROTO.with_suffix(".sha256")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    label = sys.argv[1]
    out = RUN / "entry" / f"core_source_pin_check_{label}.json"
    proto_sha = sha256(PROTO)
    recorded = PROTO_SHA_FILE.read_text().split()[0] if PROTO_SHA_FILE.exists() else None
    pins = json.loads(PROTO.read_text())["source_at_preregistration"]
    rows = []
    for rel, exp in pins.items():
        p = REPO / rel
        act = sha256(p) if p.is_file() else None
        rows.append({"path": rel, "expected_sha256": exp, "actual_sha256": act,
                     "match": act == exp})
    res = {
        "schema": "ed-fp05a-core-pin-check/1",
        "label": label,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pin_source": str(PROTO.relative_to(REPO)),
        "pin_source_sha256": proto_sha,
        "pin_source_sha256_recorded_sidecar": recorded,
        "pin_source_sidecar_match": recorded == proto_sha if recorded else None,
        "n_pinned": len(rows),
        "n_match": sum(r["match"] for r in rows),
        "all_match": len(rows) == 17 and all(r["match"] for r in rows),
        "files": rows,
    }
    with out.open("x") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(json.dumps({k: res[k] for k in ("label", "n_pinned", "n_match", "all_match",
                                          "pin_source_sha256", "pin_source_sidecar_match")}))
    return 0 if res["all_match"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
