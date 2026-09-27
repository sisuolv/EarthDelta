"""Deterministic shards and verified resume helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def shard_issue_ids(issue_ids: Iterable[str], *, max_issues: int = 8) -> list[list[str]]:
    values = list(issue_ids)
    if max_issues <= 0:
        raise ValueError("max_issues must be positive")
    return [values[i:i + max_issues] for i in range(0, len(values), max_issues)]


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_completed_shard(path: str | Path, *, expected_input_hash: str,
                           expected_spec_sha256: str) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
    except Exception:
        return False
    if row.get("status") != "OBSERVED":
        return False
    if row.get("input_sha256") != expected_input_hash:
        return False
    if row.get("spec_sha256") != expected_spec_sha256:
        return False
    payload = row.get("payload_file")
    if not payload or not Path(payload).is_file():
        return False
    return row.get("payload_sha256") == _digest(Path(payload))
