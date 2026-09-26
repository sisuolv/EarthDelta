"""Small append-only and hashing helpers used by the v8 runner."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path: str | os.PathLike[str], value: Any) -> None:
    """Write a JSON file atomically, refusing to replace a different receipt."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if target.exists():
        old = target.read_text(encoding="utf-8")
        if old != payload:
            raise FileExistsError(f"append-only receipt already differs: {target}")
        return
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def replace_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically replace a mutable status file (receipts use atomic_json)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_jsonl(path: str | os.PathLike[str], row: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
