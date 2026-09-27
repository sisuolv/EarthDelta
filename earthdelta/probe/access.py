"""Fail-closed access guard and append-only ledger for the 2020 probe."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .contracts import ProbeContractError, ProbeSpec, issue_index, required_indices


class AccessViolation(RuntimeError):
    pass


def _canonical(path: str | Path) -> str:
    return str(Path(path).resolve())


class ProbeAccessController:
    """Authorize only decoded payloads from the bound 2020 store.

    Every read is recorded before the payload is returned.  The controller is
    intentionally strict: a path outside the enrolled store, an index outside
    the registered issue windows, or a forbidden year raises ``AccessViolation``.
    """

    def __init__(self, spec: ProbeSpec, ledger_path: str | Path, *, run_id: str = "local"):
        self.spec = spec
        self.store = _canonical(spec.store_path)
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.allowed_indices = frozenset(required_indices(spec))
        self.opens = 0

    def _check_store(self, path: str | Path) -> None:
        actual = _canonical(path)
        if actual != self.store:
            raise AccessViolation(f"payload path is not the bound store: {actual}")

    def _check_indices(self, indices: Iterable[int]) -> list[int]:
        values = sorted({int(x) for x in indices})
        if not values:
            raise AccessViolation("empty payload read is not registered")
        if any(x < 0 or x not in self.allowed_indices for x in values):
            raise AccessViolation(f"payload index outside registered windows: {values}")
        if any(x in self.spec.absent_indices for x in values):
            raise AccessViolation(f"payload index is absent in bound store: {values}")
        return values

    def _append(self, row: dict) -> None:
        self.ledger_path.touch(exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def authorize(self, path: str | Path, indices: Iterable[int], *, purpose: str,
                  stage: str, job: str = "local") -> list[int]:
        self._check_store(path)
        values = self._check_indices(indices)
        self._append({
            "schema": "earthdelta.probe.access.v1",
            "run_id": self.run_id,
            "path": self.store,
            "indices": values,
            "purpose": str(purpose),
            "stage": str(stage),
            "job": str(job),
            "array_payload": True,
        })
        return values

    def authorize_coordinate(self, path: str | Path, name: str, *, purpose: str,
                             stage: str, job: str = "local") -> None:
        self._check_store(path)
        self._append({
            "schema": "earthdelta.probe.access.v1",
            "run_id": self.run_id,
            "path": self.store,
            "coordinate": str(name),
            "purpose": str(purpose),
            "stage": str(stage),
            "job": str(job),
            "array_payload": False,
        })

    def open_zarr(self, path: str | Path, *, stage: str, job: str = "local"):
        self._check_store(path)
        try:
            import zarr
        except ImportError as exc:  # pragma: no cover - deployment image
            raise AccessViolation("zarr is required in the execution image") from exc
        self.opens += 1
        self._append({
            "schema": "earthdelta.probe.access.v1",
            "run_id": self.run_id,
            "path": self.store,
            "purpose": "open_bound_store",
            "stage": str(stage),
            "job": str(job),
            "array_payload": False,
        })
        return zarr.open_group(self.store, mode="r")

    def read(self, path: str | Path, indices: Iterable[int], *, purpose: str,
             stage: str, job: str = "local") -> np.ndarray:
        values = self.authorize(path, indices, purpose=purpose, stage=stage, job=job)
        group = self.open_zarr(path, stage=stage, job=job)
        try:
            data = np.asarray(group["data"][values])
        except Exception as exc:
            raise AccessViolation(f"bound store read failed: {type(exc).__name__}: {exc}") from exc
        if data.shape[0] != len(values):
            raise AccessViolation("decoded payload length differs from authorized indices")
        digest = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
        self._append({
            "schema": "earthdelta.probe.access.v1",
            "run_id": self.run_id,
            "path": self.store,
            "indices": values,
            "purpose": str(purpose),
            "stage": str(stage),
            "job": str(job),
            "array_payload": True,
            "decoded_sha256": digest,
            "shape": list(data.shape),
        })
        return data

    def read_coordinate(self, path: str | Path, name: str, *, purpose: str,
                        stage: str, job: str = "local") -> np.ndarray:
        self.authorize_coordinate(path, name, purpose=purpose, stage=stage, job=job)
        group = self.open_zarr(path, stage=stage, job=job)
        if name not in group:
            raise AccessViolation(f"missing coordinate: {name}")
        data = np.asarray(group[name][:])
        self._append({
            "schema": "earthdelta.probe.access.v1",
            "run_id": self.run_id,
            "path": self.store,
            "coordinate": str(name),
            "purpose": str(purpose),
            "stage": str(stage),
            "job": str(job),
            "array_payload": False,
            "decoded_sha256": hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest(),
            "shape": list(data.shape),
        })
        return data


BoundStore = ProbeAccessController
