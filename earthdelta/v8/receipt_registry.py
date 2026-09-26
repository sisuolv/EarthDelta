"""Closed-world receipt dependency validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import read_json, sha256_file


class ReceiptError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReceiptSpec:
    receipt_id: str
    stage_id: str
    files: tuple[str, ...]
    required_fields: tuple[str, ...]
    release_gate: str


class ReceiptRegistry:
    def __init__(self, contract_root: str | Path):
        self.root = Path(contract_root)
        raw = read_json(self.root / "RECEIPT_REGISTRY.json")
        if raw.get("closed_world") is not True:
            raise ReceiptError("receipt registry must be closed-world")
        self.specs = {}
        for rid, spec in raw.get("receipts", {}).items():
            if rid != spec.get("receipt_id"):
                raise ReceiptError(f"receipt id mismatch: {rid}")
            self.specs[rid] = ReceiptSpec(rid, spec["stage_id"], tuple(spec["files"]),
                                          tuple(spec["required_fields"]), spec.get("release_gate", ""))
        self._check_stage_graph()

    def _check_stage_graph(self) -> None:
        autorun = read_json(self.root / "AUTORUN_R4.json")
        stages = {s["stage_id"]: s for s in autorun["stages"]}
        graph = {sid: set() for sid in stages}
        for stage in stages.values():
            for rid in stage.get("depends_on_receipts", []):
                if rid not in self.specs:
                    raise ReceiptError(f"unknown dependency receipt: {rid}")
                graph[stage["stage_id"]].add(self.specs[rid].stage_id)
        visiting, done = set(), set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ReceiptError("receipt/stage dependency cycle")
            if node in done:
                return
            visiting.add(node)
            for parent in graph[node]:
                visit(parent)
            visiting.remove(node)
            done.add(node)

        for node in graph:
            visit(node)

    def spec(self, receipt_id: str) -> ReceiptSpec:
        try:
            return self.specs[receipt_id]
        except KeyError as exc:
            raise ReceiptError(f"unknown receipt id: {receipt_id}") from exc

    def load(self, receipt_id: str, run_dir: str | Path) -> dict[str, Any]:
        spec = self.spec(receipt_id)
        run = Path(run_dir)
        paths = [run / f for f in spec.files]
        if not all(p.is_file() for p in paths):
            raise ReceiptError(f"missing receipt files for {receipt_id}")
        data = read_json(paths[0])
        if data.get("status") != "OBSERVED":
            raise ReceiptError(f"receipt {receipt_id} is not OBSERVED")
        missing = [field for field in spec.required_fields if field not in data]
        if missing:
            raise ReceiptError(f"receipt {receipt_id} missing fields: {missing}")
        return data

    def observed(self, receipt_id: str, run_dir: str | Path) -> bool:
        try:
            self.load(receipt_id, run_dir)
            return True
        except ReceiptError:
            return False

    def hash_manifest(self) -> dict[str, str]:
        return {name: sha256_file(self.root / name) for name in (
            "RECEIPT_REGISTRY.json", "AUTORUN_R4.json", "TASKS_R4.json", "METRIC_CONTRACT_R4.json",
            "ROLES_R4.json", "RECIPES_R4.json")}

