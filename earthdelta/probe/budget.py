"""Cumulative ACP card-hour accounting for probe stages."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
from typing import Any


class BudgetBlocked(RuntimeError):
    pass


@dataclass
class BudgetLedger:
    path: Path
    h100_cap: float = 12.0
    rtx5090_cap: float = 64.0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _sum(self, gpu: str) -> float:
        total = 0.0
        if not self.path.exists():
            return total
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("gpu") == gpu:
                total += float(row.get("card_hours", 0.0))
        return total

    def used(self, gpu: str) -> float:
        return self._sum(gpu)

    def ensure(self, gpu: str, projected_card_hours: float) -> None:
        cap = self.h100_cap if gpu == "H100" else self.rtx5090_cap if gpu == "RTX5090" else 0.0
        if cap <= 0:
            raise BudgetBlocked(f"unknown GPU class: {gpu}")
        if self.used(gpu) + float(projected_card_hours) > cap + 1e-9:
            raise BudgetBlocked(f"{gpu} budget would exceed cap {cap}")

    def record(self, *, job_id: str, gpu: str, card_hours: float, stage: str,
               status: str, projected: bool = False, **extra: Any) -> None:
        if card_hours < 0:
            raise ValueError("card_hours must be nonnegative")
        if not projected:
            self.ensure(gpu, card_hours)
        row = {
            "schema": "earthdelta.probe.budget.v1",
            "job_id": str(job_id), "gpu": str(gpu), "card_hours": float(card_hours),
            "stage": str(stage), "status": str(status), "projected": bool(projected),
            **extra,
        }
        with self.path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
