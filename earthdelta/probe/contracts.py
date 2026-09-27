"""Machine-checked contracts for the approved 2020 headroom probe."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ProbeContractError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_probe_spec(path: str | Path, *, expected_sha256: str | None = None) -> "ProbeSpec":
    path = Path(path)
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ProbeContractError(f"spec hash mismatch: {actual} != {expected_sha256}")
    raw = json.loads(path.read_text())
    if raw.get("schema") != "earthdelta.probe.headroom.spec.v1":
        raise ProbeContractError("unsupported probe specification schema")
    return ProbeSpec(raw=raw, path=str(path), sha256=actual)


@dataclass(frozen=True)
class ProbeSpec:
    raw: Mapping[str, Any]
    path: str
    sha256: str

    @property
    def spec_id(self) -> str:
        return str(self.raw["spec_id"])

    @property
    def store_path(self) -> str:
        return str(self.raw["bindings"]["data_store"]["path"])

    @property
    def variables(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.raw["metrics"]["variables"])

    @property
    def leads(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.raw["metrics"]["leads_hours"])

    @property
    def rollout_steps(self) -> tuple[int, ...]:
        return tuple(int(self.raw["metrics"]["rollout_steps"][str(h)]) for h in self.leads)

    @property
    def primary_cells(self) -> tuple[str, ...]:
        return tuple(self.raw["metrics"]["primary_cells"])

    @property
    def absent_indices(self) -> frozenset[int]:
        return frozenset(int(x) for x in self.raw["bindings"]["data_store"].get("absent_indices", []))

    @property
    def issue_sets(self) -> Mapping[str, Mapping[str, Any]]:
        return self.raw["issue_sets"]

    @property
    def h100_cap(self) -> float:
        return float(self.raw["resources"]["h100"]["card_hours_cap_total"])

    @property
    def rtx5090_cap(self) -> float:
        return float(self.raw["resources"]["rtx5090"]["card_hours_cap_total"])

    @property
    def decision(self) -> Mapping[str, str]:
        return self.raw["decision"]


def issue_sets(spec: ProbeSpec) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, value in spec.issue_sets.items():
        rows = list(value.get("issue_times_utc", []))
        if len(rows) != int(value["n"]):
            raise ProbeContractError(f"{name}: declared n does not match issue list")
        if rows != sorted(rows) or len(set(rows)) != len(rows):
            raise ProbeContractError(f"{name}: issue list must be sorted and unique")
        digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
        if digest != value.get("list_sha256"):
            raise ProbeContractError(f"{name}: issue list hash mismatch")
        out[name] = rows
    return out


def issue_index(issue_time: str, *, year: int = 2020) -> int:
    value = issue_time.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value).astimezone(timezone.utc)
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    hours = (dt - start).total_seconds() / 3600.0
    if hours % 6 != 0:
        raise ProbeContractError(f"issue is not aligned to 6h grid: {issue_time}")
    return int(hours // 6)


def required_indices(spec: ProbeSpec, *, include_sets: tuple[str, ...] | None = None) -> set[int]:
    sets = issue_sets(spec)
    names = include_sets or tuple(sets)
    indices: set[int] = set()
    max_step = max(spec.rollout_steps)
    for name in names:
        if name not in sets:
            raise ProbeContractError(f"unknown issue set: {name}")
        for issue in sets[name]:
            start = issue_index(issue)
            indices.update(range(start, start + max_step + 1))
    return indices


def decision_from_values(*, menu_grad_primary: Mapping[str, float], static_primary: Mapping[str, float],
                         random_primary: Mapping[str, float], ttt_primary: Mapping[str, float],
                         all_menu_grad: Mapping[str, float]) -> str:
    """Apply the frozen GO/RELAX/STOP rule to point estimates."""
    primary = ("Z500@72h", "T850@72h")
    go = (
        all(float(menu_grad_primary[c]) >= 2.0 for c in primary)
        and all(float(menu_grad_primary[c]) - float(static_primary[c]) >= 1.0 for c in primary)
        and all(float(menu_grad_primary[c]) - float(random_primary[c]) >= 1.0 for c in primary)
        and all(float(v) >= -0.5 for v in all_menu_grad.values())
    )
    if go:
        return "GO_V8"
    relax = (
        max(float(menu_grad_primary[c]) for c in primary) >= 0.5
        or all(float(ttt_primary[c]) >= 2.0 for c in primary)
    )
    return "RELAX_ONCE" if relax else "STOP_PARAM_EDIT"
