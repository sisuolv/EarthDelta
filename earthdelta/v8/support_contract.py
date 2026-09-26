"""Role support windows and replacement rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping


UTC = timezone.utc


def parse_time(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("issue time must include timezone")
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class SupportCertificate:
    role: str
    issue_time: str
    start: str
    end: str
    source_id: str
    finite: bool
    replacement: bool = False

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def support_window(issue_time: str) -> tuple[datetime, datetime]:
    t = parse_time(issue_time)
    return t - timedelta(hours=12), t + timedelta(hours=120)


def validate_issue_list(role: str, issue_times: Iterable[str], *, role_year: int | None = None,
                        min_isolation_hours: float = 24.0) -> list[str]:
    values = [parse_time(x) for x in issue_times]
    if values != sorted(values) or len(set(values)) != len(values):
        raise ValueError(f"{role}: issue times must be unique and chronological")
    if role_year is not None and any(t.year != role_year for t in values):
        raise ValueError(f"{role}: issue time year does not match role year")
    for a, b in zip(values, values[1:]):
        if b - a < timedelta(hours=min_isolation_hours):
            raise ValueError(f"{role}: issue isolation below {min_isolation_hours}h")
    for t in values:
        start, end = support_window(t.isoformat())
        if start.year != t.year or end.year != t.year:
            raise ValueError(f"{role}: support window crosses calendar year")
    return [t.isoformat().replace("+00:00", "Z") for t in values]


def make_certificates(role: str, issue_times: Iterable[str], source_id: str, *, finite: bool = True,
                      role_year: int | None = None) -> list[SupportCertificate]:
    times = validate_issue_list(role, issue_times, role_year=role_year)
    out = []
    for issue in times:
        start, end = support_window(issue)
        out.append(SupportCertificate(role, issue, start.isoformat().replace("+00:00", "Z"),
                                      end.isoformat().replace("+00:00", "Z"), source_id, bool(finite)))
    return out


def validate_replacement(original: str, replacement: str, used: set[str], *, max_hours: int = 48) -> None:
    a, b = parse_time(original), parse_time(replacement)
    if replacement in used:
        raise ValueError("replacement issue time already used")
    if abs((b - a).total_seconds()) > max_hours * 3600:
        raise ValueError("replacement exceeds signed +/-48h window")
