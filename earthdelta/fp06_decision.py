"""Pure FP06 first-match decision rules for the frozen FP-05 protocol."""
from __future__ import annotations

from typing import Any, Mapping


STATUSES = frozenset({"ABOVE", "BELOW", "STRADDLE"})


def _status(comparisons: Mapping[str, Mapping[str, Any]], name: str) -> str | None:
    value = comparisons.get(name, {}).get("status_vs_delta_min")
    return str(value) if value is not None else None


def decide_fp06(evidence: Mapping[str, Any], *, delta_min: float = 0.0034) -> dict[str, Any]:
    """Apply FP06-RULES-v1 in order and fail closed on uncovered states.

    ``evidence_valid`` represents the upstream V-check/E0/freeze gate. The
    function deliberately does not infer validity from numeric statuses.
    """
    if evidence.get("evidence_valid") is not True:
        return {"verdict": "INVALID", "reason": "INVALID_EVIDENCE", "delta_min": delta_min}
    comparisons = evidence.get("comparisons") or {}
    names = ("oracle_vs_Fs", "oracle_vs_M1", "M1_vs_Fs", "M3_vs_Fs", "M3_vs_M1")
    statuses = {name: _status(comparisons, name) for name in names}
    guard = comparisons.get("M3_vs_Fs_72h", {}).get("harm72_guard_pass")
    if any(value not in STATUSES for value in statuses.values()):
        return {
            "verdict": "INCONCLUSIVE_RULE_COVERAGE",
            "reason": "required 24h status is missing or invalid",
            "statuses": statuses,
            "delta_min": delta_min,
        }
    if statuses["oracle_vs_Fs"] == "BELOW":
        verdict = "STOP_CURRENT_BANK"
    elif statuses["oracle_vs_M1"] == "BELOW":
        verdict = "PIVOT_STATIC"
    elif statuses["M3_vs_M1"] == "BELOW":
        verdict = "STOP_CURRENT_SELECTOR"
    elif (
        statuses["M3_vs_M1"] == "ABOVE"
        and statuses["M3_vs_Fs"] == "ABOVE"
        and guard is True
    ):
        verdict = "CONTINUE_NEXT_ITERATION"
    elif "STRADDLE" in statuses.values():
        verdict = "INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE_RULE_COVERAGE"
    return {
        "verdict": verdict,
        "reason": "FP06-RULES-v1 first-match",
        "statuses": statuses,
        "harm72_guard_pass": guard,
        "delta_min": delta_min,
        "primary_24h_only": True,
    }
