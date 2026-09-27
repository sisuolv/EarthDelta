"""Approved, 2020-only headroom probe namespace.

The repository historically exposed ``earthdelta.probe`` as a module.  The
package keeps that API by loading the old implementation from ``probe.py``
under a private name, while all new probe code lives in this namespace.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_legacy_path = Path(__file__).resolve().parents[1] / "probe.py"
_legacy_spec = importlib.util.spec_from_file_location("earthdelta._legacy_probe", _legacy_path)
if _legacy_spec is None or _legacy_spec.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load legacy probe API from {_legacy_path}")
_legacy = importlib.util.module_from_spec(_legacy_spec)
_legacy_spec.loader.exec_module(_legacy)

ProbeResult = _legacy.ProbeResult
CachedResponses = _legacy.CachedResponses
central_response = _legacy.central_response
local_linearity_error = _legacy.local_linearity_error
cached_responses = _legacy.cached_responses
finite_vector = _legacy.finite_vector

from .access import AccessViolation, BoundStore, ProbeAccessController  # noqa: E402,F401
from .budget import BudgetBlocked, BudgetLedger  # noqa: E402,F401
from .contracts import (  # noqa: E402,F401
    ProbeSpec,
    load_probe_spec,
    issue_sets,
    issue_index,
    required_indices,
    decision_from_values,
)
from .edits import DirectWeightEditor, build_delta_modules, pristine_state_digest  # noqa: E402,F401
from .shards import shard_issue_ids, verify_completed_shard  # noqa: E402,F401
from .estimators import per_issue_oracle, static_best_arm, ttt_oracle  # noqa: E402,F401

__all__ = [
    "ProbeResult", "CachedResponses", "central_response", "local_linearity_error",
    "cached_responses", "finite_vector", "AccessViolation", "BoundStore",
    "ProbeAccessController", "BudgetBlocked", "BudgetLedger", "ProbeSpec",
    "load_probe_spec", "issue_sets", "issue_index", "required_indices",
    "decision_from_values", "DirectWeightEditor", "build_delta_modules",
    "pristine_state_digest", "shard_issue_ids", "verify_completed_shard",
    "per_issue_oracle", "static_best_arm", "ttt_oracle",
]
