"""Formal candidate registry: the durable record of what may actually be run.

`earthdelta.selection` plans. Planning is generic by design: given a benefit
vector, a Gram matrix and a table of candidate offsets, it returns the best
feasible candidate, and when the table qualifies nothing it returns the all-zero
program. That is correct behaviour for a planner -- an empty feasible set has a
well-defined answer, "do not edit" -- and it stays.

It is the wrong behaviour for a REGISTRY. A registry is the record that says
which trained candidates exist, what their realized coefficients are, and which
entry is the no-edit reference those candidates are measured against. If a
registry can be empty and still produce a no-op plan, then "no candidate was
good enough" and "no candidate was ever registered" become the same observation
after the fact, and a run that trained nothing is indistinguishable from a run
whose candidates all lost. So this module refuses, rather than defaults:

  * an empty registry is refused, not answered with a no-op;
  * a registry with no EXPLICIT no-edit entry is refused, because a reference
    that is merely implied is a reference nobody can point at afterwards;
  * complex coefficients are refused. `_plan_from_finite_candidates` accepts a
    complex candidate table today -- `torch.isfinite` is defined on complex
    tensors, `np.abs(...)` is defined on complex values, and the conversion to
    float64 drops the imaginary part with a UserWarning. The plan that comes
    back is the real projection of a candidate nobody meant to propose;
  * non-finite and non-real coefficients (bools, strings, arrays) are refused,
    whatever container they arrive in.

`register_single_expert_plans` additionally records what
`contracts.single_expert_plans` REALIZED rather than what its caller asked for.
That function clips with `min(coefficient, rho)`, so a nominal 1.0 under
rho=0.25 realizes 0.25, and a nominal 0.1 realizes 0.1 -- the same call
signature, two different experiments. The registry stores the realized value and
asserts it equals the expected a0, so a silently-weakened edit cannot be
reported as the edit that was specified.
"""
from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import EditPlan, reference_plan, single_expert_plans
from .pilot_contract import assert_valid_data_role

__all__ = [
    "RegistryViolation",
    "RegistryEntry",
    "CandidateRegistry",
    "DEFAULT_RHO",
    "DEFAULT_A0",
    "REGISTRY_SCHEMA_VERSION",
]


REGISTRY_SCHEMA_VERSION = "ed-candidate-registry/1"

#: The coefficient bound shared by `EditPlan.rho`, `ProgramSpec.bound` and
#: `plan_from_prediction(bound=...)`.
DEFAULT_RHO = 0.25

#: The realized single-expert coefficient the pilot specifies. It equals the
#: bound because `single_expert_plans` asks for full strength and is clipped to
#: it; the registry asserts the realized value rather than assuming the clip.
DEFAULT_A0 = DEFAULT_RHO

_COEFFICIENT_TOLERANCE = 1e-12


class RegistryViolation(ValueError):
    """A registry entry, or a registry as a whole, failed a formal check."""

    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.detail: Dict[str, Any] = dict(detail or {})


# =============================================================================
# Coefficient admission
# =============================================================================

def _describe(value: Any) -> str:
    return f"{value!r} (type {type(value).__name__})"


def _coerce_real_float(value: Any, *, plan_id: str, position: int) -> float:
    """Return a real, finite Python float, or refuse the value.

    Every rejection here is a value that survives at least one of the planner's
    existing checks: a complex number passes `torch.isfinite`, a bool passes
    `isinstance(x, int)`, a 0-d array passes `float()`.
    """
    if isinstance(value, bool):
        raise RegistryViolation(
            "REGISTRY_COEFFICIENT_NOT_REAL_FLOAT",
            f"Plan {plan_id!r} coefficient {position} is {_describe(value)}. A "
            "registry stores real floating-point coefficients; a bool is a mask, "
            "not a coefficient, and would silently register as 0.0 or 1.0.",
            {"plan_id": plan_id, "position": position, "value": repr(value)},
        )
    if isinstance(value, complex):
        raise RegistryViolation(
            "REGISTRY_COEFFICIENT_COMPLEX",
            f"Plan {plan_id!r} coefficient {position} is {_describe(value)}. "
            "Complex coefficients are refused: the finite-candidate planner "
            "converts them to float64 and keeps only the real part, so the plan "
            "that would run is not the plan that was registered.",
            {"plan_id": plan_id, "position": position, "value": repr(value),
             "real": float(value.real), "imag": float(value.imag)},
        )

    # numpy / torch scalars, without importing either as a hard dependency
    module = type(value).__module__.split(".")[0]
    if module in ("numpy", "torch"):
        dtype = str(getattr(value, "dtype", ""))
        if "complex" in dtype:
            raise RegistryViolation(
                "REGISTRY_COEFFICIENT_COMPLEX",
                f"Plan {plan_id!r} coefficient {position} has complex dtype "
                f"{dtype}. Complex coefficients are refused: the finite-candidate "
                "planner would keep only the real part.",
                {"plan_id": plan_id, "position": position, "dtype": dtype},
            )
        if "bool" in dtype:
            raise RegistryViolation(
                "REGISTRY_COEFFICIENT_NOT_REAL_FLOAT",
                f"Plan {plan_id!r} coefficient {position} has boolean dtype "
                f"{dtype}; a mask is not a coefficient.",
                {"plan_id": plan_id, "position": position, "dtype": dtype},
            )
        if hasattr(value, "numel"):  # torch tensors
            size = value.numel()
        else:
            size = getattr(value, "size", 1)
            size = size() if callable(size) else size
        try:
            size = int(size)
        except Exception:  # noqa: BLE001 - treated as non-scalar below
            size = -1
        if size != 1:
            raise RegistryViolation(
                "REGISTRY_COEFFICIENT_NOT_SCALAR",
                f"Plan {plan_id!r} coefficient {position} is an array of "
                f"{size} element(s); a coefficient is one number.",
                {"plan_id": plan_id, "position": position, "size": size},
            )

    if not isinstance(value, (int, float)) and module not in ("numpy", "torch"):
        raise RegistryViolation(
            "REGISTRY_COEFFICIENT_NOT_REAL_FLOAT",
            f"Plan {plan_id!r} coefficient {position} is {_describe(value)}, "
            "which is not a real number.",
            {"plan_id": plan_id, "position": position, "value": repr(value)},
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RegistryViolation(
            "REGISTRY_COEFFICIENT_NOT_REAL_FLOAT",
            f"Plan {plan_id!r} coefficient {position} is {_describe(value)} and "
            f"could not be read as a real float: {exc}",
            {"plan_id": plan_id, "position": position, "value": repr(value)},
        ) from exc

    if not math.isfinite(result):
        raise RegistryViolation(
            "REGISTRY_COEFFICIENT_NON_FINITE",
            f"Plan {plan_id!r} coefficient {position} is {result}; registry "
            "coefficients must be finite.",
            {"plan_id": plan_id, "position": position, "value": result},
        )
    return result


# =============================================================================
# Entries
# =============================================================================

@dataclass(frozen=True)
class RegistryEntry:
    """One registered candidate, with the coefficients it REALLY carries."""

    plan_id: str
    num_experts: int
    coefficients: Tuple[float, ...]
    rho: float = DEFAULT_RHO
    hold_steps: int = 4
    interval_hours: int = 6
    continuation: str = "reference_after_hold"
    is_reference: bool = False
    source: str = ""
    data_role: Optional[str] = None
    artifact_ref: Optional[Dict[str, Any]] = None
    notes: str = ""
    registered_at: str = ""

    @property
    def support(self) -> Tuple[int, ...]:
        return tuple(i for i, a in enumerate(self.coefficients) if a != 0.0)

    def to_edit_plan(self) -> EditPlan:
        """Rebuild the executable plan. Raises if the entry is not runnable."""
        return EditPlan(
            plan_id=self.plan_id,
            num_experts=self.num_experts,
            coefficients=tuple(self.coefficients),
            hold_steps=self.hold_steps,
            interval_hours=self.interval_hours,
            continuation=self.continuation,
            rho=self.rho,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "num_experts": self.num_experts,
            "coefficients": [float(a) for a in self.coefficients],
            "rho": float(self.rho),
            "hold_steps": self.hold_steps,
            "interval_hours": self.interval_hours,
            "continuation": self.continuation,
            "is_reference": self.is_reference,
            "source": self.source,
            "data_role": self.data_role,
            "artifact_ref": self.artifact_ref,
            "notes": self.notes,
            "registered_at": self.registered_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RegistryEntry":
        data = dict(payload)
        plan_id = str(data.get("plan_id", ""))
        raw = data.get("coefficients")
        if not isinstance(raw, (list, tuple)):
            raise RegistryViolation(
                "REGISTRY_COEFFICIENTS_NOT_A_SEQUENCE",
                f"Plan {plan_id!r} has coefficients {_describe(raw)}; expected a "
                "sequence of real floats.",
                {"plan_id": plan_id},
            )
        coefficients = tuple(
            _coerce_real_float(value, plan_id=plan_id, position=i)
            for i, value in enumerate(raw)
        )
        return cls(
            plan_id=plan_id,
            num_experts=int(data.get("num_experts", len(coefficients))),
            coefficients=coefficients,
            rho=float(data.get("rho", DEFAULT_RHO)),
            hold_steps=int(data.get("hold_steps", 4)),
            interval_hours=int(data.get("interval_hours", 6)),
            continuation=str(data.get("continuation", "reference_after_hold")),
            is_reference=bool(data.get("is_reference", False)),
            source=str(data.get("source", "")),
            data_role=assert_valid_data_role(data.get("data_role")),
            artifact_ref=data.get("artifact_ref"),
            notes=str(data.get("notes", "")),
            registered_at=str(data.get("registered_at", "")),
        )


# =============================================================================
# Registry
# =============================================================================

class CandidateRegistry:
    """A non-empty, explicitly-referenced, real-valued table of candidates."""

    def __init__(
        self,
        name: str = "candidates",
        num_experts: Optional[int] = None,
        rho: float = DEFAULT_RHO,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not name:
            raise RegistryViolation(
                "REGISTRY_NAME_EMPTY", "A registry needs a name.", {}
            )
        if num_experts is not None and (type(num_experts) is not int or num_experts <= 0):
            raise RegistryViolation(
                "REGISTRY_NUM_EXPERTS_INVALID",
                f"num_experts must be a positive integer, got {num_experts!r}.",
                {"num_experts": repr(num_experts)},
            )
        if not isinstance(rho, (int, float)) or isinstance(rho, bool) or not math.isfinite(rho) or rho <= 0:
            raise RegistryViolation(
                "REGISTRY_RHO_INVALID",
                f"rho must be a positive finite float, got {_describe(rho)}.",
                {"rho": repr(rho)},
            )
        self.name = name
        self.num_experts = num_experts
        self.rho = float(rho)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self._entries: Dict[str, RegistryEntry] = {}
        self.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # -- inspection --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, plan_id: object) -> bool:
        return str(plan_id) in self._entries

    @property
    def entries(self) -> Tuple[RegistryEntry, ...]:
        return tuple(self._entries.values())

    def plan_ids(self) -> Tuple[str, ...]:
        return tuple(self._entries.keys())

    def get(self, plan_id: str) -> RegistryEntry:
        if plan_id not in self._entries:
            raise RegistryViolation(
                "REGISTRY_PLAN_NOT_FOUND",
                f"No plan {plan_id!r} in registry {self.name!r}; registered: "
                f"{list(self._entries)}.",
                {"plan_id": plan_id, "registered": list(self._entries)},
            )
        return self._entries[plan_id]

    @property
    def reference_entry(self) -> Optional[RegistryEntry]:
        for entry in self._entries.values():
            if entry.is_reference:
                return entry
        return None

    def candidate_entries(self) -> Tuple[RegistryEntry, ...]:
        return tuple(e for e in self._entries.values() if not e.is_reference)

    # -- registration ------------------------------------------------------

    def register(
        self,
        plan: Any,
        *,
        source: str = "",
        data_role: Optional[Any] = None,
        artifact_ref: Optional[Dict[str, Any]] = None,
        notes: str = "",
        is_reference: Optional[bool] = None,
    ) -> RegistryEntry:
        """Register one plan after checking what it actually carries.

        Args:
            plan: An `EditPlan` or a `RegistryEntry`.
            source: How these coefficients came to exist -- the call that
                produced them, or the fit they came out of. Required in formal
                use; an entry with no provenance is a number with no history.
            data_role: Frozen role of the data this candidate was fitted on.
            artifact_ref: Reference to the trained artifact (path, hash, ...).
            notes: Free text.
            is_reference: Force the no-edit flag. When omitted, an all-zero plan
                is taken to BE the no-edit reference.

        Raises:
            RegistryViolation: On a duplicate id, a shape/bound disagreement, a
                second reference, or any non-real / non-finite coefficient.
        """
        plan_id = str(getattr(plan, "plan_id", "") or "")
        if not plan_id:
            raise RegistryViolation(
                "REGISTRY_PLAN_ID_EMPTY",
                "A registry entry needs a non-empty plan_id.",
                {},
            )
        if plan_id in self._entries:
            raise RegistryViolation(
                "REGISTRY_DUPLICATE_PLAN_ID",
                f"Plan {plan_id!r} is already registered in {self.name!r}. A "
                "registry is a record, not a mutable cache; re-registering an "
                "id would silently rewrite what an earlier result referred to.",
                {"plan_id": plan_id},
            )

        raw = getattr(plan, "coefficients", None)
        if not isinstance(raw, (list, tuple)):
            raise RegistryViolation(
                "REGISTRY_COEFFICIENTS_NOT_A_SEQUENCE",
                f"Plan {plan_id!r} has coefficients {_describe(raw)}; expected a "
                "sequence of real floats.",
                {"plan_id": plan_id},
            )
        if len(raw) == 0:
            raise RegistryViolation(
                "REGISTRY_COEFFICIENTS_EMPTY",
                f"Plan {plan_id!r} has no coefficients.",
                {"plan_id": plan_id},
            )
        coefficients = tuple(
            _coerce_real_float(value, plan_id=plan_id, position=i)
            for i, value in enumerate(raw)
        )

        num_experts = int(getattr(plan, "num_experts", len(coefficients)))
        if num_experts != len(coefficients):
            raise RegistryViolation(
                "REGISTRY_NUM_EXPERTS_MISMATCH",
                f"Plan {plan_id!r} declares num_experts={num_experts} but "
                f"carries {len(coefficients)} coefficients.",
                {"plan_id": plan_id, "num_experts": num_experts,
                 "n_coefficients": len(coefficients)},
            )
        if self.num_experts is None:
            self.num_experts = num_experts
        elif num_experts != self.num_experts:
            raise RegistryViolation(
                "REGISTRY_NUM_EXPERTS_MISMATCH",
                f"Plan {plan_id!r} has {num_experts} experts; registry "
                f"{self.name!r} holds plans over {self.num_experts}. Candidates "
                "measured against different bank sizes are not comparable.",
                {"plan_id": plan_id, "num_experts": num_experts,
                 "registry_num_experts": self.num_experts},
            )

        rho = float(getattr(plan, "rho", self.rho))
        if not math.isfinite(rho) or rho <= 0:
            raise RegistryViolation(
                "REGISTRY_RHO_INVALID",
                f"Plan {plan_id!r} declares rho={rho!r}.",
                {"plan_id": plan_id, "rho": rho},
            )
        if abs(rho - self.rho) > _COEFFICIENT_TOLERANCE:
            raise RegistryViolation(
                "REGISTRY_RHO_MISMATCH",
                f"Plan {plan_id!r} declares rho={rho} but registry "
                f"{self.name!r} is bound at rho={self.rho}.",
                {"plan_id": plan_id, "rho": rho, "registry_rho": self.rho},
            )
        over = [
            (i, a) for i, a in enumerate(coefficients)
            if abs(a) > self.rho + 1e-7
        ]
        if over:
            raise RegistryViolation(
                "REGISTRY_COEFFICIENT_OUT_OF_BOUND",
                f"Plan {plan_id!r} has coefficient(s) outside [-{self.rho}, "
                f"{self.rho}]: {over[:4]}.",
                {"plan_id": plan_id, "rho": self.rho, "offending": over[:8]},
            )

        all_zero = not any(a != 0.0 for a in coefficients)
        reference_flag = all_zero if is_reference is None else bool(is_reference)
        if reference_flag and not all_zero:
            raise RegistryViolation(
                "REGISTRY_REFERENCE_NOT_NO_EDIT",
                f"Plan {plan_id!r} is registered as the no-edit reference but "
                f"has nonzero coefficients {list(coefficients)}.",
                {"plan_id": plan_id, "coefficients": list(coefficients)},
            )
        if all_zero and not reference_flag:
            raise RegistryViolation(
                "REGISTRY_NO_EDIT_NOT_DECLARED",
                f"Plan {plan_id!r} is all-zero -- it IS the no-edit plan -- but "
                "was registered with is_reference=False. The no-edit entry must "
                "be explicit, so nothing later has to infer which row it was.",
                {"plan_id": plan_id},
            )
        if reference_flag and self.reference_entry is not None:
            raise RegistryViolation(
                "REGISTRY_DUPLICATE_REFERENCE",
                f"Registry {self.name!r} already has the no-edit reference "
                f"{self.reference_entry.plan_id!r}; {plan_id!r} would be a "
                "second one, leaving 'the reference' ambiguous.",
                {"plan_id": plan_id, "existing": self.reference_entry.plan_id},
            )

        entry = RegistryEntry(
            plan_id=plan_id,
            num_experts=num_experts,
            coefficients=coefficients,
            rho=self.rho,
            hold_steps=int(getattr(plan, "hold_steps", 4)),
            interval_hours=int(getattr(plan, "interval_hours", 6)),
            continuation=str(getattr(plan, "continuation", "reference_after_hold")),
            is_reference=reference_flag,
            source=str(source or getattr(plan, "source", "")),
            data_role=assert_valid_data_role(data_role),
            artifact_ref=dict(artifact_ref) if artifact_ref else None,
            notes=str(notes),
            registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        self._entries[plan_id] = entry
        return entry

    def register_reference(
        self,
        num_experts: Optional[int] = None,
        *,
        plan_id: str = "reference",
        source: str = "contracts.reference_plan",
        data_role: Optional[Any] = None,
        notes: str = "",
    ) -> RegistryEntry:
        """Register the explicit no-edit plan every registry must carry."""
        experts = num_experts if num_experts is not None else self.num_experts
        if experts is None:
            raise RegistryViolation(
                "REGISTRY_NUM_EXPERTS_UNKNOWN",
                "num_experts must be given for the first registration.",
                {},
            )
        plan = reference_plan(int(experts), rho=self.rho)
        if plan_id != plan.plan_id:
            plan = EditPlan(
                plan_id=plan_id,
                num_experts=plan.num_experts,
                coefficients=plan.coefficients,
                hold_steps=plan.hold_steps,
                interval_hours=plan.interval_hours,
                continuation=plan.continuation,
                rho=plan.rho,
            )
        return self.register(
            plan, source=source, data_role=data_role, notes=notes, is_reference=True
        )

    def register_single_expert_plans(
        self,
        num_experts: Optional[int] = None,
        *,
        coefficient: float = 1.0,
        rho: Optional[float] = None,
        expected_coefficient: float = DEFAULT_A0,
        source: Optional[str] = None,
        data_role: Optional[Any] = None,
        artifact_refs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[RegistryEntry, ...]:
        """Register `contracts.single_expert_plans`, asserting what it realized.

        `single_expert_plans` sets each active coefficient to
        `min(coefficient, rho)`. The nominal argument therefore does not say
        what will run: 1.0 under rho=0.25 realizes 0.25, and 0.1 under the same
        rho realizes 0.1. This registers the REALIZED value and refuses it
        unless it equals `expected_coefficient`, so an experiment that was
        specified at a0 cannot quietly run at something weaker.

        Raises:
            RegistryViolation: If any realized coefficient differs from
                `expected_coefficient`, or if a realized plan fails any of the
                registry's own entry checks.
        """
        experts = num_experts if num_experts is not None else self.num_experts
        if experts is None:
            raise RegistryViolation(
                "REGISTRY_NUM_EXPERTS_UNKNOWN",
                "num_experts must be given for the first registration.",
                {},
            )
        bound = self.rho if rho is None else float(rho)
        if abs(bound - self.rho) > _COEFFICIENT_TOLERANCE:
            raise RegistryViolation(
                "REGISTRY_RHO_MISMATCH",
                f"single_expert_plans was asked for rho={bound} but registry "
                f"{self.name!r} is bound at rho={self.rho}.",
                {"rho": bound, "registry_rho": self.rho},
            )

        plans = single_expert_plans(int(experts), coefficient=coefficient, rho=bound)
        provenance = source or (
            f"contracts.single_expert_plans(num_experts={int(experts)}, "
            f"coefficient={coefficient!r}, rho={bound!r})"
        )

        registered: List[RegistryEntry] = []
        for plan in plans:
            active = [a for a in plan.coefficients if a != 0.0]
            if len(active) != 1:
                raise RegistryViolation(
                    "REGISTRY_SINGLE_EXPERT_SUPPORT_INVALID",
                    f"Plan {plan.plan_id!r} from single_expert_plans has "
                    f"{len(active)} active coefficient(s); expected exactly 1.",
                    {"plan_id": plan.plan_id, "coefficients": list(plan.coefficients)},
                )
            realized = float(active[0])
            if abs(realized - float(expected_coefficient)) > _COEFFICIENT_TOLERANCE:
                raise RegistryViolation(
                    "REGISTRY_REALIZED_COEFFICIENT_MISMATCH",
                    f"Plan {plan.plan_id!r} realized coefficient {realized} but "
                    f"a0={expected_coefficient} was expected. "
                    f"single_expert_plans(coefficient={coefficient!r}, "
                    f"rho={bound!r}) clips with min(coefficient, rho), so the "
                    "nominal argument is not what runs. The registry stores the "
                    "realized value and refuses to record this plan as the a0 "
                    "candidate it is not.",
                    {"plan_id": plan.plan_id, "realized": realized,
                     "expected": float(expected_coefficient),
                     "nominal_coefficient": coefficient, "rho": bound},
                )
            registered.append(self.register(
                plan,
                source=f"{provenance} -> realized a0={realized}",
                data_role=data_role,
                artifact_ref=(artifact_refs or {}).get(plan.plan_id),
            ))
        return tuple(registered)

    # -- validation --------------------------------------------------------

    def validate(self, *, require_candidates: bool = True) -> Dict[str, Any]:
        """Re-check the whole registry and summarise it.

        Every entry is re-validated, not trusted: a registry may have been
        loaded from JSON written by another process.

        Raises:
            RegistryViolation: If the registry is empty, has no explicit no-edit
                entry, has no candidate besides the reference (unless
                `require_candidates=False`), or holds any invalid coefficient.
        """
        if not self._entries:
            raise RegistryViolation(
                "REGISTRY_EMPTY",
                f"Registry {self.name!r} is empty. An empty registry is refused "
                "rather than answered with a no-op plan: 'every candidate lost' "
                "and 'no candidate was ever registered' must not produce the "
                "same result.",
                {"registry": self.name},
            )

        reference = self.reference_entry
        if reference is None:
            raise RegistryViolation(
                "REGISTRY_NO_EDIT_MISSING",
                f"Registry {self.name!r} has no explicit no-edit entry. Every "
                "reported gain is measured against the no-edit reference, so it "
                "must be a registered row, not an implied one. Registered: "
                f"{list(self._entries)}.",
                {"registry": self.name, "registered": list(self._entries)},
            )

        candidates = self.candidate_entries()
        if require_candidates and not candidates:
            raise RegistryViolation(
                "REGISTRY_NO_CANDIDATES",
                f"Registry {self.name!r} holds only the no-edit reference "
                f"{reference.plan_id!r}. There is nothing to select between.",
                {"registry": self.name},
            )

        for entry in self._entries.values():
            for position, value in enumerate(entry.coefficients):
                _coerce_real_float(value, plan_id=entry.plan_id, position=position)
            if len(entry.coefficients) != self.num_experts:
                raise RegistryViolation(
                    "REGISTRY_NUM_EXPERTS_MISMATCH",
                    f"Plan {entry.plan_id!r} carries "
                    f"{len(entry.coefficients)} coefficients; the registry holds "
                    f"plans over {self.num_experts} experts.",
                    {"plan_id": entry.plan_id},
                )
            # Executability: an entry that cannot rebuild its EditPlan is not a
            # candidate anything could run.
            entry.to_edit_plan()

        return {
            "registry": self.name,
            "schema": REGISTRY_SCHEMA_VERSION,
            "num_experts": self.num_experts,
            "rho": self.rho,
            "n_entries": len(self._entries),
            "reference_plan_id": reference.plan_id,
            "candidate_plan_ids": [e.plan_id for e in candidates],
        }

    def assert_ready(self, *, require_candidates: bool = True) -> Dict[str, Any]:
        """Alias for `validate`, named for use at a consumption boundary."""
        return self.validate(require_candidates=require_candidates)

    # -- consumption -------------------------------------------------------

    def edit_plans(self) -> Tuple[EditPlan, ...]:
        """Every registered entry as an executable plan, reference first."""
        self.validate(require_candidates=False)
        reference = self.reference_entry
        ordered = [reference] + [e for e in self._entries.values() if e is not reference]
        return tuple(entry.to_edit_plan() for entry in ordered)

    def candidate_offsets(self, dtype: Any = None) -> Any:
        """Candidate coefficients as a `[K, num_experts]` torch tensor.

        Reference first, so row 0 of the table is always the no-edit program.
        """
        import torch

        self.validate(require_candidates=False)
        rows = [list(plan.coefficients) for plan in self.edit_plans()]
        return torch.tensor(rows, dtype=dtype or torch.float64)

    # -- durability --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": REGISTRY_SCHEMA_VERSION,
            "name": self.name,
            "num_experts": self.num_experts,
            "rho": self.rho,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "entries": [entry.to_dict() for entry in self._entries.values()],
        }

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CandidateRegistry":
        if not isinstance(payload, dict):
            raise RegistryViolation(
                "REGISTRY_PAYLOAD_INVALID",
                f"Registry payload is {_describe(payload)}; expected an object.",
                {},
            )
        schema = payload.get("schema")
        if schema is not None and schema != REGISTRY_SCHEMA_VERSION:
            raise RegistryViolation(
                "REGISTRY_SCHEMA_MISMATCH",
                f"Registry was written under schema {schema!r}; this code reads "
                f"{REGISTRY_SCHEMA_VERSION!r}. Refusing to reinterpret it.",
                {"schema": schema, "expected": REGISTRY_SCHEMA_VERSION},
            )
        registry = cls(
            name=str(payload.get("name", "candidates")),
            num_experts=payload.get("num_experts"),
            rho=float(payload.get("rho", DEFAULT_RHO)),
            metadata=payload.get("metadata") or {},
        )
        registry.created_at = str(payload.get("created_at", registry.created_at))
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, (list, tuple)):
            raise RegistryViolation(
                "REGISTRY_ENTRIES_NOT_A_SEQUENCE",
                f"Registry {registry.name!r} has entries {_describe(raw_entries)}; "
                "expected a list.",
                {"registry": registry.name},
            )
        for raw in raw_entries:
            entry = RegistryEntry.from_dict(raw)
            registry.register(
                entry.to_edit_plan(),
                source=entry.source,
                data_role=entry.data_role,
                artifact_ref=entry.artifact_ref,
                notes=entry.notes,
                is_reference=entry.is_reference,
            )
        return registry

    @classmethod
    def from_json(cls, path: Path) -> "CandidateRegistry":
        return cls.from_dict(json.loads(Path(path).read_text()))


def build_pilot_registry(
    num_experts: int,
    *,
    name: str = "pilot_candidates",
    rho: float = DEFAULT_RHO,
    expected_coefficient: float = DEFAULT_A0,
    data_role: Optional[Any] = None,
    artifact_refs: Optional[Dict[str, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> CandidateRegistry:
    """The explicit no-edit reference plus one a0 plan per expert."""
    registry = CandidateRegistry(
        name=name, num_experts=num_experts, rho=rho, metadata=metadata
    )
    registry.register_reference(num_experts, data_role=data_role)
    registry.register_single_expert_plans(
        num_experts,
        expected_coefficient=expected_coefficient,
        data_role=data_role,
        artifact_refs=artifact_refs,
    )
    registry.validate()
    return registry
