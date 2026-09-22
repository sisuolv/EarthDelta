"""B15: a formal registry refuses what a generic planner may legally default.

`_plan_from_finite_candidates` answers an empty or wholly-infeasible candidate
table with the all-zero program. That is legal for a PLANNER and is left
untouched here -- `test_generic_planner_still_defaults_to_no_op` pins that
behaviour so this work cannot be mistaken for a change to it.

A REGISTRY is a different object: a durable record of which trained candidates
exist and what coefficients they really carry. Each negative case below feeds
the registry entry point a value that survives at least one of the planner's
existing checks and asserts the specific refusal, including the two the audit
named: complex coefficients (which the planner silently projects onto the reals)
and the realized-vs-nominal coefficient of `single_expert_plans`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta.contracts import EditPlan, reference_plan, single_expert_plans  # noqa: E402
from earthdelta.registry import (  # noqa: E402
    DEFAULT_A0,
    DEFAULT_RHO,
    CandidateRegistry,
    RegistryEntry,
    RegistryViolation,
    build_pilot_registry,
)
from earthdelta.selection import (  # noqa: E402
    RegistrySelection,
    _plan_from_finite_candidates,
    select_from_registry,
)


class RawPlan:
    """A plan-shaped object, so a registry check can be reached with values an
    `EditPlan` would reject in its own constructor."""

    def __init__(self, plan_id, num_experts, coefficients, rho=DEFAULT_RHO):
        self.plan_id = plan_id
        self.num_experts = num_experts
        self.coefficients = coefficients
        self.rho = rho
        self.hold_steps = 4
        self.interval_hours = 6
        self.continuation = "reference_after_hold"


@pytest.fixture
def registry() -> CandidateRegistry:
    return CandidateRegistry(name="test_candidates", num_experts=3, rho=DEFAULT_RHO)


# =============================================================================
# Positive control
# =============================================================================

def test_pilot_registry_holds_the_reference_and_one_a0_plan_per_expert():
    built = build_pilot_registry(3)
    summary = built.validate()
    assert summary["reference_plan_id"] == "reference"
    assert summary["candidate_plan_ids"] == ["expert_0", "expert_1", "expert_2"]
    assert summary["n_entries"] == 4
    for entry in built.candidate_entries():
        active = [a for a in entry.coefficients if a != 0.0]
        assert active == [DEFAULT_A0]
        assert all(isinstance(a, float) for a in entry.coefficients)


def test_registry_round_trips_through_json(tmp_path):
    built = build_pilot_registry(2, data_role="bank_fit")
    path = built.to_json(tmp_path / "registry.json")
    restored = CandidateRegistry.from_json(path)
    assert restored.plan_ids() == built.plan_ids()
    assert restored.validate()["n_entries"] == 4 - 1  # reference + 2 experts
    assert restored.get("expert_0").data_role == "bank_fit"
    payload = json.loads(path.read_text())
    assert payload["entries"][1]["coefficients"] == [DEFAULT_A0, 0.0]
    assert "realized a0=0.25" in payload["entries"][1]["source"]


def test_registered_source_records_the_realizing_call():
    built = build_pilot_registry(2)
    source = built.get("expert_1").source
    assert "single_expert_plans" in source
    assert "realized a0=0.25" in source


# =============================================================================
# Negative: empty / referenceless registries
# =============================================================================

def test_negative_empty_registry_is_refused(registry):
    with pytest.raises(RegistryViolation) as excinfo:
        registry.validate()
    assert excinfo.value.code == "REGISTRY_EMPTY"
    assert "no candidate was ever registered" in str(excinfo.value)


def test_negative_registry_without_explicit_no_edit_is_refused(registry):
    registry.register(EditPlan("expert_0", 3, (0.25, 0.0, 0.0)), source="fit")
    with pytest.raises(RegistryViolation) as excinfo:
        registry.validate()
    assert excinfo.value.code == "REGISTRY_NO_EDIT_MISSING"
    assert "must be a registered row, not an implied one" in str(excinfo.value)


def test_negative_registry_with_only_the_reference_has_nothing_to_select(registry):
    registry.register_reference(3)
    with pytest.raises(RegistryViolation) as excinfo:
        registry.validate()
    assert excinfo.value.code == "REGISTRY_NO_CANDIDATES"


def test_negative_all_zero_plan_must_be_declared_as_the_reference(registry):
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(EditPlan("quiet", 3, (0.0, 0.0, 0.0)), source="fit",
                          is_reference=False)
    assert excinfo.value.code == "REGISTRY_NO_EDIT_NOT_DECLARED"


def test_negative_a_nonzero_plan_cannot_be_declared_the_reference(registry):
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(EditPlan("expert_0", 3, (0.25, 0.0, 0.0)), source="fit",
                          is_reference=True)
    assert excinfo.value.code == "REGISTRY_REFERENCE_NOT_NO_EDIT"


def test_negative_second_reference_is_refused(registry):
    registry.register_reference(3)
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register_reference(3, plan_id="reference_2")
    assert excinfo.value.code == "REGISTRY_DUPLICATE_REFERENCE"


def test_negative_duplicate_plan_id_is_refused(registry):
    registry.register(EditPlan("expert_0", 3, (0.25, 0.0, 0.0)), source="fit")
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(EditPlan("expert_0", 3, (0.0, 0.25, 0.0)), source="fit")
    assert excinfo.value.code == "REGISTRY_DUPLICATE_PLAN_ID"


# =============================================================================
# Negative: coefficient types the planner does not refuse
# =============================================================================

def test_generic_planner_accepts_a_complex_candidate_table():
    """The behaviour the registry exists to close off: the planner converts a
    complex table to float64 and keeps the real part, so it plans a candidate
    nobody proposed. Left unchanged on purpose."""
    benefit = torch.tensor([1.0, 0.5])
    gram = torch.eye(2)
    candidates = torch.tensor([[0.2 + 5.0j, 0.0]], dtype=torch.complex64)
    with pytest.warns(UserWarning, match="Casting complex values"):
        plan = _plan_from_finite_candidates(
            benefit, gram, candidates, None, 2.0, 1e-4,
            bound=0.25, max_active=2, max_candidates=16,
        )
    assert plan.coefficients.dtype == torch.float32
    assert pytest.approx(float(plan.coefficients[0]), abs=1e-6) == 0.2


def test_generic_planner_still_defaults_to_no_op_on_an_empty_table():
    """Pinned so this work cannot be read as a change to the planner."""
    benefit = torch.tensor([1.0, 0.5])
    plan = _plan_from_finite_candidates(
        benefit, torch.eye(2), torch.zeros(0, 2), None, 2.0, 1e-4,
        bound=0.25, max_active=2, max_candidates=16,
    )
    assert torch.equal(plan.coefficients, torch.zeros(2))
    assert plan.support == ()


def test_negative_registry_refuses_complex_coefficients(registry):
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(RawPlan("complex_expert", 3, (0.2 + 5.0j, 0.0, 0.0)),
                          source="fit")
    assert excinfo.value.code == "REGISTRY_COEFFICIENT_COMPLEX"
    assert "keeps only the real part" in str(excinfo.value)
    assert excinfo.value.detail["imag"] == 5.0


def test_negative_registry_refuses_numpy_complex_coefficients(registry):
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(
            RawPlan("np_complex", 3, (np.complex128(0.25 + 1j), 0.0, 0.0)),
            source="fit",
        )
    assert excinfo.value.code == "REGISTRY_COEFFICIENT_COMPLEX"


def test_negative_registry_refuses_complex_torch_coefficients(registry):
    coefficient = torch.tensor(0.25 + 1j, dtype=torch.complex64)
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(RawPlan("torch_complex", 3, (coefficient, 0.0, 0.0)),
                          source="fit")
    assert excinfo.value.code == "REGISTRY_COEFFICIENT_COMPLEX"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_negative_registry_refuses_non_finite_coefficients(registry, value):
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(RawPlan("nonfinite", 3, (value, 0.0, 0.0)), source="fit")
    assert excinfo.value.code == "REGISTRY_COEFFICIENT_NON_FINITE"


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_negative_registry_refuses_boolean_coefficients(registry, value):
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(RawPlan("masky", 3, (value, 0.0, 0.0)), source="fit")
    assert excinfo.value.code == "REGISTRY_COEFFICIENT_NOT_REAL_FLOAT"


@pytest.mark.parametrize("value", ["0.25", None, {"a": 1}])
def test_negative_registry_refuses_non_numeric_coefficients(registry, value):
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(RawPlan("stringy", 3, (value, 0.0, 0.0)), source="fit")
    assert excinfo.value.code == "REGISTRY_COEFFICIENT_NOT_REAL_FLOAT"


def test_negative_registry_refuses_array_valued_coefficients(registry):
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(RawPlan("arrayish", 3, (np.array([0.25, 0.1]), 0.0, 0.0)),
                          source="fit")
    assert excinfo.value.code == "REGISTRY_COEFFICIENT_NOT_SCALAR"


def test_negative_registry_refuses_an_empty_coefficient_table(registry):
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(RawPlan("empty", 0, ()), source="fit")
    assert excinfo.value.code == "REGISTRY_COEFFICIENTS_EMPTY"


def test_negative_registry_refuses_coefficients_outside_rho(registry):
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(RawPlan("hot", 3, (0.9, 0.0, 0.0)), source="fit")
    assert excinfo.value.code == "REGISTRY_COEFFICIENT_OUT_OF_BOUND"


def test_negative_registry_refuses_mismatched_bank_sizes(registry):
    registry.register_reference(3)
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register(EditPlan("expert_0", 4, (0.25, 0.0, 0.0, 0.0)), source="fit")
    assert excinfo.value.code == "REGISTRY_NUM_EXPERTS_MISMATCH"


def test_negative_loading_a_registry_with_a_complex_string_coefficient(tmp_path):
    """A registry written by another process is revalidated, not trusted."""
    payload = build_pilot_registry(2).to_dict()
    payload["entries"][1]["coefficients"] = ["0.25+1j", 0.0]
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RegistryViolation) as excinfo:
        CandidateRegistry.from_json(path)
    assert excinfo.value.code == "REGISTRY_COEFFICIENT_NOT_REAL_FLOAT"


def test_negative_loading_a_registry_written_under_another_schema(tmp_path):
    payload = build_pilot_registry(2).to_dict()
    payload["schema"] = "ed-candidate-registry/0"
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RegistryViolation) as excinfo:
        CandidateRegistry.from_json(path)
    assert excinfo.value.code == "REGISTRY_SCHEMA_MISMATCH"


# =============================================================================
# Negative: the realized single-expert coefficient
# =============================================================================

def test_single_expert_plans_clips_to_rho_not_to_the_nominal_argument():
    """The upstream behaviour the registry has to observe rather than assume."""
    assert single_expert_plans(2)[0].coefficients == (0.25, 0.0)
    assert single_expert_plans(2, coefficient=0.1)[0].coefficients == (0.1, 0.0)
    # min(), not abs-clipping: a negative nominal is not pulled up to -rho, it
    # falls outside the bound and EditPlan refuses it.
    with pytest.raises(ValueError, match=r"Coefficients must be in"):
        single_expert_plans(2, coefficient=-5.0)


def test_registry_records_the_realized_coefficient_and_asserts_it_equals_a0(registry):
    registry.register_reference(3)
    entries = registry.register_single_expert_plans(3, coefficient=1.0)
    assert [e.plan_id for e in entries] == ["expert_0", "expert_1", "expert_2"]
    for i, entry in enumerate(entries):
        assert entry.coefficients[i] == DEFAULT_A0
        assert isinstance(entry.coefficients[i], float)


def test_negative_a_weaker_nominal_coefficient_is_refused_not_recorded_as_a0(registry):
    """A caller that passes a nominal below rho silently gets a weaker edit.
    The registry refuses to file it as the a0 candidate."""
    registry.register_reference(3)
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register_single_expert_plans(3, coefficient=0.1)
    assert excinfo.value.code == "REGISTRY_REALIZED_COEFFICIENT_MISMATCH"
    assert excinfo.value.detail["realized"] == 0.1
    assert excinfo.value.detail["expected"] == 0.25
    assert "min(coefficient, rho)" in str(excinfo.value)


def test_negative_rho_disagreement_between_registry_and_plans(registry):
    registry.register_reference(3)
    with pytest.raises(RegistryViolation) as excinfo:
        registry.register_single_expert_plans(3, rho=0.5)
    assert excinfo.value.code == "REGISTRY_RHO_MISMATCH"


def test_expected_coefficient_can_be_stated_explicitly(registry):
    """A deliberately weaker experiment is registrable -- but only when it is
    declared, which is the whole point."""
    registry.register_reference(3)
    entries = registry.register_single_expert_plans(
        3, coefficient=0.1, expected_coefficient=0.1
    )
    assert entries[0].coefficients[0] == 0.1


# =============================================================================
# The formal selection entry point
# =============================================================================

def test_select_from_registry_names_the_registered_plan_it_chose():
    built = build_pilot_registry(3)
    benefit = torch.tensor([1.0, 0.1, 0.0])
    selection = select_from_registry(built, benefit, torch.eye(3))
    assert isinstance(selection, RegistrySelection)
    assert selection.plan_id == "expert_0"
    assert selection.is_reference is False
    assert selection.coefficients == (0.25, 0.0, 0.0)


def test_select_from_registry_maps_a_no_op_result_to_the_registered_reference():
    """An all-zero result now means the REGISTERED reference won, which is a
    different statement from 'nothing was registered'."""
    built = build_pilot_registry(3)
    selection = select_from_registry(built, torch.tensor([-1.0, -1.0, -1.0]),
                                     torch.eye(3))
    assert selection.plan_id == "reference"
    assert selection.is_reference is True


def test_select_matches_a_float32_round_tripped_coefficient(registry):
    """0.1 is not exactly representable in float32, so the selected program
    comes back a rounding step away from the registered float64 value."""
    registry.register_reference(3)
    registry.register_single_expert_plans(3, coefficient=0.1,
                                          expected_coefficient=0.1)
    benefit = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    selection = select_from_registry(registry, benefit, torch.eye(3, dtype=torch.float32))
    assert selection.plan_id == "expert_1"
    assert selection.coefficients == (0.0, 0.1, 0.0)


def test_negative_select_from_an_empty_registry_is_refused():
    empty = CandidateRegistry(name="empty", num_experts=3)
    with pytest.raises(RegistryViolation) as excinfo:
        select_from_registry(empty, torch.tensor([1.0, 0.1, 0.0]), torch.eye(3))
    assert excinfo.value.code == "REGISTRY_EMPTY"


def test_negative_select_from_a_registry_without_the_no_edit_entry(registry):
    registry.register(EditPlan("expert_0", 3, (0.25, 0.0, 0.0)), source="fit")
    with pytest.raises(RegistryViolation) as excinfo:
        select_from_registry(registry, torch.tensor([1.0, 0.1, 0.0]), torch.eye(3))
    assert excinfo.value.code == "REGISTRY_NO_EDIT_MISSING"


def test_negative_select_with_a_benefit_of_the_wrong_width():
    built = build_pilot_registry(3)
    with pytest.raises(ValueError, match="registry holds plans over 3 experts"):
        select_from_registry(built, torch.tensor([1.0, 0.1]), torch.eye(2))


def test_candidate_offsets_put_the_reference_in_row_zero():
    built = build_pilot_registry(2)
    offsets = built.candidate_offsets()
    assert offsets.shape == (3, 2)
    assert torch.equal(offsets[0], torch.zeros(2, dtype=offsets.dtype))
    assert offsets.dtype == torch.float64


def test_entries_rebuild_executable_edit_plans():
    built = build_pilot_registry(2)
    plans = built.edit_plans()
    assert [p.plan_id for p in plans] == ["reference", "expert_0", "expert_1"]
    assert all(isinstance(p, EditPlan) for p in plans)
    assert plans[0].coefficients == reference_plan(2).coefficients


def test_registry_entry_from_dict_rejects_a_non_sequence_coefficient_field():
    with pytest.raises(RegistryViolation) as excinfo:
        RegistryEntry.from_dict({"plan_id": "x", "coefficients": 0.25})
    assert excinfo.value.code == "REGISTRY_COEFFICIENTS_NOT_A_SEQUENCE"
