"""EarthDelta v6 research primitives.

Amortized forecast sensitivity to parameter edits. This package provides:
- contracts: Versioning, edit plans, and program specifications
- paired: Reference-error/edit-response factorization
- probe: Finite difference response computation
- geometry: Training-side local geometry
- teacher: Offline box-constrained local teacher
- selection: Budgeted plan selection
- heads: Prediction heads for response-based control
- memory: Verified records and memory features
- spectral: Spherical harmonic diagnostics
- lowrank: K-expert low-rank adapters

Real-weather integration is not included. The bridge module (stormer_bridge)
is developed separately.
"""

# Contracts
from .contracts import (
    ArtifactVersion,
    EditPlan,
    Slot,
    ProgramSpec,
    pilot_plans,
    reference_plan,
    single_expert_plans,
)

# Paired targets
from .paired import (
    PairedTargets,
    edit_responses,
    quadratic_gain,
    make_training_targets,
)

# Probe
from .probe import (
    ProbeResult,
    CachedResponses,
    central_response,
    local_linearity_error,
    cached_responses,
    finite_vector,
)

# Geometry
from .geometry import (
    ResponseGeometry,
    response_distillation,
)

# Teacher
from .teacher import (
    Candidate,
    VerifiedResult,
    box_candidates,
    verify_candidates,
)

# Selection
from .selection import (
    SurrogatePlan,
    select_plan,
    selection_regret,
    plan_from_prediction,
    unified_select,
)

# Heads
from .heads import (
    BoundedProgramHead,
    InteractionUtilityHead,
    ReferenceErrorHead,
    EditResponseHead,
    GainCalibrationHead,
    ComposedPredictionHead,
    PairedEditPredictor,  # Legacy alias
    quadratic_gain as heads_quadratic_gain,
)

# Memory
from .memory import (
    VerifiedRecord,
    eligible_records,
    gated_delta_replay,
    read_memory,
    ewma_error_feature,
    ewma_error_feature_batched,
)

# Spectral
from .spectral import (
    coefficient_diagnostics,
    assert_same_latitude_nodes,
)

# Low-rank
from .lowrank import (
    ExpertLoRA,
    GroupedLowRankResidual,
    verify_dense_sparse_equivalence,
)

__version__ = "0.6.0"
