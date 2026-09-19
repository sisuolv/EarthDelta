"""EarthDelta v5 research primitives. Real-weather integration is not included."""
from .contracts import ArtifactVersion, EditPlan, pilot_plans
from .paired import make_training_targets, edit_responses, quadratic_gain
from .rank_groups import GroupedLowRankResidual
from .memory import VerifiedRecord, eligible_records, gated_delta_replay, read_memory
from .selection import select_plan, selection_regret
from .model import PairedEditPredictor
