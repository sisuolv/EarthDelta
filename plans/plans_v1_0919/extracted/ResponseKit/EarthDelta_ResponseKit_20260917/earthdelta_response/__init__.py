"""Research primitives; not a weather model or an operational forecasting system."""
from .probe import ProbeResult, central_response, local_linearity_error
from .geometry import ResponseGeometry, response_distillation
from .teacher import Candidate, VerifiedResult, box_candidates, verify_candidates
from .program import Slot, ProgramSpec
from .student import BoundedProgramHead
from .utility import InteractionUtilityHead, quadratic_gain, plan_from_prediction
