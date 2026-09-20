"""Offline box-constrained local teacher, reusing SciPy's lsq_linear.

These teacher routines deliberately require labels and MUST NOT be CALLED
by an operational controller. Import separation alone is not a data sandbox.
It is NOT a global oracle. Candidates are optimal only for a local linear
surrogate, a frozen dictionary, and declared bounds.

The cost model is injectable: pass a dict mapping slot indices to costs,
a callable that returns costs, or None for unit costs.
"""
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence
import math
import numpy as np
from scipy.optimize import lsq_linear
import torch
from torch import Tensor
from .geometry import ResponseGeometry
from .probe import finite_vector


# Type alias for cost sources
CostSource = dict[int, float] | Callable[[int], float] | Sequence[float] | None


def _resolve_costs(cost_source: CostSource, dimension: int) -> np.ndarray:
    """Resolve cost source to a numpy array of costs.

    Args:
        cost_source: Can be:
            - None: unit costs (all 1.0)
            - Sequence[float]: direct array of costs
            - dict[int, float]: mapping from index to cost (missing = 1.0)
            - Callable[[int], float]: function returning cost for index

        dimension: Number of dimensions

    Returns:
        [d] numpy array of positive costs
    """
    if cost_source is None:
        return np.ones(dimension)
    elif isinstance(cost_source, dict):
        costs = np.ones(dimension)
        for idx, cost in cost_source.items():
            if not 0 <= idx < dimension:
                raise ValueError(f'cost index {idx} out of range [0, {dimension})')
            costs[idx] = cost
        return costs
    elif callable(cost_source):
        costs = np.array([cost_source(i) for i in range(dimension)], dtype=float)
        return costs
    else:
        costs = np.asarray(cost_source, dtype=float)
        if costs.shape != (dimension,):
            raise ValueError(f'cost sequence length {len(costs)} != dimension {dimension}')
        return costs


@dataclass(frozen=True)
class Candidate:
    """A candidate edit solution."""
    offset: Tensor
    support: tuple[int, ...]
    predicted_gain: float
    objective: float
    declared_cost: float  # ABSTRACT cost; NOT measured CUDA time
    valid: bool = True
    reason: str = ''


def box_candidates(geometry: ResponseGeometry, *, bound: float | Tensor = 0.25,
                   ridge: float = 1e-4, max_active: int | None = None,
                   costs: CostSource = None,
                   budget: float | None = None,
                   enumerate_subsets: bool = True,
                   max_candidates: int = 4096) -> list[Candidate]:
    """Solve weighted ridge LS on each allowed support, including no-op.

    Bound is componentwise: |offset_j|<=bound_j, NOT an L2 ball. The two
    constraints are not interchangeable. First experiments can use the full
    support only (enumerate_subsets=False); budgeted experiments enumerate.

    Args:
        geometry: ResponseGeometry from from_error
        bound: Scalar or [d] componentwise bounds (positive)
        ridge: Ridge regularization (nonnegative)
        max_active: Maximum number of active dimensions (None = all)
        costs: Injectable cost source. Can be:
            - None: unit costs (all 1.0, default)
            - Sequence[float]: direct array of costs per slot
            - dict[int, float]: mapping from slot index to cost (missing = 1.0)
            - Callable[[int], float]: function returning cost for slot index
        budget: Maximum total cost (None = no limit)
        enumerate_subsets: If True, enumerate all subsets up to max_active
        max_candidates: Safety limit on number of candidates

    Returns:
        List of Candidates sorted by (validity, objective, support size, support)
    """
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError('ridge must be finite and nonnegative')
    d = geometry.response.shape[1]
    if max_active is None:
        max_active = d
    if type(max_active) is not int or not 0 <= max_active <= d:
        raise ValueError('invalid max_active')
    limit = math.inf if budget is None else float(budget)
    if math.isnan(limit) or limit < 0:
        raise ValueError('invalid budget')

    # Resolve costs from the injectable source
    c = _resolve_costs(costs, d)
    if not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError('costs must be positive [d]')

    bounds_arr = torch.as_tensor(bound, dtype=torch.float64).detach().cpu().numpy()
    bounds_arr = np.broadcast_to(bounds_arr, (d,)).copy()
    if not np.isfinite(bounds_arr).all() or (bounds_arr <= 0).any():
        raise ValueError('bound must be positive scalar or [d]')
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError('invalid max_candidates')

    theoretical = sum(math.comb(d, k) for k in range(max_active + 1)) if enumerate_subsets else 2
    if theoretical > max_candidates:
        raise ValueError('support enumeration exceeds max_candidates')

    r = geometry.response.cpu().numpy()
    e = geometry.error.cpu().numpy()
    w = geometry.weights.cpu().numpy()
    zero = torch.zeros(d, dtype=torch.float64)
    base = float(geometry.baseline_loss)
    answers = [Candidate(zero, (), 0., base, 0.)]

    if enumerate_subsets:
        supports = (s for k in range(1, max_active + 1) for s in combinations(range(d), k))
    else:
        if max_active != d:
            raise ValueError('full-support mode requires max_active=d')
        supports = [tuple(range(d))]

    for support in supports:
        idx = np.asarray(support, dtype=int)
        cost = float(c[idx].sum())
        if cost > limit + 1e-12:
            continue
        a = np.concatenate((np.sqrt(w)[:, None] * r[:, idx], np.sqrt(ridge) * np.eye(len(idx))), axis=0)
        y = np.concatenate((np.sqrt(w) * e, np.zeros(len(idx))))
        solution = lsq_linear(a, y, bounds=(-bounds_arr[idx], bounds_arr[idx]), tol=1e-11, lsq_solver='exact', max_iter=200)
        delta = zero.clone()
        delta[list(support)] = torch.from_numpy(solution.x.copy())
        gain = float(geometry.predicted_gain(delta))
        objective = base - gain + ridge * float(delta.square().sum())
        answers.append(Candidate(delta, support, gain, objective, cost,
                                 bool(solution.success), str(solution.message)))

    return sorted(answers, key=lambda x: (not x.valid, x.objective, len(x.support), x.support))


@dataclass(frozen=True)
class VerifiedResult:
    """Result of nonlinear verification."""
    offset: Tensor
    accepted: bool
    baseline_loss: float
    selected_loss: float
    records: tuple[dict, ...]
    forward_calls: int


@torch.no_grad()
def verify_candidates(evaluate: Callable[[Tensor], Tensor], reference: Tensor,
                      candidates: Sequence[Candidate], truth: Tensor, weights: Tensor,
                      max_nonzero: int = 3, min_gain: float = 0.,
                      costs: CostSource = None) -> VerifiedResult:
    """TRAINING-SIDE nonlinear validation with a no-op fallback.

    evaluate returns the FULL fixed validation vector. It may be larger than
    the probe summary: this is recommended to check summary/full-field gaps.
    The callback must be a fresh deterministic replay. Invalid candidates are
    recorded, never silently called a success. A runtime/no-label 'guarantee'
    must NOT be inferred from this oracle-assisted offline acceptance.

    Args:
        evaluate: Function mapping coefficients to validation vector
        reference: [d] reference coefficients
        candidates: Sequence of Candidates to verify
        truth: [m] ground truth vector
        weights: [m] nonnegative weights
        max_nonzero: Maximum number of nonzero candidates to try
        min_gain: Minimum required improvement over baseline
        costs: Injectable cost source (for logging, not filtering here)

    Returns:
        VerifiedResult with the best accepted offset and verification records
    """
    finite_vector(reference, 'reference')
    finite_vector(truth, 'truth')
    finite_vector(weights, 'weights')
    if truth.shape != weights.shape or bool((weights < 0).any()) or not bool(weights.sum() > 0):
        raise ValueError('invalid truth/weights')
    if type(max_nonzero) is not int or max_nonzero < 0 or not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError('invalid verification budget/margin')
    calls = 0

    def loss(a):
        nonlocal calls
        calls += 1
        y = evaluate(a.detach().clone())
        finite_vector(y, 'validation vector')
        if y.shape != truth.shape:
            raise ValueError('validation vector shape mismatch')
        # Move labels by DEVICE only: converting to a BF16 prediction's dtype
        # before double accumulation would silently quantize the reference.
        w = weights.to(device=y.device, dtype=torch.float64)
        target = truth.to(device=y.device, dtype=torch.float64)
        return float((w * (target - y.double()).square()).sum())

    base = loss(reference)
    best_loss = base
    best = torch.zeros_like(reference)
    records = []
    tried = 0

    for c in candidates:
        if not c.valid:
            records.append({'support': c.support, 'status': 'solver_failed', 'reason': c.reason})
            continue
        if not c.support or not bool((c.offset != 0).any()):
            continue
        if tried >= max_nonzero:
            break
        tried += 1
        delta = c.offset.to(reference)
        try:
            value = loss(reference + delta)
        except (ValueError, RuntimeError) as exc:
            records.append({'support': c.support, 'status': 'replay_failed', 'reason': str(exc)})
            continue
        accepted = value < best_loss and base - value > min_gain
        records.append({'support': c.support, 'status': 'evaluated', 'loss': value,
                        'actual_gain': base - value, 'predicted_gain': c.predicted_gain})
        if accepted:
            best_loss = value
            best = delta.clone()

    return VerifiedResult(best, bool((best != 0).any()), base, best_loss, tuple(records), calls)
