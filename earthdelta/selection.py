"""Finite, budgeted plan selection; evaluation truth is not an input.

Provides both finite-candidate selection and continuous box-QP optimization,
unified through a common interface where finite candidates are treated as
a special case of the continuous program restricted to a discrete support set.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
from scipy.optimize import minimize
import torch
from torch import Tensor

from .metrics_contract import quadratic_gain_numpy as _canonical_qg_numpy


def select_plan(predicted_gain: Tensor, total_cost: Tensor, budget: float,
                plan_ids: tuple[str, ...], cost_penalty: float = 0.0) -> Tensor:
    """Select best plan from discrete candidates under budget constraint.

    Non-differentiable selection. Ties: lower total cost, then stable plan ID.
    No-edit must be part of the candidate set supplied by the caller.

    Args:
        predicted_gain: [B, K] predicted gains for K candidates
        total_cost: [K] costs (must INCLUDE controller + selected rollout costs)
        budget: Maximum allowed cost
        plan_ids: Unique identifiers for each candidate
        cost_penalty: Additional penalty per unit cost (for regularization)

    Returns:
        [B] indices of selected plans
    """
    if predicted_gain.ndim != 2 or total_cost.shape != (predicted_gain.shape[1],):
        raise ValueError('Shape mismatch for gain/cost.')
    if len(plan_ids) != predicted_gain.shape[1] or len(set(plan_ids)) != len(plan_ids):
        raise ValueError('Plan IDs must be unique and aligned with columns.')
    if not math.isfinite(budget) or budget < 0 or not math.isfinite(cost_penalty) or cost_penalty < 0:
        raise ValueError('Invalid budget/cost penalty.')
    if not torch.isfinite(predicted_gain).all() or not torch.isfinite(total_cost).all() or (total_cost < 0).any():
        raise ValueError('Finite gain and finite nonnegative costs are required.')
    cost = total_cost.detach().cpu().tolist()
    feasible = [i for i, c in enumerate(cost) if c <= budget]
    if not feasible:
        raise ValueError('Budget cannot run any candidate, including the reference.')
    scores = (predicted_gain - cost_penalty * total_cost.to(predicted_gain)[None]).detach().cpu().tolist()
    chosen = [min(feasible, key=lambda i: (-row[i], cost[i], plan_ids[i])) for row in scores]
    return torch.tensor(chosen, device=predicted_gain.device, dtype=torch.long)


def selection_regret(true_gain: Tensor, selected: Tensor, costs: Tensor,
                     budget: float, penalty: float = 0.0) -> Tensor:
    """POST-HOC diagnostic only. Uses realized gains, never deployment input.

    Args:
        true_gain: [B, K] realized gains
        selected: [B] indices of selected plans
        costs: [K] costs
        budget: Budget used for selection
        penalty: Cost penalty used for selection

    Returns:
        [B] regret = best_feasible_utility - selected_utility
    """
    util = true_gain - penalty * costs.to(true_gain)[None]
    feasible = costs.to(true_gain.device) <= budget
    if not feasible.any() or selected.shape != true_gain.shape[:1]:
        raise ValueError('Invalid feasible set or selected shape.')
    if not feasible[selected].all():
        raise ValueError('Selected plan is outside budget.')
    best = util.masked_fill(~feasible[None], -torch.inf).max(-1).values
    return best - util.gather(1, selected[:, None]).squeeze(1)


@dataclass(frozen=True)
class SurrogatePlan:
    """Result of surrogate-based planning."""
    coefficients: Tensor
    support: tuple[int, ...]
    predicted_gain: float
    objective_gain: float
    declared_cost: float
    solves: int
    solver_failures: int


def quadratic_gain_numpy(benefit: np.ndarray, gram: np.ndarray, program: np.ndarray) -> float:
    """Compute gain = 2*b.a - a.T H a in numpy.

    This computes gain in coefficient space where spatial weights are already
    baked into benefit (b = R.T Q e) and gram (H = R.T Q R).

    Convention: WEIGHTED_SUM - weights are embedded in gram/benefit, so no
    additional normalization is applied. This matches ResponseGeometry.predicted_gain.
    """
    return _canonical_qg_numpy(benefit, gram, program)


@torch.no_grad()
def plan_from_prediction(benefit: Tensor, gram: Tensor, *, bound: float = 0.25,
                         max_active: int = 2, costs: tuple[float, ...] | None = None,
                         budget: float = 2., ridge: float = 1e-4,
                         max_candidates: int = 4096,
                         candidate_offsets: Tensor | None = None) -> SurrogatePlan:
    """Plan from predicted benefit and Gram matrix.

    Supports both finite-candidate selection (when candidate_offsets is provided)
    and continuous box-QP optimization (when candidate_offsets is None).

    For finite candidates, this is equivalent to evaluating the quadratic gain
    at each candidate and selecting the best feasible one, treating it as a
    continuous program restricted to a discrete support set.

    For continuous optimization, enumerate small supports and solve convex box QP
    using SciPy L-BFGS-B.

    Does NOT call a weather model, access labels, or run online teacher probes.
    Requires one sample [d], [d,d].

    Args:
        benefit: [d] predicted benefit vector (b in gain = 2*b.a - a.T H a)
        gram: [d, d] predicted Gram matrix (H, must be symmetric PSD)
        bound: Componentwise coefficient bound
        max_active: Maximum number of active dimensions
        costs: Per-dimension costs (None = unit costs)
        budget: Maximum total cost
        ridge: Ridge regularization
        max_candidates: Safety limit
        candidate_offsets: [K, d] finite candidate set (None = continuous QP)

    Returns:
        SurrogatePlan with optimal coefficients and metadata
    """
    if benefit.ndim != 1 or benefit.numel() == 0 or gram.shape != (benefit.numel(), benefit.numel()):
        raise ValueError('expected [d] and [d,d]')
    if any(not x.is_floating_point() or not bool(torch.isfinite(x).all()) for x in (benefit, gram)):
        raise ValueError('nonfinite/nonfloating inputs')
    d = benefit.numel()
    if type(max_active) is not int or not 0 <= max_active <= d:
        raise ValueError('invalid support size')
    if not math.isfinite(bound) or bound <= 0 or not math.isfinite(ridge) or ridge < 0 or not math.isfinite(budget) or budget < 0:
        raise ValueError('invalid bounds/ridge/budget')

    b = benefit.detach().double().cpu().numpy()
    h = gram.detach().double().cpu().numpy()
    if not np.allclose(h, h.T, atol=1e-8, rtol=1e-8) or np.linalg.eigvalsh(h).min() < -1e-8:
        raise ValueError('predicted gram must be symmetric PSD')

    c = np.ones(d) if costs is None else np.asarray(costs, dtype=float)
    if c.shape != (d,) or not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError('invalid costs')

    # Handle finite candidate set as a special case
    if candidate_offsets is not None:
        return _plan_from_finite_candidates(
            benefit, gram, candidate_offsets, costs, budget, ridge,
            bound=bound, max_active=max_active, max_candidates=max_candidates
        )

    # Continuous optimization path
    if sum(math.comb(d, k) for k in range(max_active + 1)) > max_candidates:
        raise ValueError('support space too large')

    best = np.zeros(d)
    best_gain = 0.
    best_predicted = 0.
    best_support = ()
    best_cost = 0.
    solves = 0
    failures = 0

    for k in range(1, max_active + 1):
        for support in combinations(range(d), k):
            idx = np.asarray(support)
            cost = float(c[idx].sum())
            if cost > budget + 1e-12:
                continue
            a = h[np.ix_(idx, idx)] + ridge * np.eye(k)
            z = b[idx]
            sol = minimize(lambda x: float(x @ a @ x - 2 * z @ x),
                           np.zeros(k),
                           jac=lambda x: 2 * (a @ x - z),
                           bounds=[(-bound, bound)] * k,
                           method='L-BFGS-B',
                           options={'ftol': 1e-13, 'gtol': 1e-10, 'maxiter': 200})
            solves += 1
            if not sol.success or not np.isfinite(sol.x).all():
                failures += 1
                continue
            gain = -float(sol.fun)
            if gain > best_gain + 1e-12:
                best = np.zeros(d)
                best[idx] = sol.x
                best_support = support
                best_cost = cost
                best_gain = gain
                best_predicted = quadratic_gain_numpy(b, h, best)

    return SurrogatePlan(torch.from_numpy(best).to(benefit), best_support, best_predicted,
                         best_gain, best_cost, solves, failures)


@torch.no_grad()
def _plan_from_finite_candidates(
    benefit: Tensor,
    gram: Tensor,
    candidate_offsets: Tensor,
    costs: tuple[float, ...] | None,
    budget: float,
    ridge: float,
    *,
    bound: float,
    max_active: int,
    max_candidates: int,
) -> SurrogatePlan:
    """Select best plan from finite candidate set.

    This treats finite candidates as a continuous program restricted to a
    discrete support set: we evaluate the quadratic gain at each candidate
    and select the best feasible one.

    Hardening checks:
    - Finiteness: Rejects candidates containing NaN or Inf
    - Bound enforcement: Rejects candidates with |coefficient| > bound
    - Support size: Rejects candidates with more than max_active nonzero coefficients
    - Candidate count: Raises error if K exceeds max_candidates

    Args:
        benefit: [d] benefit vector
        gram: [d, d] Gram matrix
        candidate_offsets: [K, d] candidate offset vectors
        costs: Per-dimension costs
        budget: Maximum total cost
        ridge: Ridge regularization
        bound: Componentwise coefficient bound (candidates must satisfy |a_i| <= bound)
        max_active: Maximum number of nonzero coefficients per candidate
        max_candidates: Maximum allowed candidate count (raises if exceeded)

    Returns:
        SurrogatePlan with best feasible candidate

    Raises:
        ValueError: If candidate_offsets contains non-finite values, exceeds
                    max_candidates, or has other shape/type issues
    """
    if candidate_offsets.ndim != 2 or candidate_offsets.shape[1] != benefit.numel():
        raise ValueError('candidate_offsets must be [K, d]')

    k_count = candidate_offsets.shape[0]
    d = benefit.numel()

    # Hardening: max_candidates cap enforcement
    if k_count > max_candidates:
        raise ValueError(
            f'candidate count K={k_count} exceeds max_candidates={max_candidates}; '
            f'reduce candidates or increase max_candidates limit'
        )

    # Hardening: finiteness check on entire candidate array
    if not torch.isfinite(candidate_offsets).all():
        nonfinite_count = (~torch.isfinite(candidate_offsets)).sum().item()
        raise ValueError(
            f'candidate_offsets contains {nonfinite_count} non-finite values (NaN/Inf); '
            f'all candidates must be finite'
        )

    b = benefit.detach().double().cpu().numpy()
    h = gram.detach().double().cpu().numpy()
    c = np.ones(d) if costs is None else np.asarray(costs, dtype=float)
    candidates = candidate_offsets.detach().double().cpu().numpy()

    best = np.zeros(d)
    best_gain = 0.
    best_predicted = 0.
    best_support = ()
    best_cost = 0.
    solves = 0
    rejected_bound = 0
    rejected_support = 0

    for k in range(candidates.shape[0]):
        offset = candidates[k]
        support = tuple(i for i in range(d) if offset[i] != 0)
        if not support:
            continue

        # Hardening: max_active support-size check
        if len(support) > max_active:
            rejected_support += 1
            continue

        # Hardening: bound enforcement per-coefficient
        if np.abs(offset).max() > bound + 1e-12:
            rejected_bound += 1
            continue

        idx = np.asarray(support)
        cost = float(c[idx].sum())
        if cost > budget + 1e-12:
            continue
        solves += 1
        gain = quadratic_gain_numpy(b, h, offset) - ridge * float((offset ** 2).sum())
        if gain > best_gain + 1e-12:
            best = offset.copy()
            best_support = support
            best_cost = cost
            best_gain = gain
            best_predicted = quadratic_gain_numpy(b, h, best)

    return SurrogatePlan(torch.from_numpy(best).to(benefit), best_support, best_predicted,
                         best_gain, best_cost, solves, rejected_bound + rejected_support)


@dataclass(frozen=True)
class RegistrySelection:
    """What a formal registry selection chose, and which registered row it was."""
    plan_id: str
    coefficients: tuple[float, ...]
    is_reference: bool
    surrogate: SurrogatePlan


def select_from_registry(registry, benefit: Tensor, gram: Tensor, *,
                         costs: tuple[float, ...] | None = None,
                         budget: float = 2., ridge: float = 1e-4,
                         max_active: int = 2,
                         max_candidates: int = 4096) -> RegistrySelection:
    """Select from a validated registry, never from an implied candidate set.

    `plan_from_prediction` is a planner: handed an empty or wholly-infeasible
    candidate table it returns the all-zero program, which is the right answer
    to "what should I do" and a useless answer to "what did I choose". This
    entry point closes that gap for formal consumption by requiring the
    candidate table to come from a registry that has already been validated --
    non-empty, real-float coefficients, an explicit no-edit row -- so an
    all-zero result means the registered reference won, not that nothing was
    ever registered.

    The registry's reference occupies row 0 of the offsets table, so the chosen
    coefficients can always be mapped back to a registered plan_id.

    Args:
        registry: A `earthdelta.registry.CandidateRegistry`.
        benefit: [d] predicted benefit vector.
        gram: [d, d] predicted Gram matrix.
        costs: Per-dimension costs (None = unit costs).
        budget: Maximum total cost.
        ridge: Ridge regularization.
        max_active: Maximum number of active dimensions.
        max_candidates: Safety limit on the candidate count.

    Returns:
        RegistrySelection naming the registered plan that was selected.

    Raises:
        RegistryViolation: If the registry is empty, lacks the explicit no-edit
            entry, or holds any non-real / non-finite coefficient.
        ValueError: From the planner, on invalid benefit/gram/costs.
    """
    summary = registry.assert_ready(require_candidates=True)
    offsets = registry.candidate_offsets(dtype=benefit.dtype).to(benefit.device)
    if offsets.shape[1] != benefit.numel():
        raise ValueError(
            f'registry holds plans over {offsets.shape[1]} experts but benefit '
            f'has {benefit.numel()} dimensions'
        )

    plan = plan_from_prediction(
        benefit, gram, bound=float(summary['rho']), max_active=max_active,
        costs=costs, budget=budget, ridge=ridge, max_candidates=max_candidates,
        candidate_offsets=offsets,
    )

    chosen = plan.coefficients.detach().to(torch.float64).cpu()
    # The plan comes back in the benefit's dtype, so a coefficient that is not
    # exactly representable there (0.1 in float32) differs from the registered
    # float64 value by a rounding step. Match at the precision the round trip
    # actually has, not at float64 precision it never had.
    atol = 1e-12 if offsets.dtype == torch.float64 else 1e-6
    reference = registry.reference_entry
    for entry in ([reference] + [e for e in registry.entries if e is not reference]):
        row = torch.tensor(entry.coefficients, dtype=torch.float64)
        if torch.allclose(row, chosen, rtol=0., atol=atol):
            return RegistrySelection(
                plan_id=entry.plan_id,
                coefficients=tuple(float(a) for a in entry.coefficients),
                is_reference=bool(entry.is_reference),
                surrogate=plan,
            )

    raise ValueError(
        f'selected coefficients {chosen.tolist()} match no registered plan in '
        f'{summary["registry"]!r}; the planner returned a program the registry '
        'does not contain'
    )


def unified_select(predicted_gain: Tensor | None = None,
                   total_cost: Tensor | None = None,
                   budget: float = 2.0,
                   plan_ids: tuple[str, ...] | None = None,
                   benefit: Tensor | None = None,
                   gram: Tensor | None = None,
                   candidate_offsets: Tensor | None = None,
                   costs: tuple[float, ...] | None = None,
                   bound: float = 0.25,
                   max_active: int = 2,
                   ridge: float = 1e-4,
                   cost_penalty: float = 0.0) -> SurrogatePlan | Tensor:
    """Unified selection interface for both finite and continuous programs.

    This is the main entry point that dispatches to either:
    - select_plan: when predicted_gain and plan_ids are provided (batch discrete selection)
    - plan_from_prediction: when benefit and gram are provided (single-sample continuous QP)

    For finite candidates as a special case of continuous program:
    - Provide benefit, gram, and candidate_offsets
    - This evaluates the quadratic at each candidate and returns the best feasible one

    Args:
        predicted_gain: [B, K] predicted gains (for discrete selection)
        total_cost: [K] total costs (for discrete selection)
        budget: Maximum allowed cost
        plan_ids: Unique identifiers (for discrete selection)
        benefit: [d] benefit vector (for continuous/finite QP)
        gram: [d, d] Gram matrix (for continuous/finite QP)
        candidate_offsets: [K, d] finite candidates (None = continuous QP)
        costs: Per-dimension costs
        bound: Componentwise coefficient bound
        max_active: Maximum active dimensions
        ridge: Ridge regularization
        cost_penalty: Cost penalty for discrete selection

    Returns:
        Either [B] selection indices (discrete) or SurrogatePlan (continuous/finite QP)
    """
    # Discrete batch selection path
    if predicted_gain is not None and plan_ids is not None:
        if total_cost is None:
            raise ValueError('total_cost required for discrete selection')
        return select_plan(predicted_gain, total_cost, budget, plan_ids, cost_penalty)

    # Continuous/finite QP path
    if benefit is not None and gram is not None:
        return plan_from_prediction(benefit, gram, bound=bound, max_active=max_active,
                                    costs=costs, budget=budget, ridge=ridge,
                                    candidate_offsets=candidate_offsets)

    raise ValueError('Either (predicted_gain, total_cost, plan_ids) or (benefit, gram) must be provided')
