"""Finite, budgeted plan selection; evaluation truth is not an input."""
from __future__ import annotations
import math
import torch
from torch import Tensor


def select_plan(predicted_gain: Tensor, total_cost: Tensor, budget: float,
                plan_ids: tuple[str,...], cost_penalty: float = 0.0) -> Tensor:
    """Inputs [B,K] and [K]. Costs must INCLUDE controller + selected rollout costs.

    Non-differentiable selection. Ties: lower total cost, then stable plan ID.
    No-edit must be part of the candidate set supplied by the caller.
    """
    if predicted_gain.ndim != 2 or total_cost.shape != (predicted_gain.shape[1],):
        raise ValueError('Shape mismatch for gain/cost.')
    if len(plan_ids) != predicted_gain.shape[1] or len(set(plan_ids)) != len(plan_ids):
        raise ValueError('Plan IDs must be unique and aligned with columns.')
    if not math.isfinite(budget) or budget < 0 or not math.isfinite(cost_penalty) or cost_penalty < 0:
        raise ValueError('Invalid budget/cost penalty.')
    if not torch.isfinite(predicted_gain).all() or not torch.isfinite(total_cost).all() or (total_cost<0).any():
        raise ValueError('Finite gain and finite nonnegative costs are required.')
    cost = total_cost.detach().cpu().tolist()
    feasible = [i for i,c in enumerate(cost) if c <= budget]
    if not feasible:
        raise ValueError('Budget cannot run any candidate, including the reference.')
    scores = (predicted_gain-cost_penalty*total_cost.to(predicted_gain)[None]).detach().cpu().tolist()
    chosen = [min(feasible,key=lambda i:(-row[i],cost[i],plan_ids[i])) for row in scores]
    return torch.tensor(chosen,device=predicted_gain.device,dtype=torch.long)


def selection_regret(true_gain: Tensor, selected: Tensor, costs: Tensor,
                     budget: float, penalty: float = 0.0) -> Tensor:
    """POST-HOC diagnostic only. Uses realized gains, never deployment input."""
    util = true_gain-penalty*costs.to(true_gain)[None]
    feasible = costs.to(true_gain.device) <= budget
    if not feasible.any() or selected.shape != true_gain.shape[:1]:
        raise ValueError('Invalid feasible set or selected shape.')
    if not feasible[selected].all():
        raise ValueError('Selected plan is outside budget.')
    best = util.masked_fill(~feasible[None],-torch.inf).max(-1).values
    return best-util.gather(1,selected[:,None]).squeeze(1)
