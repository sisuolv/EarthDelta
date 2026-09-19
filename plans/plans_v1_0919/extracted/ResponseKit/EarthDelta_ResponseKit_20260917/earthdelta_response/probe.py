"""Finite differences of a pure, deterministic forecast-summary callable.

The callable receives only a program coefficient vector, NEVER future truth.
It must replay from the same initial state (including history and RNG state).
Output is a fixed, aligned 1-D physical/normalized summary, not a loss.
"""
from dataclasses import dataclass
from typing import Callable
import math
import torch
from torch import Tensor


def finite_vector(x: Tensor, name: str) -> None:
    if not isinstance(x, Tensor) or x.ndim != 1 or x.numel() == 0:
        raise ValueError(f'{name} must be a nonempty 1-D tensor')
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError(f'{name} must be finite floating point')


@dataclass(frozen=True)
class ProbeResult:
    reference: Tensor
    reference_output: Tensor
    response: Tensor  # [summary_dimension, TOTAL program_dimension]
    epsilon: Tensor
    forward_calls: int


@torch.no_grad()
def central_response(evaluate: Callable[[Tensor], Tensor], reference: Tensor,
                     epsilon: float | Tensor = 1e-2,
                     check_repeatability: bool = True,
                     repeat_atol: float = 0.0) -> ProbeResult:
    """2d+1 full trajectories, plus ONE if repeatability is checked.

    Subtractions are accumulated in float64; forward dtype is unchanged.
    This cannot recover information lost in fp16/bfloat16 forward passes.
    Caller must test epsilon/2, epsilon, 2epsilon and mixed perturbations.
    """
    finite_vector(reference, 'reference')
    if not math.isfinite(repeat_atol) or repeat_atol < 0:
        raise ValueError('repeat_atol must be nonnegative and finite')
    eps = torch.as_tensor(epsilon, device=reference.device, dtype=reference.dtype)
    if eps.ndim == 0:
        eps = eps.expand_as(reference).clone()
    if eps.shape != reference.shape or not bool(torch.isfinite(eps).all()) or not bool((eps > 0).all()):
        raise ValueError('epsilon must be positive, finite, scalar or [d]')
    a0 = reference.detach().clone()
    calls = 0

    def run(a: Tensor) -> Tensor:
        nonlocal calls
        argument = a.detach().clone()
        old = argument.clone()
        calls += 1
        y = evaluate(argument)
        if not torch.equal(argument, old):
            raise ValueError('evaluate mutated its coefficient argument')
        finite_vector(y, 'forecast summary')
        return y.detach().to(dtype=torch.float64).clone()

    y0 = run(a0)
    if check_repeatability:
        y1 = run(a0)
        if y1.shape != y0.shape or not torch.allclose(y0, y1, rtol=0., atol=repeat_atol):
            raise ValueError('non-repeatable replay; check state, randomness and caches')
    columns = []
    for j in range(a0.numel()):
        p, m = a0.clone(), a0.clone()
        p[j] += eps[j]
        m[j] -= eps[j]
        if p[j] == m[j]:
            raise ValueError('epsilon is below forward coefficient precision')
        yp, ym = run(p), run(m)
        if yp.shape != y0.shape or ym.shape != y0.shape:
            raise ValueError('summary shape changed across branches')
        columns.append((yp-ym)/(2*eps[j].double()))
    return ProbeResult(a0, y0, torch.stack(columns, dim=1), eps, calls)


@torch.no_grad()
def local_linearity_error(evaluate: Callable[[Tensor], Tensor], probe: ProbeResult,
                         offset: Tensor, floor: float = 1e-10) -> float:
    """Relative error of a mixed perturbation; costs one additional trajectory.

    No truth is needed. Agreement only concerns the neural model's response,
    not atmospheric causality or whether the resulting prediction is useful.
    """
    finite_vector(offset, 'offset')
    if offset.shape != probe.reference.shape or floor <= 0 or not math.isfinite(floor):
        raise ValueError('invalid offset/floor')
    y = evaluate((probe.reference + offset.to(probe.reference)).clone())
    finite_vector(y, 'forecast summary')
    if y.shape != probe.reference_output.shape:
        raise ValueError('summary shape mismatch')
    actual = y.double() - probe.reference_output
    linear = probe.response @ offset.to(probe.response)
    denominator = max(float(torch.linalg.vector_norm(actual)),
                      float(torch.linalg.vector_norm(linear)), floor)
    return float(torch.linalg.vector_norm(actual-linear))/denominator
