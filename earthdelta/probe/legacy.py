"""Finite differences of a pure, deterministic forecast-summary callable.

The callable receives only a program coefficient vector, NEVER future truth.
It must replay from the same initial state (including history and RNG state).
Output is a fixed, aligned 1-D physical/normalized summary, not a loss.

For finite candidate sets, use cached_responses to precompute responses rather
than recomputing via central difference each time.
"""
from dataclasses import dataclass
from typing import Callable
import math
import torch
from torch import Tensor


def finite_vector(x: Tensor, name: str) -> None:
    """Validate that x is a nonempty, finite, floating-point 1-D tensor."""
    if not isinstance(x, Tensor) or x.ndim != 1 or x.numel() == 0:
        raise ValueError(f'{name} must be a nonempty 1-D tensor')
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError(f'{name} must be finite floating point')


@dataclass(frozen=True)
class ProbeResult:
    """Result of a central difference probe."""
    reference: Tensor           # [d] reference coefficient vector
    reference_output: Tensor    # [m] output at reference
    response: Tensor            # [m, d] Jacobian: d(output)/d(coefficient)
    epsilon: Tensor             # [d] perturbation sizes used
    forward_calls: int          # number of forward passes made


@torch.no_grad()
def central_response(evaluate: Callable[[Tensor], Tensor], reference: Tensor,
                     epsilon: float | Tensor = 1e-2,
                     check_repeatability: bool = True,
                     repeat_atol: float = 0.0) -> ProbeResult:
    """Compute response matrix via central finite differences.

    Costs 2d+1 full trajectories, plus ONE if repeatability is checked.

    Subtractions are accumulated in float64; forward dtype is unchanged.
    This cannot recover information lost in fp16/bfloat16 forward passes.
    Caller must test epsilon/2, epsilon, 2epsilon and mixed perturbations.

    Args:
        evaluate: Pure function mapping [d] coefficients to [m] summary
        reference: [d] reference coefficient vector
        epsilon: Scalar or [d] perturbation sizes (positive, finite)
        check_repeatability: If True, verify the function is deterministic
        repeat_atol: Absolute tolerance for repeatability check

    Returns:
        ProbeResult containing the response matrix
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
        columns.append((yp - ym) / (2 * eps[j].double()))
    return ProbeResult(a0, y0, torch.stack(columns, dim=1), eps, calls)


@torch.no_grad()
def local_linearity_error(evaluate: Callable[[Tensor], Tensor], probe: ProbeResult,
                          offset: Tensor, floor: float = 1e-10) -> float:
    """Compute relative error of a mixed perturbation vs linear prediction.

    Costs one additional trajectory. No truth is needed. Agreement only concerns
    the neural model's response, not atmospheric causality or whether the
    resulting prediction is useful.

    Args:
        evaluate: Same callable used for the probe
        probe: ProbeResult from central_response
        offset: [d] perturbation from reference
        floor: Minimum denominator to avoid division by zero

    Returns:
        Relative error ||actual - linear|| / max(||actual||, ||linear||, floor)
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
    return float(torch.linalg.vector_norm(actual - linear)) / denominator


@dataclass(frozen=True)
class CachedResponses:
    """Precomputed responses for a finite candidate set."""
    reference: Tensor           # [d] reference coefficient vector
    reference_output: Tensor    # [m] output at reference
    candidate_offsets: Tensor   # [K, d] offset vectors for K candidates
    candidate_outputs: Tensor   # [K, m] outputs for K candidates
    responses: Tensor           # [K, m] nonlinear responses (output - reference_output)


@torch.no_grad()
def cached_responses(evaluate: Callable[[Tensor], Tensor], reference: Tensor,
                     candidate_offsets: Tensor,
                     check_repeatability: bool = True,
                     repeat_atol: float = 0.0) -> CachedResponses:
    """Precompute nonlinear responses for a finite candidate set.

    This caches the actual nonlinear du directly rather than relying on
    linear approximations from central differences. Use this when you have
    a known finite set of candidates and want exact responses.

    Args:
        evaluate: Pure function mapping [d] coefficients to [m] summary
        reference: [d] reference coefficient vector
        candidate_offsets: [K, d] offset vectors for K candidates
        check_repeatability: If True, verify the function is deterministic
        repeat_atol: Absolute tolerance for repeatability check

    Returns:
        CachedResponses containing exact nonlinear responses for each candidate
    """
    finite_vector(reference, 'reference')
    if candidate_offsets.ndim != 2 or candidate_offsets.shape[1] != reference.numel():
        raise ValueError('candidate_offsets must be [K, d] where d matches reference')
    if not candidate_offsets.is_floating_point() or not bool(torch.isfinite(candidate_offsets).all()):
        raise ValueError('candidate_offsets must be finite floating point')
    if not math.isfinite(repeat_atol) or repeat_atol < 0:
        raise ValueError('repeat_atol must be nonnegative and finite')

    a0 = reference.detach().clone()

    def run(a: Tensor) -> Tensor:
        argument = a.detach().clone()
        old = argument.clone()
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

    outputs = []
    for k in range(candidate_offsets.shape[0]):
        offset = candidate_offsets[k].to(a0)
        yk = run(a0 + offset)
        if yk.shape != y0.shape:
            raise ValueError('summary shape changed across candidates')
        outputs.append(yk)

    candidate_outputs = torch.stack(outputs, dim=0)
    responses = candidate_outputs - y0[None, :]

    return CachedResponses(
        reference=a0,
        reference_output=y0,
        candidate_offsets=candidate_offsets.detach().clone(),
        candidate_outputs=candidate_outputs,
        responses=responses
    )


def jvp_response(evaluate: Callable[[Tensor], Tensor], reference: Tensor,
                 direction: Tensor) -> Tensor:
    """Compute directional derivative via torch.func.jvp.

    TODO: Implement when torch.func.jvp is available and performant for our use case.
    This would allow computing Jacobian-vector products without the 2d+1 trajectory
    cost of full finite differences, useful for high-dimensional coefficient spaces.

    Args:
        evaluate: Pure function mapping [d] coefficients to [m] summary
        reference: [d] reference coefficient vector
        direction: [d] direction for the JVP

    Returns:
        [m] directional derivative of output w.r.t. coefficients in the given direction

    Raises:
        NotImplementedError: This is a documented placeholder for future torch.func.jvp support
    """
    raise NotImplementedError(
        "JVP-based response computation is not yet implemented. "
        "This is a placeholder for a future torch.func.jvp-based path that would "
        "compute directional derivatives more efficiently than central differences "
        "for high-dimensional coefficient spaces. For now, use central_response or "
        "cached_responses for finite candidate sets."
    )
