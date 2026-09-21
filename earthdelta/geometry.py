"""Training-side local geometry. These are standard quadratic identities."""
from dataclasses import dataclass
import torch
from torch import Tensor
from .probe import finite_vector
from .metrics_contract import quadratic_gain_from_benefit_gram, WeightConvention


@dataclass(frozen=True)
class ResponseGeometry:
    """Local quadratic geometry for response-based optimization.

    gram = R.T Q R is a Gauss-Newton/local linear metric, NOT the exact
    nonlinear loss Hessian. 'Benefit' is label dependent, not available
    at inference.

    Weight convention: WEIGHTED_SUM (no automatic normalization). The spatial
    weights Q are baked into gram and benefit at construction time, so
    predicted_gain computes 2*b.a - a.T H a directly without further weighting.
    This is appropriate because the weights are already embedded in the geometry.
    """
    response: Tensor     # [m, d] response matrix
    error: Tensor        # [m] error vector
    weights: Tensor      # [m] weights
    gram: Tensor         # [d, d] R.T Q R
    benefit: Tensor      # [d] R.T (Q * e)
    baseline_loss: Tensor  # scalar

    @classmethod
    def from_error(cls, response: Tensor, error: Tensor, weights: Tensor):
        """Construct geometry from error and response.

        error = truth_summary - reference_prediction. TRAINING ONLY.

        gram = R.T Q R is a Gauss-Newton/local linear metric, NOT the exact
        nonlinear loss Hessian. 'Benefit' is label dependent, not available
        at inference.

        Convention: WEIGHTED_SUM - No automatic normalization of weights is
        performed. The weights Q are incorporated directly into gram and benefit
        as R.T Q R and R.T (Q * e) respectively. This is appropriate because
        the caller provides weights representing absolute importance, not
        relative frequencies. For spatial averaging, normalize weights before
        calling.

        Args:
            response: [m, d] response matrix
            error: [m] error vector
            weights: [m] nonnegative weights (not normalized)

        Returns:
            ResponseGeometry instance
        """
        finite_vector(error, 'error')
        finite_vector(weights, 'weights')
        if response.ndim != 2 or response.shape[0] != error.numel() or response.shape[1] == 0:
            raise ValueError('response must be [m,d]')
        if not response.is_floating_point() or not bool(torch.isfinite(response).all()):
            raise ValueError('invalid response')
        if weights.shape != error.shape or bool((weights < 0).any()) or not bool(weights.sum() > 0):
            raise ValueError('weights must be nonnegative, aligned, and nonzero')
        r = response.detach().double().clone()
        e = error.detach().to(r).clone()
        w = weights.detach().to(r).clone()
        return cls(r, e, w, r.T @ (w[:, None] * r), r.T @ (w * e), (w * e.square()).sum())

    def predicted_gain(self, offset: Tensor) -> Tensor:
        """Predict gain for a given offset: 2*b.a - a.T H a.

        This computes gain in coefficient space where spatial weights are
        already baked into benefit (b = R.T Q e) and gram (H = R.T Q R).
        Uses the canonical quadratic_gain_from_benefit_gram function.

        Args:
            offset: [d] coefficient offset

        Returns:
            Scalar predicted gain
        """
        a = offset.to(self.gram)
        if a.ndim != 1 or a.shape[0] != self.gram.shape[0] or not bool(torch.isfinite(a).all()):
            raise ValueError('invalid offset')
        # Use canonical coefficient-space gain function (weights already in gram/benefit)
        return quadratic_gain_from_benefit_gram(self.benefit, self.gram, a)


def response_distillation(student: Tensor, teacher: Tensor, gram: Tensor) -> Tensor:
    """Compute response distillation loss: mean (a_student-a_teacher)^T H (a_student-a_teacher).

    Supports [d] with [d,d], or [B,d] with shared [d,d] / per-sample [B,d,d].
    Teacher and geometry are detached; ONLY student receives gradients.
    Non-uniqueness is intentional. Add separate amplitude/domain constraints.

    Args:
        student: [d] or [B, d] student coefficients
        teacher: [d] or [B, d] teacher coefficients (same shape as student)
        gram: [d, d] or [B, d, d] Gram matrix (must be symmetric PSD)

    Returns:
        Scalar mean distillation loss
    """
    if student.ndim not in (1, 2) or student.shape != teacher.shape:
        raise ValueError('student and teacher must match [d] or [B,d]')
    d = student.shape[-1]
    if gram.shape not in ((d, d), (*student.shape[:-1], d, d)):
        raise ValueError('invalid Gram shape')
    if any(not t.is_floating_point() or not bool(torch.isfinite(t).all()) for t in (student, teacher, gram)):
        raise ValueError('nonfinite or nonfloating values')
    # Keep the teacher/metric precision rather than rounding labels to a
    # reduced-precision student's dtype. Student casts retain autograd.
    work_dtype = torch.float64 if (gram.dtype == torch.float64 or student.dtype == torch.float64) else torch.float32
    h = gram.detach().to(device=student.device, dtype=work_dtype)
    if not torch.allclose(h, h.transpose(-1, -2), atol=1e-6, rtol=1e-6):
        raise ValueError('Gram must be symmetric')
    if bool((torch.linalg.eigvalsh(h.double()).min(dim=-1).values < -1e-7).any()):
        raise ValueError('Gram must be PSD')
    delta = student.to(dtype=work_dtype) - teacher.detach().to(device=student.device, dtype=work_dtype)
    return torch.einsum('...i,...ij,...j->...', delta, h, delta).mean()
