"""Training-side local geometry. These are standard quadratic identities."""
from dataclasses import dataclass
import torch
from torch import Tensor
from .probe import finite_vector


@dataclass(frozen=True)
class ResponseGeometry:
    response: Tensor
    error: Tensor
    weights: Tensor
    gram: Tensor
    benefit: Tensor
    baseline_loss: Tensor

    @classmethod
    def from_error(cls, response: Tensor, error: Tensor, weights: Tensor):
        """error = truth_summary - reference_prediction. TRAINING ONLY.

        gram = R.T Q R is a Gauss-Newton/local linear metric, NOT the exact
        nonlinear loss Hessian. 'Benefit' is label dependent, not available
        at inference. No automatic normalization of weights is performed.
        """
        finite_vector(error, 'error'); finite_vector(weights, 'weights')
        if response.ndim != 2 or response.shape[0] != error.numel() or response.shape[1] == 0:
            raise ValueError('response must be [m,d]')
        if not response.is_floating_point() or not bool(torch.isfinite(response).all()):
            raise ValueError('invalid response')
        if weights.shape != error.shape or bool((weights < 0).any()) or not bool(weights.sum() > 0):
            raise ValueError('weights must be nonnegative, aligned, and nonzero')
        r = response.detach().double().clone()
        e = error.detach().to(r).clone(); w = weights.detach().to(r).clone()
        return cls(r, e, w, r.T @ (w[:,None]*r), r.T @ (w*e), (w*e.square()).sum())

    def predicted_gain(self, offset: Tensor) -> Tensor:
        a = offset.to(self.gram)
        if a.ndim != 1 or a.shape[0] != self.gram.shape[0] or not bool(torch.isfinite(a).all()):
            raise ValueError('invalid offset')
        return 2*self.benefit.dot(a) - a @ self.gram @ a


def response_distillation(student: Tensor, teacher: Tensor, gram: Tensor) -> Tensor:
    """Mean (a_student-a_teacher)^T H (a_student-a_teacher).

    Supports [d] with [d,d], or [B,d] with shared [d,d] / per-sample [B,d,d].
    Teacher and geometry are detached; ONLY student receives gradients.
    Non-uniqueness is intentional. Add separate amplitude/domain constraints.
    """
    if student.ndim not in (1,2) or student.shape != teacher.shape:
        raise ValueError('student and teacher must match [d] or [B,d]')
    d = student.shape[-1]
    if gram.shape not in ((d,d), (*student.shape[:-1],d,d)):
        raise ValueError('invalid Gram shape')
    if any(not t.is_floating_point() or not bool(torch.isfinite(t).all()) for t in (student,teacher,gram)):
        raise ValueError('nonfinite or nonfloating values')
    # Keep the teacher/metric precision rather than rounding labels to a
    # reduced-precision student's dtype. Student casts retain autograd.
    work_dtype = torch.float64 if (gram.dtype == torch.float64 or student.dtype == torch.float64) else torch.float32
    h = gram.detach().to(device=student.device, dtype=work_dtype)
    if not torch.allclose(h,h.transpose(-1,-2),atol=1e-6,rtol=1e-6):
        raise ValueError('Gram must be symmetric')
    if bool((torch.linalg.eigvalsh(h.double()).min(dim=-1).values < -1e-7).any()):
        raise ValueError('Gram must be PSD')
    delta=student.to(dtype=work_dtype)-teacher.detach().to(device=student.device, dtype=work_dtype)
    return torch.einsum('...i,...ij,...j->...',delta,h,delta).mean()
