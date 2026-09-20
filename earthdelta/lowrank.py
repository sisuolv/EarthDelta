"""K-expert low-rank adapters targeting attn.proj-style linear layers.

Generalizes from rank-group masks to K experts that can share or have independent
A/B factors, with explicit coefficient tensors. Provides both a dense/differentiable
path (computes all K experts' contributions weighted by coefficients, always) and
a sparse/inference path (skips zero-coefficient experts entirely).

These two paths must produce identical outputs - this is verified by numerical
equivalence tests.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn


class ExpertLoRA(nn.Module):
    """K low-rank experts targeting linear layers like attn.proj.

    Each expert is a low-rank factorization: expert_k = scale * B_k @ A_k
    Experts can share A factors (shared_down=True) or B factors (shared_up=True).
    Coefficients are explicit tensor inputs, not implicit from which expert is "on".

    The dense path always computes all K expert contributions weighted by coefficients.
    The sparse path skips experts with zero coefficients.
    Both paths produce identical results - this is a numerical requirement, not optimization.
    """

    def __init__(self, in_features: int, out_features: int, num_experts: int = 8,
                 rank_per_expert: int = 4, scale: float = 1.0,
                 shared_down: bool = False, shared_up: bool = False):
        """Initialize ExpertLoRA.

        Args:
            in_features: Input dimension of target linear layer
            out_features: Output dimension of target linear layer
            num_experts: Number of experts K
            rank_per_expert: Rank of each expert's low-rank factorization
            scale: Scaling factor for expert outputs
            shared_down: If True, all experts share the same A (down) projection
            shared_up: If True, all experts share the same B (up) projection
        """
        super().__init__()
        if min(in_features, out_features, num_experts, rank_per_expert) <= 0 or not math.isfinite(scale):
            raise ValueError('Invalid feature/expert/rank/scale configuration.')

        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.rank_per_expert = rank_per_expert
        self.scale = scale
        self.shared_down = shared_down
        self.shared_up = shared_up

        # Create down (A) projections
        if shared_down:
            self.down = nn.ModuleList([nn.Linear(in_features, rank_per_expert, bias=False)])
        else:
            self.down = nn.ModuleList([
                nn.Linear(in_features, rank_per_expert, bias=False)
                for _ in range(num_experts)
            ])

        # Create up (B) projections
        if shared_up:
            self.up = nn.ModuleList([nn.Linear(rank_per_expert, out_features, bias=False)])
        else:
            self.up = nn.ModuleList([
                nn.Linear(rank_per_expert, out_features, bias=False)
                for _ in range(num_experts)
            ])

        # Initialize: Kaiming for down, zeros for up (like LoRA)
        for a in self.down:
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in self.up:
            nn.init.zeros_(b.weight)

    def _get_down(self, expert_idx: int) -> nn.Linear:
        """Get down projection for expert."""
        return self.down[0] if self.shared_down else self.down[expert_idx]

    def _get_up(self, expert_idx: int) -> nn.Linear:
        """Get up projection for expert."""
        return self.up[0] if self.shared_up else self.up[expert_idx]

    def forward_dense(self, x: Tensor, coefficients: Tensor) -> Tensor:
        """Dense/differentiable forward pass.

        Computes ALL K experts' contributions and weights by coefficients.
        This is the gradient-correct path and does NOT skip any computation.
        Zero coefficients still trigger matmuls (the result is zeroed after).

        Args:
            x: [B, N, in_features] input tensor
            coefficients: [B, K] expert coefficients

        Returns:
            [B, N, out_features] weighted sum of expert outputs
        """
        if x.ndim != 3 or x.shape[-1] != self.in_features:
            raise ValueError('x must be [B, N, in_features].')
        if coefficients.shape != (x.shape[0], self.num_experts):
            raise ValueError('coefficients must be [B, K].')
        if not torch.isfinite(coefficients).all():
            raise ValueError('coefficients must be finite.')

        out = x.new_zeros(x.shape[0], x.shape[1], self.out_features)
        for k in range(self.num_experts):
            a = self._get_down(k)
            b = self._get_up(k)
            # Compute expert output and scale by coefficient
            expert_out = b(a(x)) * self.scale  # [B, N, out]
            out = out + expert_out * coefficients[:, k, None, None]
        return out

    def forward_sparse(self, x: Tensor, coefficients: Tensor) -> Tensor:
        """Sparse/inference forward pass.

        Skips experts with zero coefficients entirely. This is numerically
        equivalent to forward_dense but may be faster when many experts are inactive.

        Args:
            x: [B, N, in_features] input tensor
            coefficients: [B, K] expert coefficients

        Returns:
            [B, N, out_features] weighted sum of active expert outputs
        """
        if x.ndim != 3 or x.shape[-1] != self.in_features:
            raise ValueError('x must be [B, N, in_features].')
        if coefficients.shape != (x.shape[0], self.num_experts):
            raise ValueError('coefficients must be [B, K].')
        if not torch.isfinite(coefficients).all():
            raise ValueError('coefficients must be finite.')

        coefficients = coefficients.to(x.device)
        out = x.new_zeros(x.shape[0], x.shape[1], self.out_features)

        for k in range(self.num_experts):
            # Find rows where this expert is active
            active_mask = coefficients[:, k] != 0
            if not active_mask.any():
                continue

            a = self._get_down(k)
            b = self._get_up(k)

            # Process only active rows
            rows = active_mask.nonzero(as_tuple=True)[0]
            x_active = x.index_select(0, rows)
            expert_out = b(a(x_active)) * coefficients[rows, k, None, None] * self.scale
            out = out.index_add(0, rows, expert_out)

        return out

    def forward(self, x: Tensor, coefficients: Tensor, sparse: bool = False) -> Tensor:
        """Forward pass with selectable path.

        Args:
            x: [B, N, in_features] input tensor
            coefficients: [B, K] expert coefficients
            sparse: If True, use sparse path (skip zero-coefficient experts)

        Returns:
            [B, N, out_features] output
        """
        if sparse:
            return self.forward_sparse(x, coefficients)
        else:
            return self.forward_dense(x, coefficients)

    def rank_upper_bound(self, active: Tensor) -> Tensor:
        """Compute upper bound on effective rank given active expert mask.

        Args:
            active: [B, K] boolean mask of active experts

        Returns:
            [B] upper bound on rank for each batch element
        """
        num_active = active.sum(-1)
        if self.shared_down or self.shared_up:
            # Shared factors limit the effective rank
            return (num_active * self.rank_per_expert).clamp(max=min(self.in_features, self.out_features, self.rank_per_expert))
        else:
            return (num_active * self.rank_per_expert).clamp(max=min(self.in_features, self.out_features))


def verify_dense_sparse_equivalence(module: ExpertLoRA, x: Tensor, coefficients: Tensor,
                                    atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    """Verify that dense and sparse paths produce identical outputs.

    This is a numerical correctness check, not an optimization check.
    Both paths must produce the same result.

    Args:
        module: ExpertLoRA module to test
        x: [B, N, in_features] input tensor
        coefficients: [B, K] coefficients (should include zeros for a meaningful test)
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        True if outputs match, False otherwise
    """
    with torch.no_grad():
        dense_out = module.forward_dense(x, coefficients)
        sparse_out = module.forward_sparse(x, coefficients)
        return torch.allclose(dense_out, sparse_out, atol=atol, rtol=rtol)


# Legacy compatibility: GroupedLowRankResidual wraps ExpertLoRA with boolean masks
class GroupedLowRankResidual(nn.Module):
    """Legacy grouped low-rank residual (wraps ExpertLoRA).

    This provides backward compatibility with the v5 API that uses boolean
    active masks. Internally it uses ExpertLoRA with the mask converted to
    coefficient zeros/ones.
    """

    def __init__(self, in_features: int, out_features: int, groups: int = 4,
                 rank_per_group: int = 4, scale: float = 1.0):
        super().__init__()
        if min(in_features, out_features, groups, rank_per_group) <= 0 or not math.isfinite(scale):
            raise ValueError('Invalid feature/group/rank/scale configuration.')
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.rank_per_group = rank_per_group
        self.scale = scale

        # Use separate down/up for exact v5 compatibility
        self.down = nn.ModuleList([nn.Linear(in_features, rank_per_group, bias=False) for _ in range(groups)])
        self.up = nn.ModuleList([nn.Linear(rank_per_group, out_features, bias=False) for _ in range(groups)])
        for a, b in zip(self.down, self.up):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)

    def forward(self, x: Tensor, active: Tensor, coefficients: Tensor | None = None) -> Tensor:
        """Forward pass with boolean active mask.

        Args:
            x: [B, N, in_features] input
            active: [B, groups] boolean mask
            coefficients: Optional [B, groups] coefficients (defaults to float(active))

        Returns:
            [B, N, out_features] output
        """
        if x.ndim != 3 or x.shape[-1] != self.in_features:
            raise ValueError('x must be [B,N,in_features].')
        if active.dtype != torch.bool or active.shape != (x.shape[0], self.groups):
            raise ValueError('active must be bool [B,groups].')
        active = active.to(x.device)
        c = active.to(x.dtype) if coefficients is None else coefficients.to(x)
        if c.shape != active.shape or not torch.isfinite(c).all():
            raise ValueError('coefficients must be finite [B,groups].')
        if ((~active) & (c != 0)).any():
            raise ValueError('Inactive coefficients must equal zero.')

        out = x.new_zeros(x.shape[0], x.shape[1], self.out_features)
        for g, (a, b) in enumerate(zip(self.down, self.up)):
            rows = active[:, g].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            y = b(a(x.index_select(0, rows))) * c[rows, g, None, None] * self.scale
            out = out.index_add(0, rows, y)
        return out

    def rank_upper_bound(self, active: Tensor) -> Tensor:
        """Upper bound on effective rank."""
        return (active.sum(-1) * self.rank_per_group).clamp(max=min(self.in_features, self.out_features))
