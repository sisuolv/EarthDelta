"""Prediction heads for response-based control.

Splits the v5 PairedEditPredictor monolithic model into composable nn.Modules:
- BoundedProgramHead: Predicts bounded program coefficients
- InteractionUtilityHead: Predicts (benefit, Gram) for quadratic gain
- ReferenceErrorHead: Predicts e0 in summary space
- EditResponseHead: Predicts du in summary space
- GainCalibrationHead: Calibration adjustment for gain predictions

The EMA target encoder and latent-space JEPA loss are available as an OPTIONAL
branch, gated by use_jepa_latent. The default path regresses e0 and du directly
in the (already-computed) summary space.

Controller input in the default path is "frozen-backbone step-0 pooled features".
The head receives these as plain tensor input; backbone pooling belongs to the
bridge module.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .metrics_contract import (
    quadratic_gain as _canonical_quadratic_gain,
    quadratic_gain_from_benefit_gram,
    WeightConvention,
)


class BoundedProgramHead(nn.Module):
    """[B, features] -> bounded [B, d] coefficients.

    Encodes a whole precommitted program once at issue time, NOT a closed-loop
    controller. Pair with the original TemporalPatchState and a fixed pooling
    rule. Zero output initially is intentional.
    """
    def __init__(self, features: int, dimension: int, width: int = 64, bound: float = 0.25):
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in (features, dimension, width)) or not math.isfinite(bound) or bound <= 0:
            raise ValueError('invalid dimensions or bound')
        self.features = features
        self.dimension = dimension
        self.bound = float(bound)
        self.net = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, dimension))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        """Predict bounded coefficients from pooled features.

        Args:
            features: [B, features] frozen-backbone step-0 pooled features

        Returns:
            [B, d] coefficients in [-bound, bound]
        """
        if features.ndim != 2 or features.shape[-1] != self.features:
            raise ValueError('invalid features')
        return self.bound * torch.tanh(self.net(features))


class InteractionUtilityHead(nn.Module):
    """Predict b:[B, d], H:[B, d, d] for gain(a) = 2b.a - a.T H a.

    H is PSD via L L.T + eps I. Off-diagonals describe response overlap,
    NOT exact nonlinear mixed derivatives or atmosphere causality. d=8 uses
    8+36=44 outputs. Inputs must not contain future residual/teacher labels.
    """
    def __init__(self, features: int, dimension: int, width: int = 64, eps: float = 1e-5):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (features, dimension, width)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError('invalid dimensions/eps')
        self.features = features
        self.dimension = dimension
        self.eps = float(eps)
        rows, cols = torch.tril_indices(dimension, dimension)
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)
        self.net = nn.Sequential(nn.Linear(features, width), nn.GELU(),
                                 nn.Linear(width, dimension + len(rows)))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Predict benefit and Gram matrix.

        Args:
            features: [B, features] frozen-backbone step-0 pooled features

        Returns:
            (benefit, gram): [B, d] and [B, d, d] where gram is PSD
        """
        if features.ndim != 2 or features.shape[-1] != self.features:
            raise ValueError('features must be [batch,features]')
        out = self.net(features)
        d = self.dimension
        b = out[:, :d]
        raw = out[:, d:]
        values = torch.where((self.rows == self.cols)[None, :], F.softplus(raw), raw)
        lower = out.new_zeros((out.shape[0], d, d))
        lower[:, self.rows, self.cols] = values
        identity = torch.eye(d, device=out.device, dtype=out.dtype)
        return b, lower @ lower.transpose(-1, -2) + self.eps * identity


class ReferenceErrorHead(nn.Module):
    """Predict reference error e0 in summary space.

    Takes pooled features from frozen backbone and predicts the error
    between truth and reference forecast in the summary space.
    """
    def __init__(self, feature_dim: int, target_dim: int, latent_dim: int = 32):
        super().__init__()
        if min(feature_dim, target_dim, latent_dim) <= 0:
            raise ValueError('All dimensions must be positive.')
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.decoder = nn.Linear(latent_dim, target_dim)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """Predict reference error.

        Args:
            context: [B, ...] pooled context features

        Returns:
            (latent, error): latent representation and decoded error prediction
        """
        z = self.predictor(context)
        e = self.decoder(z)
        return z, e


class EditResponseHead(nn.Module):
    """Predict edit response du in summary space.

    Takes pooled features and edit descriptors, predicts the response
    (change in forecast) due to each edit.
    """
    def __init__(self, feature_dim: int, edit_dim: int, target_dim: int, latent_dim: int = 32):
        super().__init__()
        if min(feature_dim, edit_dim, target_dim, latent_dim) <= 0:
            raise ValueError('All dimensions must be positive.')
        self.feature_dim = feature_dim
        self.edit_dim = edit_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.edit_encoder = nn.Sequential(
            nn.Linear(edit_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim + latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.decoder = nn.Linear(latent_dim, target_dim)

    def forward(self, context: Tensor, edit_descriptors: Tensor, enabled: Tensor) -> tuple[Tensor, Tensor]:
        """Predict edit responses.

        Args:
            context: [B, feature_dim] pooled context features
            edit_descriptors: [K, edit_dim] edit descriptors
            enabled: [K] boolean mask of enabled edits

        Returns:
            (latent, response): [B, K, latent_dim] and [B, K, target_dim]
        """
        b = context.shape[0]
        k = edit_descriptors.shape[0]
        edit_z = self.edit_encoder(edit_descriptors)  # [K, latent]
        # Combine context with each edit
        context_expanded = context[:, None, :].expand(b, k, -1)  # [B, K, feature_dim]
        edit_expanded = edit_z[None, :, :].expand(b, k, -1)  # [B, K, latent]
        combined = torch.cat([context_expanded, edit_expanded], dim=-1)  # [B, K, feature_dim + latent]
        z = self.predictor(combined)  # [B, K, latent]
        response = self.decoder(z) * enabled.to(context)[None, :, None]  # [B, K, target]
        return z, response


class GainCalibrationHead(nn.Module):
    """Calibration head for gain predictions.

    Takes latent representations of reference error and edit response,
    outputs a calibration adjustment to the geometric gain.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError('latent_dim must be positive.')
        self.calibration = nn.Linear(2 * latent_dim, 1)
        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, reference_z: Tensor, response_z: Tensor) -> Tensor:
        """Compute calibration adjustment.

        Args:
            reference_z: [B, K, latent] reference latents expanded to K edits
            response_z: [B, K, latent] response latents

        Returns:
            [B, K] calibration adjustments
        """
        combined = torch.cat([reference_z, response_z], dim=-1)
        return self.calibration(combined).squeeze(-1)


class JEPAEncoder(nn.Module):
    """Encoder for JEPA-style latent space training.

    This is an OPTIONAL component for ablation studies, not the default path.
    The default path regresses e0 and du directly in summary space.
    """
    def __init__(self, target_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(target_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class JEPATargetEncoder(nn.Module):
    """EMA target encoder for JEPA training.

    This is an OPTIONAL component for ablation studies.
    """
    def __init__(self, encoder: JEPAEncoder):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.encoder.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: JEPAEncoder, momentum: float = 0.99) -> None:
        """Update target encoder with EMA of online encoder.

        Args:
            online_encoder: The online encoder to copy from
            momentum: EMA momentum (higher = slower update)
        """
        if not 0 <= momentum <= 1:
            raise ValueError('EMA momentum must be in [0,1].')
        for p, q in zip(online_encoder.parameters(), self.parameters()):
            q.lerp_(p, 1 - momentum)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class ComposedPredictionHead(nn.Module):
    """Composed prediction head combining all sub-heads.

    This replaces the monolithic PairedEditPredictor with composable components.
    The JEPA latent-space path is optional and gated by use_jepa_latent.

    Default path: regress e0 and du directly in summary space.
    JEPA path (ablation): use EMA target encoders and latent-space JEPA loss.
    """
    def __init__(self, feature_dim: int, memory_dim: int, edit_dim: int,
                 target_dim: int, latent_dim: int = 32,
                 use_jepa_latent: bool = False,
                 use_gain_calibration: bool = False):
        super().__init__()
        if min(feature_dim, memory_dim, edit_dim, target_dim, latent_dim) <= 0:
            raise ValueError('All dimensions must be positive.')
        self.feature_dim = feature_dim
        self.memory_dim = memory_dim
        self.target_dim = target_dim
        self.use_jepa_latent = use_jepa_latent
        # Analytic gain is the protocol quantity.  Calibration is an optional
        # learned diagnostic and must never silently contaminate that quantity.
        self.use_gain_calibration = bool(use_gain_calibration)

        # Context processing
        self.history = nn.GRU(feature_dim, latent_dim, batch_first=True)
        self.memory = nn.Linear(memory_dim, latent_dim)
        self.cross_scale = nn.Linear(latent_dim, latent_dim, bias=False)
        self.time = nn.Sequential(nn.Linear(1, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))
        self.edit = nn.Sequential(nn.Linear(edit_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))

        # Prediction heads
        self.base_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))
        self.response_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))
        self.base_decoder = nn.Linear(latent_dim, target_dim)
        self.response_decoder = nn.Linear(latent_dim, target_dim)
        self.gain_calibration = nn.Linear(2 * latent_dim, 1)
        nn.init.zeros_(self.gain_calibration.weight)
        nn.init.zeros_(self.gain_calibration.bias)

        # JEPA components (optional)
        if use_jepa_latent:
            self.base_encoder = nn.Sequential(nn.Linear(target_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))
            self.response_encoder = copy.deepcopy(self.base_encoder)
            self.base_target = copy.deepcopy(self.base_encoder).requires_grad_(False)
            self.response_target = copy.deepcopy(self.response_encoder).requires_grad_(False)

    def forward(self, history: Tensor, memory: Tensor, edit_descriptors: Tensor,
                leads_hours: Tensor, enabled: Tensor,
                weights: Tensor | None = None) -> dict[str, Tensor]:
        """Forward pass for prediction.

        Args:
            history: [B, L, S, F] history of pooled features
            memory: [B, S, M] memory features
            edit_descriptors: [K, A] edit descriptors
            leads_hours: [H] lead time hours
            enabled: [K] boolean mask of enabled edits
            weights: Optional [target_dim] Q-weights for gain computation.
                     Convention: WEIGHTED_MEAN - weights are normalized to sum to 1.
                     This matches the training target convention in paired.py.
                     For uniform weights (None), output is bit-identical to the
                     previous unweighted .mean(-1) implementation.

        Returns:
            Dict with base_z, response_z, reference_error, edit_response, gain
        """
        if history.ndim != 4 or history.shape[-1] != self.feature_dim:
            raise ValueError('history must be [B,L,S,feature_dim].')
        b, l, s, _ = history.shape
        if l < 1 or memory.shape != (b, s, self.memory_dim) or edit_descriptors.ndim != 2:
            raise ValueError('Memory/edit/history shape mismatch.')
        k = edit_descriptors.shape[0]
        if enabled.dtype != torch.bool or enabled.shape != (k,) or leads_hours.ndim != 1 or (leads_hours <= 0).any():
            raise ValueError('Invalid enabled mask or positive horizon vector.')

        _, hidden = self.history(history.permute(0, 2, 1, 3).reshape(b * s, l, self.feature_dim))
        context = hidden[-1].reshape(b, s, -1) + self.memory(memory)
        context = context + self.cross_scale(context.mean(1, keepdim=True))

        t = self.time(leads_hours.to(history)[:, None] / 24.0)
        z0 = self.base_predictor(context[:, None] + t[None, :, None])  # [B, H, S, D]
        ze = self.response_predictor(
            context[:, None, None] + t[None, None, :, None] +
            self.edit(edit_descriptors.to(history))[None, :, None, None]
        )

        error = self.base_decoder(z0)
        response = self.response_decoder(ze) * enabled.to(history)[None, :, None, None, None]

        # Compute geometric gain using canonical implementation with WEIGHTED_MEAN convention.
        # For uniform weights (weights=None), this is mathematically equivalent to .mean(-1),
        # preserving bit-identical behavior for backward compatibility.
        geometric = _canonical_quadratic_gain(
            error, response, weights, convention=WeightConvention.WEIGHTED_MEAN
        )

        reference_z = z0[:, None].expand(-1, k, -1, -1, -1)
        calibration = self.gain_calibration(torch.cat((reference_z, ze), -1)).squeeze(-1)
        enabled_gain = enabled.to(history)[None, :, None, None]
        gain_analytic = geometric * enabled_gain
        gain_calibrated = (geometric + calibration) * enabled_gain
        gain = gain_calibrated if self.use_gain_calibration else gain_analytic

        return {
            'base_z': z0, 'response_z': ze, 'reference_error': error,
            'edit_response': response, 'gain': gain,
            'gain_analytic': gain_analytic,
            'gain_calibrated': gain_calibrated,
            'gain_calibration': calibration * enabled_gain,
        }

    @torch.no_grad()
    def update_targets(self, momentum: float = 0.99) -> None:
        """Update EMA target encoders (JEPA path only).

        Args:
            momentum: EMA momentum in [0, 1]
        """
        if not self.use_jepa_latent:
            return
        if not 0 <= momentum <= 1:
            raise ValueError('EMA momentum must be in [0,1].')
        for online, target in [(self.base_encoder, self.base_target), (self.response_encoder, self.response_target)]:
            for p, q in zip(online.parameters(), target.parameters()):
                q.lerp_(p, 1 - momentum)

    def training_loss(self, predictions: dict[str, Tensor], targets,
                      enabled: Tensor) -> dict[str, Tensor]:
        """Compute training losses.

        Args:
            predictions: Output from forward()
            targets: PairedTargets from paired.make_training_targets
            enabled: [K] boolean mask of enabled edits

        Returns:
            Dict of losses: total, jepa (if enabled), anchors, gain, variance
        """
        losses = {}

        if self.use_jepa_latent:
            ez = self.base_encoder(targets.reference_error)
            dz = self.response_encoder(targets.edit_response)
            with torch.no_grad():
                et = self.base_target(targets.reference_error)
                dt = self.response_target(targets.edit_response)
            jepa = F.mse_loss(predictions['base_z'], et)
            if enabled.any():
                jepa = jepa + F.mse_loss(predictions['response_z'][:, enabled], dt[:, enabled])
            losses['jepa'] = jepa
        else:
            jepa = torch.tensor(0., device=predictions['base_z'].device)
            losses['jepa'] = jepa

        anchors = (F.mse_loss(predictions['reference_error'], targets.reference_error) +
                   F.mse_loss(predictions['edit_response'], targets.edit_response))
        if self.use_jepa_latent:
            ez = self.base_encoder(targets.reference_error)
            dz = self.response_encoder(targets.edit_response)
            anchors = anchors + F.mse_loss(self.base_decoder(ez), targets.reference_error)
            anchors = anchors + F.mse_loss(self.response_decoder(dz), targets.edit_response)
        losses['anchors'] = anchors

        gain = F.smooth_l1_loss(predictions['gain'], targets.quadratic_gain)
        losses['gain'] = gain

        # Variance regularization
        z_samples = predictions['base_z'].reshape(-1, predictions['base_z'].shape[-1])
        variance = F.relu(1.0 - torch.sqrt(z_samples.var(0, unbiased=False) + 1e-4)).mean()
        if enabled.any():
            r_samples = predictions['response_z'][:, enabled].reshape(-1, predictions['response_z'].shape[-1])
            variance = variance + F.relu(1.0 - torch.sqrt(r_samples.var(0, unbiased=False) + 1e-4)).mean()
        losses['variance'] = variance

        total = losses['jepa'] + anchors + gain + 0.01 * variance
        losses['total'] = total

        return losses


# Legacy alias for compatibility
PairedEditPredictor = ComposedPredictionHead


def quadratic_gain(benefit: Tensor, gram: Tensor, program: Tensor) -> Tensor:
    """Batched differentiable surrogate gain; program can be [B, d] or [d].

    This computes gain in coefficient space: 2*b.a - a.T H a
    where benefit (b = R.T Q e) and gram (H = R.T Q R) already have spatial
    weights baked in. No additional weighting convention is applied here.

    Convention note: This is a coefficient-space operation, distinct from the
    physical-space quadratic_gain in metrics_contract.py. The spatial weighting
    was already applied when constructing benefit and gram matrices.
    """
    return quadratic_gain_from_benefit_gram(benefit, gram, program)
