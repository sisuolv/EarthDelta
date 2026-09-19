"""Small paired-target predictive prototype on PRECOMPUTED physical features.

Not a global weather encoder and not an upstream LeWM reproduction. It learns
reference-error and edit-response embeddings with EMA targets, physical anchors,
and a small gain calibration head. No future truth is accepted by forward().
The first prototype predicts all requested horizons directly; recurrent latent
rollout and SG-JEPA-style experiments remain separate planned extensions.
"""
from __future__ import annotations
import copy
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .paired import PairedTargets

class PairedEditPredictor(nn.Module):
    def __init__(self, feature_dim: int, memory_dim: int, edit_dim: int,
                 target_dim: int, latent_dim: int = 32):
        super().__init__()
        if min(feature_dim,memory_dim,edit_dim,target_dim,latent_dim) <= 0:
            raise ValueError('All dimensions must be positive.')
        self.feature_dim,self.memory_dim,self.target_dim = feature_dim,memory_dim,target_dim
        self.history = nn.GRU(feature_dim,latent_dim,batch_first=True)
        self.memory = nn.Linear(memory_dim,latent_dim)
        self.cross_scale = nn.Linear(latent_dim,latent_dim,bias=False)
        self.edit = nn.Sequential(nn.Linear(edit_dim,latent_dim),nn.SiLU(),nn.Linear(latent_dim,latent_dim))
        self.time = nn.Sequential(nn.Linear(1,latent_dim),nn.SiLU(),nn.Linear(latent_dim,latent_dim))
        self.base_predictor = nn.Sequential(nn.Linear(latent_dim,latent_dim),nn.SiLU(),nn.Linear(latent_dim,latent_dim))
        self.response_predictor = nn.Sequential(nn.Linear(latent_dim,latent_dim),nn.SiLU(),nn.Linear(latent_dim,latent_dim))
        self.base_encoder = nn.Sequential(nn.Linear(target_dim,latent_dim),nn.SiLU(),nn.Linear(latent_dim,latent_dim))
        self.response_encoder = copy.deepcopy(self.base_encoder)
        self.base_target = copy.deepcopy(self.base_encoder).requires_grad_(False)
        self.response_target = copy.deepcopy(self.response_encoder).requires_grad_(False)
        self.base_decoder = nn.Linear(latent_dim,target_dim)
        self.response_decoder = nn.Linear(latent_dim,target_dim)
        self.gain_calibration = nn.Linear(2*latent_dim,1)
        nn.init.zeros_(self.gain_calibration.weight);nn.init.zeros_(self.gain_calibration.bias)

    def forward(self, history: Tensor, memory: Tensor, edit_descriptors: Tensor,
                leads_hours: Tensor, enabled: Tensor) -> dict[str,Tensor]:
        # history [B,L,S,F], memory [B,S,M], edits [K,A], leads [H], enabled [K]
        if history.ndim != 4 or history.shape[-1] != self.feature_dim:
            raise ValueError('history must be [B,L,S,feature_dim].')
        b,l,s,_ = history.shape
        if l < 1 or memory.shape != (b,s,self.memory_dim) or edit_descriptors.ndim != 2:
            raise ValueError('Memory/edit/history shape mismatch.')
        k = edit_descriptors.shape[0]
        if enabled.dtype != torch.bool or enabled.shape != (k,) or leads_hours.ndim != 1 or (leads_hours<=0).any():
            raise ValueError('Invalid enabled mask or positive horizon vector.')
        _,hidden = self.history(history.permute(0,2,1,3).reshape(b*s,l,self.feature_dim))
        context = hidden[-1].reshape(b,s,-1)+self.memory(memory)
        context = context+self.cross_scale(context.mean(1,keepdim=True))
        t = self.time(leads_hours.to(history)[:,None]/24.0)
        z0 = self.base_predictor(context[:,None]+t[None,:,None]) # [B,H,S,D]
        ze = self.response_predictor(context[:,None,None]+t[None,None,:,None]+self.edit(edit_descriptors.to(history))[None,:,None,None])
        error = self.base_decoder(z0)
        response = self.response_decoder(ze)*enabled.to(history)[None,:,None,None,None]
        geometric = (2*error[:,None]*response-response.square()).mean(-1)
        reference_z = z0[:,None].expand(-1,k,-1,-1,-1)
        calibration = self.gain_calibration(torch.cat((reference_z,ze),-1)).squeeze(-1)
        gain = (geometric+calibration)*enabled.to(history)[None,:,None,None]
        return {'base_z':z0,'response_z':ze,'reference_error':error,'edit_response':response,'gain':gain}

    @torch.no_grad()
    def update_targets(self, momentum: float = 0.99) -> None:
        if not 0 <= momentum <= 1:
            raise ValueError('EMA momentum must be in [0,1].')
        for online,target in [(self.base_encoder,self.base_target),(self.response_encoder,self.response_target)]:
            for p,q in zip(online.parameters(),target.parameters()):
                q.lerp_(p,1-momentum)

    def training_loss(self, predictions: dict[str,Tensor], targets: PairedTargets,
                      enabled: Tensor) -> dict[str,Tensor]:
        # Weights here are pilot defaults, not validated hyperparameters.
        ez = self.base_encoder(targets.reference_error)
        dz = self.response_encoder(targets.edit_response)
        with torch.no_grad():
            et = self.base_target(targets.reference_error)
            dt = self.response_target(targets.edit_response)
        jepa = F.mse_loss(predictions['base_z'],et)
        if enabled.any():
            jepa = jepa+F.mse_loss(predictions['response_z'][:,enabled],dt[:,enabled])
        anchors = (F.mse_loss(predictions['reference_error'],targets.reference_error)
                   +F.mse_loss(predictions['edit_response'],targets.edit_response)
                   +F.mse_loss(self.base_decoder(ez),targets.reference_error)
                   +F.mse_loss(self.response_decoder(dz),targets.edit_response))
        gain = F.smooth_l1_loss(predictions['gain'],targets.quadratic_gain)
        samples = ez.reshape(-1,ez.shape[-1])
        # An elementary variance floor, NOT claimed to reproduce SIGReg.
        variance = F.relu(1.0-torch.sqrt(samples.var(0,unbiased=False)+1e-4)).mean()
        if enabled.any():
            d_samples = dz[:,enabled].reshape(-1,dz.shape[-1])
            variance = variance+F.relu(1.0-torch.sqrt(d_samples.var(0,unbiased=False)+1e-4)).mean()
        total = jepa+anchors+gain+0.01*variance
        return {'total':total,'jepa':jepa,'anchors':anchors,'gain':gain,'variance':variance}
