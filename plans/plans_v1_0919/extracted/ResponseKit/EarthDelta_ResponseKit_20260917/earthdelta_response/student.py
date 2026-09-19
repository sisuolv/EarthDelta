"""Minimal amortized coefficient student. No labels/response probing at runtime."""
import math
import torch
from torch import Tensor, nn


class BoundedProgramHead(nn.Module):
    """[B,features] -> bounded [B,d] coefficients.

    Encodes a whole precommitted program once at issue time, NOT a closed-loop
    controller. Pair with the original TemporalPatchState and a fixed pooling
    rule. The optional interaction-value head and a small surrogate planner live
    in utility.py; neither student has been trained on real weather here. Zero output initially is intentional.
    """
    def __init__(self,features: int,dimension: int,width: int=64,bound: float=.25):
        super().__init__()
        if any(type(x) is not int or x<=0 for x in (features,dimension,width)) or not math.isfinite(bound) or bound<=0:
            raise ValueError('invalid dimensions or bound')
        self.features=features; self.dimension=dimension; self.bound=float(bound)
        self.net=nn.Sequential(nn.Linear(features,width),nn.GELU(),nn.Linear(width,dimension))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,features: Tensor) -> Tensor:
        if features.ndim!=2 or features.shape[-1]!=self.features: raise ValueError('invalid features')
        return self.bound*torch.tanh(self.net(features))
