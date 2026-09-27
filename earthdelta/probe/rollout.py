"""Torch rollouts and the signed six-variable/four-lead objective."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from earthdelta.bridge import (
    DEFAULT_VARIABLES,
    POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    NormalizationContract,
    WeatherStepBridge,
    load_stormer_checkpoint_detailed,
)

from .edits import DirectWeightEditor


TARGET_BLOCKS = (18, 19, 20, 21, 22, 23)


@dataclass
class LoadedBridge:
    bridge: WeatherStepBridge
    model: torch.nn.Module
    checkpoint_sha256: str


def load_bridge(checkpoint: str | Path, norm_dir: str | Path, device: torch.device) -> LoadedBridge:
    loaded = load_stormer_checkpoint_detailed(str(checkpoint), patch_size=4, variables=list(DEFAULT_VARIABLES),
                                              in_img_size=(128, 256), hidden_size=1024, depth=24,
                                              num_heads=16, mlp_ratio=4.0)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"checkpoint identity failed: missing={loaded.missing_keys} unexpected={loaded.unexpected_keys}")
    norm = NormalizationContract.from_npz_dir(
        str(norm_dir), variables=list(DEFAULT_VARIABLES), intervals=(6,),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )
    version = loaded.version
    bridge = WeatherStepBridge(loaded.model.to(device), norm, version)
    bridge.model.eval()
    return LoadedBridge(bridge=bridge, model=bridge.model, checkpoint_sha256=loaded.checkpoint_sha256)


def rollout_trajectory(bridge: WeatherStepBridge, x_norm: torch.Tensor, *, steps: int,
                       deltas: Mapping[int, torch.Tensor] | None = None,
                       target_blocks: Sequence[int] = TARGET_BLOCKS,
                       differentiable: bool = False) -> torch.Tensor:
    """Return normalized states [B, steps+1, 69, 128, 256]."""
    if x_norm.ndim != 4 or x_norm.shape[1:] != (69, 128, 256):
        raise ValueError(f"unexpected normalized input shape: {tuple(x_norm.shape)}")
    interval = 6
    interval_tensor = torch.full((x_norm.shape[0],), interval / 10.0,
                                 device=x_norm.device, dtype=x_norm.dtype)
    context = DirectWeightEditor(bridge.model, deltas or {}, blocks=target_blocks)
    grad_context = context if deltas else torch.no_grad()
    with grad_context:
        states = [x_norm]
        x = x_norm
        for _ in range(steps):
            padded, pad_size = bridge.pad(x)
            output = bridge.model(padded, list(DEFAULT_VARIABLES), interval_tensor)
            diff = output[:, :, pad_size:]
            diff = bridge.normalization.replace_constant(diff, list(DEFAULT_VARIABLES))
            diff = bridge.normalization.denormalize_diff(diff, interval)
            raw = bridge.normalization.denormalize(x) + diff
            x = bridge.normalization.normalize(raw)
            states.append(x)
        return torch.stack(states, dim=1)


def area_weighted_mse(pred_raw: torch.Tensor, truth_raw: torch.Tensor, latitude: np.ndarray,
                      channel_index: int) -> torch.Tensor:
    if pred_raw.shape != truth_raw.shape or pred_raw.ndim != 5:
        raise ValueError("expected [B,T,C,H,W] tensors")
    lat = torch.as_tensor(latitude, dtype=pred_raw.dtype, device=pred_raw.device)
    weights = torch.cos(torch.deg2rad(lat))[:, None]
    weights = weights / weights.sum()
    error = pred_raw[:, :, channel_index] - truth_raw[:, :, channel_index]
    return (error.square() * weights).sum(dim=(-2, -1))


def l6_per_issue(pred_norm: torch.Tensor, truth_raw: torch.Tensor, bridge: WeatherStepBridge,
                 latitude: np.ndarray, variables: Sequence[Mapping[str, object]],
                 leads: Sequence[int], denominator: Mapping[str, float]) -> torch.Tensor:
    pred_raw = bridge.denormalize(pred_norm.reshape(-1, 69, 128, 256)).reshape_as(pred_norm)
    terms = []
    for var in variables:
        name = str(var["name"]); channel = int(var["channel_index"])
        for j, lead in enumerate(leads):
            key = f"{name}@{lead}h"
            mse = area_weighted_mse(pred_raw[:, [int(lead // 6)], :, :, :],
                                    truth_raw[:, [int(lead // 6)], :, :, :], latitude, channel).mean()
            denom = float(denominator[key]) ** 2
            if denom <= 0 or not np.isfinite(denom):
                raise ValueError(f"invalid L6 denominator: {key}")
            terms.append(mse / denom)
    return torch.stack(terms).mean()


def l6_cells(pred_norm: torch.Tensor, truth_raw: torch.Tensor, bridge: WeatherStepBridge,
              latitude: np.ndarray, variables: Sequence[Mapping[str, object]],
              leads: Sequence[int], denominator: Mapping[str, float]) -> np.ndarray:
    pred_raw = bridge.denormalize(pred_norm.reshape(-1, 69, 128, 256)).reshape_as(pred_norm)
    out = np.empty((len(variables), len(leads)), dtype=np.float64)
    for vi, var in enumerate(variables):
        channel = int(var["channel_index"]); name = str(var["name"])
        for li, lead in enumerate(leads):
            key = f"{name}@{lead}h"; step = int(lead // 6)
            mse = area_weighted_mse(pred_raw[:, [step]], truth_raw[:, [step]], latitude, channel)
            out[vi, li] = float(mse.mean().detach().cpu()) / float(denominator[key] ** 2)
    return out


def raw_truth_from_store(array: np.ndarray, *, endpoint_steps: Sequence[int]) -> torch.Tensor:
    selected = np.asarray(array)[list(endpoint_steps)]
    return torch.from_numpy(np.ascontiguousarray(selected)).unsqueeze(0).float()
