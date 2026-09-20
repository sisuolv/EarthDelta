"""Frozen-backbone bridge to the Stormer weather model.

This package provides the integration between EarthDelta's low-rank adapters
(ExpertLoRA) and the Stormer weather forecasting model, enabling controlled
rollouts with parameter edits injected at attention output projections.

Main components:
- Stormer: SDPA-based reimplementation of the Stormer architecture
- load_stormer_checkpoint: Load official checkpoints with strict=True
- NormalizationContract: Input and diff normalization matching official inference
- WeatherStepBridge: Wrap model + normalization for autoregressive rollout
- controlled_rollout: Rollout with ExpertLoRA injection at blocks 18-23

The architecture adaptation replaces xformers.ops.memory_efficient_attention with
torch.nn.functional.scaled_dot_product_attention (equivalent scaling and semantics).
"""

from .stormer_arch import (
    Stormer,
    MemEffAttention,
    ExplicitAttention,
    Block,
    FinalLayer,
    WeatherEmbedding,
    TimestepEmbedder,
    modulate,
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
)

from .stormer_bridge import (
    DEFAULT_VARIABLES,
    CONSTANTS,
    NormalizationContract,
    WeatherStepBridge,
    load_stormer_checkpoint,
    controlled_rollout,
    check_version_match,
)

__all__ = [
    # Architecture
    "Stormer",
    "MemEffAttention",
    "ExplicitAttention",
    "Block",
    "FinalLayer",
    "WeatherEmbedding",
    "TimestepEmbedder",
    "modulate",
    "get_2d_sincos_pos_embed",
    "get_1d_sincos_pos_embed_from_grid",
    # Bridge
    "DEFAULT_VARIABLES",
    "CONSTANTS",
    "NormalizationContract",
    "WeatherStepBridge",
    "load_stormer_checkpoint",
    "controlled_rollout",
    "check_version_match",
]
