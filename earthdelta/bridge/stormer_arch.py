"""Faithful re-implementation of Stormer architecture with a checked backend.

Adapted from tung-nd/stormer (commit 58dfee5a6037399a40fefd492bc00421e0c885a8, MIT license).
The default CPU-safe backend is PyTorch SDPA. When the official xformers CUDA
operator is available, the same tensor layout is used so the S0 bridge path can
be compared at the official numerical precision rather than accepting an
unbounded accumulated SDPA difference.

xformers expects q/k/v shaped (B, N, num_heads, head_dim) and returns same.
F.scaled_dot_product_attention expects (B, num_heads, N, head_dim) and returns same.
Both apply 1/sqrt(head_dim) scaling by default.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_, Mlp, PatchEmbed


# =============================================================================
# Positional embedding utilities (adapted from stormer/utils/pos_embed.py)
# =============================================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size_h: int, grid_size_w: int,
                            cls_token: bool = False) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension
        grid_size_h: Grid height
        grid_size_w: Grid width
        cls_token: Whether to include CLS token position

    Returns:
        Positional embeddings of shape [grid_size_h*grid_size_w, embed_dim]
        or [1+grid_size_h*grid_size_w, embed_dim] with cls_token
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size_h, grid_size_w])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate 2D sincos embeddings from grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sincos embeddings from positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


# =============================================================================
# Modulation helper
# =============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# =============================================================================
# Timestep Embedder
# =============================================================================

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Linear(1, hidden_size)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t.unsqueeze(-1))


# =============================================================================
# Memory Efficient Attention with SDPA
# =============================================================================

class MemEffAttention(nn.Module):
    """Memory-efficient attention using xformers when available, else SDPA.

    Structurally equivalent to the xformers version in the original Stormer.
    The only difference is the attention implementation:
    - xformers: q/k/v shaped (B, N, num_heads, head_dim)
    - SDPA: q/k/v shaped (B, num_heads, N, head_dim)

    Both apply 1/sqrt(head_dim) scaling by default, so outputs are equivalent.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # scale is computed internally by SDPA, but we store it for reference
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, C)
            attn_bias: Optional attention bias (not used in Stormer, kept for compatibility)

        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        # xformers is the official Stormer operator and expects [B,N,H,D].
        # Import lazily so the bridge remains usable on CPU-only development
        # machines; the S0 gate records which backend actually ran.
        use_xformers = x.device.type == "cuda"
        if use_xformers:
            try:
                from xformers.ops import memory_efficient_attention, unbind
                q, k, v = unbind(qkv, 2)
                x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
                x = x.reshape(B, N, C)
            except (ImportError, ModuleNotFoundError):
                use_xformers = False
        if not use_xformers:
            # Reshape from (B,N,3,H,D) to (B,H,N,D) for SDPA. Dropout is
            # disabled in eval mode, matching the official module.
            qkv_sdpa = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv_sdpa[0], qkv_sdpa[1], qkv_sdpa[2]
            dropout_p = self.attn_drop.p if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# Weather Embedding
# =============================================================================

class WeatherEmbedding(nn.Module):
    """Variable-aware weather embedding with per-variable patch embeddings."""

    def __init__(
        self,
        variables: List[str],
        img_size: Tuple[int, int],
        patch_size: int = 2,
        embed_dim: int = 1024,
        num_heads: int = 16,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables

        # Separate embedding layer for each variable
        self.token_embeds = nn.ModuleList([
            PatchEmbed(None, patch_size, 1, embed_dim) for _ in range(len(variables))
        ])
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # Variable embedding
        self.channel_embed, self.channel_map = self._create_var_embedding(embed_dim)

        # Variable aggregation via cross attention
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )

        self._initialize_weights()

    def _create_var_embedding(self, dim: int) -> Tuple[nn.Parameter, dict]:
        var_embed = nn.Parameter(
            torch.zeros(1, len(self.variables), dim), requires_grad=True
        )
        var_map = {var: idx for idx, var in enumerate(self.variables)}
        return var_embed, var_map

    def _initialize_weights(self):
        # Initialize positional embedding with sincos
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize variable embedding with 1D sincos
        channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1], np.arange(len(self.variables))
        )
        self.channel_embed.data.copy_(
            torch.from_numpy(channel_embed).float().unsqueeze(0)
        )

        # Initialize token embeddings
        for token_embed in self.token_embeds:
            w = token_embed.proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # Initialize linear and layernorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars: tuple, device: torch.device) -> torch.Tensor:
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb: torch.Tensor, vars: tuple) -> torch.Tensor:
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def aggregate_variables(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate variables using cross attention.

        Args:
            x: Input of shape (B, V, L, D)

        Returns:
            Aggregated output of shape (B, L, D)
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.channel_agg(var_query, x, x)  # BxL, 1, D
        x = x.squeeze(1)  # BxL, D

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward(self, x: torch.Tensor, variables: List[str]) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (B, V, H, W)
            variables: List of variable names

        Returns:
            Embedded output of shape (B, L, D)
        """
        if isinstance(variables, list):
            variables = tuple(variables)

        # Tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i, vid in enumerate(var_ids):
            embed_variable = self.token_embeds[vid](x[:, i : i + 1])  # B, L, D
            embeds.append(embed_variable)
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # Add variable and positional embeddings
        var_embed = self.get_var_emb(self.channel_embed, variables)
        x = x + var_embed.unsqueeze(2)
        x = x + self.pos_embed.unsqueeze(1)

        # Variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        return x


# =============================================================================
# Transformer Block with AdaLN-Zero
# =============================================================================

class Block(nn.Module):
    """Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (B, N, D)
            c: Conditioning of shape (B, D)

        Returns:
            Output of shape (B, N, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


# =============================================================================
# Final Layer
# =============================================================================

class FinalLayer(nn.Module):
    """Final layer with adaLN modulation."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        # Original uses Identity here (commented out LayerNorm)
        self.norm_final = nn.Identity()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# =============================================================================
# Stormer Model
# =============================================================================

class Stormer(nn.Module):
    """Stormer weather forecasting model.

    Adapted from tung-nd/stormer with SDPA attention.
    """

    def __init__(
        self,
        in_img_size: Tuple[int, int],
        variables: List[str],
        patch_size: int = 2,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        # Handle padding for image size
        if in_img_size[0] % patch_size != 0:
            pad_size = patch_size - in_img_size[0] % patch_size
            in_img_size = (in_img_size[0] + pad_size, in_img_size[1])
        self.in_img_size = in_img_size
        self.variables = variables
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Embedding
        self.embedding = WeatherEmbedding(
            variables=variables,
            img_size=in_img_size,
            patch_size=patch_size,
            embed_dim=hidden_size,
            num_heads=num_heads,
        )
        self.embed_norm_layer = nn.LayerNorm(hidden_size)

        # Interval embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Backbone
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # Prediction layer
        self.head = FinalLayer(hidden_size, patch_size, len(variables))

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        trunc_normal_(self.t_embedder.mlp.weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, h: Optional[int] = None,
                   w: Optional[int] = None) -> torch.Tensor:
        """Convert patches back to image.

        Args:
            x: Patches of shape (B, L, V * patch_size**2)
            h: Optional output height
            w: Optional output width

        Returns:
            Image of shape (B, V, H, W)
        """
        p = self.patch_size
        v = len(self.variables)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def forward(self, x: torch.Tensor, variables: List[str],
                time_interval: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (B, V, H, W)
            variables: List of variable names
            time_interval: Time interval tensor of shape (B,)

        Returns:
            Output of shape (B, V, H, W)
        """
        x = self.embedding(x, variables)  # B, L, D
        x = self.embed_norm_layer(x)

        time_interval_emb = self.t_embedder(time_interval)
        for block in self.blocks:
            x = block(x, time_interval_emb)

        x = self.head(x, time_interval_emb)
        x = self.unpatchify(x)

        return x


# =============================================================================
# Reference Attention for Testing (explicit softmax implementation)
# =============================================================================

class ExplicitAttention(nn.Module):
    """Explicit softmax attention for testing SDPA equivalence."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Explicit attention: softmax(QK^T / sqrt(d)) V
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
