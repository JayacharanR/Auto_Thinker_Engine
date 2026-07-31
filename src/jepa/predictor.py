"""
Lightweight Transformer Predictor for JEPA.

Takes context encoder representations (from unmasked patches) and
positional embeddings of masked regions, then predicts the target
encoder's output at those masked positions.

Optionally action-conditioned (V-JEPA-AC style): injects steering
and speed telemetry from comma2k19 to enable "predict what happens
given this action" rather than just "predict occluded patches."
"""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class JEPAPredictor(nn.Module):
    """
    Lightweight transformer predictor for JEPA masked prediction.

    The predictor takes:
    1. Context representations from the context encoder (unmasked patches)
    2. Positional embeddings for masked patch positions (learnable mask tokens)
    3. Optionally, action embeddings (steering + speed at each timestep)

    And predicts the target encoder's representations at the masked positions.

    Design: intentionally smaller than the encoder (half embed_dim, fewer layers)
    to prevent the predictor from becoming too powerful, which would reduce
    the encoder's need to learn good representations.

    Args:
        context_dim: Dimension of context encoder output.
        embed_dim: Predictor's internal embedding dimension (typically context_dim/2).
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        num_patches: Total number of patches (for positional embedding).
        action_conditioning: Whether to inject action information.
        action_dim: Dimension of action input (e.g., 2 for [steering, speed]).
        action_embed_dim: Dimension to project actions to before injection.
    """

    def __init__(
        self,
        context_dim: int = 384,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_patches: int = 784,
        action_conditioning: bool = True,
        action_dim: int = 2,
        action_embed_dim: int = 64,
    ):
        super().__init__()

        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.action_conditioning = action_conditioning

        # Project context encoder output to predictor dimension
        self.context_proj = nn.Linear(context_dim, embed_dim)

        # Learnable mask tokens — one per masked position
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embeddings for ALL patches (context + masked)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Action conditioning (V-JEPA-AC style)
        if action_conditioning:
            self.action_proj = nn.Sequential(
                nn.Linear(action_dim, action_embed_dim),
                nn.GELU(),
                nn.Linear(action_embed_dim, embed_dim),
            )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                PredictorBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Project back to context_dim (match target encoder output dim)
        self.output_proj = nn.Linear(embed_dim, context_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        mask_indices: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict target representations at masked positions.

        Args:
            context_tokens: (B, N_ctx, D_ctx) context encoder output for unmasked patches.
            context_indices: (B, N_ctx) indices of unmasked (context) patches.
            mask_indices: (B, N_mask) indices of masked patches to predict.
            actions: Optional (B, T, A) action sequence (steering, speed per timestep).
                     Only used if action_conditioning=True.

        Returns:
            (B, N_mask, D_ctx) predicted target representations at masked positions.
        """
        batch_size = context_tokens.shape[0]
        n_ctx = context_tokens.shape[1]
        n_mask = mask_indices.shape[1]

        # Project context tokens to predictor dimension
        ctx = self.context_proj(context_tokens)  # (B, N_ctx, D_pred)

        # Add positional embeddings to context tokens
        ctx_pos = torch.gather(
            self.pos_embed.expand(batch_size, -1, -1),
            dim=1,
            index=context_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )
        ctx = ctx + ctx_pos

        # Create mask tokens with positional embeddings
        mask_tokens = self.mask_token.expand(batch_size, n_mask, -1)
        mask_pos = torch.gather(
            self.pos_embed.expand(batch_size, -1, -1),
            dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )
        mask_tokens = mask_tokens + mask_pos

        # Optionally inject action conditioning
        if self.action_conditioning and actions is not None:
            action_embed = self.action_proj(actions)  # (B, T, D_pred)
            # Pool action embeddings over time and add to mask tokens
            # This injects "what action was taken" into the prediction
            action_pooled = action_embed.mean(dim=1, keepdim=True)  # (B, 1, D_pred)
            mask_tokens = mask_tokens + action_pooled

        # Concatenate context and mask tokens
        # Context tokens come first, mask tokens second
        x = torch.cat([ctx, mask_tokens], dim=1)  # (B, N_ctx + N_mask, D_pred)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract only the mask token predictions
        predictions = x[:, n_ctx:, :]  # (B, N_mask, D_pred)

        # Project back to context encoder dimension
        predictions = self.output_proj(predictions)  # (B, N_mask, D_ctx)

        return predictions


class PredictorBlock(nn.Module):
    """Transformer block for the predictor (pre-norm architecture)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        residual = x
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + x_attn

        # Pre-norm MLP
        residual = x
        x = residual + self.mlp(self.norm2(x))
        return x
