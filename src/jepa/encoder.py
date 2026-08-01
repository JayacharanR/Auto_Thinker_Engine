"""
ViT-Small Context Encoder for JEPA pretraining.

Processes unmasked video tubelets into patch embeddings. This is the
encoder whose quality is measured by the Phase 2 linear probe and
whose weights are transferred to DreamerV3 in Phase 3 (Arm 2).

Architecture: ViT-Small with 3D tubelet embedding for spatiotemporal patches.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed3D(nn.Module):
    """
    3D patch embedding for video tubelets.

    Converts a video clip (B, C, T, H, W) into a sequence of patch
    embeddings by applying a 3D convolution with kernel size
    (tubelet_size, patch_size, patch_size).

    Args:
        img_size: Spatial resolution (H=W).
        patch_size: Spatial patch size.
        tubelet_size: Temporal tubelet size (frames per patch).
        in_channels: Input channels (3 for RGB).
        embed_dim: Output embedding dimension.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_patches_spatial = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) video tensor.

        Returns:
            (B, N, D) patch embeddings where N = T/tubelet_size * (H/patch * W/patch).
        """
        x = self.proj(x)  # (B, D, T', H', W')
        x = rearrange(x, "b d t h w -> b (t h w) d")
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder for JEPA pretraining.

    Processes unmasked video tubelets and produces latent representations.
    Designed as the context encoder in the JEPA framework — the target
    encoder is an EMA copy of this (see target_encoder.py).

    Args:
        img_size: Spatial resolution of input frames.
        patch_size: Size of spatial patches.
        tubelet_size: Number of frames per temporal patch.
        in_channels: Input channels (3 for RGB).
        embed_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim ratio (mlp_dim = embed_dim * mlp_ratio).
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        num_frames: Total number of input frames.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        num_frames: int = 16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Compute number of patches
        num_patches_spatial = (img_size // patch_size) ** 2
        num_patches_temporal = num_frames // tubelet_size
        self.num_patches = num_patches_spatial * num_patches_temporal

        # Learnable positional embeddings (spatiotemporal)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following ViT conventions."""
        # Positional embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Apply to all submodules
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
        elif isinstance(m, nn.Conv3d):
            # Fan-out initialization for conv layers
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _get_pos_embed(self, num_patches_actual: int) -> torch.Tensor:
        """
        Get positional embeddings, with interpolation if temporal length differs.

        Phase 2 pretrains with 16 frames (8 temporal patches × 196 spatial = 1568).
        Phase 3 uses 4 frames (2 temporal patches × 196 spatial = 392).
        Without interpolation, this is a hard shape mismatch — same bug class
        as the spatial resolution mismatch, on the time axis.

        The fix: reshape pos_embed into a (T_train, S, D) grid, interpolate
        T_train → T_actual, then flatten back to (1, N_actual, D). This is
        the standard approach used by VideoMAE, TimeSformer, V-JEPA, etc.

        Args:
            num_patches_actual: Total number of patches from patch_embed output.

        Returns:
            (1, num_patches_actual, D) positional embeddings.
        """
        num_patches_stored = self.pos_embed.shape[1]

        # Fast path: no interpolation needed
        if num_patches_actual == num_patches_stored:
            return self.pos_embed

        # Compute spatial and temporal dimensions
        num_spatial = self.patch_embed.num_patches_spatial
        num_temporal_stored = num_patches_stored // num_spatial
        num_temporal_actual = num_patches_actual // num_spatial

        if num_temporal_actual == num_temporal_stored:
            # Spatial mismatch (shouldn't happen after resolution fix, but safe)
            return self.pos_embed[:, :num_patches_actual]

        # Reshape to (1, T, S, D) for temporal interpolation
        pos = self.pos_embed.reshape(1, num_temporal_stored, num_spatial, self.embed_dim)

        # Interpolate temporal dimension: (1, T_stored, S, D) → (1, T_actual, S, D)
        # Permute to (1, D, T_stored, S) for F.interpolate, then back
        pos = pos.permute(0, 3, 1, 2)  # (1, D, T_stored, S)
        pos = torch.nn.functional.interpolate(
            pos,
            size=(num_temporal_actual, num_spatial),
            mode="bilinear",
            align_corners=False,
        )
        pos = pos.permute(0, 2, 3, 1)  # (1, T_actual, S, D)

        # Flatten back to (1, N_actual, D)
        return pos.reshape(1, num_patches_actual, self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: (B, C, T, H, W) video tensor.
            mask_indices: Optional (B, N_keep) indices of patches to KEEP
                         (i.e., the unmasked/context patches). If None,
                         all patches are processed.

        Returns:
            (B, N_keep, D) encoder representations for unmasked patches.
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add positional embeddings (with temporal interpolation if needed)
        x = x + self._get_pos_embed(x.shape[1])

        # Apply masking: only keep context (unmasked) patches
        if mask_indices is not None:
            # mask_indices: (B, N_keep) — indices of patches to keep
            batch_size = x.shape[0]
            x = torch.gather(
                x,
                dim=1,
                index=mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            )

        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x

    def get_pos_embed_for_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get positional embeddings for specific patch indices.

        Used by the predictor to get positional info for masked patches.

        Args:
            indices: (B, N_mask) indices of patches.

        Returns:
            (B, N_mask, D) positional embeddings.
        """
        return torch.gather(
            self.pos_embed.expand(indices.shape[0], -1, -1),
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )


class TransformerBlock(nn.Module):
    """
    Standard transformer block with pre-norm architecture.

    Uses pre-LayerNorm (more stable training) and GELU activation
    in the MLP, following the ViT convention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        # Pre-norm MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
