"""Explicit temporal/height/width positions for navigation's rectangular video.

Historical JEPA/Dreamer paths are unchanged. Navigation never infers the time
axis from the total token count: equal counts can represent different grids.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def grid_positions(position, stored_grid, actual_grid):
    if len(stored_grid) != 3 or len(actual_grid) != 3 or min(*stored_grid, *actual_grid) <= 0:
        raise ValueError("Explicit positive temporal/height/width grids required")
    if position.ndim != 3 or position.shape[0] != 1 or math.prod(stored_grid) != position.shape[1]:
        raise ValueError("Stored positions do not match the declared source grid")
    if tuple(stored_grid) == tuple(actual_grid):
        return position
    volume = position.reshape(1, *stored_grid, position.shape[-1]).permute(0, 4, 1, 2, 3)
    # Interpolate coordinates independently; flattening spatial positions would
    # mix unrelated image rows and can silently reinterpret space as elapsed time.
    interpolated = F.interpolate(
        volume.float(), size=tuple(actual_grid), mode="trilinear", align_corners=False
    )
    return (
        interpolated.permute(0, 2, 3, 4, 1)
        .reshape(1, math.prod(actual_grid), -1)
        .to(position.dtype)
    )


def causal_clip_indices(index, episode_start, length=8, stride=1):
    if index < episode_start or min(length, stride) < 1:
        raise ValueError("Past-only indices need a valid episode and positive context")
    return [max(episode_start, index - k * stride) for k in range(length - 1, -1, -1)]


class RectangularJEPA(nn.Module):
    """Wrap a source ViTEncoder, retaining every temporal and spatial token."""

    def __init__(self, backbone, stored_grid):
        super().__init__()
        self.backbone = backbone
        self.stored_grid = tuple(stored_grid)
        grid_positions(backbone.pos_embed, self.stored_grid, self.stored_grid)

    def forward(self, video, context_indices=None):
        if video.ndim != 5:
            raise ValueError("Expected batch/channel/time/height/width video")
        base = self.backbone
        _, _, time, height, width = video.shape
        if time % base.tubelet_size or height % base.patch_size or width % base.patch_size:
            raise ValueError("Video dimensions must divide exactly into declared tubelets")
        grid = (time // base.tubelet_size, height // base.patch_size, width // base.patch_size)
        tokens = base.patch_embed(video)
        if tokens.shape[1] != math.prod(grid):
            raise ValueError("Embedding order/dimensions differ from explicit T/H/W grid")
        tokens = tokens + grid_positions(base.pos_embed, self.stored_grid, grid)
        if context_indices is not None:
            if (
                context_indices.ndim != 2
                or context_indices.shape[0] != video.shape[0]
                or context_indices.dtype != torch.long
            ):
                raise ValueError("Context indices require explicit per-batch int64 positions")
            if (
                context_indices.numel() == 0
                or context_indices.min() < 0
                or context_indices.max() >= tokens.shape[1]
            ):
                raise ValueError("Context indices outside the actual video grid")
            tokens = torch.gather(
                tokens, 1, context_indices[..., None].expand(-1, -1, tokens.shape[-1])
            )
        tokens = base.pos_drop(tokens)
        for block in base.blocks:
            tokens = block(tokens)
        return base.norm(tokens)

    def spatial_tokens(self, video, pool=(4, 8)):
        """Pool temporal context; retain a fixed, declared4x8 spatial grid per view."""
        tokens = self(video)
        base = self.backbone
        t, h, w = (
            video.shape[2] // base.tubelet_size,
            video.shape[3] // base.patch_size,
            video.shape[4] // base.patch_size,
        )
        spatial = tokens.reshape(video.shape[0], t, h, w, -1).mean(1).permute(0, 3, 1, 2)
        return F.adaptive_avg_pool2d(spatial, pool).flatten(2).transpose(1, 2)
