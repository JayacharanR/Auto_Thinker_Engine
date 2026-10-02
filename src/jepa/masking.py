"""
Masking strategies for JEPA pretraining.

The default is V-JEPA-style tube masking: a union of random spatial blocks,
each repeated across *every* temporal patch. Masking a block in some frames
only would let the model copy the hidden content from neighbouring frames of
the same location, which teaches it little. ``temporal_last`` (predict the
last frames from the earlier ones) is kept as an alternative.

CRITICAL: masking must never leak target information into context.
The unit test in test_jepa_components.py verifies this invariant.
"""

import math
from typing import Optional

import torch

# V-JEPA's two mask families: many small blocks, or a few large ones.
DEFAULT_TUBE_GROUPS = [
    {"num_blocks": 8, "scale": [0.15, 0.15], "aspect_ratio": [0.75, 1.5]},
    {"num_blocks": 2, "scale": [0.7, 0.7], "aspect_ratio": [0.75, 1.5]},
]


class MaskGenerator:
    """
    Generates masks for JEPA pretraining.

    Strategies:
    1. ``tube``: each batch draws one block group from ``config["tube"]["groups"]``.
       A group places ``num_blocks`` rectangles covering ``scale`` of the
       spatial grid (aspect ratio in ``aspect_ratio``); their union is masked
       in every temporal patch.
    2. ``multi_block``: the same tube masks configured with the older keys
       ``num_masks``, ``min_mask_ratio`` and ``max_mask_ratio``.
    3. ``temporal_last``: mask the last N temporal patches (causal prediction).

    Args:
        num_patches: Total number of patches in the sequence.
        num_patches_spatial: Number of patches in the spatial dimensions (H' * W').
        num_patches_temporal: Number of patches in the temporal dimension (T').
        strategy: 'tube', 'multi_block' or 'temporal_last'.
        config: Strategy-specific configuration dict.
    """

    def __init__(
        self,
        num_patches: int,
        num_patches_spatial: int,
        num_patches_temporal: int,
        strategy: str = "tube",
        config: Optional[dict] = None,
    ):
        self.num_patches = num_patches
        self.num_patches_spatial = num_patches_spatial
        self.num_patches_temporal = num_patches_temporal
        self.strategy = strategy
        self.config = config or {}

        # Spatial grid dimensions (assume square)
        self.grid_h = int(num_patches_spatial**0.5)
        self.grid_w = self.grid_h
        assert self.grid_h * self.grid_w == num_patches_spatial, (
            f"Spatial patches ({num_patches_spatial}) must form a square grid"
        )
        assert num_patches == num_patches_spatial * num_patches_temporal

        if strategy == "tube":
            cfg = self.config.get("tube", {})
            self.groups = cfg.get("groups", DEFAULT_TUBE_GROUPS)
            self.min_context_ratio = cfg.get("min_context_ratio", 0.1)
        elif strategy == "multi_block":
            cfg = self.config.get("multi_block", {})
            self.groups = [{
                "num_blocks": cfg.get("num_masks", 4),
                "scale": [cfg.get("min_mask_ratio", 0.15), cfg.get("max_mask_ratio", 0.30)],
                "aspect_ratio": cfg.get("aspect_ratio", [0.75, 1.5]),
            }]
            self.min_context_ratio = cfg.get("min_context_ratio", 0.1)
        elif strategy != "temporal_last":
            raise ValueError(f"Unknown masking strategy: {strategy}")

    def __call__(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Generate masks for a batch.

        Returns:
            Dict with:
            - 'context_indices': (B, N_ctx) indices of unmasked patches
            - 'mask_indices': (B, N_mask) indices of masked patches
            - 'mask_ratio': actual mask ratio applied
        """
        if self.strategy == "temporal_last":
            return self._temporal_last_mask(batch_size)
        return self._tube_mask(batch_size)

    def _spatial_masks(self, batch_size: int, group: dict) -> torch.Tensor:
        """(B, H*W) bool: union of ``num_blocks`` random rectangles per sample."""
        k = int(group["num_blocks"])
        lo_s, hi_s = group["scale"]
        lo_a, hi_a = group.get("aspect_ratio", [0.75, 1.5])

        area = (lo_s + (hi_s - lo_s) * torch.rand(batch_size, k)) * self.num_patches_spatial
        log_lo, log_hi = math.log(lo_a), math.log(hi_a)
        aspect = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(batch_size, k))
        h = (area * aspect).sqrt().round().clamp(1, self.grid_h).long()
        w = (area / aspect).sqrt().round().clamp(1, self.grid_w).long()
        top = (torch.rand(batch_size, k) * (self.grid_h - h + 1)).long()
        left = (torch.rand(batch_size, k) * (self.grid_w - w + 1)).long()

        rows = torch.arange(self.grid_h).view(1, 1, -1, 1)
        cols = torch.arange(self.grid_w).view(1, 1, 1, -1)
        top, left = top[..., None, None], left[..., None, None]
        h, w = h[..., None, None], w[..., None, None]
        inside = (rows >= top) & (rows < top + h) & (cols >= left) & (cols < left + w)
        return inside.any(dim=1).flatten(1)

    def _tube_mask(self, batch_size: int) -> dict[str, torch.Tensor]:
        group = self.groups[torch.randint(len(self.groups), (1,)).item()]
        min_context = max(1, int(self.min_context_ratio * self.num_patches_spatial))

        spatial = self._spatial_masks(batch_size, group)
        # Resample the rare samples whose blocks leave too little context.
        for _ in range(100):
            short = (~spatial).sum(1) < min_context
            if not short.any():
                break
            spatial[short] = self._spatial_masks(int(short.sum()), group)
        else:
            raise RuntimeError("Mask blocks leave no context; reduce scale or num_blocks")

        masked = spatial.repeat(1, self.num_patches_temporal)  # same blocks in every frame
        context_indices = _random_subset(~masked)
        mask_indices = _random_subset(masked)
        return {
            "context_indices": context_indices,
            "mask_indices": mask_indices,
            "mask_ratio": mask_indices.shape[1] / self.num_patches,
        }

    def _temporal_last_mask(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Temporal masking: mask the last N temporal patches.

        This is a causal prediction task — predict future frames from past.
        """
        cfg = self.config.get("temporal", {})
        num_future = min(cfg.get("num_future_frames", 4), self.num_patches_temporal)
        split = (self.num_patches_temporal - num_future) * self.num_patches_spatial
        indices = torch.arange(self.num_patches)
        return {
            "context_indices": indices[:split].unsqueeze(0).expand(batch_size, -1),
            "mask_indices": indices[split:].unsqueeze(0).expand(batch_size, -1),
            "mask_ratio": (self.num_patches - split) / self.num_patches,
        }


def _random_subset(selected: torch.Tensor) -> torch.Tensor:
    """
    (B, N) bool -> (B, n) sorted indices, with n the smallest per-row count.

    Rows with more selected positions keep a random subset of them, so every
    row has the same length without favouring early patches.
    """
    n = int(selected.sum(1).min())
    keys = selected.float() + torch.rand(selected.shape)  # selected rank above unselected
    return keys.topk(n, dim=1).indices.sort(dim=1).values


def verify_no_leak(
    context_indices: torch.Tensor,
    mask_indices: torch.Tensor,
) -> bool:
    """
    Verify that context and mask indices don't overlap.

    This is a critical invariant: the context encoder must not see
    any of the patches that the predictor is trying to predict.

    Args:
        context_indices: (B, N_ctx) context patch indices.
        mask_indices: (B, N_mask) masked patch indices.

    Returns:
        True if no overlap (correct behavior).

    Raises:
        AssertionError if any overlap is found.
    """
    overlap = (context_indices.unsqueeze(2) == mask_indices.unsqueeze(1)).any(dim=2)
    for b in torch.where(overlap.any(dim=1))[0].tolist():
        leaked = context_indices[b][overlap[b]].tolist()
        raise AssertionError(
            f"MASKING BUG: Batch {b} has {len(leaked)} overlapping indices "
            f"between context and mask: {leaked}. "
            f"Target information is leaking into context!"
        )
    return True
