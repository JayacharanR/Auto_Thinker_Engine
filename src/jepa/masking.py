"""
Masking strategies for JEPA pretraining.

Implements both spatial patch masking and temporal masking as
configurable strategies. Multi-block masking (multiple masked
regions rather than one large block) is the default — it tends
to be more robust.

CRITICAL: masking must never leak target information into context.
The unit test in test_jepa_components.py verifies this invariant.
"""

from typing import Optional

import torch


class MaskGenerator:
    """
    Generates masks for JEPA pretraining.

    Supports two strategies:
    1. multi_block: Multiple random rectangular regions masked in the
       spatial-temporal patch grid. More robust than single-block masking.
    2. temporal_last: Mask the last N frames (causal prediction).

    Args:
        num_patches: Total number of patches in the sequence.
        num_patches_spatial: Number of patches in the spatial dimensions (H' * W').
        num_patches_temporal: Number of patches in the temporal dimension (T').
        strategy: 'multi_block' or 'temporal_last'.
        config: Strategy-specific configuration dict.
    """

    def __init__(
        self,
        num_patches: int,
        num_patches_spatial: int,
        num_patches_temporal: int,
        strategy: str = "multi_block",
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

    def __call__(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Generate masks for a batch.

        Returns:
            Dict with:
            - 'context_indices': (B, N_ctx) indices of unmasked patches
            - 'mask_indices': (B, N_mask) indices of masked patches
            - 'mask_ratio': actual mask ratio applied
        """
        if self.strategy == "multi_block":
            return self._multi_block_mask(batch_size)
        elif self.strategy == "temporal_last":
            return self._temporal_last_mask(batch_size)
        else:
            raise ValueError(f"Unknown masking strategy: {self.strategy}")

    def _multi_block_mask(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Multi-block masking: mask multiple random rectangular regions.

        Generates several mask blocks in the spatiotemporal grid. Each block
        is a contiguous rectangle in the spatial dimensions, applied across
        a random subset of temporal patches.
        """
        cfg = self.config.get("multi_block", {})
        num_masks = cfg.get("num_masks", 4)
        min_ratio = cfg.get("min_mask_ratio", 0.15)
        max_ratio = cfg.get("max_mask_ratio", 0.30)
        target_total_ratio = cfg.get("total_mask_ratio", 0.65)

        all_context_indices = []
        all_mask_indices = []

        for _ in range(batch_size):
            # Start with all patches unmasked
            masked = torch.zeros(self.num_patches, dtype=torch.bool)

            for _ in range(num_masks):
                # Random mask ratio for this block
                mask_ratio = min_ratio + (max_ratio - min_ratio) * torch.rand(1).item()
                num_patches_to_mask = int(self.num_patches_spatial * mask_ratio)

                # Random rectangular block in spatial dimensions
                block_h = max(1, int((num_patches_to_mask / self.grid_w) ** 0.5 * 1.5))
                block_w = max(1, num_patches_to_mask // block_h)
                block_h = min(block_h, self.grid_h)
                block_w = min(block_w, self.grid_w)

                # Random top-left corner
                top = torch.randint(0, max(1, self.grid_h - block_h + 1), (1,)).item()
                left = torch.randint(0, max(1, self.grid_w - block_w + 1), (1,)).item()

                # Random temporal range
                t_start = torch.randint(0, self.num_patches_temporal, (1,)).item()
                t_len = max(1, torch.randint(1, self.num_patches_temporal + 1, (1,)).item())
                t_end = min(t_start + t_len, self.num_patches_temporal)

                # Mark patches as masked
                for t in range(t_start, t_end):
                    for h in range(top, min(top + block_h, self.grid_h)):
                        for w in range(left, min(left + block_w, self.grid_w)):
                            patch_idx = t * self.num_patches_spatial + h * self.grid_w + w
                            masked[patch_idx] = True

            # Enforce target total mask ratio (add/remove patches randomly)
            current_ratio = masked.float().mean().item()
            target_count = int(self.num_patches * target_total_ratio)

            if masked.sum() > target_count:
                # Too many masked — unmask some randomly
                masked_indices = torch.where(masked)[0]
                keep_masked = masked_indices[
                    torch.randperm(len(masked_indices))[:target_count]
                ]
                masked = torch.zeros(self.num_patches, dtype=torch.bool)
                masked[keep_masked] = True
            elif masked.sum() < target_count:
                # Too few masked — mask additional random patches
                unmasked_indices = torch.where(~masked)[0]
                additional = target_count - masked.sum().item()
                additional = min(additional, len(unmasked_indices))
                extra = unmasked_indices[
                    torch.randperm(len(unmasked_indices))[:additional]
                ]
                masked[extra] = True

            # Split into context and mask indices
            context_idx = torch.where(~masked)[0]
            mask_idx = torch.where(masked)[0]

            all_context_indices.append(context_idx)
            all_mask_indices.append(mask_idx)

        # Pad to same length across batch (different samples may have
        # slightly different numbers of context/mask patches due to
        # block overlap)
        context_indices = _pad_indices(all_context_indices)
        mask_indices = _pad_indices(all_mask_indices)

        actual_ratio = mask_indices.shape[1] / self.num_patches

        return {
            "context_indices": context_indices,
            "mask_indices": mask_indices,
            "mask_ratio": actual_ratio,
        }

    def _temporal_last_mask(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Temporal masking: mask the last N frames.

        This is a causal prediction task — predict future frames from past.
        """
        cfg = self.config.get("temporal", {})
        num_future_frames = cfg.get("num_future_frames", 4)
        num_future_frames = min(num_future_frames, self.num_patches_temporal)

        # Mask the last num_future_frames temporal patches
        mask_start_t = self.num_patches_temporal - num_future_frames

        context_indices_list = []
        mask_indices_list = []

        for t in range(self.num_patches_temporal):
            for s in range(self.num_patches_spatial):
                patch_idx = t * self.num_patches_spatial + s
                if t >= mask_start_t:
                    mask_indices_list.append(patch_idx)
                else:
                    context_indices_list.append(patch_idx)

        context_indices = torch.tensor(context_indices_list, dtype=torch.long)
        mask_indices = torch.tensor(mask_indices_list, dtype=torch.long)

        # Expand for batch
        context_indices = context_indices.unsqueeze(0).expand(batch_size, -1)
        mask_indices = mask_indices.unsqueeze(0).expand(batch_size, -1)

        actual_ratio = len(mask_indices_list) / self.num_patches

        return {
            "context_indices": context_indices,
            "mask_indices": mask_indices,
            "mask_ratio": actual_ratio,
        }


def _pad_indices(indices_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Pad a list of index tensors to the same length.

    Uses the minimum length across the batch (truncation) to ensure
    valid indexing. This means some patches at the boundary may be
    reassigned, but the total mask ratio stays approximately correct.
    """
    min_len = min(len(idx) for idx in indices_list)
    padded = torch.stack([idx[:min_len] for idx in indices_list])
    return padded


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
    for b in range(context_indices.shape[0]):
        ctx_set = set(context_indices[b].tolist())
        mask_set = set(mask_indices[b].tolist())
        overlap = ctx_set & mask_set
        assert len(overlap) == 0, (
            f"MASKING BUG: Batch {b} has {len(overlap)} overlapping indices "
            f"between context and mask: {overlap}. "
            f"Target information is leaking into context!"
        )
    return True
