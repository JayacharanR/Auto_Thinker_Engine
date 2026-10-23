"""Staged exact context extraction; not imported by any deployed navigation code."""

import numpy as np


class PreparedContext:
    """Reuse integer 3x3 averages while preserving the original patch memory layout.

    A 96x96 window reduces to 32x32 by averaging disjoint 3x3 cells.
    Precomputing all possible cell origins preserves all three alignment phases.
    Nine integer pixels cannot average to a half integer, so (sum+4)//9
    equals the original float64 mean followed by np.rint, including at borders.
    """

    def __init__(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Native HWC uint8 RGB required")
        if not rgb.shape[0] or not rgb.shape[1]:
            raise ValueError("Nonempty image required")
        self.height, self.width = rgb.shape[:2]
        self.padded = np.pad(rgb, ((48, 48), (48, 48), (0, 0)), mode="edge")
        h, w = self.padded.shape[:2]
        summed = np.zeros((h - 2, w - 2, 3), dtype=np.uint16)
        for dy in range(3):
            for dx in range(3):
                summed += self.padded[dy : h - 2 + dy, dx : w - 2 + dx]
        self.small = ((summed + 4) // 9).astype(np.uint8)

    def patches(self, points):
        points = np.asarray(points)
        if not len(points):
            return np.empty((0, 6, 32, 32), np.uint8)
        if points.ndim != 2 or points.shape[1] != 2 or not np.issubdtype(points.dtype, np.integer):
            raise ValueError("Integer RGB proposal coordinates required")
        if (
            (points < 0).any()
            or (points[:, 0] >= self.width).any()
            or (points[:, 1] >= self.height).any()
        ):
            raise ValueError("Proposal outside image")
        # Convert before addition: uint16 input must not wrap at large offsets.
        px, py = points[:, 0].astype(np.int64), points[:, 1].astype(np.int64)
        rows, cols = np.arange(32)[None, :, None], np.arange(32)[None, None, :]
        local = self.padded[py[:, None, None] + 32 + rows, px[:, None, None] + 32 + cols]
        wide = self.small[py[:, None, None] + 3 * rows, px[:, None, None] + 3 * cols]
        return np.concatenate([local, wide], axis=-1).transpose(0, 3, 1, 2)


def exact_rgb_features(rgb, points, chunk_size=256):
    """Bound temporary allocation; reuse the registered NumPy feature reduction."""
    from src.navigation.signal_pretrained_verifier import rgb_features

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Positive integer chunk size required")
    prepared = PreparedContext(rgb)
    if not len(points):
        return np.empty((0, 87), np.float32)
    return np.concatenate(
        [
            rgb_features(prepared.patches(points[i : i + chunk_size]))
            for i in range(0, len(points), chunk_size)
        ]
    )
