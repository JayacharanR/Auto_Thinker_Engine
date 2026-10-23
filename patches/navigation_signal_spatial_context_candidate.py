"""Spatial RGB context for prospective TRAIN comparison; no metadata features."""

import numpy as np

SCALES = (16, 32, 96)
FEATURES = 288


def spatial_context(rgb, points):
    """4x4 RGB mean/std grids at three centered native-pixel scales.

    Every coordinate sees the same descriptor. Integral sums are exact integers,
    moments are evaluated in float64, and the output is float32 before caching.
    Layout: scale, row, column, statistic(mean then std), RGB channel.
    """
    rgb, points = np.asarray(rgb), np.asarray(points)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("HWC uint8 RGB required")
    if points.ndim != 2 or points.shape[1] != 2 or not np.issubdtype(points.dtype, np.integer):
        raise ValueError("Integer Nx2 point coordinates required")
    h, w = rgb.shape[:2]
    if (
        not h
        or not w
        or ((points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any())
    ):
        raise ValueError("Nonempty RGB image and in-bounds coordinates required")
    if not len(points):
        return np.empty((0, FEATURES), np.float32)
    padded = np.pad(rgb, ((48, 48), (48, 48), (0, 0)), mode="edge").astype(np.int64)
    integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    square = np.pad((padded * padded).cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    x, y = points[:, 0].astype(np.int64), points[:, 1].astype(np.int64)
    features = []
    for scale in SCALES:
        cell = scale // 4
        for row in range(4):
            for col in range(4):
                left, top = x + 48 - scale // 2 + col * cell, y + 48 - scale // 2 + row * cell
                right, bottom = left + cell, top + cell
                total = (
                    integral[bottom, right]
                    - integral[top, right]
                    - integral[bottom, left]
                    + integral[top, left]
                )
                squared = (
                    square[bottom, right]
                    - square[top, right]
                    - square[bottom, left]
                    + square[top, left]
                )
                mean = total / (cell * cell)
                std = np.sqrt(np.maximum(squared / (cell * cell) - mean * mean, 0))
                features.extend([mean / 255, std / 255])
    return np.concatenate(features, axis=1).astype(np.float32)
