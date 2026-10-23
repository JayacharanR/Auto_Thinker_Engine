"""Staged compiled RGB feature reductions; no deployed imports."""

import numpy as np
from numba import njit


# Dynamic staging imports have no stable module name for Numba's disk cache.
# Compile once per process and measure that startup separately from inference.
@njit(cache=False, fastmath=False)
def _features(padded, averaged, points):
    result = np.empty((len(points), 87), np.float32)
    for n in range(len(points)):
        x, y = points[n, 0], points[n, 1]
        for c in range(3):
            for dy in range(5):
                for dx in range(5):
                    result[n, c * 25 + dy * 5 + dx] = padded[y + 46 + dy, x + 46 + dx, c]
            local_sum, wide_sum = np.float32(0), np.float32(0)
            for dy in range(32):
                for dx in range(32):
                    local_sum += padded[y + 32 + dy, x + 32 + dx, c]
                    wide_sum += averaged[y + 3 * dy, x + 3 * dx, c]
            local_mean = local_sum / np.float32(1024)
            wide_mean = wide_sum / np.float32(1024)
            local_square, wide_square = np.float32(0), np.float32(0)
            for dy in range(32):
                for dx in range(32):
                    local_delta = padded[y + 32 + dy, x + 32 + dx, c] - local_mean
                    wide_delta = averaged[y + 3 * dy, x + 3 * dx, c] - wide_mean
                    local_square += local_delta * local_delta
                    wide_square += wide_delta * wide_delta
            result[n, 75 + c] = local_mean
            result[n, 78 + c] = np.sqrt(local_square / np.float32(1024))
            result[n, 81 + c] = wide_mean
            result[n, 84 + c] = np.sqrt(wide_square / np.float32(1024))
    return result


def fused_features(prepared, points):
    """Use a validated PreparedContext instance; include normalization each call."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2 or not np.issubdtype(points.dtype, np.integer):
        raise ValueError("Integer Nx2 coordinates required")
    if (
        (points < 0).any()
        or (points[:, 0] >= prepared.width).any()
        or (points[:, 1] >= prepared.height).any()
    ):
        raise ValueError("Proposal outside image")
    return _features(
        prepared.padded.astype(np.float32) / np.float32(255),
        prepared.small.astype(np.float32) / np.float32(255),
        points.astype(np.int64),
    )
