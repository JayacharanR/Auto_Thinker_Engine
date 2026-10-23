"""Relative-color features computed only from the original 87 RGB measurements."""

import numpy as np


def opponent_features(features):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 87 or not np.isfinite(x).all():
        raise ValueError("Finite original 87 RGB features required")
    color = x[:, :75].reshape(-1, 3, 25)
    if (color < 0).any() or (color > 1).any():
        raise ValueError("Original RGB intensities must be within 0..1")
    red, green, blue = color[:, 0], color[:, 1], color[:, 2]
    opponent = np.stack([red - green, green - blue, red - blue], axis=1)
    chromaticity = color / np.maximum(color.sum(1, keepdims=True), 1e-6)
    local_contrast = color - color.mean(2, keepdims=True)
    return np.concatenate(
        [x, opponent.reshape(-1, 75), chromaticity.reshape(-1, 75), local_contrast.reshape(-1, 75)],
        axis=1,
    )
