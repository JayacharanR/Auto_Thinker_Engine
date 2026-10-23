"""Registered TRAIN-only photometric transforms; no spatial or phase manipulation."""

import numpy as np

GAMMAS = (0.7, 1.4)


def gamma_rgb(rgb, gamma):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Native HWC uint8 RGB required")
    if gamma not in (1.0, *GAMMAS):
        raise ValueError("Unregistered photometric transform")
    lut = np.rint(255 * (np.arange(256, dtype=np.float64) / 255) ** gamma).astype(np.uint8)
    return lut[rgb]
