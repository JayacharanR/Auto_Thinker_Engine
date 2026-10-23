"""Fixed continuous RGB/opponent fields; no signal labels, thresholds or rules."""

import torch
from torch.nn import functional as F

FIELD_CHANNELS = 11
FIELD_GRID = (9, 16)


def optical_fields(rgb):
    """Preserve small saturated pixels in32x32 cells of each native RGB view.

    Shape (...,3,288,512) uint8 -> (...,144,11). Every pixel/view is treated
    identically; no map, detected box, annotation or hardcoded stop action.
    """
    if rgb.dtype != torch.uint8 or rgb.shape[-3:] != (3, 288, 512):
        raise ValueError("Native uint8 three-channel512x288 camera images required")
    shape = rgb.shape[:-3]
    images = rgb.reshape(-1, 3, 288, 512).float() / 255
    r, g, b = images.unbind(1)
    channels = torch.stack([r, g, b, r - g, g - r, r - b, g - b, torch.minimum(r, g) - b], 1)
    maximum = F.max_pool2d(channels, 32, 32)
    average = F.avg_pool2d(images, 32, 32)
    return (
        torch.cat([maximum, average], 1)
        .flatten(2)
        .transpose(1, 2)
        .reshape(*shape, 144, FIELD_CHANNELS)
    )


def past_average(features, window=8):
    """Time-first sequence; repeat only its first frame at episode start."""
    if features.ndim < 2 or features.shape[0] < 1 or window < 1:
        raise ValueError("Nonempty time-first features and positive past window required")
    channels = features.reshape(features.shape[0], -1).T[None]
    padded = F.pad(channels, (window - 1, 0), mode="replicate")
    return F.avg_pool1d(padded, window, 1)[0].T.reshape_as(features)
