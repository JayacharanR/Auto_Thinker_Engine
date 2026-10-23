"""Native RGB proposals independent of the learned CNN's existing candidates."""

import numpy as np
from scipy.ndimage import label, maximum_filter, minimum_filter


def global_rgb_points(rgb):
    """All per-channel local maxima with at least16/255 contrast in a5x5window.

    This is a fixed candidate rule, not a signal classification threshold. All
    plateau pixels satisfying it remain. No color/state, top-K or position filter.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (288, 512, 3):
        raise ValueError("Native512x288 HWC uint8 RGB required")
    values = rgb.astype(np.int16)
    high = maximum_filter(values, size=(5, 5, 1), mode="nearest")
    low = minimum_filter(values, size=(5, 5, 1), mode="nearest")
    candidate = ((values == high) & (high - low >= 16)).any(-1)
    y, x = np.nonzero(candidate)
    return np.column_stack([x, y]).astype(np.int64)


def plateau_rgb_points(rgb):
    """One centroid-nearest native pixel per per-channel local-maximum plateau.

    Uses the same5x5/16contrast rule; no component is discarded or ranked by score.
    Separate channels preserve distinct color maxima. Eight-connectivity; centroid
    ties prefer y then x. The chosen coordinate must be a member of its plateau.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (288, 512, 3):
        raise ValueError("Native512x288 HWC uint8 RGB required")
    values = rgb.astype(np.int16)
    high = maximum_filter(values, size=(5, 5, 1), mode="nearest")
    low = minimum_filter(values, size=(5, 5, 1), mode="nearest")
    mask = (values == high) & (high - low >= 16)
    points = []
    for channel in range(3):
        components, count = label(mask[:, :, channel], structure=np.ones((3, 3), bool))
        if not count:
            continue
        y, x = np.nonzero(mask[:, :, channel])
        ids = components[y, x]
        size = np.bincount(ids, minlength=count + 1)
        center_x = np.bincount(ids, weights=x, minlength=count + 1) / np.maximum(size, 1)
        center_y = np.bincount(ids, weights=y, minlength=count + 1) / np.maximum(size, 1)
        distance = (x - center_x[ids]) ** 2 + (y - center_y[ids]) ** 2
        ordered = np.lexsort((x, y, distance, ids))
        first = np.r_[True, ids[ordered][1:] != ids[ordered][:-1]]
        selected = ordered[first]
        if len(selected) != count:
            raise ValueError("Every channel plateau must retain exactly one representative")
        points.append(np.column_stack([x[selected], y[selected]]))
    if not points:
        return np.empty((0, 2), np.int64)
    union = np.unique(np.concatenate(points), axis=0)
    return union[np.lexsort((union[:, 0], union[:, 1]))]
