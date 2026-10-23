"""RGB-only proposal refinement; no labels, map, phase or temporal inputs."""

import numpy as np


def refine_rgb_points(rgb, points):
    """Keep all points; add each channel maximum in the native 5x5 neighborhood.

    Ties prefer the nearest pixel to the original point, then y, then x. Duplicate
    coordinates are merged. There is no score threshold, top-K or class inference.
    """
    rgb, points = np.asarray(rgb), np.asarray(points)
    if rgb.dtype != np.uint8 or rgb.shape != (288, 512, 3):
        raise ValueError("Native512x288 HWC uint8 RGB required")
    if points.ndim != 2 or points.shape[1] != 2 or not np.issubdtype(points.dtype, np.integer):
        raise ValueError("Integer Nx2 native proposal coordinates required")
    if not len(points):
        return np.empty((0, 2), np.int64)
    if (points < 0).any() or (points[:, 0] >= 512).any() or (points[:, 1] >= 288).any():
        raise ValueError("Proposal outside native image")
    # Ordering gives a deterministic geometric tie-break, independent of class.
    offsets = np.array(
        sorted(
            ((x, y) for y in range(-2, 3) for x in range(-2, 3)),
            key=lambda p: (p[0] ** 2 + p[1] ** 2, p[1], p[0]),
        )
    )
    xy = points[:, None, :] + offsets[None, :, :]
    valid = (xy[:, :, 0] >= 0) & (xy[:, :, 0] < 512) & (xy[:, :, 1] >= 0) & (xy[:, :, 1] < 288)
    clipped = np.clip(xy, [0, 0], [511, 287])
    colors = rgb[clipped[:, :, 1], clipped[:, :, 0]].astype(np.int16)
    colors[~valid] = -1
    choice = colors.argmax(1)
    added = xy[np.arange(len(points))[:, None], choice].reshape(-1, 2)
    all_points = np.unique(np.concatenate([points, added]), axis=0)
    return all_points[np.lexsort((all_points[:, 0], all_points[:, 1]))]
