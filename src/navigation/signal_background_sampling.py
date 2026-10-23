"""TRAIN-only background sampling independent of all classifier predictions."""

import numpy as np


def select_background(points, background, excluded, sample_id, limit=64):
    """Uniform fixed-seed sample of new known-background candidate coordinates."""
    points = np.asarray(points)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.issubdtype(points.dtype, np.integer)
        or background.shape != (288, 512)
        or background.dtype != np.bool_
    ):
        raise ValueError("Native integer points and TRAIN known-background mask required")
    if (points < 0).any() or (points[:, 0] >= 512).any() or (points[:, 1] >= 288).any():
        raise ValueError("Point outside image")
    if sample_id < 0 or limit != 64:
        raise ValueError("Registered sample identity and64point limit required")
    codes = points[:, 1].astype(np.int64) * 512 + points[:, 0]
    if len(np.unique(codes)) != len(points):
        raise ValueError("Unique frozen candidate coordinates required")
    ignored = np.array([int(y) * 512 + int(x) for x, y in excluded], np.int64)
    eligible = np.flatnonzero(background[points[:, 1], points[:, 0]] & ~np.isin(codes, ignored))
    rng = np.random.default_rng(np.random.SeedSequence([42, int(sample_id)]))
    return np.sort(rng.choice(eligible, size=min(64, len(eligible)), replace=False))


def append_fold_background(x, y, extra, extra_fold, excluded_fold):
    """Append matching original/gamma extra negatives, never the held-out junction."""
    if (
        x.ndim != 2
        or x.shape[1] != 87
        or y.shape != (len(x),)
        or extra.ndim != 3
        or extra.shape[0] != 3
        or extra.shape[2] != 87
        or extra_fold.shape != (extra.shape[1],)
        or not np.isin(extra_fold, [0, 1, 2]).all()
        or excluded_fold not in (0, 1, 2)
    ):
        raise ValueError("Registered three-version features and internal TRAIN fold required")
    kept = extra_fold != excluded_fold
    addition = np.concatenate([version[kept] for version in extra]).astype(np.float32)
    return np.concatenate([x, addition]), np.concatenate(
        [y, np.zeros(len(addition), dtype=y.dtype)]
    )
