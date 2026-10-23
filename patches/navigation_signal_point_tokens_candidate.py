"""Staged RGB point representation; not imported by any deployed policy.

No depth, housing box, stop line, applicability or traffic state is synthesized.
Each point is an observed camera ray with the classifier's full probabilities.
Only current-frame observations are accepted; a future trained recurrent policy
must learn memory, rather than treating an old ray as current-frame geometry.
"""

import math

import numpy as np

VIEWS = ("front", "left", "right")
TOKEN_DIM = 16
FIELDS = frozenset(("view", "x", "y", "probabilities", "frame", "timestamp"))
SCHEMA = "camera-signal-point-rays-v1-staged"


def calibrated_ray(x, y, calibration):
    """Return unit ego-frame ray direction and camera origin, never a 3D point."""
    values = np.array([*calibration.xyz, *calibration.pitch_yaw_roll, calibration.fov])
    if (
        not np.isfinite(values).all()
        or not 0 < calibration.fov < 180
        or calibration.width != 512
        or calibration.height != 288
    ):
        raise ValueError("Finite registered native camera calibration required")
    if (
        isinstance(x, (bool, np.bool_))
        or isinstance(y, (bool, np.bool_))
        or not isinstance(x, (int, np.integer))
        or not isinstance(y, (int, np.integer))
        or not 0 <= x < 512
        or not 0 <= y < 288
    ):
        raise ValueError("Native integer RGB point required")
    focal = 512 / (2 * math.tan(math.radians(calibration.fov) / 2))
    ray = np.array([1.0, (x - 256) / focal, (144 - y) / focal])
    pitch, yaw, roll = np.deg2rad(calibration.pitch_yaw_roll)
    cp, sp, cy, sy, cr, sr = (
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
        math.cos(roll),
        math.sin(roll),
    )
    rotation = np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ]
    )
    direction = rotation @ ray
    direction /= np.linalg.norm(direction)
    return direction, np.asarray(calibration.xyz, dtype=float)


def pack_current_points(packet, observations):
    """One token per accepted detection, plus null; no capacity truncation.

    Layout: unit ray3, origin3, unconditional BG/red/yellow/green probabilities4,
    view one-hot3, normalized nativeXY2, null-indicator1. Count is dynamic; inputs
    may contain several detected points on the same physical housing. No supplied
    object or teacher ID performs deduplication/association.
    """
    packet.validate()
    tokens = np.zeros((len(observations) + 1, TOKEN_DIM), np.float32)
    tokens[0, -1] = 1
    for i, row in enumerate(observations, 1):
        if set(row) != FIELDS or row["view"] not in VIEWS:
            raise ValueError("Exactly the declared RGB-point fields required")
        if (
            row["frame"] != packet.frame
            or not np.isfinite(row["timestamp"])
            or abs(row["timestamp"] - packet.timestamp) > 1e-6
        ):
            raise ValueError("Only current-frame point observations allowed")
        p = np.asarray(row["probabilities"], dtype=float)
        if (
            p.shape != (4,)
            or not np.isfinite(p).all()
            or (p < 0).any()
            or (p > 1).any()
            or not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-6)
            or p[1:].sum() < 0.5
        ):
            raise ValueError("Valid accepted BG/red/yellow/green probabilities required")
        ray, origin = calibrated_ray(row["x"], row["y"], packet.calibration[row["view"]])
        tokens[i] = [
            *ray,
            *origin,
            *p,
            *[float(row["view"] == v) for v in VIEWS],
            row["x"] / 512,
            row["y"] / 288,
            0,
        ]
    return tokens, np.ones(len(tokens), dtype=bool)


def pad_point_batch(items):
    """Pad varying detection counts, retaining every token and a valid null."""
    if not items:
        raise ValueError("A nonempty batch is required")
    for tokens, valid in items:
        if (
            tokens.ndim != 2
            or tokens.shape[1] != TOKEN_DIM
            or not len(tokens)
            or valid.shape != (len(tokens),)
            or valid.dtype != np.bool_
            or not np.isfinite(tokens).all()
            or not valid.all()
            or tokens[0, -1] != 1
            or tokens[1:, -1].any()
        ):
            raise ValueError("Unpadded current-point tokens with one null required")
    maximum = max(len(t) for t, _ in items)
    batch = np.zeros((len(items), maximum, TOKEN_DIM), np.float32)
    mask = np.zeros((len(items), maximum), bool)
    for i, (tokens, valid) in enumerate(items):
        batch[i, : len(tokens)], mask[i, : len(tokens)] = tokens, valid
    return batch, mask
