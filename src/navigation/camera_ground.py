"""Approximate local-ground ranges from declared camera rays, without simulator state."""

import numpy as np


def mount_rotation(calibration):
    pitch, yaw, roll = np.deg2rad(calibration.pitch_yaw_roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ]
    )


def ground_contact(box, calibration, maximum_range_m=60.0):
    """Bottom-center ray/flat local z=0 intersection; unknown when invalid.

    Ego origin is approximated at the road plane. Grades, ego pitch/height,
    truncated/occluded boxes and non-ground contact create systematic error.
    This is not stereo depth or measured3D actor position.
    """
    box = np.asarray(box, float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Finite image box required")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= calibration.width and 0 <= y0 < y1 < calibration.height - 2):
        return None
    intrinsic = calibration.intrinsic
    direction = (
        np.array(
            [
                1.0,
                ((x0 + x1) * 0.5 - intrinsic[0, 2]) / intrinsic[0, 0],
                -(y1 - intrinsic[1, 2]) / intrinsic[1, 1],
            ]
        )
        @ mount_rotation(calibration).T
    )
    origin = np.array(calibration.xyz, float)
    if direction[2] >= -1e-4 or origin[2] <= 0:
        return None
    scale = -origin[2] / direction[2]
    point = origin + scale * direction
    if np.linalg.norm(point[:2]) > maximum_range_m:
        return None
    return point[:2]
