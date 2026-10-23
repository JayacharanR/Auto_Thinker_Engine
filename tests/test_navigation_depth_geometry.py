"""Analytic box test distinguishes axial depth from a radial misinterpretation."""

import numpy as np

from scripts.probe_navigation_depth_geometry import outside_box, pixel_points
from src.navigation.contracts import CameraCalibration


def test_off_axis_depth_cannot_be_interpreted_as_range():
    cal = CameraCalibration(512, 288, 100.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    mask = np.zeros((288, 512), bool)
    mask[144, 436] = True
    depth = np.full(mask.shape, 25.0)
    correct = pixel_points(depth, mask, cal)
    transform = np.eye(4)
    transform[:3, 3] = correct[0]
    extent = np.array([2.0, 1.0, 1.0])
    assert outside_box(correct, np.eye(4), transform, extent)[0] == 0
    wrong = pixel_points(depth, mask, cal, radial=True)
    assert np.isclose(np.linalg.norm(wrong[0]), 25)
    assert outside_box(wrong, np.eye(4), transform, extent)[0] > 4
    # Translate/rotate both native frames: the geometric comparison must be invariant.
    basis = np.array(
        [
            [0.0, -1.0, 0.0, 100.0],
            [1.0, 0.0, 0.0, -42.0],
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(
        outside_box(wrong, basis, basis @ transform, extent),
        outside_box(wrong, np.eye(4), transform, extent),
    )
