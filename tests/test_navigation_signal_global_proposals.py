import numpy as np
import pytest

from src.navigation.signal_global_proposals import global_rgb_points, plateau_rgb_points


def test_each_rgb_channel_is_equal_and_threshold_is_exact():
    rgb = np.zeros((288, 512, 3), np.uint8)
    rgb[10, 10, 0], rgb[20, 20, 1], rgb[30, 30, 2] = 16, 200, 255
    rgb[40, 40, 0] = 15
    assert np.array_equal(global_rgb_points(rgb), [[10, 10], [20, 20], [30, 30]])


def test_plateaus_and_border_not_truncated_flat_image_has_no_peak():
    rgb = np.full((288, 512, 3), 100, np.uint8)
    assert global_rgb_points(rgb).shape == (0, 2)
    rgb[0, :3] = 200
    assert np.array_equal(global_rgb_points(rgb), [[0, 0], [1, 0], [2, 0]])
    with pytest.raises(ValueError, match="Native"):
        global_rgb_points(rgb.astype(np.float32))


def test_plateau_representative_is_actual_centroid_nearest_member_and_all_regions_remain():
    rgb = np.zeros((288, 512, 3), np.uint8)
    rgb[0, :3] = 200
    rgb[10, 10, 0] = 16
    rgb[20, 20, 1] = 200
    rgb[30, 30, 2] = 255
    assert np.array_equal(plateau_rgb_points(rgb), [[1, 0], [10, 10], [20, 20], [30, 30]])
    rgb = np.zeros_like(rgb)
    # An L-shaped plateau: do not invent the center's missing pixel.
    rgb[10, 10, 0] = rgb[11, 10, 0] = rgb[10, 11, 0] = 255
    assert np.array_equal(plateau_rgb_points(rgb), [[10, 10]])
    # Ties use y before x. Different channels remain independent regions.
    rgb = np.zeros_like(rgb)
    rgb[10, 10:12, 0] = 255
    rgb[11, 11, 1] = 255
    assert np.array_equal(plateau_rgb_points(rgb), [[10, 10], [11, 11]])
