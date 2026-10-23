import numpy as np
import pytest

from src.navigation.signal_rgb_refinement import refine_rgb_points


def test_all_original_points_remain_and_each_channel_can_add_a_point():
    rgb = np.zeros((288, 512, 3), np.uint8)
    rgb[9, 10, 0] = 255
    rgb[10, 12, 1] = 255
    rgb[11, 9, 2] = 255
    got = refine_rgb_points(rgb, np.array([[10, 10], [100, 100], [10, 10]]))
    assert {tuple(p) for p in got} == {(10, 10), (100, 100), (10, 9), (12, 10), (9, 11)}
    assert np.array_equal(got, got[np.lexsort((got[:, 0], got[:, 1]))])


def test_ties_edges_and_empty_are_deterministic_without_fake_padding_peaks():
    rgb = np.zeros((288, 512, 3), np.uint8)
    points = np.array([[0, 0], [511, 287]])
    assert np.array_equal(refine_rgb_points(rgb, points), points)
    rgb[0, 1] = 50
    rgb[1, 0] = 50
    assert np.array_equal(refine_rgb_points(rgb, np.array([[0, 0]])), [[0, 0], [1, 0]])
    assert refine_rgb_points(rgb, np.empty((0, 2), np.int64)).shape == (0, 2)
    with pytest.raises(ValueError, match="outside"):
        refine_rgb_points(rgb, np.array([[512, 0]]))
