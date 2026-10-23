import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.navigation.signal_context import context_patches
from src.navigation.signal_pretrained_verifier import rgb_features

spec = importlib.util.spec_from_file_location(
    "staged_context_fast",
    Path(__file__).resolve().parents[1] / "patches/navigation_signal_context_fast_candidate.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_all_alignment_phases_and_edges_reproduce_original_patch_bytes_and_features():
    rng = np.random.default_rng(42)
    for shape in ((1, 1, 3), (7, 11, 3), (288, 512, 3)):
        rgb = rng.integers(0, 256, shape, dtype=np.uint8)
        h, w = shape[:2]
        # Exhaust the small image and include every alignment phase, four corners,
        # repeated points and random interiors in the native-sized image.
        if h < 10:
            y, x = np.mgrid[:h, :w]
            points = np.stack([x.ravel(), y.ravel()], -1)
        else:
            y, x = np.mgrid[50:56, 60:66]
            points = np.concatenate(
                [
                    np.stack([x.ravel(), y.ravel()], -1),
                    [[0, 0], [w - 1, h - 1], [0, h - 1], [w - 1, 0], [0, 0]],
                    np.stack([rng.integers(w, size=100), rng.integers(h, size=100)], -1),
                ]
            ).astype(np.uint16)
        expected = context_patches(rgb, points)
        actual = module.PreparedContext(rgb).patches(points)
        assert actual.strides == expected.strides
        np.testing.assert_array_equal(actual, expected)
        original = rgb_features(expected)
        for chunk in (1, 7, 256):
            np.testing.assert_array_equal(module.exact_rgb_features(rgb, points, chunk), original)


def test_uniform_extreme_pixels_rounding_and_input_bounds():
    for value in (0, 1, 127, 128, 254, 255):
        rgb = np.full((50, 60, 3), value, np.uint8)
        prepared = module.PreparedContext(rgb)
        assert (prepared.patches(np.array([[0, 0], [59, 49]])) == value).all()
        assert prepared.patches(np.empty((0, 2), np.int64)).shape == (0, 6, 32, 32)
        for points in ([[60, 0]], [[-1, 0]], [[0, 50]], [[1.5, 2]]):
            with pytest.raises(ValueError):
                prepared.patches(points)
    # Every possible integer sum of nine uint8 pixels has the same rounded mean.
    sums = np.arange(9 * 255 + 1)
    np.testing.assert_array_equal((sums + 4) // 9, np.rint(sums.astype(float) / 9))
    with pytest.raises(ValueError):
        module.PreparedContext(np.zeros((5, 5, 4), np.uint8))
