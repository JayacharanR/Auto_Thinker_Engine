import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "staged_spatial_context",
    Path(__file__).resolve().parents[1] / "patches/navigation_signal_spatial_context_candidate.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reference(rgb, points):
    padded = np.pad(rgb, ((48, 48), (48, 48), (0, 0)), mode="edge")
    result = []
    for x, y in points:
        features = []
        for scale in (16, 32, 96):
            cell = scale // 4
            for row in range(4):
                for col in range(4):
                    left, top = x + 48 - scale // 2 + col * cell, y + 48 - scale // 2 + row * cell
                    patch = padded[top : top + cell, left : left + cell].astype(np.float64)
                    features.extend((patch.mean((0, 1)) / 255).tolist())
                    features.extend((patch.std((0, 1)) / 255).tolist())
        result.append(features)
    return np.asarray(result, np.float32)


def test_integral_descriptor_matches_direct_pixels_at_borders_and_all_scales():
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (288, 512, 3), np.uint8)
    points = np.array([[0, 0], [511, 287], [0, 287], [511, 0], [257, 134], [257, 134]], np.uint16)
    np.testing.assert_array_equal(module.spatial_context(rgb, points), reference(rgb, points))
    for value in (0, 255):
        uniform = np.full((1, 1, 3), value, np.uint8)
        np.testing.assert_array_equal(
            module.spatial_context(uniform, np.array([[0, 0]])), reference(uniform, [[0, 0]])
        )


def test_descriptor_preserves_spatial_arrangement_hidden_by_global_moments():
    left = np.zeros((96, 96, 3), np.uint8)
    left[:, :48, 0] = 255
    right = left[:, ::-1].copy()
    np.testing.assert_array_equal(left.mean((0, 1)), right.mean((0, 1)))
    assert not np.array_equal(
        module.spatial_context(left, [[48, 48]]), module.spatial_context(right, [[48, 48]])
    )
    assert module.spatial_context(left, np.empty((0, 2), np.int64)).shape == (0, 288)
    for invalid in ([[96, 0]], [[-1, 0]], [[1.5, 1]], [[1, 1, 2]]):
        with pytest.raises(ValueError):
            module.spatial_context(left, invalid)
