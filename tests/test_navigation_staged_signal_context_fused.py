import importlib.util
from pathlib import Path

import numpy as np

from src.navigation.signal_context import context_patches
from src.navigation.signal_pretrained_verifier import rgb_features


def load_staged(name):
    path = Path(__file__).resolve().parents[1] / "patches" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compiled_reductions_preserve_float32_order_not_just_fp16_rounding():
    prepare = load_staged("navigation_signal_context_fast_candidate").PreparedContext
    fused = load_staged("navigation_signal_context_fused_candidate").fused_features
    rng = np.random.default_rng(43)
    for shape in ((1, 1, 3), (9, 13, 3), (288, 512, 3)):
        rgb = rng.integers(0, 256, shape, dtype=np.uint8)
        h, w = shape[:2]
        points = np.concatenate(
            [
                np.stack([rng.integers(w, size=200), rng.integers(h, size=200)], -1),
                [[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]],
            ]
        )
        original = rgb_features(context_patches(rgb, points))
        actual = fused(prepare(rgb), points)
        np.testing.assert_array_equal(actual, original)
        assert fused(prepare(rgb), np.empty((0, 2), np.int64)).shape == (0, 87)
