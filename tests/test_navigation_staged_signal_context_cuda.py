import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.navigation.signal_context import context_patches
from src.navigation.signal_pretrained_verifier import rgb_features


def load_staged(name):
    path = Path(__file__).resolve().parents[1] / "patches" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires authoritative host GPU")
def test_cuda_sequential_reductions_exactly_match_original_float32_features():
    prepare = load_staged("navigation_signal_context_fast_candidate").PreparedContext
    gpu = load_staged("navigation_signal_context_cuda_candidate").cuda_features
    rng = np.random.default_rng(44)
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
        actual = gpu(prepare(rgb), points).cpu().numpy()
        np.testing.assert_array_equal(actual, original)
        assert gpu(prepare(rgb), np.empty((0, 2), np.int64)).shape == (0, 87)
