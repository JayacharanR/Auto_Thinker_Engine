import numpy as np
import pytest
import torch

from src.navigation.signal_pixel_study import NativeSignalGrid, point_candidates


def test_single_rgb_native_resolution_translation_and_no_extra_inputs():
    torch.set_num_threads(1)
    model = NativeSignalGrid().eval()
    rgb = torch.randint(0, 255, (1, 3, 48, 48), dtype=torch.uint8)
    with torch.inference_mode():
        full = model(rgb)
        crop = model(rgb[:, :, 3:45, 2:44])
    assert full.shape == (1, 4, 48, 48)
    torch.testing.assert_close(full[:, :, 13:35, 12:34], crop[:, :, 10:32, 10:32])
    with pytest.raises(ValueError):
        model(torch.zeros(1, 4, 48, 48))


def test_connected_peak_keeps_separate_objects_and_merges_adjacent_pixels():
    p = np.zeros((4, 10, 12), np.float32)
    p[0] = 1
    p[1, 2, 2:4] = [0.6, 0.9]
    p[3, 7, 9] = 0.8
    candidates = point_candidates(p)
    assert [(c["x"], c["y"], c["cls"]) for c in candidates] == [(3, 2, 1), (9, 7, 3)]
    assert not point_candidates(np.zeros_like(p))
