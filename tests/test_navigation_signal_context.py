import numpy as np
import pytest
import torch

from src.navigation.signal_context import (
    SignalContextVerifier,
    context_patches,
    label_proposals,
    verified_candidates,
)


def test_context_has_native_center_and_no_boundary_wrap():
    rgb = np.zeros((100, 120, 3), np.uint8)
    rgb[50, 60] = [120, 60, 30]
    rgb[-1, -1] = 255
    patches = context_patches(rgb, np.array([[60, 50], [0, 0]]))
    assert patches.shape == (2, 6, 32, 32)
    assert patches[0, :3, 16, 16].tolist() == [120, 60, 30]
    assert patches[0, 3:, 16, 16].tolist() == [13, 7, 3]
    assert not patches[1].any()
    with pytest.raises(ValueError):
        context_patches(rgb, np.array([[120, 50]]))


def test_unknown_regions_and_ambiguous_housings_never_become_negatives():
    background = np.ones((10, 10), bool)
    background[3, 3] = False
    peaks = [dict(x=x, y=x) for x in [1, 3, 5, 8]]
    boxes = [[4, 4, 7, 7, 1], [4, 4, 6, 6, 3], [7, 7, 9, 9, 2]]
    assert label_proposals(peaks, boxes, background).tolist() == [0, -1, -1, 2]


def test_verifier_scores_all_candidates_and_accepts_only_rgb_patches():
    model = SignalContextVerifier()
    assert model(torch.zeros(2, 6, 32, 32)).shape == (2, 4)
    with pytest.raises(ValueError):
        model(torch.zeros(2, 7, 32, 32))
    peaks = [dict(x=1, y=1), dict(x=2, y=2)]
    result = verified_candidates(peaks, np.array([[0.1, 0.1, 0.1, 0.7], [0.7, 0.1, 0.1, 0.1]]))
    assert len(result) == 1 and result[0]["cls"] == 3
    with pytest.raises(ValueError):
        verified_candidates(peaks, np.ones((1, 4)))
