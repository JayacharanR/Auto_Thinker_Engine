import numpy as np
import pytest
import torch

from src.navigation.signal_context import context_patches
from src.navigation.signal_tight_verifier import tight_views


def test_tight_views_use_native_pixels_and_keep_point_alignment_at_boundary():
    image = np.zeros((40, 40, 3), np.uint8)
    image[:, :, 0] = np.arange(40)[None, :]
    image[:, :, 1] = np.arange(40)[:, None]
    patches = torch.from_numpy(context_patches(image, np.array([[0, 0], [20, 20]])))
    views = tight_views(patches)
    assert views.shape == (4, 3, 128, 128)
    # Edge padding: negative coordinates reproduce the original border.
    assert views[0, :2, 0, 0].tolist() == [0, 0]
    torch.testing.assert_close(views[1, :2, 0, 0], torch.tensor([16, 16]) / 255)
    torch.testing.assert_close(views[3, :2, 0, 0], torch.tensor([12, 12]) / 255)
    torch.testing.assert_close(views[1, :2, -1, -1], torch.tensor([23, 23]) / 255)
    # The area-averaged context must not supply the new high-resolution views.
    patches[:, 3:] = 255
    torch.testing.assert_close(tight_views(patches), views, rtol=0, atol=0)


def test_tight_views_refuse_extra_inputs_and_float_patches():
    with pytest.raises(ValueError):
        tight_views(torch.zeros(1, 7, 32, 32, dtype=torch.uint8))
    with pytest.raises(ValueError):
        tight_views(torch.zeros(1, 6, 32, 32))
