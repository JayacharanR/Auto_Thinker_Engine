import numpy as np
import pytest

from src.navigation.signal_rgb_augmentation import GAMMAS, gamma_rgb


def test_photometric_transform_keeps_geometry_channel_order_and_identity():
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    assert np.array_equal(gamma_rgb(ramp, 1.), ramp)
    for gamma in GAMMAS:
        result = gamma_rgb(ramp, gamma)
        assert result.shape == ramp.shape and result.dtype == np.uint8
        assert result[0, 0, 0] == 0 and result[0, -1, 0] == 255
        assert (np.diff(result[0, :, 0].astype(int)) >= 0).all()
        rgb = np.array([[[250, 100, 25], [40, 240, 30]]], dtype=np.uint8)
        transformed = gamma_rgb(rgb, gamma)
        assert np.array_equal(rgb.argmax(-1), transformed.argmax(-1))


def test_transform_rejects_extra_channels_and_unregistered_gamma():
    with pytest.raises(ValueError):
        gamma_rgb(np.zeros((4, 4, 4), np.uint8), .7)
    with pytest.raises(ValueError):
        gamma_rgb(np.zeros((4, 4, 3), np.uint8), 2.)
