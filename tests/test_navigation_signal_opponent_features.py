import numpy as np
import pytest

from src.navigation.signal_opponent_features import opponent_features


def test_relative_color_distinguishes_yellow_from_gray_without_metadata():
    x = np.zeros((2, 87), np.float32)
    x[0, :75] = np.repeat([0.8, 0.8, 0.2], 25)
    x[1, :75] = 0.8
    y = opponent_features(x)
    assert y.shape == (2, 312) and np.array_equal(x, y[:, :87])
    assert y[0, 87:112].max() == 0  # R-G: both channels high.
    assert y[0, 112:137].min() == pytest.approx(0.6)  # G-B distinguishes yellow.
    assert np.all(y[1, 87:162] == 0)  # Neutral gray has no color difference.
    assert np.allclose(y[:, 237:], 0, atol=1e-7)  # Uniform patches have no contrast.


def test_chromaticity_is_scale_invariant_and_black_is_finite():
    x = np.full((1, 87), 0.1, np.float32)
    x[:, :25] = 0.2
    a, b = opponent_features(x), opponent_features(x * 2)
    np.testing.assert_allclose(a[:, 162:237], b[:, 162:237], atol=1e-7)
    assert np.isfinite(opponent_features(np.zeros((2, 87), np.float32))).all()
    with pytest.raises(ValueError):
        opponent_features(np.zeros((1, 88), np.float32))
