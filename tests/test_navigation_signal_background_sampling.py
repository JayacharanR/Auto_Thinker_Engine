import numpy as np
import pytest

from src.navigation.signal_background_sampling import append_fold_background, select_background


def test_sampling_is_deterministic_known_background_only_without_old_training_rows():
    points = np.array([[x, 10] for x in range(200)])
    background = np.zeros((288, 512), bool)
    background[10, 50:180] = True
    excluded = [(x, 10) for x in range(50, 60)]
    first = select_background(points, background, excluded, 42)
    assert len(first) == 64 and len(set(first)) == 64
    assert np.array_equal(first, select_background(points, background, excluded, 42))
    assert (first >= 60).all() and (first < 180).all()
    assert not np.array_equal(first, select_background(points, background, excluded, 43))
    assert len(select_background(points[:55], background, [], 1)) == 5


def test_every_gamma_version_excludes_heldout_fold_and_appends_only_background():
    original = np.full((3, 87), -1, np.float32)
    y = np.array([1, 2, 3])
    extra = np.zeros((3, 6, 87), np.float16)
    for version in range(3):
        extra[version] = np.arange(6)[:, None] + 100 * version
    folds = np.array([0, 1, 2, 0, 1, 2])
    x, new_y = append_fold_background(original, y, extra, folds, 1)
    assert x[3:, 0].tolist() == [0, 2, 3, 5, 100, 102, 103, 105, 200, 202, 203, 205]
    assert new_y.tolist() == [1, 2, 3] + [0] * 12
    with pytest.raises(ValueError, match="internal TRAIN fold"):
        append_fold_background(original, y, extra, folds, -1)
