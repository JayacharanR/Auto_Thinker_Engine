import numpy as np
import pytest

from scripts.train_navigation_signal_rgb_boost import fuse_proposer_training


def test_fusion_keeps_augmented_rows_aligned_and_excluded_locations_out_of_fit():
    original = np.repeat(np.arange(4, dtype=np.float32)[:, None], 87, axis=1)
    mask = np.array([True, False, True, False])
    augmented = np.concatenate([original[mask] + offset for offset in (0, 10, 20)])
    learned = np.stack(
        [np.repeat((np.arange(4) + offset)[:, None], 36, axis=1) for offset in (100, 200, 300)]
    ).astype(np.float32)
    fit, inference = fuse_proposer_training(original, augmented, learned, mask)
    assert fit.shape == (6, 123) and inference.shape == (4, 123)
    assert fit[:, 0].tolist() == [0, 2, 10, 12, 20, 22]
    assert fit[:, 87].tolist() == [100, 102, 200, 202, 300, 302]
    assert inference[:, 87].tolist() == [100, 101, 102, 103]
    with pytest.raises(ValueError, match="alignment"):
        fuse_proposer_training(original, augmented[:-1], learned, mask)
    with pytest.raises(ValueError, match="alignment"):
        fuse_proposer_training(original, augmented, learned[:, :-1], mask)
