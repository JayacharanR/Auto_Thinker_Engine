import numpy as np
import pytest

from src.navigation.signal_phase_supervision import (
    joint_candidates,
    phase_balanced_weights,
    phase_targets,
)


def test_phase_targets_are_training_labels_and_do_not_change_background():
    samples = [dict(phase=p, role="train") for p in ("red", "red_repeat", "green", "yellow")]
    rows = [dict(sample=i, role="train", label=1) for i in range(4)]
    data = dict(samples=samples, train=list(range(4)))
    assert phase_targets(rows, data, np.ones(4, int)).tolist() == [1, 1, 3, 2]
    rows[1]["label"] = 0
    assert phase_targets(rows, data, np.array([1, 0, 1, 1])).tolist() == [1, 0, 3, 2]
    with pytest.raises(ValueError, match="Only TRAIN"):
        phase_targets(rows, dict(data, train=[1, 2, 3]), np.array([1, 0, 1, 1]))


def test_state_weighting_preserves_half_total_mass_for_foreground():
    labels = np.array([0] * 20 + [1] * 8 + [2] * 3 + [3] * 2)
    weights = phase_balanced_weights(labels)
    assert weights.sum() == pytest.approx(len(labels))
    assert weights[labels == 0].sum() == pytest.approx(len(labels) / 2)
    for state in (1, 2, 3):
        assert weights[labels == state].sum() == pytest.approx(len(labels) / 6)
    with pytest.raises(ValueError):
        phase_balanced_weights(np.array([0, 1, 2]))


def test_joint_inference_needs_only_rgb_model_probabilities_and_original_points():
    peaks = [dict(x=1, y=2, cls=1, score=0.7), dict(x=5, y=6, cls=3, score=0.9)]
    result = joint_candidates(peaks, np.array([[0.4, 0.1, 0.45, 0.05], [0.8, 0.05, 0.05, 0.1]]))
    assert len(result) == 1 and result[0]["cls"] == 2
    assert result[0]["score"] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        joint_candidates(peaks, np.array([[0.4, 0.1, 0.45, 0.05]]))
