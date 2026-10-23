import numpy as np

from scripts.summarize_navigation_factual_dynamics import constant_baselines


def test_constant_prior_uses_only_train_and_masks_nan_before_arithmetic():
    outcomes = np.zeros((4, 8, 3), dtype=np.float32)
    valid = np.ones_like(outcomes)
    outcomes[0, :, 0] = 1
    outcomes[2, :, 0] = 1
    outcomes[3, 0, 0] = np.nan
    valid[3, 0, 0] = 0
    windows = dict(outcomes=outcomes, cost_valid=valid,
                   is_validation=np.array([False, False, True, True]),
                   episode_id=np.arange(4))
    result = constant_baselines(windows)
    assert result["train_prior_probability_by_horizon"] == [0.5] * 8
    assert result["always_zero_brier_by_horizon"] == [1.0] + [0.5] * 7
    assert np.isfinite(result["train_prior_collision_bce"])
    assert result["train_prior_brier_by_horizon"] == [0.25] * 8
    outcomes[2, :, 0] = 0  # VAL outcomes cannot alter the fitted constant prior.
    assert constant_baselines(windows)["train_prior_probability_by_horizon"] == [0.5] * 8
