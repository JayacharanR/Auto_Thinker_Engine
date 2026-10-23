import numpy as np

from src.navigation.dynamics_learning import member_population, sample_batch, tensor_batch


def synthetic_windows():
    count = 80
    return dict(
        is_validation=np.arange(count) >= 64,
        episode_id=np.arange(count) // 8,
        input_index=np.arange(count),
        future_indices=np.tile(np.arange(8), (count, 1)),
        controls=np.zeros((count, 80, 3), np.float32),
        conditioning=np.zeros((count, 35), np.float32),
        latent_valid=np.ones((count, 8), np.float32),
        outcomes=np.zeros((count, 8, 3), np.float32),
        cost_valid=np.ones((count, 8, 3), np.float32),
    )


def test_bootstrap_keeps_train_contact_stratum_and_draws_match_across_models():
    windows = synthetic_windows()
    windows["outcomes"][8:16, :, 0] = 1
    for seed in (42, 1051, 2060):
        groups, positives, episodes = member_population(windows, seed)
        assert len(episodes) == 8 and 1 in episodes and len(positives) > 0
        first = sample_batch(groups, positives, np.random.default_rng(seed + 17))
        second = sample_batch(groups, positives, np.random.default_rng(seed + 17))
        assert np.array_equal(first, second)
        assert not windows["is_validation"][first].any()
        assert (windows["outcomes"][first[8:], :, 0] == 1).any(1).all()


def test_future_and_privileged_outcomes_are_separate_from_forward_inputs():
    windows = synthetic_windows()
    features = np.zeros((80, 48, 384), np.float16)
    inputs, targets = tensor_batch(windows, features, [0, 1], "cpu")
    assert set(inputs) == {"past_tokens", "ordered_controls", "conditioning"}
    assert targets["future_tokens"].shape == (2, 8, 48, 384)
    assert not targets["cost_valid"][..., 1].any()
    assert windows["cost_valid"][..., 1].all()  # Loss masking must not rewrite original labels.
