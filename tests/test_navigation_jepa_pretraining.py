import numpy as np
import torch

from src.navigation.jepa_pretraining import (
    sample_clips,
    tube_masks,
    unlabeled_split,
    variance_diagnostics,
)


def test_masks_partition_actual_grid_and_repeat_spatially():
    context, masked = tube_masks(4, np.random.default_rng(42), "cpu")
    assert context.shape == masked.shape == (4, 288)
    for visible, hidden in zip(context, masked):
        assert sorted(torch.cat([visible, hidden]).tolist()) == list(range(576))
        for index in (visible, hidden):
            assert all(
                torch.equal(index[:72], index[t * 72 : (t + 1) * 72] - t * 144) for t in range(4)
            )


def test_unsupervised_holdout_never_uses_driving_validation():
    episodes = [
        dict(id=str(i), town="Town03", junction_ids=[i], partition="train", role="train")
        for i in range(10)
    ]
    episodes.append(
        dict(
            id="drivingval", town="Town03", junction_ids=[500], partition="train", role="validation"
        )
    )
    train, held = unlabeled_split(dict(episodes=episodes))
    assert len(train) == 8 and len(held) == 2
    assert {e["id"] for e in train}.isdisjoint(e["id"] for e in held)
    assert all(e["id"] != "drivingval" for e in train + held)


def test_pretrain_clips_are_past_only_and_keep_views():
    class Raw:
        def rgb(self, episode):
            x = np.zeros((20, 3, 3, 2, 2), np.uint8)
            x[:] = np.arange(20)[:, None, None, None, None]
            return x

    clips, sources = sample_clips(Raw(), [dict(id="e")], np.random.default_rng(1), 2)
    assert clips.shape == (6, 3, 8, 2, 2)
    for i, source in enumerate(sources):
        assert clips[i * 3 : i * 3 + 3, :, -1].max() == source["endpoint"]
        assert clips[i * 3 : i * 3 + 3].max() <= source["endpoint"]
    positional_only = torch.randn(1, 10, 8).expand(12, -1, -1)
    diagnostics = variance_diagnostics(positional_only)
    assert diagnostics["within_clip_variance"] > 0
    assert diagnostics["between_scene_variance"] == 0
