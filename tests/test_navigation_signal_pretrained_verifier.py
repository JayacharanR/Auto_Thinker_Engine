import numpy as np
import pytest

from src.navigation.signal_pretrained_verifier import retain_original_states, rgb_features


def test_rgb_features_keep_center_color_without_absolute_coordinates():
    patches = np.zeros((1, 6, 32, 32), np.uint8)
    patches[0, 0, 16, 16] = 255
    result = rgb_features(patches)
    assert result.shape == (1, 87) and result[0, :75].sum() == 1
    assert result[0, 12] == 1
    with pytest.raises(ValueError):
        rgb_features(np.zeros((1, 7, 32, 32), np.uint8))


def test_binary_verification_cannot_change_original_phase_or_hide_unscored_points():
    peaks = [dict(x=1, y=1, cls=2, score=0.7), dict(x=3, y=4, cls=3, score=0.9)]
    result = retain_original_states(peaks, np.array([0.8, 0.4]))
    assert result == [dict(x=1, y=1, cls=2, score=0.8)]
    with pytest.raises(ValueError):
        retain_original_states(peaks, np.array([0.8]))
    with pytest.raises(ValueError):
        retain_original_states(peaks, np.array([0.8, np.nan]))


def test_real_frozen_encoder_preserves_batchnorm_and_has_no_training_parameters():
    import json
    from pathlib import Path

    import torch

    from src.navigation.signal_pretrained_verifier import FrozenSignalResNet

    root = Path(__file__).resolve().parents[1]
    descriptor = json.loads(
        (root / "outputs/navigation_models/navigation_backbones_20261019.json").read_text()
    )["imagenet_resnet18"]
    torch.set_num_threads(2)
    model = FrozenSignalResNet(descriptor)
    before = {k: v.clone() for k, v in model.named_buffers()}
    model.train(True)
    assert not any(m.training for m in model.modules())
    assert not any(p.requires_grad for p in model.parameters())
    output = model(torch.full((2, 6, 32, 32), 128, dtype=torch.uint8))
    assert output.shape == (2, 1024) and torch.isfinite(output).all()
    assert all(torch.equal(v, dict(model.named_buffers())[k]) for k, v in before.items())
