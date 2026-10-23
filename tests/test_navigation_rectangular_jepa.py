"""Rectangular positions must preserve axes, source parity and past-only windows."""

import pytest
import torch

from src.jepa.encoder import ViTEncoder
from src.navigation.rectangular_jepa import RectangularJEPA, causal_clip_indices, grid_positions


def test_same_token_count_with_different_aspect_is_not_a_position_fast_path():
    coordinates = torch.zeros(1, 2, 2, 4, 3)
    for t in range(2):
        for y in range(2):
            for x in range(4):
                coordinates[0, t, y, x] = torch.tensor([t * 10, y * 100, x])
    position = coordinates.flatten(1, 3).requires_grad_()
    assert grid_positions(position, (2, 2, 4), (2, 2, 4)) is position
    changed = grid_positions(position, (2, 2, 4), (2, 4, 2)).reshape(1, 2, 4, 2, 3)
    assert torch.all(changed[0, 0, :, :, 0] == 0) and torch.all(changed[0, 1, :, :, 0] == 10)
    assert changed[0, 0, 0, 0, 2].detach().item() == pytest.approx(0.5)
    assert changed[0, 0, 3, 1, 2].detach().item() == pytest.approx(2.5)
    assert not torch.equal(changed.flatten(1, 3), position)
    changed.sum().backward()
    assert torch.isfinite(position.grad).all() and position.grad.abs().sum() > 0


def test_rectangular_encoder_preserves_source_outputs_masks_and_spatial_structure():
    torch.set_num_threads(2)
    torch.manual_seed(8)
    base = ViTEncoder(img_size=32, patch_size=8, num_frames=4, embed_dim=24, depth=2, num_heads=3)
    navigation = RectangularJEPA(base, (2, 4, 4))
    video = torch.randn(2, 3, 4, 32, 32)
    assert torch.equal(base(video), navigation(video))
    mask = torch.tensor([[0, 3, 6], [1, 5, 7]])
    assert torch.equal(base(video, mask), navigation(video, mask))
    rectangle = torch.randn(2, 3, 4, 16, 64)
    assert navigation(rectangle).shape == (2, 32, 24)
    assert navigation.spatial_tokens(rectangle, (2, 4)).shape == (2, 8, 24)
    with pytest.raises(ValueError, match="divide exactly"):
        navigation(torch.randn(2, 3, 4, 17, 64))
    with pytest.raises(ValueError, match="outside"):
        navigation(rectangle, torch.tensor([[32], [1]]))


def test_causal_clips_never_cross_episode_start_or_see_a_future_image():
    indices = causal_clip_indices(12, 10, length=8)
    assert indices == [10, 10, 10, 10, 10, 10, 11, 12]
    images = torch.arange(20)
    before = images[indices].clone()
    images[13:] = 1000
    assert torch.equal(before, images[indices])
    assert causal_clip_indices(20, 0, length=4, stride=2) == [14, 16, 18, 20]
    with pytest.raises(ValueError):
        causal_clip_indices(5, 6)
