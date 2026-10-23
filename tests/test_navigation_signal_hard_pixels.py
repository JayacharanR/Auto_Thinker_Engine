import torch

from src.navigation.signal_hard_pixels import hard_pixel_loss
from src.navigation.signal_model import signal_loss


def test_rare_confident_background_is_not_diluted_by_easy_pixels():
    logits = torch.full((1, 4, 10, 10), -8.0)
    logits[:, 0] = 8
    logits[0, 0, 0, 0] = -8
    logits[0, 1, 0, 0] = 8
    target = torch.zeros((1, 10, 10), dtype=torch.long)
    assert hard_pixel_loss(logits, target) > 40 * signal_loss(logits, target)


def test_ignored_lights_have_no_gradient_and_empty_support_is_finite():
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.zeros((2, 8, 8), dtype=torch.long)
    target[:, :4] = 255
    target[0, 5, 5] = 1
    hard_pixel_loss(logits, target).backward()
    assert not logits.grad[:, :, :4].any()
    assert torch.isfinite(logits.grad).all()
    loss = hard_pixel_loss(logits, torch.full_like(target, 255))
    assert loss.item() == 0
