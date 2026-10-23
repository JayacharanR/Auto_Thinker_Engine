import pytest
import torch

from src.navigation.spatial_dynamics_v2 import SpatialOrderedJEPADynamics


def example():
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261020)
        model = SpatialOrderedJEPADynamics(hidden=32).eval()
        tokens, controls = torch.randn(2, 48, 384), torch.randn(2, 80, 3)
        condition = torch.randn(2, 35)
    return model, tokens, controls, condition


def test_risk_uses_camera_identity_and_image_position():
    model, tokens, controls, condition = example()
    original = model(tokens, controls, condition)["outcome_logits"]
    swapped_views = tokens.reshape(2, 3, 16, 384).flip(1).reshape_as(tokens)
    swapped_pixels = tokens.reshape(2, 3, 16, 384).flip(2).reshape_as(tokens)
    assert not torch.allclose(original, model(swapped_views, controls, condition)["outcome_logits"])
    assert not torch.allclose(
        original, model(swapped_pixels, controls, condition)["outcome_logits"]
    )


def test_collision_gradient_reaches_attended_visual_future():
    model, tokens, controls, condition = example()
    model(tokens.requires_grad_(), controls, condition)["outcome_logits"][..., 0].sum().backward()
    # In v1 the decoder attention is disconnected from every risk output.
    assert model.attend.in_proj_weight.grad.abs().sum() > 0
    assert model.cost_attention.in_proj_weight.grad.abs().sum() > 0
    assert model.position[0].weight.grad.abs().sum() > 0
    assert tokens.grad.abs().sum() > 0


def test_later_controls_cannot_change_earlier_risk_or_other_candidate():
    model, tokens, controls, condition = example()
    original = model(tokens, controls, condition)
    changed = controls.clone()
    changed[0, 40:] = -changed[0, 40:]
    after = model(tokens, changed, condition)
    assert torch.equal(original["outcome_logits"][:, :4], after["outcome_logits"][:, :4])
    assert torch.equal(original["outcome_logits"][1], after["outcome_logits"][1])
    assert not torch.equal(original["outcome_logits"][0, 4:], after["outcome_logits"][0, 4:])
    assert torch.equal(original["future_tokens"][:, :4], after["future_tokens"][:, :4])


def test_no_future_target_or_incomplete_grid_can_enter_forward():
    model, tokens, controls, condition = example()
    with pytest.raises(TypeError):
        model(tokens, controls, condition, future_tokens=tokens)
    with pytest.raises(ValueError, match="grids"):
        model(tokens[:, :16], controls, condition)
    with pytest.raises(ValueError, match="navigation"):
        model(tokens, controls, condition[:, :32])
