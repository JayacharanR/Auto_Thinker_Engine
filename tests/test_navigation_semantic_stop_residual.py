import pytest
import torch

from src.navigation.semantic_stop_residual import SemanticStopResidual, semantic_features


def example():
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        prediction = dict(
            stop_logits=torch.randn(1, 20, 9), cruise_speeds=torch.rand(1, 20, 9) * 8,
            light=torch.randn(1, 20, 5), presence=torch.randn(1, 20, 2),
            distance=torch.rand(1, 20, 2) * 80, hazards=torch.randn(1, 20, 4),
            correction=torch.randn(1, 20, 8),
        )
        speed = torch.where(prediction["stop_logits"] >= 0, 0.0, prediction["cruise_speeds"])
        prediction.update(current_speed=speed[..., 0], speeds=speed[..., 1:],
                          brake=prediction["stop_logits"][..., 0])
        model = SemanticStopResidual()
    return model, prediction


def test_zero_initialization_exactly_preserves_the_frozen_parent():
    model, parent = example()
    result, _ = model(parent, semantic_features(parent))
    for key in parent:
        assert torch.equal(parent[key], result[key]), key
    result["stop_logits"].sum().backward()
    assert model.delta.weight.grad.abs().sum() > 0


def test_streaming_and_full_history_agree_without_future_leakage():
    model, parent = example()
    with torch.no_grad():
        model.delta.weight.fill_(0.1)
    facts = semantic_features(parent)
    full, _ = model(parent, facts)
    hidden, rows = None, []
    for t in range(20):
        step = {k: v[:, t:t + 1] for k, v in parent.items()}
        result, hidden = model(step, facts[:, t:t + 1], hidden)
        rows.append(result["stop_logits"])
    assert torch.allclose(full["stop_logits"], torch.cat(rows, 1), atol=1e-6)
    changed = facts.clone()
    changed[:, 10:] = 1 - changed[:, 10:]
    after, _ = model(parent, changed)
    assert torch.equal(full["stop_logits"][:, :10], after["stop_logits"][:, :10])
    assert not torch.equal(full["stop_logits"][:, 10:], after["stop_logits"][:, 10:])


def test_structured_features_bound_presence_state_products():
    _, parent = example()
    facts = semantic_features(parent)
    assert facts.shape == (1, 20, 17)
    assert ((facts >= 0) & (facts <= 1)).all()
    assert (facts[..., 13:16] <= parent["presence"].sigmoid()[..., :1]).all()


def test_invalid_features_cannot_be_silently_broadcast():
    model, parent = example()
    with pytest.raises(ValueError, match="17predicted"):
        model(parent, torch.zeros(1, 1, 17))
