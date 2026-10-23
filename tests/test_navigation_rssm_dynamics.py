import torch

from src.navigation.rssm_dynamics import RSSMDynamics, prediction_objective


def test_rssm_actions_are_ordered_and_future_controls_cannot_change_earlier_predictions():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = RSSMDynamics(
        token_dim=12, spatial_tokens=6, hidden=32, stochastic=8, steps=4, controls_per_step=2
    )
    past = torch.randn(2, 6, 12)
    controls = torch.randn(2, 8, 3)
    a = model(past, controls)
    altered = controls.clone()
    altered[:, 4:] *= -3
    b = model(past, altered)
    assert torch.equal(a["future_tokens"][:, :2], b["future_tokens"][:, :2])
    reversed_ = model(past, controls.flip(1))
    assert not torch.allclose(a["future_tokens"], reversed_["future_tokens"])
    assert a["future_tokens"].shape == (2, 4, 6, 12)


def test_censored_costs_and_invalid_future_targets_never_train_safe_negatives():
    torch.manual_seed(1)
    model = RSSMDynamics(
        token_dim=12, spatial_tokens=6, hidden=32, stochastic=8, steps=4, controls_per_step=2
    )
    prediction = model(torch.randn(2, 6, 12), torch.randn(2, 8, 3))
    targets = torch.randn(2, 4, 6, 12)
    valid = torch.ones(2, 4)
    valid[:, 2:] = 0
    costs = torch.zeros(2, 4, 3)
    cost_mask = torch.zeros(2, 4)
    first, parts = prediction_objective(prediction, targets, valid, costs, cost_mask)
    changed = targets.clone()
    changed[:, 2:] = float("nan")
    second, _ = prediction_objective(prediction, changed, valid, costs + float("nan"), cost_mask)
    assert torch.equal(first, second) and parts["cost"] == 0
    kl = model.posterior_kl(prediction, targets, valid)
    (first + 0.01 * kl).backward()
    assert torch.isfinite(model.transition.weight_hh.grad).all()


def test_unknown_red_cost_can_be_masked_without_discarding_collision_or_progress():
    model = RSSMDynamics(
        token_dim=12, spatial_tokens=6, hidden=32, stochastic=8, steps=4, controls_per_step=2
    )
    past, controls = torch.randn(2, 6, 12), torch.randn(2, 8, 3)
    prediction = model(past, controls, conditioning=torch.zeros(2, 35))
    changed = model(past, controls, conditioning=torch.ones(2, 35))
    assert not torch.equal(prediction["outcome_logits"], changed["outcome_logits"])
    targets = torch.randn(2, 4, 6, 12)
    mask = torch.ones(2, 4, 3)
    mask[..., 1] = 0
    outcomes = torch.zeros(2, 4, 3)
    outcomes[..., 1] = float("nan")
    loss, parts = prediction_objective(prediction, targets, torch.ones(2, 4), outcomes, mask)
    loss.backward()
    assert torch.isfinite(loss) and parts["cost"] > 0
    assert model.condition[0].weight.grad is not None
