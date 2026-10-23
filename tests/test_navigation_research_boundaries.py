"""Budget/leakage/causality checks before any M4/M5 experiment can be claimed."""

import copy

import pytest
import torch

from src.navigation.label_budget import nested_budgets
from src.navigation.ordered_dynamics import OrderedJEPADynamics, independent_ensemble


def groups():
    return [
        dict(
            id=f"episode{i:03d}", role="train", partition="train", town="Town01_Opt", labels=50 + i
        )
        for i in range(30)
    ]


def test_nested_whole_annotation_budgets_do_not_exceed_caps_or_resplit_locations():
    result = nested_budgets(groups())
    old = set()
    weights = {g["id"]: g["labels"] for g in groups()}
    for a in result["allocations"]:
        assert old <= set(a["groups"])
        assert sum(weights[g] for g in a["groups"]) == a["actual_labels"] <= a["maximum_labels"]
        assert a["actual_fraction"] <= a["nominal_fraction"]
        old = set(a["groups"])
    assert old == set(weights) and result == nested_budgets(groups())
    for change in (dict(role="validation"), dict(partition="heldout_dev"), dict(town="Town05_Opt")):
        changed = groups()
        changed[0].update(change)
        with pytest.raises(ValueError, match="TRAIN"):
            nested_budgets(changed)


def test_later_candidate_actions_cannot_change_an_earlier_future_prediction():
    torch.set_num_threads(2)
    torch.manual_seed(2)
    model = OrderedJEPADynamics(
        token_dim=24, spatial_tokens=6, hidden=32, steps=3, controls_per_step=4
    ).eval()
    context, actions = torch.randn(2, 6, 24), torch.randn(2, 12, 3)
    a = model(context, actions)
    changed = actions.clone()
    changed[:, 8:] += 20
    b = model(context, changed)
    assert torch.equal(a["future_tokens"][:, :2], b["future_tokens"][:, :2])
    assert torch.equal(a["outcome_logits"][:, :2], b["outcome_logits"][:, :2])
    assert not torch.allclose(a["future_tokens"][:, 2], b["future_tokens"][:, 2])
    reversed_actions = model(context, actions.flip(1))
    assert not torch.allclose(a["future_tokens"][:, -1], reversed_actions["future_tokens"][:, -1])
    assert a["future_tokens"].shape == (2, 3, 6, 24)


def test_ensemble_owns_separate_parameters_and_different_initializations():
    caller_rng = torch.get_rng_state().clone()
    models = independent_ensemble(token_dim=24, spatial_tokens=6, hidden=32)
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert len(models) == 3
    assert models[0].query.data_ptr() != models[1].query.data_ptr()
    assert not torch.equal(models[0].query, models[1].query)
    before = copy.deepcopy(models[1].state_dict())
    with torch.no_grad():
        models[0].query.add_(1)
    assert all(torch.equal(v, models[1].state_dict()[k]) for k, v in before.items())
