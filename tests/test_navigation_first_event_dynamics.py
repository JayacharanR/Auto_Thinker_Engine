import math

import numpy as np
import pytest
import torch

from scripts.train_navigation_first_event_study import mixture_sequence_nll
from src.navigation.first_event_dynamics import FirstEventDynamics, hazard_nll


def test_survival_likelihood_sums_prefix_instead_of_reweighting_early_event():
    logits = torch.zeros(2, 80, requires_grad=True)
    event = torch.zeros_like(logits)
    event[0, 1] = 1
    valid = torch.ones(2, 80, dtype=torch.bool)
    valid[0, 2:] = False
    assert hazard_nll(logits, event, valid).item() == pytest.approx(41 * math.log(2))
    event[0, 2:] = float("nan")
    loss = hazard_nll(logits, event, valid)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[0, 2:].abs().sum() == 0
    expected = mixture_sequence_nll(
        np.zeros((3, 2, 80)),
        [
            dict(event=event[0].numpy(), valid=valid[0].numpy()),
            dict(event=event[1].numpy(), valid=valid[1].numpy()),
        ],
    )
    assert expected == pytest.approx(41 * math.log(2))


@pytest.mark.parametrize("kind", ["jepa", "rssm"])
def test_causal_controls_spatial_risk_gradient_and_unknown_objects(kind):
    torch.manual_seed(42)
    model = FirstEventDynamics(kind, token_dim=12, hidden=16, stochastic=4)
    args = [
        torch.randn(1, 48, 12),
        torch.zeros(1, 80, 3),
        torch.zeros(1, 41),
        torch.zeros(1, 8, 3, 100, 20),
        torch.zeros(1, 8, 3, 100, dtype=torch.bool),
        torch.zeros(1, 8, 3),
    ]
    a = model(*args)
    changed = [x.clone() for x in args]
    changed[1][:, 40:, 0] = 0.5
    b = model(*changed)
    torch.testing.assert_close(a["logits"][:, :40], b["logits"][:, :40], rtol=0, atol=0)
    assert not torch.allclose(a["logits"][:, 40:], b["logits"][:, 40:])
    changed = [x.clone() for x in args]
    changed[3].fill_(999.0)  # Absent boxes cannot influence predictions.
    c = model(*changed)
    torch.testing.assert_close(a["logits"], c["logits"], rtol=0, atol=0)
    assert (a["cumulative"][:, 1:] >= a["cumulative"][:, :-1]).all()
    a["logits"][0, 20].backward()
    assert model.decode.weight.grad.abs().sum() > 0
