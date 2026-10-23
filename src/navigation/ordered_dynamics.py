"""Past-only JEPA prediction primitive with ordered candidate control sequences.

This module is not a trained planner. Future target embeddings and outcome
labels belong to a separate training loss and never enter forward inference.
"""

import torch
from torch import nn


class OrderedJEPADynamics(nn.Module):
    def __init__(self, token_dim=384, spatial_tokens=96, hidden=128, steps=8, controls_per_step=10):
        super().__init__()
        self.spatial_tokens, self.steps = spatial_tokens, steps
        self.controls_per_step = controls_per_step
        self.context = nn.Linear(token_dim, hidden)
        self.condition = nn.Sequential(nn.Linear(35, hidden), nn.SiLU())
        self.controls = nn.GRU(3, hidden, batch_first=True)
        self.transition = nn.GRUCell(hidden, hidden)
        self.query = nn.Parameter(torch.randn(spatial_tokens, hidden) * 0.02)
        self.attend = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.decode = nn.Linear(hidden, token_dim)
        self.cost = nn.Linear(hidden, 3)  # Collision, red crossing, signed route progress.

    def forward(self, past_tokens, ordered_controls, conditioning=None):
        if past_tokens.ndim != 3 or past_tokens.shape[1] != self.spatial_tokens:
            raise ValueError("Explicit past spatial token grid required")
        if ordered_controls.ndim != 3 or ordered_controls.shape != (
            past_tokens.shape[0],
            self.steps * self.controls_per_step,
            3,
        ):
            raise ValueError("Ordered steer/throttle/brake control horizon required")
        if not torch.isfinite(past_tokens).all() or not torch.isfinite(ordered_controls).all():
            raise ValueError("Dynamics inputs must be finite")
        encoded = self.context(past_tokens)
        action, _ = self.controls(ordered_controls)
        state = encoded.mean(1)
        if conditioning is not None:
            if conditioning.shape != (len(state), 35) or not torch.isfinite(conditioning).all():
                raise ValueError("Declared16-point navigation and3 ego-motion values required")
            state = state + self.condition(conditioning)
        future, costs = [], []
        for horizon in range(self.steps):
            # Only controls through this horizon can influence this prediction.
            through = (horizon + 1) * self.controls_per_step - 1
            state = self.transition(action[:, through], state)
            query = self.query[None] + state[:, None]
            attended, _ = self.attend(query, encoded, encoded, need_weights=False)
            future.append(self.decode(query + attended))
            costs.append(self.cost(state))
        return dict(future_tokens=torch.stack(future, 1), outcome_logits=torch.stack(costs, 1))


def independent_ensemble(members=3, seed=42, **kwargs):
    if not 3 <= members <= 5:
        raise ValueError("Use3–5 independently initialized complete dynamics models")
    models = []
    # Avoid changing the caller's RNG state when constructing comparison arms.
    with torch.random.fork_rng(devices=[]):
        for i in range(members):
            torch.random.default_generator.manual_seed(seed + i * 1009)
            models.append(OrderedJEPADynamics(**kwargs))
    return nn.ModuleList(models)
