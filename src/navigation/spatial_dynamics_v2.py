"""Position-aware causal JEPA candidate, separate from the retained trained v1.

This untrained module preserves camera identity and token location in the risk
path. Neither construction tests nor latent prediction qualify it as a planner.
"""

import torch
from torch import nn


class SpatialOrderedJEPADynamics(nn.Module):
    def __init__(self, token_dim=384, hidden=128):
        super().__init__()
        self.spatial_tokens, self.steps, self.controls_per_step = 48, 8, 10
        self.context = nn.Sequential(nn.Linear(token_dim, hidden), nn.LayerNorm(hidden))
        # Encoder cache order is front, left, right; each contains a row-major4x4 grid.
        self.camera = nn.Embedding(3, hidden)
        y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        grid = torch.stack((x, y), -1).float().reshape(16, 2) / 1.5 - 1
        self.register_buffer("coordinates", grid.repeat(3, 1))
        self.register_buffer("views", torch.arange(3).repeat_interleave(16))
        self.position = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.condition = nn.Sequential(nn.Linear(35, hidden), nn.SiLU())
        self.scene_queries = nn.Parameter(torch.randn(4, hidden) * 0.02)
        self.scene_attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.scene_summary = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.SiLU())
        self.controls = nn.GRU(3, hidden, batch_first=True)
        self.transition = nn.GRUCell(hidden, hidden)
        self.query = nn.Parameter(torch.randn(48, hidden) * 0.02)
        self.attend = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.decode = nn.Linear(hidden, token_dim)
        self.cost_queries = nn.Parameter(torch.randn(4, hidden) * 0.02)
        self.cost_attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.cost = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def forward(self, past_tokens, ordered_controls, conditioning=None):
        if (
            past_tokens.ndim != 3 or past_tokens.shape[1:] != (48, self.context[0].in_features)
            or ordered_controls.shape != (len(past_tokens), 80, 3)
            or not torch.isfinite(past_tokens).all()
            or not torch.isfinite(ordered_controls).all()
        ):
            raise ValueError("Past front/left/right4x4 grids and ordered80commands required")
        encoded = (self.context(past_tokens) + self.camera(self.views)[None]
                   + self.position(self.coordinates)[None])
        queries = self.scene_queries[None].expand(len(encoded), -1, -1)
        attended, _ = self.scene_attention(queries, encoded, encoded, need_weights=False)
        state = self.scene_summary(attended.flatten(1))
        if conditioning is not None:
            if conditioning.shape != (len(state), 35) or not torch.isfinite(conditioning).all():
                raise ValueError("Declared16-point navigation and3ego values required")
            state = state + self.condition(conditioning)
        action, _ = self.controls(ordered_controls)
        future, costs = [], []
        for horizon in range(8):
            state = self.transition(action[:, (horizon + 1) * 10 - 1], state)
            query = self.query[None] + state[:, None]
            visual, _ = self.attend(query, encoded, encoded, need_weights=False)
            future_hidden = query + visual
            future.append(self.decode(future_hidden))
            risk_queries = self.cost_queries[None] + state[:, None]
            risk_scene, _ = self.cost_attention(
                risk_queries, future_hidden, future_hidden, need_weights=False
            )
            costs.append(self.cost(risk_scene.flatten(1)))
        return dict(future_tokens=torch.stack(future, 1), outcome_logits=torch.stack(costs, 1))
