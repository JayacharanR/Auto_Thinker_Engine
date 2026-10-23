"""Learned stop correction from frozen camera predictions; no privileged inputs.

This module is untrained until a separate registered fit produces evidence.
The existing driving controller, speed proposals and steering stay unchanged.
"""

import torch
from torch import nn

FACT_DIM = 17


def semantic_features(prediction):
    light = prediction["light"].float().softmax(-1)
    presence = prediction["presence"].float().sigmoid()
    distance = (prediction["distance"].float() / 80).clamp(0, 1)
    hazards = prediction["hazards"].float().sigmoid()
    # These are derived prediction features, not threshold rules or actuators.
    joint_light = light[..., 1:4] * presence[..., :1]
    nearby_stop = presence[..., 1:2] * (1 - distance[..., 1:2])
    return torch.cat([light, presence, distance, hazards, joint_light, nearby_stop], -1)


class SemanticStopResidual(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(FACT_DIM + 9, hidden), nn.SiLU())
        self.memory = nn.GRU(hidden, hidden, batch_first=True)
        self.delta = nn.Linear(hidden, 9)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, prediction, facts, hidden=None):
        base = prediction["stop_logits"].float()
        if (base.ndim != 3 or base.shape[-1] != 9
                or facts.shape != (*base.shape[:2], FACT_DIM)
                or prediction["cruise_speeds"].shape != base.shape
                or not torch.isfinite(base).all() or not torch.isfinite(facts).all()):
            raise ValueError("Causal frozen9stop logits and17predicted facts required")
        inputs = torch.cat([facts, base.clamp(-10, 10) / 10], -1)
        recurrent, hidden = self.memory(self.encode(inputs), hidden)
        logits = base + 4 * self.delta(recurrent).tanh()
        speeds = torch.where(logits >= 0, 0.0, prediction["cruise_speeds"])
        result = dict(prediction, stop_logits=logits, current_speed=speeds[..., 0],
                      speeds=speeds[..., 1:], brake=logits[..., 0])
        return result, hidden
