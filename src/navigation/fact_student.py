"""Camera recurrent state predicts structured facts; privileged targets are loss-only."""

import torch
from torch import nn
from torch.nn import functional as F

from src.navigation.spatial_student import driving_loss
from src.navigation.visual_decision_student import VisualDecisionStudent


class FactHeads(nn.Module):
    def __init__(self, hidden=192):
        super().__init__()
        self.light = nn.Linear(hidden, 5)
        self.presence = nn.Linear(hidden, 2)
        self.distance = nn.Linear(hidden, 2)
        self.hazards = nn.Linear(hidden, 4)

    def forward(self, recurrent):
        return dict(
            light=self.light(recurrent),
            presence=self.presence(recurrent),
            distance=80 * torch.sigmoid(self.distance(recurrent)),
            hazards=self.hazards(recurrent),
        )


class FactStudent(VisualDecisionStudent):
    def __init__(self, encoder_dim=384, hidden=192, scratch=False, color_enabled=True):
        super().__init__(encoder_dim, hidden, scratch, color_enabled)
        self.facts = FactHeads(hidden)
        self._recurrent_sequence = None
        self.memory.register_forward_hook(self.capture_recurrent)

    def capture_recurrent(self, module, inputs, output):
        self._recurrent_sequence = output[0]

    def forward(self, features, optical, navigation, ego, hidden=None):
        self._recurrent_sequence = None
        prediction, hidden = super().forward(features, optical, navigation, ego, hidden)
        recurrent = self._recurrent_sequence
        self._recurrent_sequence = None
        if recurrent is None:
            raise RuntimeError("Fact heads require the same current camera recurrent state")
        prediction.update(self.facts(recurrent))
        return prediction, hidden


def fact_loss(prediction, targets, burn_in=8):
    total, parts = driving_loss(prediction, targets, burn_in)
    p = {k: v[:, burn_in:] for k, v in prediction.items()}
    t = {k: v[:, burn_in:] for k, v in targets.items()}
    valid_light = t["light"] != 4
    cross_entropy = F.cross_entropy(
        p["light"].flatten(0, 1), t["light"].flatten(), reduction="none"
    )
    light = (cross_entropy * valid_light.flatten()).sum() / valid_light.sum().clamp_min(1)
    presence = F.binary_cross_entropy_with_logits(p["presence"], t["presence"])
    mask = t["presence"]
    distance_error = F.smooth_l1_loss(p["distance"] / 40, t["distance"] / 40, reduction="none")
    distance = (distance_error * mask).sum() / mask.sum().clamp_min(1)
    hazards = F.binary_cross_entropy_with_logits(p["hazards"], t["hazards"])
    parts.update(
        fact_light=light, fact_presence=presence, fact_distance=distance, fact_hazards=hazards
    )
    return total + light + 0.5 * presence + distance + 0.5 * hazards, parts
