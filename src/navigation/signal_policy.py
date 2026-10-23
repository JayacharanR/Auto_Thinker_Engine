"""RGB-only object attention and a learned stop residual over a frozen driver."""

import numpy as np
import torch
from torch import nn

from src.navigation.contracts import VIEWS
from src.navigation.signal_tracking import angular_box
from src.navigation.student import CameraStopGoStudent

TOKEN_DIM = 17
MAX_OBJECTS = 300


def pack_observations(packet, observations):
    """Keep every detector output plus an always-valid learned null token."""
    if len(observations) > MAX_OBJECTS:
        raise ValueError("Detector output exceeds declared 300-object capacity")
    tokens = np.zeros((MAX_OBJECTS + 1, TOKEN_DIM), np.float32)
    valid = np.zeros(MAX_OBJECTS + 1, bool)
    valid[0] = True
    for i, row in enumerate(observations, 1):
        view = row["view"]
        p = np.asarray(row["probabilities"], dtype=np.float32)
        score = float(row["score"])
        if (
            p.shape != (3,)
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.isclose(p.sum(), 1)
        ):
            raise ValueError("Invalid RGB state distribution")
        if not np.isfinite(score) or not 0.5 <= score <= 1:
            raise ValueError("Invalid RGB detection confidence")
        bearing, elevation, width, height = angular_box(row["box"], packet.calibration[view], 0)
        box = np.asarray(row["box"], dtype=np.float32) / [512, 288, 512, 288]
        one_hot = [float(v == view) for v in VIEWS]
        tokens[i] = [
            *box,
            score,
            *p,
            np.sin(bearing),
            np.cos(bearing),
            elevation,
            width,
            height,
            *one_hot,
            0,
        ]
        valid[i] = True
    return tokens, valid


class CausalSignalFacts:
    """Infer at 2.5Hz, consume at10Hz, hold only past RGB facts with explicit age."""

    def __init__(self, perception):
        self.perception, self.previous, self.count = perception, None, 0
        self.tokens = self.valid = self.seen = None

    def step(self, packet):
        packet.validate()
        if self.previous is not None and (
            packet.frame != self.previous[0] + 2
            or abs(packet.timestamp - self.previous[1] - 0.1) > 1e-5
        ):
            raise ValueError("Facts require contiguous10Hz input; reset per episode")
        if self.count % 4 == 0:
            self.tokens, self.valid = pack_observations(packet, self.perception.infer(packet))
            self.seen = packet.timestamp
        result = self.tokens.copy()
        result[1:, -1] = (packet.timestamp - self.seen) * self.valid[1:]
        self.previous = packet.frame, packet.timestamp
        self.count += 1
        return result, self.valid.copy()


class StopResidual(nn.Module):
    def __init__(self, hidden=192):
        super().__init__()
        self.objects = nn.Sequential(nn.Linear(TOKEN_DIM, 64), nn.SiLU(), nn.Linear(64, 64))
        self.query = nn.Linear(hidden + 35, 64)
        self.memory = nn.GRU(hidden + 35 + 64, 64, batch_first=True)
        self.output = nn.Linear(64, 9)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, recurrent, navigation, ego, facts, valid, hidden=None):
        if facts.shape[-2:] != (MAX_OBJECTS + 1, TOKEN_DIM) or valid.shape != facts.shape[:-1]:
            raise ValueError("Signal fact dimensions differ from declared input")
        if not bool(valid[..., 0].all()):
            raise ValueError("An explicit valid null token is required")
        nav = torch.cat(
            [navigation.flatten(-2) / 32, ego / ego.new_tensor([10, 3.14159265, 1])], -1
        )
        query = self.query(torch.cat([recurrent, nav], -1))
        objects = self.objects(facts.float())
        weights = (objects * query.unsqueeze(-2)).sum(-1) / 8
        weights = weights.masked_fill(~valid.bool(), float("-inf")).softmax(-1)
        attended = (objects * weights.unsqueeze(-1)).sum(-2)
        output, hidden = self.memory(torch.cat([recurrent, nav, attended], -1), hidden)
        return self.output(output), hidden


def apply_stop_residual(parent, residual):
    result = dict(parent)
    stop = parent["stop_logits"] + residual
    speeds = torch.where(
        stop >= 0, torch.zeros_like(parent["cruise_speeds"]), parent["cruise_speeds"]
    )
    result.update(
        stop_logits=stop, current_speed=speeds[..., 0], speeds=speeds[..., 1:], brake=stop[..., 0]
    )
    return result


class SignalResidualStudent(CameraStopGoStudent):
    def __init__(self, hidden=192):
        super().__init__(hidden)
        self.requires_grad_(False)
        self.residual = StopResidual(hidden)

    def forward(
        self, rgb, front, navigation, ego, hidden=None, signal_facts=None, signal_valid=None
    ):
        if signal_facts is None or signal_valid is None:
            raise ValueError("Declared RGB signal facts required")
        base_hidden, residual_hidden = hidden if hidden is not None else (None, None)
        with torch.no_grad():
            recurrent, base_hidden = self.encode_history(rgb, front, navigation, ego, base_hidden)
            parent = super().decode(recurrent)
        residual, residual_hidden = self.residual(
            recurrent, navigation, ego, signal_facts, signal_valid, residual_hidden
        )
        return apply_stop_residual(parent, residual), (base_hidden, residual_hidden)
