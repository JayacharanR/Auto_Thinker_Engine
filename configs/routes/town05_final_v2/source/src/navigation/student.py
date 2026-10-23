"""Camera-only causal CNN/GRU: separate path, speed and perception heads."""

import torch
from torch import nn
from torch.nn import functional as F


def encoder(channels, strides, pool):
    layers, previous = [], 3
    for channel, stride in zip(channels, strides):
        layers.extend(
            [
                nn.Conv2d(
                    previous, channel, 5 if previous == 3 else 3, stride, 2 if previous == 3 else 1
                ),
                nn.GroupNorm(4, channel),
                nn.SiLU(),
            ]
        )
        previous = channel
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(pool), nn.Flatten())


class CameraStudent(nn.Module):
    def __init__(self, hidden=192):
        super().__init__()
        self.hidden_size = hidden
        self.context = encoder([12, 24, 40, 48], [4, 2, 2, 2], (2, 4))
        self.context_projection = nn.Linear(48 * 8, 96)
        self.signal = encoder([8, 16, 32], [2, 2, 2], (3, 8))
        self.signal_projection = nn.Linear(32 * 24, 96)
        self.navigation = nn.Sequential(nn.Linear(35, 64), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(96 * 4 + 64, hidden), nn.SiLU())
        self.memory = nn.GRU(hidden, hidden, batch_first=True)
        self.path = nn.Linear(hidden, 8)
        self.speed = nn.Linear(hidden, 9)  # immediate target plus 0.5..4s profile
        self.brake = nn.Linear(hidden, 1)
        self.light = nn.Linear(hidden, 5)
        self.hazards = nn.Linear(hidden, 4)
        self.distance = nn.Linear(hidden, 2)
        self.presence = nn.Linear(hidden, 2)
        nn.init.constant_(self.speed.bias, -1.0)

    def forward(self, rgb, front, navigation, ego, hidden=None):
        batch, steps = rgb.shape[:2]
        context = self.context_projection(self.context(rgb.flatten(0, 2).float() / 255 - 0.5))
        signal = self.signal_projection(self.signal(front.flatten(0, 1).float() / 255 - 0.5))
        scale = ego.new_tensor([10, 3.14159265, 1])
        nav = self.navigation(torch.cat([navigation.flatten(-2) / 32, ego / scale], dim=-1))
        features = torch.cat(
            [context.reshape(batch, steps, -1), signal.reshape(batch, steps, -1), nav], dim=-1
        )
        recurrent, hidden = self.memory(self.fuse(features), hidden)
        return self.decode(recurrent), hidden

    def decode(self, recurrent):
        speed = F.softplus(self.speed(recurrent)) * 5
        return {
            "correction": 2 * torch.tanh(self.path(recurrent)),
            "current_speed": speed[..., 0],
            "speeds": speed[..., 1:],
            "brake": self.brake(recurrent).squeeze(-1),
            "light": self.light(recurrent),
            "hazards": self.hazards(recurrent),
            "distance": 80 * torch.sigmoid(self.distance(recurrent)),
            "presence": self.presence(recurrent),
        }


class CameraStopGoStudent(CameraStudent):
    """Learn stop intent separately from cruising speed and comfort braking.

    The semantic stop output is also the deployed brake output. A stop therefore
    cannot request a positive target speed. Ordinary deceleration remains the
    conventional controller's job; this adds no privileged runtime rules.
    """

    def __init__(self, hidden=192):
        super().__init__(hidden)
        self.stop = nn.Linear(hidden, 9)

    def decode(self, recurrent):
        result = super().decode(recurrent)
        stop = self.stop(recurrent)
        cruise = torch.cat([result["current_speed"].unsqueeze(-1), result["speeds"]], -1)
        gated = torch.where(stop >= 0, torch.zeros_like(cruise), cruise)
        result.update(
            comfort_brake=result["brake"],
            stop_logits=stop,
            cruise_speeds=cruise,
            current_speed=gated[..., 0],
            speeds=gated[..., 1:],
            brake=stop[..., 0],
        )
        return result


def make_student(architecture, hidden=192):
    models = {"camera-student-v1": CameraStudent, "camera-stop-go-v2": CameraStopGoStudent}
    if architecture not in models:
        raise ValueError(f"Unsupported camera checkpoint architecture: {architecture}")
    return models[architecture](hidden)


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def student_loss(prediction, targets):
    # Burn in the first half of each clip before supervising its memory outputs.
    start = prediction["current_speed"].shape[1] // 2
    p = {k: v[:, start:] for k, v in prediction.items()}
    t = {k: v[:, start:] for k, v in targets.items()}
    losses = {
        "path": masked_mean(
            F.smooth_l1_loss(p["correction"], t["correction"], reduction="none"),
            t["correction_mask"],
        ),
        "speed": F.smooth_l1_loss(p["current_speed"] / 5, t["current_speed"] / 5),
        "future_speed": masked_mean(
            F.smooth_l1_loss(p["speeds"] / 5, t["speeds"] / 5, reduction="none"), t["speed_mask"]
        ),
        "brake": F.binary_cross_entropy_with_logits(p["brake"], t["brake"]),
        "light": F.cross_entropy(p["light"].flatten(0, 1), t["light"].flatten()),
        "hazards": F.binary_cross_entropy_with_logits(p["hazards"], t["hazards"]),
        "distance": masked_mean(
            F.smooth_l1_loss(p["distance"] / 80, t["distance"] / 80, reduction="none"),
            t["presence"],
        ),
        "presence": F.binary_cross_entropy_with_logits(p["presence"], t["presence"]),
    }
    weights = dict(
        path=1, speed=2, future_speed=1, brake=1, light=0.2, hazards=0.1, distance=0.2, presence=0.1
    )
    if "stop_logits" in p:
        # Zero-speed teacher intent is distinct from transient comfort braking.
        speed = torch.cat([t["current_speed"].unsqueeze(-1), t["speeds"]], -1)
        valid = torch.cat([torch.ones_like(speed[..., :1]), t["speed_mask"]], -1)
        stopped = speed < 0.1
        losses["speed"] = masked_mean(
            F.smooth_l1_loss(p["cruise_speeds"][..., 0] / 5, speed[..., 0] / 5, reduction="none"),
            (~stopped[..., 0]).float(),
        )
        losses["future_speed"] = masked_mean(
            F.smooth_l1_loss(p["cruise_speeds"][..., 1:] / 5, speed[..., 1:] / 5, reduction="none"),
            t["speed_mask"] * (~stopped[..., 1:]).float(),
        )
        losses["stop"] = F.binary_cross_entropy_with_logits(
            p["stop_logits"][..., 0], stopped[..., 0].float()
        )
        losses["future_stop"] = masked_mean(
            F.binary_cross_entropy_with_logits(p["stop_logits"], stopped.float(), reduction="none")[
                ..., 1:
            ],
            valid[..., 1:],
        )
        losses["brake"] = F.binary_cross_entropy_with_logits(p["comfort_brake"], t["brake"])
        # Distance validity is not proof of visual visibility. Avoid known lamps
        # beyond the collection's 80m range; occlusion/localization remain open.
        light_valid = (t["light"] == 0) | (t["presence"][..., 0] > 0.5)
        losses["light"] = masked_mean(
            F.cross_entropy(p["light"].transpose(1, 2), t["light"], reduction="none"),
            light_valid.float(),
        )
        weights.update(stop=2, future_stop=1, brake=0.1)
    total = sum(weights[k] * v for k, v in losses.items())
    return total, losses
