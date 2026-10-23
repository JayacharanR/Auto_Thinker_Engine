"""Spatial camera representations plus common RGB fields and a causal driving head."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.navigation.optical_fields import FIELD_CHANNELS, past_average


class ScratchGrid(nn.Module):
    def __init__(self):
        super().__init__()
        layers, previous = [], 3
        for width, stride in zip((16, 32, 64, 128), (4, 2, 2, 2)):
            layers.extend(
                [
                    nn.Conv2d(
                        previous,
                        width,
                        5 if previous == 3 else 3,
                        stride,
                        2 if previous == 3 else 1,
                    ),
                    nn.GroupNorm(4, width),
                    nn.SiLU(),
                ]
            )
            previous = width
        self.layers = nn.Sequential(*layers)
        self.project = nn.Conv2d(128, 384, 1)

    def forward(self, rgb):
        return (
            self.project(F.adaptive_avg_pool2d(self.layers(rgb.float() / 255 - 0.5), (4, 4)))
            .flatten(2)
            .transpose(1, 2)
        )


class SpatialCameraStudent(nn.Module):
    def __init__(self, encoder_dim=384, hidden=192, scratch=False, color_enabled=True):
        super().__init__()
        self.hidden_size, self.encoder_dim = hidden, encoder_dim
        self.color_enabled = color_enabled
        self.scratch = ScratchGrid() if scratch else None
        self.low = nn.Sequential(nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, 64))
        self.color = nn.Linear(FIELD_CHANNELS, 64)
        self.position = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64))
        self.view = nn.Embedding(3, 64)
        self.kind = nn.Embedding(2, 64)
        self.nav = nn.Sequential(nn.Linear(35, 64), nn.SiLU())
        self.queries = nn.Parameter(torch.randn(8, 64) * 0.02)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(8 * 64 + 64, hidden), nn.SiLU())
        self.memory = nn.GRU(hidden, hidden, batch_first=True)
        self.path = nn.Linear(hidden, 8)
        self.speed = nn.Linear(hidden, 9)
        self.stop = nn.Linear(hidden, 9)
        self.comfort = nn.Linear(hidden, 1)
        nn.init.constant_(self.speed.bias, -1.0)

    @staticmethod
    def grid(height, width, device):
        y, x = torch.meshgrid(
            (torch.arange(height, device=device) + 0.5) * 2 / height - 1,
            (torch.arange(width, device=device) + 0.5) * 2 / width - 1,
            indexing="ij",
        )
        return torch.stack([x, y], -1).reshape(height * width, 2)

    def scratch_features(self, rgb, gradients=False, chunk=8):
        """Encode each frame once, then carry the exact8-frame past mean."""
        if self.scratch is None or rgb.shape[0] != 1:
            raise ValueError("Scratch whole-episode encoding requires one episode")
        result = []
        for start in range(0, rgb.shape[1], chunk):
            images = rgb[:, start : start + chunk]
            predicted = (
                checkpoint(self.scratch, images.flatten(0, 2), use_reentrant=False)
                if gradients
                else self.scratch(images.flatten(0, 2))
            )
            result.append(predicted.reshape(-1, 3, 16, 384))
        return past_average(torch.cat(result, 0))[None]

    def forward(self, features, optical, navigation, ego, hidden=None):
        if features.shape[-3:-1] != (3, 16) or optical.shape[-3:] != (3, 144, FIELD_CHANNELS):
            raise ValueError("Declared three-view spatial/optical token grids required")
        batch, steps = features.shape[:2]
        camera = self.view.weight[None, None, :, None, :]
        low = (
            self.low(features.float())
            + camera
            + self.position(self.grid(4, 4, features.device))
            + self.kind.weight[0]
        )
        colors = (
            self.color(optical.float())
            + camera
            + self.position(self.grid(9, 16, features.device))
            + self.kind.weight[1]
        )
        keys = low.flatten(2, 3)
        if self.color_enabled:
            keys = torch.cat([keys, colors.flatten(2, 3)], 2)
        scale = ego.new_tensor([10, 3.14159265, 1])
        nav = self.nav(torch.cat([navigation.flatten(-2) / 32, ego / scale], -1))
        query = self.queries[None] + nav.flatten(0, 1)[:, None]
        attended, _ = self.attention(
            query, keys.flatten(0, 1), keys.flatten(0, 1), need_weights=False
        )
        fused = self.fuse(torch.cat([attended.reshape(batch, steps, -1), nav], -1))
        recurrent, hidden = self.memory(fused, hidden)
        cruise = F.softplus(self.speed(recurrent)) * 5
        stop = self.stop(recurrent)
        speed = torch.where(stop >= 0, torch.zeros_like(cruise), cruise)
        return dict(
            correction=2 * torch.tanh(self.path(recurrent)),
            cruise_speeds=cruise,
            stop_logits=stop,
            current_speed=speed[..., 0],
            speeds=speed[..., 1:],
            comfort_brake=self.comfort(recurrent).squeeze(-1),
            brake=stop[..., 0],
        ), hidden


def driving_loss(prediction, targets, burn_in=8):
    """Only imitation labels from the selected whole episodes; no light-state loss."""
    if not 0 <= burn_in < prediction["current_speed"].shape[1]:
        raise ValueError("Burn-in must leave supervised frames")
    p = {k: v[:, burn_in:] for k, v in prediction.items()}
    t = {k: v[:, burn_in:] for k, v in targets.items()}
    speed = torch.cat([t["current_speed"][..., None], t["speeds"]], -1)
    valid = torch.cat([torch.ones_like(speed[..., :1]), t["speed_mask"]], -1)
    stopped = speed < 0.1

    def mean(value, mask):
        return (value * mask).sum() / mask.sum().clamp_min(1)

    path = mean(
        F.smooth_l1_loss(p["correction"], t["correction"], reduction="none"), t["correction_mask"]
    )
    go = (~stopped).float() * valid
    errors = F.smooth_l1_loss(p["cruise_speeds"] / 5, speed / 5, reduction="none")
    stop_error = F.binary_cross_entropy_with_logits(
        p["stop_logits"], stopped.float(), reduction="none"
    )
    parts = dict(
        path=path,
        current_speed=mean(errors[..., 0], go[..., 0]),
        future_speed=mean(errors[..., 1:], go[..., 1:]),
        stop=stop_error[..., 0].mean(),
        future_stop=mean(stop_error[..., 1:], valid[..., 1:]),
        brake=F.binary_cross_entropy_with_logits(p["comfort_brake"], t["brake"]),
    )
    return parts["path"] + 2 * parts["current_speed"] + parts["future_speed"] + 2 * parts[
        "stop"
    ] + parts["future_stop"] + 0.1 * parts["brake"], parts
