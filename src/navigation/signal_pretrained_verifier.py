"""Frozen RGB proposal features, no metadata or teacher state at inference."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.navigation.dagger import digest


def rgb_features(patches):
    if patches.ndim != 4 or patches.shape[1:] != (6, 32, 32) or patches.dtype != np.uint8:
        raise ValueError("Only native/contextRGBuint8patches")
    x = patches.astype(np.float32) / 255
    return np.concatenate(
        [
            x[:, :3, 14:19, 14:19].reshape(len(x), 75),
            x[:, :3].mean((2, 3)),
            x[:, :3].std((2, 3)),
            x[:, 3:].mean((2, 3)),
            x[:, 3:].std((2, 3)),
        ],
        axis=1,
    )


class FrozenSignalResNet(nn.Module):
    def __init__(self, descriptor):
        super().__init__()
        from torchvision.models import resnet18

        if (
            descriptor["kind"] != "imagenet_resnet18"
            or digest(descriptor["checkpoint"]) != descriptor["sha256"]
        ):
            raise ValueError("Exact original pretrainedImageNetResNet required")
        model = resnet18(weights=None)
        model.load_state_dict(
            torch.load(descriptor["checkpoint"], map_location="cpu", weights_only=True), strict=True
        )
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.inference_mode()
    def forward(self, patches):
        if patches.ndim != 4 or patches.shape[1:] != (6, 32, 32) or patches.dtype != torch.uint8:
            raise ValueError("Only native/contextRGBuint8patches")
        both = torch.cat([patches[:, :3], patches[:, 3:]], 0).float() / 255
        x = F.interpolate(both, (128, 128), mode="bilinear", align_corners=False)
        maps = self.features((x - self.mean) / self.std)
        n = len(patches)
        local = maps[:n, :, 1:3, 1:3].mean((2, 3))
        context = maps[n:].mean((2, 3))
        return torch.cat([local, context], -1)


def retain_original_states(peaks, foreground_probability):
    probability = np.asarray(foreground_probability)
    if probability.shape != (len(peaks),) or not np.isfinite(probability).all():
        raise ValueError("Exactlyonefiniteforegroundprobabilityperproposal")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Foregroundprobabilitymustbein0..1")
    result = [
        dict(x=p["x"], y=p["y"], cls=p["cls"], score=float(value))
        for p, value in zip(peaks, probability)
        if value >= 0.5
    ]
    return sorted(result, key=lambda p: (-p["score"], p["y"], p["x"]))
