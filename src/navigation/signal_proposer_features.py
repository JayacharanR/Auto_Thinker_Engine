"""Frozen native RGB proposer features; no simulator, labels or metadata inputs."""

import json
from pathlib import Path

import torch
from torch import nn

from src.navigation.dagger import digest
from src.navigation.signal_pixel_study import NativeSignalGrid

PROPOSER_SHA256 = "27ddcc86e6be56a2edc7033e5ac9b47100f0c305d86e34893ced3cb20000b193"
FEATURE_DIM = 36


def verify_ordered_dataset(samples, train, validation, saved):
    """Compare recorded JSON semantics, including ordering, roles and all fields.

    In-memory housing points are tuples, serialized as JSON arrays. Normalize
    through the same JSON representation without reordering or dropping fields.
    """
    actual = dict(samples=samples, train=train, validation=validation)
    encoded = json.dumps(actual, sort_keys=True, allow_nan=False)
    expected = json.dumps(saved, sort_keys=True, allow_nan=False)
    if encoded != expected:
        raise ValueError("Original ordered data identity changed")


def point_features(features, probabilities, points):
    """Gather 32 learned channels and four class probabilities at every RGB point.

    Full-image convolutions preserve the original image-edge padding. Never run
    a cropped patch through the proposer and assume it matches the full image.
    Coordinates address the maps but are not appended to the returned features.
    """
    if features.ndim != 3 or features.shape[0] != 32:
        raise ValueError("32-channel spatial feature map required")
    if probabilities.shape != (4, *features.shape[1:]):
        raise ValueError("Aligned four-class probability map required")
    if points.ndim != 2 or points.shape[1] != 2 or points.dtype not in (torch.int32, torch.int64):
        raise ValueError("Integer XY proposal points required")
    if features.device != probabilities.device or features.device != points.device:
        raise ValueError("All maps and points must share a device")
    height, width = features.shape[1:]
    if len(points) and (
        (points < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] >= height).any()
    ):
        raise ValueError("Proposal outside native image")
    x, y = points[:, 0].long(), points[:, 1].long()
    return torch.cat([features[:, y, x].T.float(), probabilities[:, y, x].T.float()], -1)


class FrozenBulbFeatures(nn.Module):
    """Extract the exact existing native proposer representation without updates."""

    def __init__(self, checkpoint):
        super().__init__()
        checkpoint = Path(checkpoint)
        if digest(checkpoint) != PROPOSER_SHA256:
            raise ValueError("Exact frozen bulb proposer checkpoint required")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = NativeSignalGrid()
        self.model.load_state_dict(saved["model"], strict=True)
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.inference_mode()
    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 288, 512) or rgb.dtype != torch.uint8:
            raise ValueError("Only native BCHW 512x288 uint8 RGB allowed")
        features = self.model.features(rgb.float() / 255 - 0.5)
        logits = self.model.classifier(features)
        return features, logits.float().softmax(1)
