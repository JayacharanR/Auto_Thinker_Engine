"""Separate stronger externally supervised RGB detector; no CARLA training labels."""

import json
from pathlib import Path

import torch

from src.navigation.grid_encoders import file_sha256
from src.navigation.vehicle_perception import FrozenRoadUsers


class FrozenRoadUsersFPN(FrozenRoadUsers):
    def __init__(self, descriptor, device="cuda"):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

        if isinstance(descriptor, (str, Path)):
            descriptor = json.loads(Path(descriptor).read_text())
        if file_sha256(descriptor["checkpoint"]) != descriptor["sha256"]:
            raise ValueError("Frozen road-user detector weights changed")
        self.device = torch.device(device)
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None, min_size=512, max_size=1024
        )
        self.model.load_state_dict(
            torch.load(descriptor["checkpoint"], map_location="cpu", weights_only=True),
            strict=True,
        )
        self.model = self.model.to(device).eval().requires_grad_(False)
        self.descriptor = dict(descriptor)
