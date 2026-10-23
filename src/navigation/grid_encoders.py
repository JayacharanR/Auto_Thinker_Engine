"""Frozen named research backbones with explicit spatial layout and provenance."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.jepa.encoder import ViTEncoder
from src.navigation.rectangular_jepa import RectangularJEPA


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class FrozenGridEncoder(nn.Module):
    def __init__(self, descriptor):
        super().__init__()
        self.descriptor = dict(descriptor)
        self.kind = descriptor["kind"]
        if file_sha256(descriptor["checkpoint"]) != descriptor["sha256"]:
            raise ValueError("Frozen encoder artifact hash mismatch")
        if self.kind == "imagenet_resnet18":
            from torchvision.models import resnet18

            model = resnet18(weights=None)
            model.load_state_dict(
                torch.load(descriptor["checkpoint"], map_location="cpu", weights_only=True),
                strict=True,
            )
            self.model = nn.Sequential(*list(model.children())[:-2])
            self.output_dim = 512
        elif self.kind == "dinov2_small":
            import timm
            from timm.models.vision_transformer import checkpoint_filter_fn

            self.model = timm.create_model(
                "vit_small_patch14_dinov2", pretrained=False, img_size=(154, 266), num_classes=0
            )
            state = torch.load(descriptor["checkpoint"], map_location="cpu", weights_only=True)
            self.model.load_state_dict(checkpoint_filter_fn(state, self.model), strict=True)
            self.output_dim = 384
        elif self.kind in ("jepa_comma2k19", "jepa_carla", "jepa_random"):
            state = torch.load(descriptor["checkpoint"], map_location="cpu", weights_only=False)
            base = ViTEncoder(
                img_size=128,
                num_frames=8,
                patch_size=16,
                tubelet_size=2,
                embed_dim=384,
                depth=12,
                num_heads=6,
            )
            base.load_state_dict(state["context_encoder_state_dict"], strict=True)
            self.model = RectangularJEPA(base, (4, 8, 8))
            self.output_dim = 384
        else:
            raise ValueError(f"Unknown registered encoder:{self.kind}")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.inference_mode()
    def frame_features(self, rgb):
        if rgb.dtype != torch.uint8 or rgb.shape[-3:] != (3, 144, 256):
            raise ValueError("Declared256x144 RGB context required")
        x = (rgb.float() / 255 - self.mean) / self.std
        if self.kind == "imagenet_resnet18":
            maps = self.model(x)
        elif self.kind == "dinov2_small":
            # Preserve all pixels/aspect; mean-color padding supplies complete14px patches.
            x = F.pad(x, (0, 10, 0, 10), value=0)
            tokens = self.model.forward_features(x)[:, self.model.num_prefix_tokens :]
            maps = tokens.reshape(len(x), 11, 19, 384).permute(0, 3, 1, 2)
        else:
            raise ValueError("A video JEPA needs the full past8-frame clip")
        return F.adaptive_avg_pool2d(maps, (4, 4)).flatten(2).transpose(1, 2)

    @torch.inference_mode()
    def video_features(self, video):
        if video.dtype != torch.uint8 or video.shape[1:] != (3, 8, 3, 144, 256):
            raise ValueError("Batch/three-view/eight-past-frame RGB required")
        if self.kind in ("jepa_comma2k19", "jepa_carla", "jepa_random"):
            flat = video.flatten(0, 1).permute(0, 2, 1, 3, 4).float() / 255
            x = (flat - self.mean[:, :, None]) / self.std[:, :, None]
            tokens = self.model.spatial_tokens(x, pool=(4, 4))
            return tokens.reshape(len(video), 3, 16, 384)
        tokens = self.frame_features(video.flatten(0, 2))
        return tokens.reshape(len(video), 3, 8, 16, self.output_dim).mean(2)
