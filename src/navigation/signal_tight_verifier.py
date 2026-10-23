"""Fixed 8/16-pixel views for the frozen RGB verifier scale comparison."""

import torch
from torch.nn import functional as F

from src.navigation.signal_pretrained_verifier import FrozenSignalResNet


def tight_views(patches):
    if patches.ndim != 4 or patches.shape[1:] != (6, 32, 32) or patches.dtype != torch.uint8:
        raise ValueError("Only original native/context RGB uint8 patches")
    native = patches[:, :3].float() / 255
    return torch.cat([
        F.interpolate(native[:, :, 12:20, 12:20], (128, 128),
                      mode="bilinear", align_corners=False),
        F.interpolate(native[:, :, 8:24, 8:24], (128, 128),
                      mode="bilinear", align_corners=False),
    ])


class TightSignalResNet(FrozenSignalResNet):
    @torch.inference_mode()
    def forward(self, patches):
        views = tight_views(patches)
        maps = self.features((views - self.mean) / self.std)
        n = len(patches)
        return torch.cat([maps[:n, :, 1:3, 1:3].mean((2, 3)), maps[n:].mean((2, 3))], -1)
