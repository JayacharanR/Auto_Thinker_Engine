"""Small fully convolutional RGB signal localizer; no simulator/map/runtime rules."""

from torch import nn
from torch.nn import functional as F


class SignalGrid(nn.Module):
    """Native RGB -> stride2 background/red/yellow/green logits, receptive field21px."""

    def __init__(self):
        super().__init__()
        layers, previous = [], 3
        for index, channel in enumerate((12, 24, 32, 32, 32, 32)):
            layers.extend(
                [
                    nn.Conv2d(previous, channel, 3, stride=2 if index == 1 else 1, padding=1),
                    nn.SiLU(),
                ]
            )
            previous = channel
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(previous, 4, 1)

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.shape[2] % 2 or rgb.shape[3] % 2:
            raise ValueError("Even-sized BCHW RGB required")
        return self.classifier(self.features(rgb.float() / 255 - 0.5))


def signal_loss(logits, targets):
    """Class-balanced sparse-center supervision; projected unlabelled lamps ignored."""
    error = F.cross_entropy(logits.float(), targets.long(), reduction="none", ignore_index=255)
    positive = (targets > 0) & (targets < 4)
    background = targets == 0
    foreground_loss = (error * positive).sum() / positive.sum().clamp_min(1)
    background_loss = (error * background).sum() / background.sum().clamp_min(1)
    return foreground_loss + 0.2 * background_loss
