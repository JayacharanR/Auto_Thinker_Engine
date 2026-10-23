"""RGB-only proposal context; no privileged observation or training metadata."""

import numpy as np
from torch import nn


def context_patches(rgb, points):
    """Native32andarea-averaged96windows, both centered at integer RGBpoint."""
    rgb = np.asarray(rgb)
    points = np.asarray(points)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Native HWC uint8RGB required")
    if not len(points):
        return np.empty((0, 6, 32, 32), np.uint8)
    if points.ndim != 2 or points.shape[1] != 2 or not np.issubdtype(points.dtype, np.integer):
        raise ValueError("Integer RGBproposal coordinates required")
    if (
        (points < 0).any()
        or (points[:, 0] >= rgb.shape[1]).any()
        or (points[:, 1] >= rgb.shape[0]).any()
    ):
        raise ValueError("Proposal outside image")
    padded = np.pad(rgb, ((48, 48), (48, 48), (0, 0)), mode="edge")
    patches = []
    for x, y in points:
        local = padded[y + 32 : y + 64, x + 32 : x + 64]
        wide = padded[y : y + 96, x : x + 96]
        small = np.rint(wide.reshape(32, 3, 32, 3, 3).mean((1, 3))).astype(np.uint8)
        patches.append(np.concatenate([local, small], -1).transpose(2, 0, 1))
    return np.stack(patches)


class SignalContextVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        layers, previous = [], 6
        for index, channels in enumerate((16, 32, 64, 64)):
            layers.extend(
                [
                    nn.Conv2d(previous, channels, 3, stride=1 if index == 0 else 2, padding=1),
                    nn.SiLU(),
                ]
            )
            previous = channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(1024, 64), nn.SiLU(), nn.Linear(64, 4))

    def forward(self, patches):
        if patches.ndim != 4 or patches.shape[1:] != (6, 32, 32):
            raise ValueError("Only native/contextRGB patches; no extra channels or metadata")
        return self.head(self.features(patches.float() / 255 - 0.5))


def label_proposals(peaks, boxes, background):
    """Training-only labels: overlapping inconsistent housing states stay unknown."""
    labels = []
    for peak in peaks:
        x, y = peak["x"], peak["y"]
        known = {b[4] for b in boxes if b[0] <= x < b[2] and b[1] <= y < b[3]}
        if len(known) == 1:
            labels.append(next(iter(known)))
        elif not known and background[y, x]:
            labels.append(0)
        else:
            labels.append(-1)
    return np.array(labels, dtype=np.int64)


def verified_candidates(peaks, probabilities):
    if len(peaks) != len(probabilities):
        raise ValueError("Every original proposal needs a verifier score")
    result = []
    for peak, prob in zip(peaks, probabilities):
        cls = int(np.argmax(prob[1:]) + 1)
        if prob[cls] >= 0.5:
            result.append(dict(x=peak["x"], y=peak["y"], cls=cls, score=float(prob[cls])))
    return sorted(result, key=lambda p: (-p["score"], p["y"], p["x"]))
