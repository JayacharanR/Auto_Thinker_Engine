"""Frozen RGB-only signal components; no training labels, simulator or map access."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.navigation.contracts import VIEWS
from src.navigation.signal_detector import make_signal_detector

STATES = ("Red", "Yellow", "Green")


def state_features(rgb, box):
    """Exact deployed ROI convention used to assess the selected state export."""
    if rgb.dtype != np.uint8 or rgb.shape != (288, 512, 3):
        raise ValueError("Native512x288 RGB required")
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Finite housing box required")
    x0, y0, x1, y1 = box
    integer = [
        max(0, math.floor(x0)),
        max(0, math.floor(y0)),
        min(512, math.ceil(x1)),
        min(288, math.ceil(y1)),
    ]
    if not (0 <= integer[0] < integer[2] <= 512 and 0 <= integer[1] < integer[3] <= 288):
        raise ValueError("Empty/outside housing crop")
    image = Image.fromarray(rgb).crop(integer).resize((16, 32), Image.Resampling.BILINEAR)
    return np.asarray(image).astype(np.float32).ravel() / 255 - 0.5


def state_probabilities(linear, features):
    if not np.array_equal(linear["classes"], [0, 1, 2]):
        raise ValueError("Expected Red/Yellow/Green class order")
    logits = features @ linear["coef"].T + linear["intercept"]
    probabilities = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


class FrozenSignalPerception:
    def __init__(self, bundle, device="cuda"):
        bundle = json.loads(Path(bundle).read_text())
        if bundle["detector_architecture"] != "signal-detector-roi-residual-v1":
            raise ValueError("Unrecognized signal architecture")
        if not bundle["expanded_state_accepted"]:
            raise ValueError("Accepted state bundle required")
        for path_key, sha_key in (
            ("detector_checkpoint", "detector_sha256"),
            ("state_classifier", "state_sha256"),
        ):
            with Path(bundle[path_key]).open("rb") as stream:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
                actual = hasher.hexdigest()
            if actual != bundle[sha_key]:
                raise ValueError("Signal bundle artifact hash mismatch")
        self.device = torch.device(device)
        self.model = make_signal_detector(roi_adapter=True)
        checkpoint = torch.load(
            bundle["detector_checkpoint"], map_location="cpu", weights_only=False
        )
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        with np.load(bundle["state_classifier"], allow_pickle=False) as saved:
            self.linear = {k: saved[k].copy() for k in ("coef", "intercept", "classes")}

    @torch.inference_mode()
    def infer(self, packet):
        packet.validate()
        images = [
            torch.from_numpy(packet.rgb[v].transpose(2, 0, 1).copy()).to(self.device).float() / 255
            for v in VIEWS
        ]
        predicted = self.model(images)
        observations = []
        for view, result in zip(VIEWS, predicted):
            boxes = result["boxes"].float().cpu().numpy()
            scores = result["scores"].float().cpu().numpy()
            for box, score in zip(boxes, scores):
                probabilities = state_probabilities(
                    self.linear, state_features(packet.rgb[view], box)
                )
                observations.append(
                    dict(
                        view=view,
                        box=box.tolist(),
                        score=float(score),
                        probabilities=probabilities.tolist(),
                    )
                )
        return observations
