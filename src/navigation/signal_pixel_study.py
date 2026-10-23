"""Offline native-pixel signal experiment, outside all deployed policy imports."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from src.navigation.dagger import digest
from src.navigation.signal_learning import SignalDataset


class NativeSignalGrid(nn.Module):
    """Single RGB view only, stride1 and21pixel receptive field, no coordinates."""

    def __init__(self):
        super().__init__()
        layers, previous = [], 3
        for index, channel in enumerate((12, 24, 32, 32, 32, 32)):
            dilation = 1 if index < 2 else 2
            layers.extend(
                [
                    nn.Conv2d(previous, channel, 3, padding=dilation, dilation=dilation),
                    nn.SiLU(),
                ]
            )
            previous = channel
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(previous, 4, 1)

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("BCHW RGB required; no additional channels")
        return self.classifier(self.features(rgb.float() / 255 - 0.5))


class NativeSignalDataset:
    def __init__(self, root, audit_root):
        root, audit_root = Path(root), Path(audit_root)
        read = lambda p: json.loads(Path(p).read_text())  # noqa:E731
        assessment = read(audit_root / "assessment.json")
        for file, key in (
            ("cases.json", "case_sha256"),
            ("labels.json", "labels_sha256"),
            ("validation_identity.json", "validation_identity_sha256"),
        ):
            if digest(audit_root / file) != assessment[key]:
                raise ValueError("Audited pixel sources changed")
        cases, labels = read(audit_root / "cases.json"), read(audit_root / "labels.json")
        eligible = {
            (r["source"], r["town"], r["case_id"])
            for r in cases
            if r["disposition"] == "eligible_train"
        }
        label_map = defaultdict(list)
        for label in labels:
            label_map[(label["source"], label["town"], label["case_id"], label["view"])].append(
                label
            )
        self.samples, self.train, self.validation = [], [], []
        self.anchors = {"center": defaultdict(list), "bulb": defaultdict(list)}
        self.boxes, images, center_targets, bulb_targets, self.background = [], [], [], [], []
        sources = sorted({c["source"] for c in cases})
        validation_identity = read(audit_root / "validation_identity.json")
        for source in sources:
            supplement = "expansion" in source
            data = SignalDataset(root / source, training_only=supplement)
            original_audit = read(root / source / "data_audit.json")
            original_labels = defaultdict(list)
            for label in original_audit["labels"]:
                if label["static_phase_response_supported"]:
                    original_labels[(label["town"], label["case_id"], label["view"])].append(label)
            for old_index, sample in enumerate(data.samples):
                key = (source, sample["town"], sample["case_id"])
                valid = sample["role"] == "validation"
                if key not in eligible and not valid:
                    continue
                if (
                    valid
                    and digest(root / source / sample["path"])
                    != validation_identity["image_sha256"][sample["path"]]
                ):
                    raise ValueError("Original validation image changed")
                index = len(self.samples)
                self.samples.append(dict(sample, source=source))
                (self.validation if valid else self.train).append(index)
                images.append(data.images[old_index])
                # Preserve the original broad ignore zones; erase old center classes.
                ignored = data.targets[old_index] != 0
                target = np.where(ignored.repeat(2, 0).repeat(2, 1), 255, 0).astype(np.uint8)
                self.background.append(target == 0)
                center, bulb, boxes = target.copy(), target.copy(), []
                state = sample["phase"]
                cls = {"red": 1, "red_repeat": 1, "yellow": 2, "green": 3}[state]
                group = (sample["town"], sample["case_id"], sample["view"])
                lamps = original_labels[group] if valid else label_map[(*key, sample["view"])]
                for label in lamps:
                    yellow = label.get(
                        "yellow_color_supported", label.get("original_yellow_supported")
                    )
                    if state == "yellow" and not yellow:
                        continue
                    x0, y0, x1, y1 = label["crop_xyxy"]
                    boxes.append([x0, y0, x1, y1, cls])
                    cx, cy = min(511, (x0 + x1) // 2), min(287, (y0 + y1) // 2)
                    center[cy, cx] = cls
                    if not valid:
                        points = np.asarray(label["pixels_yx"][cls - 1], dtype=np.int64)
                        if len(points):
                            y, x = points[:, 0] + y0, points[:, 1] + x0
                            bulb[y, x] = cls
                            self.anchors["bulb"][cls].append((index, np.column_stack([x, y])))
                        self.anchors["center"][cls].append((index, np.array([[cx, cy]])))
                self.boxes.append(boxes)
                center_targets.append(center)
                bulb_targets.append(bulb)
            del data
        self.images = np.stack(images)
        self.targets = {"center": np.stack(center_targets), "bulb": np.stack(bulb_targets)}
        self.background = np.stack(self.background)
        if len(self.validation) != 144 or sum(len(self.boxes[i]) for i in self.validation) != 72:
            raise ValueError("Original validation denominator changed")
        for cls in (1, 2, 3):
            if len(self.anchors["center"][cls]) != len(self.anchors["bulb"][cls]):
                raise ValueError("Paired supervision arms require identical positive housings")

    def batch(self, arm, rng, device):
        images, targets = [], []
        for offset in range(12):
            if offset % 2 == 0:
                cls = int(rng.integers(1, 4))
                choices = self.anchors[arm][cls]
                index, positions = choices[int(rng.integers(len(choices)))]
                cx, cy = positions[int(rng.integers(len(positions)))]
                x = int(np.clip(cx - rng.integers(12, 53), 0, 448))
                y = int(np.clip(cy - rng.integers(12, 53), 0, 224))
            else:
                index = int(rng.choice(self.train))
                x, y = int(rng.integers(449)), int(rng.integers(225))
            images.append(self.images[index, :, y : y + 64, x : x + 64])
            targets.append(self.targets[arm][index, y : y + 64, x : x + 64])
        return torch.from_numpy(np.stack(images)).to(device), torch.from_numpy(
            np.stack(targets)
        ).to(device)


def point_candidates(probability, threshold=0.5):
    """One peak per8-connected active RGB region; all regions retained."""
    active = probability[1:].max(0)
    components, n = ndimage.label(active >= threshold, structure=np.ones((3, 3)))
    result = []
    for label, region in enumerate(ndimage.find_objects(components, n), 1):
        if region is None:
            continue
        local = np.where(components[region] == label, active[region], -1)
        y, x = np.unravel_index(np.argmax(local), local.shape)
        y, x = y + region[0].start, x + region[1].start
        result.append(
            dict(
                x=int(x),
                y=int(y),
                cls=int(probability[1:, y, x].argmax() + 1),
                score=float(active[y, x]),
            )
        )
    return sorted(result, key=lambda p: (-p["score"], p["y"], p["x"]))
