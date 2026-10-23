"""Training-only localized labels; complete corpus gate and whole-junction split."""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from scripts.collect_navigation_signal_pairs import validate_plan
from src.navigation.dagger import digest
from src.navigation.rule_geometry import project_box
from src.navigation.sensors import calibrations


def junction_roles(cases, seed=20261016):
    groups = sorted({(c["town"], c["junction_id"]) for c in cases})
    if len(groups) < 5 or any(t.removesuffix("_Opt") == "Town05" for t, _ in groups):
        raise ValueError("At least five approved TRAIN junctions, no Town05")
    rng = np.random.default_rng(seed)
    selected = {
        groups[int(i)] for i in rng.permutation(len(groups))[: math.ceil(0.2 * len(groups))]
    }
    return {group: "validation" if group in selected else "train" for group in groups}


def phase_targets(ignore, labels, state, cls, supervision="center"):
    """Evaluation centers stay fixed when training supervision covers the region."""
    if supervision not in ("center", "region"):
        raise ValueError("Unknown signal supervision")
    evaluation, training, points = ignore.copy(), ignore.copy(), []
    for label in labels:
        if state == "yellow" and not label["yellow_color_supported"]:
            continue
        x0, y0, x1, y1 = label["crop_xyxy"]
        cx, cy = min(255, int((x0 + x1) / 4)), min(143, int((y0 + y1) / 4))
        evaluation[cy, cx] = cls
        if supervision == "region":
            training[y0 // 2 : math.ceil(y1 / 2), x0 // 2 : math.ceil(x1 / 2)] = cls
        training[cy, cx] = cls
        points.append((cx, cy, cls))
    return evaluation, training, sorted(set(points))


class SignalDataset:
    def __init__(self, root, supervision="center", *, training_only=False):
        self.root = Path(root).resolve()

        def read(path):
            return json.loads(Path(path).read_text())

        audit = read(self.root / "data_audit.json")
        if not audit["static_label_data_gate_passed"] or not audit["all_cases_complete"]:
            raise ValueError("Completed static label-data gate required before training")
        plan, request = read(self.root / "plan.json"), read(self.root / "request.json")
        validate_plan(plan)
        if digest(self.root / "plan.json") != audit["plan_sha256"] or any(
            digest(self.root / "source" / path) != sha
            for path, sha in request["source_hashes"].items()
        ):
            raise ValueError("Static dataset provenance changed")
        if "expansion" in plan:
            if not training_only:
                raise ValueError(
                    "TRAIN-only supplement requires explicit append, never a new split"
                )
            self.roles = {(c["town"], c["junction_id"]): "train" for c in plan["cases"]}
        else:
            if training_only:
                raise ValueError("Explicit TRAIN-only expansion plan required")
            self.roles = junction_roles(plan["cases"])
        calibration = calibrations(
            yaml.safe_load((self.root / "source/configs/navigation.yaml").read_text())
        )
        supported = defaultdict(list)
        for label in audit["labels"]:
            if label["static_phase_response_supported"]:
                supported[(label["town"], label["case_id"], label["view"])].append(label)
        self.samples, images, targets, training_targets, self.positive = (
            [],
            [],
            [],
            [],
            defaultdict(list),
        )
        self.validation = []
        for town in sorted({c["town"] for c in plan["cases"]}):
            geometry = read(self.root / town / "geometry.json")
            for case in read(self.root / town / "cases.json"):
                role = self.roles[(town, case["junction_id"])]
                for view, c in calibration.items():
                    ignore = np.zeros((144, 256), np.uint8)
                    for actor in geometry["lights"]:
                        for box in actor["boxes_world"]:
                            projected = project_box(box["vertices"], case["ego_pose"], c)
                            if projected["in_image"]:
                                x0, y0, x1, y1 = projected["bbox_xyxy"]
                                x0, y0 = (
                                    max(0, math.floor(x0 / 2) - 2),
                                    max(0, math.floor(y0 / 2) - 2),
                                )
                                x1, y1 = (
                                    min(256, math.ceil(x1 / 2) + 2),
                                    min(144, math.ceil(y1 / 2) + 2),
                                )
                                ignore[y0:y1, x0:x1] = 255
                    for state, cls in (("red", 1), ("green", 3), ("yellow", 2), ("red_repeat", 1)):
                        path = self.root / town / f"route{case['route_id']}_{state}_{view}.png"
                        if digest(path) != audit["image_sha256"][str(path.relative_to(self.root))]:
                            raise ValueError("Captured native RGB changed")
                        with Image.open(path) as image:
                            rgb = np.asarray(image.convert("RGB")).copy()
                        target, training_target, points = phase_targets(
                            ignore,
                            supported[(town, case["route_id"], view)],
                            state,
                            cls,
                            supervision,
                        )
                        index = len(self.samples)
                        self.samples.append(
                            {
                                "path": str(path.relative_to(self.root)),
                                "town": town,
                                "junction_id": case["junction_id"],
                                "case_id": case["route_id"],
                                "view": view,
                                "phase": state,
                                "role": role,
                                "points": points,
                            }
                        )
                        if role == "train":
                            for cx, cy, label in points:
                                self.positive[label].append((index, cx, cy))
                        else:
                            self.validation.append(index)
                        images.append(rgb.transpose(2, 0, 1))
                        targets.append(target)
                        training_targets.append(training_target)
        if not all(self.positive[cls] for cls in (1, 2, 3)) or (
            not self.validation and not training_only
        ):
            raise ValueError("Frozen perception split lacks training classes/validation")
        self.images, self.targets = np.stack(images), np.stack(targets)
        self.training_targets = np.stack(training_targets)
        self.train = [i for i, row in enumerate(self.samples) if row["role"] == "train"]
        self.manifest = {
            "kind": "Localized RGB phase component only; no action or driving model",
            "training_supervision": supervision,
            "corpus_audit_sha256": digest(self.root / "data_audit.json"),
            "plan_sha256": digest(self.root / "plan.json"),
            "split_rule": "Seed20261016; seeded ceil20% whole junctions; independent of outcomes",
            "groups": [
                {"town": t, "junction_id": j, "role": role} for (t, j), role in self.roles.items()
            ],
            "samples": self.samples,
            "training_class_counts": {str(k): len(v) for k, v in self.positive.items()},
            "labels": "Sparse housing-center state; unlabelled lamps ignored. Background outside "
            "all projected lamp boxes; no original-traffic visibility claim",
        }

    def batch(self, indices, device):
        return torch.from_numpy(self.images[indices].copy()).to(device), torch.from_numpy(
            self.targets[indices].copy()
        ).to(device)

    def training_batch(self, rng, batch_size, device):
        images, targets = [], []
        for index in range(batch_size):
            if index % 2 == 0:
                cls = int(rng.integers(1, 4))
                sample, cx, cy = self.positive[cls][int(rng.integers(len(self.positive[cls])))]
                x = 2 * int(np.clip(cx - int(rng.integers(8, 57)), 0, 192))
                y = 2 * int(np.clip(cy - int(rng.integers(8, 57)), 0, 80))
            else:
                sample = int(rng.choice(self.train))
                x, y = 2 * int(rng.integers(193)), 2 * int(rng.integers(81))
            images.append(self.images[sample, :, y : y + 128, x : x + 128])
            targets.append(
                self.training_targets[sample, y // 2 : y // 2 + 64, x // 2 : x // 2 + 64]
            )
        return torch.from_numpy(np.stack(images)).to(device), torch.from_numpy(
            np.stack(targets)
        ).to(device)
