"""Append only independently verified strict-TRAIN phase pairs; no resplit."""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.navigation.dagger import digest
from src.navigation.rule_geometry import project_box
from src.navigation.sensors import calibrations
from src.navigation.signal_pixel_study import NativeSignalDataset


class DiverseSignalDataset(NativeSignalDataset):
    def __init__(self, root, base_audit, supplement):
        root, supplement = Path(root), Path(supplement)
        super().__init__(root, base_audit)
        original_validation = self.images[self.validation].copy()
        original_background = self.background[self.validation].copy()
        read = lambda p: json.loads(Path(p).read_text())  # noqa:E731
        assessment, manifest = (
            read(supplement / "assessment.json"),
            read(supplement / "manifest.json"),
        )
        if (
            digest(supplement / "manifest.json") != assessment["manifest_sha256"]
            or manifest["role"] != "train"
            or assessment["original_cases"] != 200
            or digest(manifest["labels_file"]) != manifest["labels_sha256"]
        ):
            raise ValueError("Exact independently audited strictTRAIN supplement required")
        grouped = defaultdict(list)
        for label in read(manifest["labels_file"]):
            if label["static_phase_response_supported"]:
                grouped[(label["task_id"], label["case_id"], label["view"])].append(label)
        source = root / "outputs/navigation_signal_diversity_20261021"
        config = calibrations(
            yaml.safe_load((source / "source/configs/navigation.yaml").read_text())
        )
        images, centers, bulbs, background, geometries = [], [], [], [], {}
        for entry in manifest["eligible_cases"]:
            if not entry["eligible"] or entry["role"] != "train" or "Town05" in entry["town"]:
                raise ValueError("No heldout/unknown source can enter TRAIN")
            metadata = root / entry["case_metadata"]
            if digest(metadata) != entry["case_metadata_sha256"]:
                raise ValueError("Closed native case changed")
            case = read(metadata)["case"]
            capture = root / entry["source_capture"]
            if entry["task_id"] not in geometries:
                if digest(capture / "geometry.json") != entry["geometry_sha256"]:
                    raise ValueError("Native light geometry changed")
                geometries[entry["task_id"]] = read(capture / "geometry.json")
            geometry = geometries[entry["task_id"]]
            for view, calibration in config.items():
                ignore = np.zeros((144, 256), np.uint8)
                for light in geometry["lights"]:
                    for box in light["boxes_world"]:
                        projected = project_box(box["vertices"], case["ego_pose"], calibration)
                        if projected["in_image"]:
                            a, b, c, d = projected["bbox_xyxy"]
                            x0, y0 = max(0, math.floor(a / 2) - 2), max(0, math.floor(b / 2) - 2)
                            x1, y1 = min(256, math.ceil(c / 2) + 2), min(144, math.ceil(d / 2) + 2)
                            ignore[y0:y1, x0:x1] = 255
                native_ignore = ignore.repeat(2, 0).repeat(2, 1)
                for phase, cls in (("red", 1), ("green", 3), ("yellow", 2), ("red_repeat", 1)):
                    path = capture / f"route{case['route_id']}_{phase}_{view}.png"
                    relative = str(path.relative_to(root))
                    if digest(path) != entry["image_sha256"][relative]:
                        raise ValueError("Verified nativeRGB changed")
                    with Image.open(path) as im:
                        rgb = np.array(im.convert("RGB")).transpose(2, 0, 1)
                    index = len(self.samples)
                    self.samples.append(
                        dict(
                            path=relative,
                            source="strict_signal_diversity",
                            town=case["town"],
                            junction_id=case["junction_id"],
                            case_id=case["route_id"],
                            view=view,
                            phase=phase,
                            role="train",
                            illumination=entry["illumination"],
                            task_id=entry["task_id"],
                        )
                    )
                    self.train.append(index)
                    center, bulb, boxes = native_ignore.copy(), native_ignore.copy(), []
                    for label in grouped[(entry["task_id"], entry["case_id"], view)]:
                        if phase == "yellow" and not label["yellow_color_supported"]:
                            continue
                        x0, y0, x1, y1 = label["crop_xyxy"]
                        boxes.append([x0, y0, x1, y1, cls])
                        cx, cy = min(511, (x0 + x1) // 2), min(287, (y0 + y1) // 2)
                        center[cy, cx] = cls
                        points = np.array(label["pixels_yx"][cls - 1], np.int64)
                        if not len(points):
                            raise ValueError(
                                "Audited supported housing unexpectedly lacks bulb pixels"
                            )
                        y, x = points[:, 0] + y0, points[:, 1] + x0
                        bulb[y, x] = cls
                        self.anchors["bulb"][cls].append((index, np.column_stack([x, y])))
                        self.anchors["center"][cls].append((index, np.array([[cx, cy]])))
                    self.boxes.append(boxes)
                    images.append(rgb)
                    centers.append(center)
                    bulbs.append(bulb)
                    background.append(native_ignore == 0)
        self.images = np.concatenate([self.images, np.stack(images)])
        self.targets["center"] = np.concatenate([self.targets["center"], np.stack(centers)])
        self.targets["bulb"] = np.concatenate([self.targets["bulb"], np.stack(bulbs)])
        self.background = np.concatenate([self.background, np.stack(background)])
        if (
            not np.array_equal(self.images[self.validation], original_validation)
            or not np.array_equal(self.background[self.validation], original_background)
            or len(self.validation) != 144
            or sum(len(self.boxes[i]) for i in self.validation) != 72
        ):
            raise ValueError("Original validation RGB/labels/ignore regions changed")
