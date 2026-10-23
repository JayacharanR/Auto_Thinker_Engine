"""Versioned RGB-derived road-user observations retaining unknown geometry.

The old detector and its callers remain unchanged. This optional shared pack is
not a collision probability and grants no permission to drive through unknowns.
"""

import numpy as np

from src.navigation.camera_ground import ground_contact, mount_rotation
from src.navigation.contracts import VIEWS

SCHEMA = "navigation-camera-road-users-v1"
CLASSES = (1, 2, 3, 4, 6, 8)
FIELDS = (
    "detector_score",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "box_x0",
    "box_y0",
    "box_x1",
    "box_y1",
    "ray_x",
    "ray_y",
    "ray_z",
    "flat_ground_valid",
    "flat_ground_x_over60",
    "flat_ground_y_over60",
    "image_truncated",
    "ground_horizon_unknown",
    "ground_range_unknown",
)


def object_observation(packet, raw):
    """Use only a declared SensorPacket and four raw detector fields."""
    if set(raw) != {"view", "label", "box", "score"}:
        raise ValueError("Only RGB detector fields may enter object observations")
    view, label, score = raw["view"], raw["label"], float(raw["score"])
    if (
        view not in VIEWS
        or label not in CLASSES
        or not np.isfinite(score)
        or not 0.35 <= score <= 1
    ):
        raise ValueError("Fixed road-user classes and original .35 threshold required")
    cal = packet.calibration[view]
    box = np.asarray(raw["box"], dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Finite original detector box required")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= cal.width and 0 <= y0 < y1 <= cal.height):
        raise ValueError("Original-image detector box bounds required")
    ray = (
        np.array(
            [
                1.0,
                ((x0 + x1) * 0.5 - cal.width / 2) / cal.intrinsic[0, 0],
                -(y1 - cal.height / 2) / cal.intrinsic[1, 1],
            ]
        )
        @ mount_rotation(cal).T
    )
    ray /= np.linalg.norm(ray)
    xy = ground_contact(box, cal)
    if xy is not None:
        status = "flat_ground_approximation"
    elif y1 >= cal.height - 2:
        status = "image_bottom_unknown"
    elif ray[2] >= -1e-4:
        status = "horizon_unknown"
    else:
        status = "range_unknown"
    return dict(
        view=view,
        label=int(label),
        score=score,
        box=box.tolist(),
        ray_ego=ray.tolist(),
        flat_ground_xy=None if xy is None else xy.tolist(),
        ground_status=status,
        image_truncated=bool(x0 <= 0 or y0 <= 0 or x1 >= cal.width or y1 >= cal.height),
    )


def observation_pack(packet, raw_rows):
    """Keep every accepted raw detection, with separate presence/ground-valid bits.

    Zero padding and unavailable coordinates are never interpreted as free space:
    consumers must respect both presence and flat_ground_valid. Rows stay in
    original per-view detector order; no cross-view or same-class suppression.
    """
    packet.validate()
    observations = [object_observation(packet, row) for row in raw_rows]
    values = np.zeros((3, 100, len(FIELDS)), dtype=np.float32)
    presence = np.zeros((3, 100), dtype=bool)
    for view_index, view in enumerate(VIEWS):
        rows = [row for row in observations if row["view"] == view]
        if len(rows) > 100:
            raise ValueError("Never truncate detections beyond the registered FPN100/image limit")
        cal = packet.calibration[view]
        for index, row in enumerate(rows):
            value = values[view_index, index]
            value[0] = row["score"]
            value[1 + CLASSES.index(row["label"])] = 1
            value[7:11] = np.asarray(row["box"]) / [cal.width, cal.height, cal.width, cal.height]
            value[11:14] = row["ray_ego"]
            known = row["flat_ground_xy"] is not None
            value[14] = known
            if known:
                value[15:17] = np.asarray(row["flat_ground_xy"]) / 60
            value[17] = row["image_truncated"]
            value[18] = row["ground_status"] == "horizon_unknown"
            value[19] = row["ground_status"] == "range_unknown"
            presence[view_index, index] = True
    return dict(
        schema=SCHEMA,
        frame=packet.frame,
        timestamp=packet.timestamp,
        values=values,
        presence=presence,
        observations=observations,
        unknown_ground_contacts=sum(row["flat_ground_xy"] is None for row in observations),
        no_collision_probability=True,
    )


class FrozenRoadUserObservations:
    """Same frozen FPN forward/classes/score; expose rows before ground filtering."""

    def __init__(self, descriptor, device="cuda"):
        from src.navigation.road_user_fpn import FrozenRoadUsersFPN

        self.detector = FrozenRoadUsersFPN(descriptor, device)

    def observe(self, packet):
        import torch

        packet.validate()
        with torch.inference_mode():
            images = [
                torch.from_numpy(packet.rgb[v].copy())
                .to(self.detector.device)
                .permute(2, 0, 1)
                .float()
                / 255
                for v in VIEWS
            ]
            predictions = self.detector.model(images)
            raw = []
            for view, result in zip(VIEWS, predictions):
                for box, score, label in zip(
                    result["boxes"].cpu().numpy(),
                    result["scores"].cpu().numpy(),
                    result["labels"].cpu().numpy(),
                ):
                    if int(label) in CLASSES and score >= 0.35:
                        raw.append(
                            dict(view=view, label=int(label), score=float(score), box=box.tolist())
                        )
        return observation_pack(packet, raw)
