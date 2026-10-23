"""Frozen RGB detector and approximate camera-derived constant-velocity tracks."""

import json
from pathlib import Path

import numpy as np
import torch

from src.navigation.camera_ground import ground_contact
from src.navigation.constant_velocity import PerceivedTrack
from src.navigation.grid_encoders import file_sha256

SIZES = {1: (0.4, 0.4), 2: (1.0, 0.4), 3: (2.5, 1.1), 4: (1.0, 0.4), 6: (6.0, 1.4), 8: (5.0, 1.4)}


class FrozenRoadUsers:
    def __init__(self, descriptor, device="cuda"):
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large

        if isinstance(descriptor, (str, Path)):
            descriptor = json.loads(Path(descriptor).read_text())
        if file_sha256(descriptor["checkpoint"]) != descriptor["sha256"]:
            raise ValueError("Frozen road-user detector changed")
        self.device = torch.device(device)
        self.model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        self.model.load_state_dict(
            torch.load(descriptor["checkpoint"], map_location="cpu", weights_only=True), strict=True
        )
        self.model = self.model.to(device).eval().requires_grad_(False)
        self.descriptor = dict(descriptor)

    @torch.inference_mode()
    def detect(self, packet):
        packet.validate()
        views = ("front", "left", "right")
        images = [
            torch.from_numpy(packet.rgb[view].copy()).to(self.device).permute(2, 0, 1).float() / 255
            for view in views
        ]
        predictions = self.model(images)
        detections = []
        unknown = 0
        for view, prediction in zip(views, predictions):
            for box, score, label in zip(
                prediction["boxes"].cpu().numpy(),
                prediction["scores"].cpu().numpy(),
                prediction["labels"].cpu().numpy(),
            ):
                if label not in SIZES or score < 0.35:
                    continue
                xy = ground_contact(box, packet.calibration[view])
                if xy is None:
                    unknown += 1
                    continue
                detections.append(
                    dict(
                        view=view,
                        box=box.tolist(),
                        score=float(score),
                        label=int(label),
                        position_xy=xy,
                        sigma_m=0.5 + 0.2 * np.linalg.norm(xy),
                    )
                )
        # Declared coarse cross-view merge; no simulator identity or GT association.
        fused = []
        for row in sorted(detections, key=lambda row: -row["score"]):
            if any(
                np.linalg.norm(row["position_xy"] - prior["position_xy"]) < 2.0 for prior in fused
            ):
                continue
            fused.append(row)
        return fused, dict(
            unknown_ground_contacts=unknown,
            raw_detections=len(detections),
            flat_local_ground_approximation=True,
            local_carla_detector_labels=0,
        )


class CameraVehicleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.previous = None
        self.ego_xy = np.zeros(2)

    def step(self, packet, detections):
        packet.validate()
        timestamp = packet.timestamp
        speed, yaw, _ = packet.ego_motion
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        if self.previous is not None:
            old_time, old_speed, old_yaw = self.previous
            dt = timestamp - old_time
            if not 0 < dt <= 0.5:
                raise ValueError("Camera tracker needs increasing odometry within.5s")
            direction = (
                np.array([np.cos(old_yaw), np.sin(old_yaw)]) + np.array([np.cos(yaw), np.sin(yaw)])
            ) * 0.5
            self.ego_xy += direction * (speed + old_speed) * 0.5 * dt
        self.previous = timestamp, float(speed), float(yaw)
        available = set(self.tracks)
        for detection in sorted(detections, key=lambda row: -row["score"]):
            world = self.ego_xy + np.asarray(detection["position_xy"]) @ rotation.T
            matches = []
            for identity in available:
                track = self.tracks[identity]
                dt = timestamp - track["time"]
                gap = float(np.linalg.norm(world - (track["position"] + track["velocity"] * dt)))
                if gap < max(3.0, 2 * detection["sigma_m"]):
                    matches.append((gap, identity))
            if matches:
                _, identity = min(matches)
                available.remove(identity)
                old = self.tracks[identity]
                dt = timestamp - old["time"]
                velocity = (world - old["position"]) / dt
                norm = np.linalg.norm(velocity)
                if norm > 25:
                    velocity *= 25 / norm
                velocity = 0.5 * velocity + 0.5 * old["velocity"]
                velocity_sigma = max(1.5, float(np.linalg.norm(velocity - old["velocity"])))
            else:
                identity = self.next_id
                self.next_id += 1
                velocity = np.zeros(2)
                velocity_sigma = 4.0
            self.tracks[identity] = dict(
                position=world,
                velocity=velocity,
                time=timestamp,
                sigma=detection["sigma_m"],
                velocity_sigma=velocity_sigma,
                label=detection["label"],
                confidence=detection["score"],
            )
        self.tracks = {
            identity: track
            for identity, track in self.tracks.items()
            if timestamp - track["time"] <= 1.0
        }
        result = []
        for identity, track in sorted(self.tracks.items()):
            age = timestamp - track["time"]
            xy = (track["position"] + track["velocity"] * age - self.ego_xy) @ rotation
            velocity = track["velocity"] @ rotation
            length, width = SIZES[track["label"]]
            result.append(
                PerceivedTrack(
                    identity,
                    xy,
                    velocity,
                    length,
                    width,
                    track["sigma"],
                    track["velocity_sigma"],
                    track["confidence"],
                    age,
                )
            )
        return result
