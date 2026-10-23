"""TRAIN-only privileged label cameras; evaluated policy inputs are unchanged."""

import json
from dataclasses import asdict

import numpy as np

from scripts.navigation_label_rig import LabelRig, LabelShards
from src.navigation.collection_agent import CollectionExpert
from src.navigation.label_collection import require_label_scope
from src.navigation.runtime import atomic_json


def get_entry_point():
    return "DenseLabelCollectionExpert"


class DenseLabelCollectionExpert(CollectionExpert):
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        spec = self.navigation_spec
        require_label_scope(
            "expert",
            spec.get("collection"),
            spec.get("student"),
            spec.get("final_registration"),
            spec.get("rule_test"),
            spec["collection"].get("background") == "empty",
            None,
        )
        if spec["collection"].get("dense_labels") is not True:
            raise ValueError("Dense labels must be explicitly declared in the collection manifest")
        self.label_rig = None
        self.label_writer = LabelShards(self.out / "labels", labels=True)
        self.label_log = (self.out / "label_alignment.jsonl").open("x")
        self.maximum_label_alignment_m = self.maximum_label_alignment_degrees = 0.0
        import carla

        enum = {name: int(value) for name, value in carla.CityObjectLabel.names.items()}
        if enum.get("Car") != 14 or enum.get("TrafficLight") != 7:
            raise ValueError("Pinned0.9.15dense-label semantic classes differ")
        atomic_json(self.out / "label_semantic_enum.json", enum)

    def collection_metadata(self, metadata):
        metadata = super().collection_metadata(metadata)
        guard = self.navigation_spec["collection"].get("learning_guard", {})
        metadata.update(
            dense_label_role=guard.get("label_role"),
            label_hz=2,
            dense_labels_separate_from_policy=True,
            label_plan_sha256=guard.get("source_plan_sha256"),
        )
        return metadata

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        # All parent controls and teacher targets use the same original RGB/world
        # route interface. No label bytes are passed into the expert or policy packet.
        control = super().run_step(input_data, timestamp, sensors, plant)
        if self.label_rig is None:
            self.label_rig = LabelRig(
                self._world, self._vehicle, self.navigation, labels=True, include_rgb=False
            )
            return control  # Sensors created now cannot label an earlier RGB frame.
        frame = int(input_data["front"][0])
        _, labels, label_time, poses = self.label_rig.collect(frame)
        snapshot = self._world.get_snapshot()
        if snapshot.frame != frame or abs(label_time - snapshot.timestamp.elapsed_seconds) > 1e-6:
            raise ValueError(
                "Dense-label image timestamp/frame differs from actual RGBphysics frame"
            )
        import carla

        for name, pose in poses.items():
            view = name.split("_", 1)[0]
            calibration = self.camera_calibration[view]
            relative = carla.Transform(
                carla.Location(*calibration.xyz),
                carla.Rotation(
                    pitch=calibration.pitch_yaw_roll[0],
                    yaw=calibration.pitch_yaw_roll[1],
                    roll=calibration.pitch_yaw_roll[2],
                ),
            )
            expected = np.asarray(self._vehicle.get_transform().get_matrix()) @ np.asarray(
                relative.get_matrix()
            )
            actual = np.asarray(
                carla.Transform(
                    carla.Location(*pose[:3]),
                    carla.Rotation(pitch=pose[3], yaw=pose[4], roll=pose[5]),
                ).get_matrix()
            )
            self.maximum_label_alignment_m = max(
                self.maximum_label_alignment_m,
                float(np.linalg.norm(expected[:3, 3] - actual[:3, 3])),
            )
            rotation = expected[:3, :3].T @ actual[:3, :3]
            self.maximum_label_alignment_degrees = max(
                self.maximum_label_alignment_degrees,
                float(np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))),
            )
        if self.maximum_label_alignment_m > 0.002 or self.maximum_label_alignment_degrees > 0.05:
            raise ValueError("Dense-label and registered RGBcamera poses differ")
        if self.frame_count % 10 == 1 and not (
            self.collection_limit and self.collector.writer.samples >= self.collection_limit
        ):
            packet = self.packet(input_data, timestamp)
            packet.validate()
            self.label_writer.append(frame, float(timestamp), packet.rgb, labels)
            self.label_log.write(
                json.dumps(
                    dict(
                        frame=frame,
                        timestamp=timestamp,
                        physical_timestamp=label_time,
                        label_role=self.navigation_spec["collection"]["learning_guard"][
                            "label_role"
                        ],
                        rig_poses=poses,
                        calibration={v: asdict(c) for v, c in self.camera_calibration.items()},
                        pixel_identity_source="Same-frame RGBcopied from the original "
                        "three native cameras",
                    )
                )
                + "\n"
            )
        return control

    def destroy(self, results=None):
        try:
            if getattr(self, "label_rig", None):
                self.label_rig.close()
            if getattr(self, "label_writer", None):
                self.label_writer.close()
                atomic_json(
                    self.out / "label_profile.json",
                    dict(
                        label_timestamps=self.label_writer.count,
                        raw_bytes=self.label_writer.raw_bytes,
                        shards=self.label_writer.shards,
                        maximum_position_error_m=self.maximum_label_alignment_m,
                        maximum_angle_error_degrees=self.maximum_label_alignment_degrees,
                        all_raw_pixel_roundtrips_verified=True,
                        labels_not_used_by_controller=True,
                        label_role=self.navigation_spec["collection"]["learning_guard"][
                            "label_role"
                        ],
                    ),
                )
            if getattr(self, "label_log", None):
                self.label_log.close()
        finally:
            super().destroy(results)
