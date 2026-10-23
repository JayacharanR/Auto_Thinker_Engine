"""Privileged expert collection adapter. Policy packets exclude teacher/world facts."""

import json
import math
from pathlib import Path

import numpy as np

from src.navigation.collection import CollectionRecorder, RouteProjector, training_position
from src.navigation.contracts import SensorPacket
from src.navigation.dataset import EpisodeWriter
from src.navigation.pdm_agent import NavigationExpert
from src.navigation.perturbations import SteeringPerturbation
from src.navigation.replay import capture_world_state
from src.navigation.runtime import atomic_json
from src.navigation.sensors import calibrations


def get_entry_point():
    return "CollectionExpert"


def controls(command):
    return float(command.steer), float(command.throttle), float(command.brake)


def finite_distance(value):
    return float(value) if np.isfinite(value) else None


class CollectionExpert(NavigationExpert):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        spec = self.navigation_spec
        catalog = json.loads((Path(spec["collection"]["catalog"]) / "catalog.json").read_text())
        current = json.loads((Path(spec["out"]) / "current_route.json").read_text())
        entry = next(e for e in catalog["encounters"] if e["route_id"] == int(current["route_id"]))
        split = json.loads(
            (Path(spec["collection"]["catalog"]).parent / "locations.json").read_text()
        )
        self.allowed_junctions = self.collection_locations(split, entry)
        self.projector = RouteProjector(
            [[t.location.x, t.location.y] for t, _ in self.org_dense_route_world_coord]
        )
        self.camera_calibration = calibrations(self.navigation)
        self.collection_mode = spec["collection"].get("mode", "expert")
        self.collection_limit = spec["collection"].get("max_samples", 0)
        self.perturbation = SteeringPerturbation(current["seed"])
        self.collector = CollectionRecorder(
            EpisodeWriter(
                self.out / "data",
                self.collection_metadata(
                    {
                        "route_id": entry["route_id"],
                        "seed": current["seed"],
                        "family": entry["family"],
                        "town": entry["town"],
                        "junction_ids": entry["crossed_junction_ids"],
                        "collection_catalog_sha256": spec["collection"]["sha256"],
                        "partition": "train",
                        "driver": "privileged_pdm_lite_modified",
                        "mode": self.collection_mode,
                        "trace_control_semantics": "Expert command before collection perturbations",
                        "perturbation_policy": "Clear approach: steer +/-0.04-0.08 for0.3s per2s; "
                        "cancel and privileged brake if teacher hazard/brake appears during pulse",
                        "scenario": entry["scenario"],
                        "physics_hz": 20,
                        "sample_hz": 10,
                        "rgb_annotations": False,
                    }
                ),
                minimum_free_gb=self.navigation["runtime"]["minimum_free_gb"],
            )
        )

    def collection_locations(self, split, entry):
        guard = self.navigation_spec["collection"].get("learning_guard")
        if guard:
            return set(guard["allowed_junctions"][entry["town"]])
        return set(split["assignments"][entry["town"]]["train"])

    def collection_metadata(self, metadata):
        guard = self.navigation_spec["collection"].get("learning_guard")
        if guard:
            metadata.update(strict_training_guard_sha256=guard["guard_sha256"])
        metadata["background"] = self.navigation_spec["collection"].get("background", "canonical")
        metadata["training_signal_wait_seconds"] = self.navigation_spec["collection"].get(
            "signal_stationary_seconds"
        )
        return metadata

    def enforce_location(self, frame):
        return training_position(
            self._world.get_map(),
            self._vehicle.get_transform(),
            self.allowed_junctions,
            frame,
            Path(self.navigation_spec["out"]) / "collection_partition_violation.json",
        )

    def packet(self, input_data, timestamp):
        pose = self.enforce_location(int(input_data["front"][0]))
        yaw = math.radians(pose.rotation.yaw)
        return SensorPacket(
            frame=int(input_data["front"][0]),
            timestamp=float(timestamp),
            rgb={
                v: np.ascontiguousarray(input_data[v][1][:, :, :3][:, :, ::-1])
                for v in self.camera_calibration
            },
            calibration=self.camera_calibration,
            navigation_waypoints=self.projector.project(
                np.array([pose.location.x, pose.location.y]), yaw
            ),
            ego_motion=np.array(
                [
                    self._vehicle.get_velocity().length(),
                    yaw,
                    math.radians(self._vehicle.get_angular_velocity().z),
                ],
                dtype=np.float32,
            ),
        )

    def select_control(self, control, input_data, timestamp):
        if self.collection_mode == "noisy":
            hazard = any(
                self.teacher_data[k]
                for k in ("vehicle_hazard", "walker_hazard", "light_hazard", "stop_sign_hazard")
            )
            requested, intervention = self.perturbation.control(
                controls(control),
                self.frame_count - 1,
                self._vehicle.get_velocity().length(),
                bool(self.teacher_data["junction"]),
                hazard,
            )
            control.steer, control.throttle, control.brake = requested
            return control, intervention
        return control, None

    def extra_facts(self):
        return {}

    def _get_control(self, input_data, plant):
        control, data = super()._get_control(input_data, plant)
        self.teacher_data = data
        return control, data

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        if self.collector.pending is not None:
            self.collector.acknowledge(
                input_data["front"][0], controls(self._vehicle.get_control())
            )
        at_limit = bool(
            self.collection_limit and self.collector.writer.samples >= self.collection_limit
        )
        if at_limit:
            atomic_json(
                Path(self.navigation_spec["out"]) / "collection_budget.json",
                {"reason": "sample_budget", "samples": self.collector.writer.samples},
            )
            (Path(self.navigation_spec["out"]) / "STOP").touch()
        control = super().run_step(input_data, timestamp, sensors, plant)
        # The first superclass call initializes the hero/world. It does not tick
        # physics, so this still checks the current packet on every20Hz control.
        self.enforce_location(int(input_data["front"][0]))
        teacher_control = controls(control)
        control, intervention = self.select_control(control, input_data, timestamp)
        if self.frame_count % 2 == 1 and not at_limit:
            pose = self._vehicle.get_transform()
            yaw = math.radians(pose.rotation.yaw)
            packet = self.packet(input_data, timestamp)
            planner = self._waypoint_planner
            i = planner.route_index
            light = planner.next_traffic_lights[i]
            facts = {
                "ego_pose_world": [pose.location.x, pose.location.y, pose.location.z, yaw],
                "teacher_target_speed_mps": float(self.teacher_data["target_speed"]),
                "teacher_planned_path_ego": self.teacher_data["route"],
                "teacher_hazards": {
                    k: self.teacher_data[k]
                    for k in ("vehicle_hazard", "walker_hazard", "light_hazard", "stop_sign_hazard")
                },
                "next_light_state": str(light.get_state()) if light is not None else None,
                "applicable_light_id": light.id if light is not None else None,
                "signal_visibility_valid": None,
                "next_light_distance_m": finite_distance(
                    planner.distances_to_next_traffic_lights[i]
                ),
                "next_stop_distance_m": finite_distance(planner.distances_to_next_stop_signs[i]),
                "world": capture_world_state(self._world),
                "fact_source": "privileged world and teacher; not sensor-only perception truth",
                "teacher_speed_reduced_by": {
                    k: self.teacher_data.get(k)
                    for k in (
                        "speed_reduced_by_obj_type",
                        "speed_reduced_by_obj_id",
                        "speed_reduced_by_obj_distance",
                    )
                },
                **self.extra_facts(),
            }
            self.collector.capture(packet, teacher_control, controls(control), facts, intervention)
        return control

    def destroy(self, results=None):
        if getattr(self, "collector", None):
            if self.collector.pending is not None and getattr(self, "_vehicle", None):
                self.collector.acknowledge(
                    self._world.get_snapshot().frame, controls(self._vehicle.get_control())
                )
            profile = self.collector.close()
            profile.update(
                noise_ticks=self.perturbation.noise_ticks,
                safety_stop_ticks=self.perturbation.safety_stop_ticks,
            )
            atomic_json(self.out / "collection_profile.json", profile)
        super().destroy(results)
