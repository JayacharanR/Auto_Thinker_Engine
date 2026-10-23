"""Leaderboard PDM-Lite adapter with synchronized camera evidence and no forced crawl.

This is explicitly a PRIVILEGED reference driver. It must never be represented
as the camera student. Vendor paths are activated by the runner before import.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from autopilot import AutoPilot

from src.navigation.contracts import VIEWS
from src.navigation.replay import capture_world_state
from src.navigation.runtime import atomic_json
from src.navigation.sensors import camera_specs
from src.navigation.video import VideoWriter


def get_entry_point():
    return "NavigationExpert"


class NavigationExpert(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        spec = json.loads(Path(path_to_conf_file).read_text())
        self.navigation = yaml.safe_load(Path(spec["config"]).read_text())
        self.out = Path(spec["out"]) / "episodes" / str(route_index)
        self.out.mkdir(parents=True, exist_ok=False)
        self.config.steer_noise = 0.0  # Collection perturbations must be explicit and logged.
        self.video = None
        self.trace = (self.out / "trace.jsonl").open("w")
        self.world_trace = (self.out / "world_trace.jsonl").open("w")
        self.timings = []
        self.first_wall = time.monotonic()
        self.first_timestamp = None
        self.last_frame = -1
        self.frame_count = 0
        self.navigation_spec = spec
        self.driver_label = "PRIVILEGED EXPERT"
        self.trace_policy_kind = "privileged_pdm_lite_modified"
        manifest = json.loads((Path(spec["out"]) / "manifest.json").read_text())
        xyz = np.array(
            [
                [t.location.x, t.location.y, t.location.z]
                for t, _ in self.org_dense_route_world_coord
            ]
        )
        length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        current = json.loads((Path(spec["out"]) / "current_route.json").read_text())
        lengths = manifest["route"].get("lengths_m")
        # A multi-route catalog registers one length per route id; never fall back.
        expected = lengths[current["route_id"]] if lengths else manifest["route"]["length_m"]
        atomic_json(
            self.out / "route_validation.json",
            {
                "route_id": current["route_id"],
                "seed": current["seed"],
                "expected_m": expected,
                "harness_m": length,
            },
        )
        if abs(length - expected) > max(10, 0.02 * expected):
            raise ValueError(
                f"Harness route length {length:.1f}m differs from manifest "
                f"{expected:.1f}m; refusing to silently change the mission"
            )

    def sensors(self):
        return super().sensors() + camera_specs(self.navigation)

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        started = time.monotonic()
        frames = [input_data[v][0] for v in VIEWS]
        if len(set(frames)) != 1 or frames[0] <= self.last_frame:
            raise RuntimeError(f"Unsynchronized or repeated camera frames: {frames}")
        self.last_frame = frames[0]
        rgb = {}
        c = self.navigation["camera"]
        for view in VIEWS:
            bgra = input_data[view][1]
            if bgra.shape != (c["height"], c["width"], 4) or bgra.dtype != np.uint8:
                raise ValueError(f"Invalid {view} camera output: {bgra.shape}/{bgra.dtype}")
            rgb[view] = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
        control = super().run_step(input_data, timestamp, sensors, plant)
        self.timings.append(time.monotonic() - started)
        self.first_timestamp = timestamp if self.first_timestamp is None else self.first_timestamp
        transform = self._vehicle.get_transform()
        location = transform.location
        point_count = len(self._waypoint_planner.route_points)
        progress = 100 * self._waypoint_planner.route_index / max(1, point_count - 1)
        record = {
            "frame": int(frames[0]),
            "timestamp": float(timestamp),
            "ego": [location.x, location.y, location.z, transform.rotation.yaw],
            "speed_mps": self._vehicle.get_velocity().length(),
            "control": [float(control.steer), float(control.throttle), float(control.brake)],
            "expert_route_progress_percent": float(progress),
            "expert_hazards": {
                "light": bool(self.traffic_light_hazard),
                "stop": bool(self.stop_sign_hazard),
                "vehicle": bool(self.vehicle_hazard),
            },
            "policy_kind": self.trace_policy_kind,
            "control_semantics": "privileged teacher proposal before collection mixing",
        }
        self.trace.write(json.dumps(record, allow_nan=False) + "\n")
        world_state = capture_world_state(self._world)
        if world_state["frame"] != int(frames[0]):
            raise RuntimeError("World audit snapshot differs from the camera frame")
        self.world_trace.write(json.dumps(world_state, allow_nan=False) + "\n")
        if self.step % 2 == 0:
            canvas = np.concatenate([rgb[v] for v in ("left", "front", "right")], axis=1)
            for i, view in enumerate(("left", "front", "right")):
                cv2.putText(
                    canvas,
                    view.upper(),
                    (i * c["width"] + 12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
            cv2.putText(
                canvas,
                f"{self.driver_label} | {record['speed_mps'] * 3.6:.1f} km/h | frame {frames[0]}",
                (12, c["height"] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            if self.video is None:
                self.video = VideoWriter(
                    self.out / "three_cameras.mp4", c["width"] * 3, c["height"], fps=10
                )
            self.video.write(canvas)
        self.frame_count += 1
        if self.frame_count % 20 == 0:
            self.trace.flush()
            self.world_trace.flush()
            elapsed = time.monotonic() - self.first_wall
            progress_path = Path(self.navigation_spec["out"]) / "progress.json"
            progress = json.loads(progress_path.read_text())
            progress.update(
                {
                    "status": "running",
                    "stage": "M0 expert",
                    "frames": self.frame_count,
                    "expert_route_progress_percent": record["expert_route_progress_percent"],
                    "elapsed_seconds": elapsed,
                    "frames_per_second": self.frame_count / elapsed,
                    "speed_kmh": record["speed_mps"] * 3.6,
                    "episode": str(self.out),
                    "updated_at": time.time(),
                },
            )
            atomic_json(progress_path, progress)
        return control

    def destroy(self, results=None):
        if getattr(self, "video", None):
            self.video.close()
        if getattr(self, "trace", None):
            self.trace.close()
        if getattr(self, "world_trace", None):
            self.world_trace.close()
        if getattr(self, "timings", None):
            elapsed = time.monotonic() - self.first_wall
            atomic_json(
                self.out / "profile.json",
                {
                    "frames": self.frame_count,
                    "wall_seconds": elapsed,
                    "simulation_seconds": (self.frame_count - 1) / 20,
                    "frames_per_second": self.frame_count / elapsed,
                    "expert_and_camera_p95_ms": float(np.percentile(self.timings, 95) * 1000),
                    "all_camera_frames_aligned": True,
                    "last_frame": self.last_frame,
                    "note": "Expert latency excludes video encoding and simulator/render time",
                },
            )
        super().destroy(results)
