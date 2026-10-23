"""CARLA adapter: navigation/ego acquisition and scoring stay outside camera inference."""

import json
import math
import time
from pathlib import Path

import carla
import cv2
import numpy as np
import yaml
from leaderboard.autoagents.autonomous_agent import Track
from leaderboard.autoagents.autonomous_agent_local import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from src.navigation.collection import RouteProjector
from src.navigation.contracts import VIEWS, ControlCommand, SensorPacket
from src.navigation.policy_ipc import PolicyClient
from src.navigation.replay import capture_world_state
from src.navigation.runtime import atomic_json
from src.navigation.sensors import calibrations, camera_specs
from src.navigation.video import VideoWriter


def get_entry_point():
    return "NavigationCameraAgent"


class NavigationCameraAgent(AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        self.track = Track.MAP
        self.spec = json.loads(Path(path_to_conf_file).read_text())
        self.config = yaml.safe_load(Path(self.spec["config"]).read_text())
        self.out = Path(self.spec["out"]) / "episodes" / str(route_index)
        self.out.mkdir(parents=True, exist_ok=False)
        self.vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.vehicle.get_world()  # Audit only; never sent to the policy worker.
        points = [[t.location.x, t.location.y] for t, _ in self.org_dense_route_world_coord]
        self.projector = RouteProjector(points)
        self.calibration = calibrations(self.config)
        manifest = json.loads((Path(self.spec["out"]) / "manifest.json").read_text())
        self.policy_kind = manifest["policy_kind"]
        current = json.loads((Path(self.spec["out"]) / "current_route.json").read_text())
        route = manifest["route"]
        expected = (
            route["lengths_m"][current["route_id"]] if route.get("lengths_m") else route["length_m"]
        )
        xyz = np.array(
            [
                [t.location.x, t.location.y, t.location.z]
                for t, _ in self.org_dense_route_world_coord
            ]
        )
        length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        if abs(length - expected) > max(10, 0.02 * expected):
            raise ValueError("Camera mission length differs from frozen route manifest")
        atomic_json(
            self.out / "route_validation.json",
            {
                "route_id": current["route_id"],
                "seed": current["seed"],
                "expected_m": expected,
                "harness_m": length,
            },
        )
        self.trace = (self.out / "trace.jsonl").open("w")
        self.world_trace = (self.out / "world_trace.jsonl").open("w")
        self.video = None
        self.client = PolicyClient(
            self.spec["student"]["checkpoint"],
            self.out / "policy_stderr.log",
            self.spec["student"]["rule_overrides"],
        )
        self.frames, self.last_frame = 0, -1
        self.timings, self.interventions = [], 0
        self.started = time.monotonic()

    def sensors(self):
        return camera_specs(self.config)

    def run_step(self, input_data, timestamp, sensors=None):
        before = time.monotonic()
        frames = [input_data[v][0] for v in VIEWS]
        if len(set(frames)) != 1 or frames[0] <= self.last_frame:
            raise RuntimeError("Camera student received stale or misaligned views")
        self.last_frame = int(frames[0])
        pose = self.vehicle.get_transform()
        yaw = math.radians(pose.rotation.yaw)
        rgb = {v: np.ascontiguousarray(input_data[v][1][:, :, :3][:, :, ::-1]) for v in VIEWS}
        # Declared ground-truth ego odometry/localization matches collection. No other actor
        # state, traffic-light state, teacher path/speed or hazard enters SensorPacket.
        packet = SensorPacket(
            self.last_frame,
            float(timestamp),
            rgb,
            self.calibration,
            self.projector.project(np.array([pose.location.x, pose.location.y]), yaw),
            np.array(
                [
                    self.vehicle.get_velocity().length(),
                    yaw,
                    math.radians(self.vehicle.get_angular_velocity().z),
                ],
                np.float32,
            ),
        )
        result = self.client.step(packet)
        command = ControlCommand(*result["control"])
        command.validate()
        self.timings.append(time.monotonic() - before)
        self.interventions += int(bool(result["interventions"]))
        self.frames += 1
        self.trace.write(
            json.dumps(
                {
                    "frame": self.last_frame,
                    "timestamp": float(timestamp),
                    "ego": [pose.location.x, pose.location.y, pose.location.z, pose.rotation.yaw],
                    "speed_mps": float(packet.ego_motion[0]),
                    **result,
                    "policy_kind": self.policy_kind,
                },
                allow_nan=False,
            )
            + "\n"
        )
        world = capture_world_state(self.world)
        if world["frame"] != self.last_frame:
            raise RuntimeError("Audit-only world snapshot differs from camera frame")
        self.world_trace.write(json.dumps(world, allow_nan=False) + "\n")
        if self.frames % 2 == 1:
            canvas = np.concatenate([rgb[v] for v in ("left", "front", "right")], axis=1)
            cv2.putText(
                canvas,
                f"CAMERA STUDENT | {packet.ego_motion[0] * 3.6:.1f} km/h | "
                f"rule overrides {'ON' if self.spec['student']['rule_overrides'] else 'OFF'}",
                (12, 272),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            if self.video is None:
                self.video = VideoWriter(self.out / "three_cameras.mp4", 1536, 288, fps=10)
            self.video.write(canvas)
        if self.frames % 20 == 0:
            self.trace.flush()
            self.world_trace.flush()
            elapsed = time.monotonic() - self.started
            progress_path = Path(self.spec["out"]) / "progress.json"
            progress = json.loads(progress_path.read_text())
            progress.update(
                status="running",
                stage="M2 camera driving",
                frames=self.frames,
                frames_per_second=self.frames / elapsed,
                speed_kmh=float(packet.ego_motion[0] * 3.6),
                updated_at=time.time(),
            )
            atomic_json(progress_path, progress)
        control = carla.VehicleControl()
        control.steer, control.throttle, control.brake = (
            command.steer,
            command.throttle,
            command.brake,
        )
        return control

    def destroy(self, results=None):
        if getattr(self, "client", None):
            self.client.close()
        for name in ("trace", "world_trace", "video"):
            handle = getattr(self, name, None)
            if handle:
                handle.close()
        if getattr(self, "timings", None):
            elapsed = time.monotonic() - self.started
            atomic_json(
                self.out / "profile.json",
                {
                    "frames": self.frames,
                    "wall_seconds": elapsed,
                    "frames_per_second": self.frames / elapsed,
                    "camera_policy_ipc_p95_ms": float(np.percentile(self.timings, 95) * 1000),
                    "all_camera_frames_aligned": True,
                    "last_frame": self.last_frame,
                    "rule_overrides_enabled": self.spec["student"]["rule_overrides"],
                    "intervention_ticks": self.interventions,
                    "policy_inputs": "Three RGB views, fixed route geometry, ego odometry, "
                    "timestamp",
                    "note": "Inference/IPC/packet latency excludes simulator, video "
                    "and audit recording",
                },
            )
