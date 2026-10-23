"""Array/JSON-only policy IPC. No pickle, training labels or CARLA objects cross it."""

import io
import json
import os
import select
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.navigation.contracts import VIEWS, CameraCalibration, SensorPacket
from src.navigation.policy_input_contract import LEGACY_SCHEMA, SUPPORTED_SCHEMAS

FIELDS = {"metadata", "navigation", "ego", *(f"rgb_{v}" for v in VIEWS)}
PACKET_FIELDS = {"frame", "timestamp", "rgb", "calibration", "navigation_waypoints", "ego_motion"}
MAX_PAYLOAD = 8 * 1024 * 1024


def encode_packet(packet):
    if type(packet) is not SensorPacket or set(vars(packet)) != PACKET_FIELDS:
        raise ValueError("Only exact SensorPacket fields may cross the policy boundary")
    packet.validate()
    metadata = {
        "frame": packet.frame,
        "timestamp": packet.timestamp,
        "calibration": {k: asdict(v) for k, v in packet.calibration.items()},
    }
    buffer = io.BytesIO()
    np.savez(
        buffer,
        metadata=np.frombuffer(json.dumps(metadata).encode(), np.uint8),
        navigation=packet.navigation_waypoints,
        ego=packet.ego_motion,
        **{f"rgb_{v}": packet.rgb[v] for v in VIEWS},
    )
    data = buffer.getvalue()
    if len(data) > MAX_PAYLOAD:
        raise ValueError("Policy payload too large")
    return data


def decode_packet(payload):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("Policy payload too large")
    with np.load(io.BytesIO(payload), allow_pickle=False) as values:
        if set(values.files) != FIELDS:
            raise ValueError("Unexpected policy payload fields")
        meta = json.loads(values["metadata"].tobytes())
        if set(meta) != {"frame", "timestamp", "calibration"}:
            raise ValueError("Unexpected metadata fields")
        packet = SensorPacket(
            meta["frame"],
            meta["timestamp"],
            {v: values[f"rgb_{v}"] for v in VIEWS},
            {v: CameraCalibration(**c) for v, c in meta["calibration"].items()},
            values["navigation"],
            values["ego"],
        )
    packet.validate()
    return packet


class PolicyClient:
    def __init__(
        self,
        checkpoint,
        stderr_path,
        rule_overrides=False,
        device="cuda",
        input_schema=LEGACY_SCHEMA,
    ):
        if type(input_schema) is not str or input_schema not in SUPPORTED_SCHEMAS:
            raise ValueError("Unsupported requested policy input schema")
        self.input_schema = input_schema
        root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        for key in ("CARLA_ROOT", "SCENARIO_RUNNER_ROOT", "LEADERBOARD_ROOT"):
            environment.pop(key, None)
        environment["PYTHONPATH"] = str(root)
        self.log = Path(stderr_path).open("w")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "src.navigation.policy_worker",
                "--checkpoint",
                str(checkpoint),
                "--device",
                device,
                "--input-schema",
                input_schema,
                *(["--rule-overrides"] if rule_overrides else []),
            ],
            cwd=root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
        )
        try:
            ready = self.read(30)
            if (
                ready.get("status") != "ready"
                or ready.get("policy_input_schema") != self.input_schema
            ):
                raise RuntimeError("Camera policy failed readiness")
        except BaseException:
            self.close()
            raise

    def read(self, timeout=10):
        readable, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not readable:
            raise TimeoutError("Camera policy IPC timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Camera policy exited; see policy_stderr.log")
        return json.loads(line)

    def step(self, packet, feedback=None):
        if self.input_schema == LEGACY_SCHEMA:
            if feedback is not None:
                raise ValueError("Legacy policy cannot receive actuator feedback")
            data = encode_packet(packet)
        else:
            from src.navigation.ego_action_feedback import encode_step

            data = encode_step(packet, feedback)
        self.process.stdin.write(struct.pack("<Q", len(data)))
        self.process.stdin.write(data)
        self.process.stdin.flush()
        result = self.read()
        if result["frame"] != packet.frame:
            raise ValueError("Stale camera policy reply")
        if self.input_schema != LEGACY_SCHEMA and (
            result.get("policy_input_schema") != self.input_schema
            or result.get("ego_action_feedback") != vars(feedback)
        ):
            raise ValueError("Missing or inconsistent actuator feedback receipt")
        return result

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(struct.pack("<Q", 0))
                    self.process.stdin.flush()
                except BrokenPipeError:
                    pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.terminate()  # Only this owned, unresponsive inference child.
                    self.process.wait(timeout=10)
        finally:
            self.log.close()
