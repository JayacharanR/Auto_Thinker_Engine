"""Fresh inference process: approved SensorPacket payloads and forbidden oracle imports."""

import argparse
import importlib.abc
import json
import struct
import sys

FORBIDDEN = (
    "carla",
    "srunner",
    "leaderboard",
    "autopilot",
    "agents",
    "src.navigation.replay",
    "src.navigation.pdm_agent",
    "src.navigation.collection_agent",
    "src.navigation.collection",
    "src.navigation.dataset",
    "src.navigation.learning_data",
    "src.navigation.dagger",
    "src.navigation.dagger_agent",
    "src.navigation.correction_learning",
    "src.navigation.rule_geometry",
    "src.navigation.signal_data",
    "src.navigation.signal_learning",
    "src.navigation.signal_detection_data",
    "src.navigation.signal_label_probe",
    "src.navigation.signal_expansion",
    "src.navigation.signal_roi_learning",
    "src.navigation.signal_temporal_audit",
    "src.navigation.approach_data",
)


class PrivilegeGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in FORBIDDEN):
            raise ImportError(f"Forbidden simulator/training import in camera policy: {fullname}")
        return None


def read_exact(stream, size):
    parts = bytearray()
    while len(parts) < size:
        chunk = stream.read(size - len(parts))
        if not chunk:
            raise EOFError("Truncated policy packet")
        parts.extend(chunk)
    return bytes(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rule-overrides", action="store_true")
    args = parser.parse_args()
    sys.meta_path.insert(0, PrivilegeGuard())
    from src.navigation.policy_ipc import MAX_PAYLOAD, decode_packet
    from src.navigation.policy_runtime import CameraPolicy

    policy = CameraPolicy(args.checkpoint, args.device, args.rule_overrides)
    print(json.dumps({"status": "ready", "forbidden_imports": list(FORBIDDEN)}), flush=True)
    while True:
        header = sys.stdin.buffer.read(8)
        if not header:
            return
        if len(header) != 8:
            raise ValueError("Truncated policy header")
        (size,) = struct.unpack("<Q", header)
        if size == 0:
            return
        if size > MAX_PAYLOAD:
            raise ValueError("Policy payload too large")
        packet = decode_packet(read_exact(sys.stdin.buffer, size))
        result = policy.step(packet)
        if any(name in sys.modules for name in FORBIDDEN):
            raise RuntimeError("Privileged module entered camera inference process")
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
