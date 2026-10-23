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
    "src.navigation.training_gap_agent",
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
    "src.navigation.causal_learning",
    "src.navigation.spatial_data",
    "src.navigation.jepa_pretraining",
    "src.navigation.rule_prefix",
    "src.navigation.rule_suite",
    "src.navigation.rule_case_contract",
    "src.navigation.rule_case_runtime",
    "src.navigation.trace_storage",
    "src.navigation.native_opportunities",
    "src.navigation.final_audit",
    "src.navigation.recorder_collisions",
    "src.navigation.label_budget",
    "src.navigation.final_evaluation",
    "src.navigation.recorder_actions",
    "src.navigation.dynamics_data",
    "src.navigation.dynamics_learning",
    "src.navigation.label_collection",
    "src.navigation.labeled_collection_agent",
    "src.navigation.intervention_collection",
    "src.navigation.native_cleanup",
    "src.navigation.native_session",
    "src.navigation.native_process_owner",
    "src.navigation.native_process_client",
    "scripts.audit_navigation_interventions",
    "scripts.probe_navigation_depth_geometry",
    "scripts.assess_navigation_dense_perception",
    "scripts.fit_navigation_box_depth",
    "scripts.navigation_label_rig",
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
    parser.add_argument("--input-schema", default="navigation-sensor-packet-v1")
    args = parser.parse_args()
    sys.meta_path.insert(0, PrivilegeGuard())
    import torch

    from src.navigation.policy_input_contract import (
        ACTUATOR_SCHEMA,
        policy_input_schema,
    )
    from src.navigation.policy_ipc import MAX_PAYLOAD, decode_packet
    from src.navigation.policy_runtime import CameraPolicy

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    schema = policy_input_schema(state)
    if args.input_schema != schema:
        raise ValueError("Checkpoint and transport input schema differ")
    if state.get("architecture") == "spatial-visual-facts-v3":
        from src.navigation.fact_policy import FactSpatialPolicy

        policy = FactSpatialPolicy(state, args.device, args.rule_overrides)
    else:
        policy = CameraPolicy(args.checkpoint, args.device, args.rule_overrides)
    del state
    if schema == ACTUATOR_SCHEMA:
        from src.navigation.ego_action_feedback import decode_step
        from src.navigation.feedback_policy import FeedbackPolicySession

        policy = FeedbackPolicySession(policy)
    print(
        json.dumps(
            {"status": "ready", "forbidden_imports": list(FORBIDDEN), "policy_input_schema": schema}
        ),
        flush=True,
    )
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
        payload = read_exact(sys.stdin.buffer, size)
        if schema == ACTUATOR_SCHEMA:
            packet, feedback = decode_step(payload)
            result = policy.step(packet, feedback)
        else:
            packet = decode_packet(payload)
            result = policy.step(packet)
        if any(name in sys.modules for name in FORBIDDEN):
            raise RuntimeError("Privileged module entered camera inference process")
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
