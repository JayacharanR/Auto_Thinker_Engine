#!/usr/bin/env python3
"""Independently verify versioned ego feedback against the official native recorder."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.ego_action_feedback import EgoActionFeedback  # noqa:E402
from src.navigation.policy_input_contract import ACTUATOR_SCHEMA  # noqa:E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_feedback_rows(rows, aligned):
    if not rows:
        raise ValueError("A nonempty real versioned actuator trace is required")
    previous = None
    checked = 0
    for row in rows:
        frame = row["frame"]
        if row.get("policy_input_schema") != ACTUATOR_SCHEMA:
            raise ValueError("Every trace row must declare the versioned own-actuator schema")
        raw = row.get("ego_action_feedback")
        if type(raw) is not dict:
            raise ValueError("Missing exact actuator acknowledgment")
        feedback = EgoActionFeedback(**raw)
        feedback.validate(frame)
        actual = np.array([feedback.steer, feedback.throttle, feedback.brake])
        if frame not in aligned:
            raise ValueError("Native observation-frame actuator acknowledgment unavailable")
        if not np.allclose(actual, aligned[frame]["control"], rtol=0, atol=1e-5):
            raise ValueError("Own feedback differs from native observation-frame control")
        if row.get("actuator_history_ready") is not (previous is not None):
            raise ValueError("History readiness must use two real consecutive acknowledgments")
        if previous is not None:
            if frame != previous["frame"] + 1:
                raise ValueError("Versioned control trace has a stale, duplicate or missing frame")
            # Prior trace control includes an explicitly logged scripted prefix when present.
            # It is audit-only; neither that tag nor the proposed command enters feedback.
            if not np.allclose(actual, previous["control"], rtol=0, atol=1e-5):
                raise ValueError("Acknowledgment differs from the preceding applied command")
        previous = row
        checked += 1
    return dict(
        observations=checked,
        consecutive_prior_commands=checked - 1,
        own_actuator_differences=0,
        all_feedback_has_native_ack=True,
        no_safety_or_driving_qualification=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--recorder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    trace = args.episode / "trace.jsonl"
    world_path = args.episode / "world_trace.jsonl"
    try:
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        world = [json.loads(line) for line in world_path.read_text().splitlines()]
        if [r["frame"] for r in rows] != [r["frame"] for r in world]:
            raise ValueError("Full camera and world observation sequences differ")
        hero = [a["id"] for a in world[0]["actors"] if a["role"] == "hero"]
        if len(hero) != 1:
            raise ValueError("One physical hero is required for independent native audit")
        native, receipt = align_world_frames(vehicle_frames(args.recorder, hero[0]), world)
        result = verify_feedback_rows(rows, native)
        result.update(status="complete", alignment=receipt)
    except Exception as error:
        atomic_json(args.out / "assessment.json", dict(status="failed", error=str(error)))
        raise
    result.update(
        source_sha256=digest(__file__),
        trace_sha256=digest(trace),
        world_trace_sha256=digest(world_path),
        recorder_sha256=digest(args.recorder),
    )
    atomic_json(args.out / "assessment.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
