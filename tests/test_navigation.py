"""Integrity checks for the separate navigation contract and runtime."""

import json
from copy import deepcopy

import numpy as np
import pytest
import yaml

from src.navigation.contracts import ControlCommand
from src.navigation.metrics import execution_status, summarize_records, wilson_interval
from src.navigation.replay import compare_world_prefixes
from src.navigation.routes import verify_route
from src.navigation.runtime import RunLock, atomic_json
from src.navigation.sensors import FrameBarrier, SensorTimeout, calibrations, camera_specs


def test_barrier_rejects_missing_frame_instead_of_reusing_latest():
    barrier = FrameBarrier(names=("front",))
    barrier.put("front", 9, 0.45, "old")
    barrier.put("front", 11, 0.55, "future")
    with pytest.raises(SensorTimeout, match="received 11"):
        barrier.collect(10, 0.01)
    assert barrier.collect(11, 0.01)["front"][2] == "future"


def test_barrier_accepts_only_matching_frames_and_timestamps():
    barrier = FrameBarrier(names=("front", "left"))
    barrier.put("left", 5, 0.25, "left")
    barrier.put("front", 5, 0.25, "front")
    assert set(barrier.collect(5)) == {"front", "left"}
    barrier.put("left", 6, 0.31, "left")
    barrier.put("front", 6, 0.30, "front")
    with pytest.raises(SensorTimeout, match="timestamps"):
        barrier.collect(6)


def test_barrier_missing_and_overflow_are_explicit():
    barrier = FrameBarrier(names=("front",), capacity=1)
    with pytest.raises(SensorTimeout, match="Missing"):
        barrier.collect(1, 0.001)
    barrier.put("front", 1, 0.05, None)
    barrier.put("front", 2, 0.10, None)
    with pytest.raises(SensorTimeout, match="overflow"):
        barrier.collect(1)


def test_camera_intrinsics_and_distinct_side_rotations():
    with open("configs/navigation.yaml") as f:
        config = yaml.safe_load(f)
    assert [s["yaw"] for s in camera_specs(config)] == [0, -60, 60]
    c = calibrations(config)["front"]
    assert c.intrinsic[0, 0] == pytest.approx(512 / (2 * np.tan(np.deg2rad(50))))
    assert c.intrinsic[0, 2] == 256
    assert c.intrinsic[1, 2] == 144


@pytest.mark.parametrize("values", [(np.nan, 0, 0), (2, 0, 0), (0, 0.3, 0.8)])
def test_invalid_controls_do_not_silently_reach_actuators(values):
    with pytest.raises(ValueError):
        ControlCommand(*values).validate()


def test_safe_completion_requires_arrival_and_no_infractions():
    records = [
        {"status": "Completed", "scores": {"score_route": 100}, "infractions": {}},
        {"status": "Failed - Agent crashed", "scores": {"score_route": 100}},
        {
            "status": "Completed",
            "scores": {"score_route": 100},
            "infractions": {
                "collisions_vehicle": ["rear-ended", "crossing"],
                "red_light": ["crossed red"],
            },
        },
        {"status": "Failed - Agent blocked", "scores": {"score_route": 0}},
    ]
    summary = summarize_records(records)
    assert summary["safe_completed"] == 1
    assert summary["all_collisions"] == 2  # Fault does not filter the count.
    assert summary["episodes"][2]["red_violations"] == 1
    assert not summary["episodes"][1]["completed"]


def test_confidence_interval_does_not_claim_zero_risk():
    lower, upper = wilson_interval(36, 36)
    assert 0.90 < lower < 0.91
    assert upper == pytest.approx(1)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_profile_setup_failure_cannot_report_success():
    failed = [{"status": "Failed - Agent couldn't be set up"}]
    assert execution_status(failed, False, None, 40) == "failed"
    assert execution_status(failed, False, None) == "failed"
    assert execution_status([], False, "profile_step_budget", 40) == "failed"
    truncated = [{"status": "Failed"}]
    assert execution_status(truncated, False, "profile_step_budget", 40) == "profile_complete"
    assert execution_status(truncated, False, "user_stop", 40) == "stopped"
    assert execution_status(truncated, False, "wall_budget", 40) == "failed"
    assert execution_status(truncated, False, None) == "evaluation_complete"


def test_repeatability_checks_traffic_and_lights_not_only_ego():
    actor = {
        "id": 1,
        "type": "vehicle.test",
        "role": "background",
        "pose": [0, 0, 0, 0, 0, 0],
        "velocity": [0, 0, 0],
    }
    light = dict(actor, id=2, type="traffic.traffic_light", signal="Red", phase_elapsed=2.0)
    trace = [{"frame": 100, "actors": [actor, light]}]
    other = deepcopy(trace)
    other[0]["actors"][0]["id"] = 200  # IDs themselves are not physical state.
    assert compare_world_prefixes(trace, other)["observable_prefix_within_tolerance"]
    other[0]["actors"][1]["signal"] = "Green"
    assert not compare_world_prefixes(trace, other)["observable_prefix_within_tolerance"]
    other = deepcopy(trace)
    other[0]["actors"][0]["pose"][0] = 1
    assert not compare_world_prefixes(trace, other)["observable_prefix_within_tolerance"]
    other[0]["actors"].pop()
    assert not compare_world_prefixes(trace, other)["actor_type_role_sequence_equal"]


def test_upstream_perfect_arrival_and_runtime_failure_precedence():
    record = {
        "status": "Perfect",
        "scores": {"score_route": 100},
        "infractions": {"collisions_vehicle": [], "red_light": []},
    }
    assert summarize_records([record])["safe_completed"] == 1
    assert summarize_records([record], execution_failure=True)["safe_completed"] == 0


def test_lock_rejects_duplicate_and_releases_on_exit(tmp_path):
    lock = tmp_path / ".lock"
    with RunLock(lock):
        with pytest.raises(RuntimeError, match="Another navigation"):
            with RunLock(lock):
                pass
    with RunLock(lock):
        pass


def test_atomic_records_and_route_tamper_detection(tmp_path):
    path = tmp_path / "status.json"
    atomic_json(path, {"stage": "ready"})
    assert json.loads(path.read_text()) == {"stage": "ready"}
    assert not path.with_suffix(".json.tmp").exists()
    route = tmp_path / "route.xml"
    route.write_text("changed")
    atomic_json(route.with_suffix(".json"), {"route_sha256": "wrong"})
    with pytest.raises(ValueError, match="registered manifest"):
        verify_route(route)


def test_encounter_geometry_helpers():
    from src.navigation.encounters import candidate_valid, path_length, summarize, yaw_delta

    assert yaw_delta(170, -170) == 20
    assert yaw_delta(-170, 170) == -20
    assert yaw_delta(0, 180) == 180 and yaw_delta(0, -180) == 180
    assert path_length([[0, 0, 0], [3, 4, 0], [3, 4, 2]]) == 7
    base = {
        "approach_m": 50.0,
        "planner_through_junction": True,
        "geometric_length_m": 100.0,
        "planner_length_m": 105.0,
        "control": ["signal"],
        "junction_id": 7,
        "signal_ids": [1],
        "stop_ids": [],
    }
    assert candidate_valid(base)
    # Unfair approach, a planner detour, or a different junction are never valid.
    assert not candidate_valid({**base, "approach_m": 29.0})
    assert not candidate_valid({**base, "planner_length_m": 140.0})
    assert not candidate_valid({**base, "planner_through_junction": False})
    stop = {**base, "control": ["stop"], "signal_ids": [], "stop_ids": [3], "junction_id": 8}
    summary = summarize([base, stop, {**base, "approach_m": 10.0}])
    assert summary["movements"] == 3 and summary["valid_movements"] == 2
    assert summary["valid_by_control"] == {"signal": 1, "stop": 1}
    assert summary["valid_stop_actors"] == 1 and summary["valid_junctions"] == 2
