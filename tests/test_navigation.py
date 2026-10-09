"""Integrity checks for the separate navigation contract and runtime."""

import json
from copy import deepcopy
from pathlib import Path

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


def _movement(town, junction, entry, control, turn, approach=50.0, signal=(), stop=()):
    return {
        "town": town,
        "junction_id": junction,
        "entry_lane": [entry, 0, -1],
        "exit_lane": [entry + 100, 0, -1],
        "control": control,
        "planner_options": ["LANEFOLLOW", turn],
        "signal_ids": list(signal),
        "stop_ids": list(stop),
        "approach_m": approach,
        "valid": True,
        "has_opposite_entry": True,
        "has_left_entry": True,
    }


def test_catalog_selection_is_seeded_balanced_and_outcome_blind():
    from src.navigation.catalog import coverage, select_catalog

    movements = [
        _movement(t, j, e, ["signal"], turn, signal=[j * 10 + e])
        for t in ("TownA", "TownB")
        for j in range(3)
        for e, turn in enumerate(("LEFT", "RIGHT", "STRAIGHT"))
    ]
    movements += [_movement("TownA", 9, e, ["stop"], "STRAIGHT", stop=[90 + e]) for e in range(2)]
    movements.append({**_movement("TownB", 7, 1, ["stop"], "LEFT", stop=[99]), "excluded": True})
    families = ["signal_stop", "stop_sign", "route_junction"]
    first = select_catalog(movements, families, 6, seed=1)
    assert first == select_catalog(movements, families, 6, seed=1)
    cov = coverage(first)
    assert cov["signal_stop"]["towns"] == {"TownA": 3, "TownB": 3}
    # Two stop actors share six encounters round-robin; the excluded stop is never used.
    stops = [p["movement"]["stop_ids"][0] for p in first if p["family"] == "stop_sign"]
    assert sorted(stops) == [90, 90, 90, 91, 91, 91]
    assert cov["stop_sign"]["reused_movements"] == 4
    assert all(p["occurrence"] == 0 for p in first if p["family"] == "signal_stop")
    # Selection never reads outcome fields, so adding them cannot change the catalog.
    with_outcomes = [{**m, "expert_result": "collision"} for m in movements]
    assert [
        p["movement"]["junction_id"] for p in select_catalog(with_outcomes, families, 6, 1)
    ] == [p["movement"]["junction_id"] for p in first]


def test_min_speed_is_reported_separately_but_never_hides_safety_infractions():
    from src.navigation.metrics import summarize_records

    def record(**infractions):
        return {"status": "Completed", "scores": {"score_route": 100.0}, "infractions": infractions}

    summary = summarize_records(
        [
            record(min_speed_infractions=["slow"]),
            record(min_speed_infractions=["slow"], red_light=["ran"]),
            record(collisions_vehicle=["hit"]),
        ]
    )
    assert summary["safe_completed"] == 0
    assert summary["rule_safe_completed"] == 1
    assert summary["all_collisions"] == 1
    assert [e["min_speed_infractions"] for e in summary["episodes"]] == [1, 1, 0]


def _load_script(name):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_encounter_runner_retries_only_crashes_once_and_drops_interrupted_routes(tmp_path):
    from types import SimpleNamespace

    runner = _load_script("run_encounters")

    def episode(rid, status):
        return {"route_id": f"RouteScenario_{rid}_rep0", "status": status}

    plan = {
        # first attempt: 5 finishes, 6 crashes the simulator, 7 never gets a record
        "chunk_a0": [episode(5, "Completed"), episode(6, "Failed - Simulation crashed")],
        # retry: 6 now finishes (agent failure is a result), 7 still has no record
        "chunk_a1": [episode(6, "Failed - Agent timed out")],
    }
    calls = []

    def fake_run(args, xml, subset, chunk, slot=0, gate=None):
        calls.append((chunk.name, subset))
        chunk.mkdir()
        (chunk / "manifest.json").write_text("{}")
        (chunk / "summary.json").write_text(json.dumps({"episodes": plan[chunk.name]}))

    runner.run_chunk = fake_run
    args = SimpleNamespace(out=tmp_path)
    attempts, finals = runner.settle_group(args, None, [5, 6, 7], tmp_path / "chunk")
    # 7 failed to record twice, so the one-retry budget is spent and it stays unsettled.
    assert calls == [("chunk_a0", "5-7"), ("chunk_a1", "6,7")]
    assert set(finals) == {5, 6} and "Agent" in finals[6]["status"]
    assert [a["status"] for a in attempts[7]] == ["no evaluator record"] * 2
    # Resuming re-reads the same attempts without running anything again.
    calls.clear()
    assert runner.settle_group(args, None, [5, 6, 7], tmp_path / "chunk")[1].keys() == {5, 6}
    assert calls == []

    stopped = tmp_path / "stopped"
    stopped.mkdir()
    (stopped / "summary.json").write_text(
        json.dumps({"episodes": [episode(1, "Completed"), episode(2, "Failed")]})
    )
    (stopped / "stopped.json").write_text(json.dumps({"reason": "user_stop"}))
    records, was_stopped = runner.chunk_records(stopped)
    assert was_stopped and set(records) == {1}  # route 2 was interrupted, not a result


def test_encounter_runner_keeps_raw_records_written_before_a_native_abort(tmp_path):
    runner = _load_script("run_encounters")
    chunk = tmp_path / "chunk_a0"
    chunk.mkdir()
    raw = {"_checkpoint": {"records": [
        {"route_id": "RouteScenario_4_rep0", "status": "Perfect", "scores": {"score_route": 100},
         "infractions": {}},
        {"route_id": "RouteScenario_5_rep0", "status": "Failed - Simulation crashed",
         "scores": {"score_route": 10}, "infractions": {}},
    ]}}  # fmt: skip
    (chunk / "leaderboard.json").write_text(json.dumps(raw))
    records, stopped = runner.chunk_records(chunk)
    assert not stopped and set(records) == {4, 5}
    assert records[4]["completed"] and runner.CRASH_STATUS in records[5]["status"]


def test_multi_route_stop_fails_only_the_route_that_was_running():
    from src.navigation.metrics import summarize_records

    done = {"status": "Perfect", "scores": {"score_route": 100.0}, "infractions": {}}
    summary = summarize_records([done, done], execution_failure=True, last_route_only=True)
    assert [e["completed"] for e in summary["episodes"]] == [True, False]
    assert summarize_records([done, done], execution_failure=True)["safe_completed"] == 0


def test_refused_attempts_are_not_driving_failures_and_are_bounded(tmp_path):
    from types import SimpleNamespace

    runner = _load_script("run_encounters")
    calls = []

    def refused(args, xml, subset, chunk, slot=0, gate=None):
        calls.append(chunk.name)  # e.g. lock busy or resources short: no manifest written

    runner.run_chunk = refused
    attempts, finals = runner.settle_group(
        SimpleNamespace(out=tmp_path), None, [1, 2], tmp_path / "c"
    )
    assert finals == {} and len(calls) == 3
    assert {a["status"] for a in attempts[1]} == {"not started"}


def test_slot_ports_and_sibling_servers(monkeypatch, tmp_path):
    import psutil

    from src.navigation import runtime as rt

    config = {"port": 2100, "traffic_manager_port": 8100, "max_servers": 3, "slot_port_stride": 100}
    assert rt.slot_ports(config, 0) == (2100, 8100)
    assert rt.slot_ports(config, 2) == (2300, 8300)
    assert rt.navigation_ports(config) == {2100, 2200, 2300}
    with pytest.raises(ValueError):
        rt.slot_ports(config, 3)

    class Proc:
        def __init__(self, cmdline):
            self.pid, self.info = 7, {"pid": 7, "cmdline": cmdline, "status": "running"}

    carla = "/x/CarlaUE4-Linux-Shipping CarlaUE4 /Game/Carla/Maps/Town03_Opt -carla-rpc-port={}"
    for port, allowed in ((2200, True), (2000, False)):
        procs = [Proc(carla.format(port).split())]
        monkeypatch.setattr(psutil, "process_iter", lambda attrs, procs=procs: procs)
        if allowed:
            rt.check_resources(tmp_path, 0, {2200, 2300})
        else:
            with pytest.raises(RuntimeError):  # a foreign (e.g. historical) server
                rt.check_resources(tmp_path, 0, {2200, 2300})
    trainer = [Proc("python scripts/train_cardreamer.py".split())]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: trainer)
    with pytest.raises(RuntimeError):
        rt.check_resources(tmp_path, 0, {2200})


def test_externally_killed_attempt_keeps_records_and_spends_no_retry(tmp_path):
    from types import SimpleNamespace

    runner = _load_script("run_encounters")
    base = tmp_path / "chunk"
    killed = tmp_path / "chunk_a0"
    killed.mkdir()
    (killed / "manifest.json").write_text("{}")
    raw = {"_checkpoint": {"records": [
        {"route_id": "RouteScenario_1_rep0", "status": "Completed",
         "scores": {"score_route": 100}, "infractions": {}},
    ]}}  # fmt: skip
    (killed / "leaderboard.json").write_text(json.dumps(raw))
    (killed / "interrupted.json").write_text(json.dumps({"reason": "host memory pressure"}))
    (tmp_path / "chunk_a0.routes.json").write_text(json.dumps([1, 2, 3]))
    calls = []

    def rerun(args, xml, subset, chunk, slot=0, gate=None):
        calls.append(subset)
        chunk.mkdir()
        (chunk / "manifest.json").write_text("{}")
        episodes = [{"route_id": f"RouteScenario_{r}_rep0", "status": "Completed"} for r in (2, 3)]
        (chunk / "summary.json").write_text(json.dumps({"episodes": episodes}))

    runner.run_chunk = rerun
    attempts, finals = runner.settle_group(SimpleNamespace(out=tmp_path), None, [1, 2, 3], base)
    assert calls == ["2,3"] and set(finals) == {1, 2, 3}
    assert len(attempts[1]) == 1 and len(attempts[2]) == 1  # no failure entries for the kill


def test_memory_watchdog_pauses_only_the_newest_chunk(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    runner = _load_script("run_encounters")
    gate = runner.LaunchGate({}, tmp_path / "STOP")
    old, new = tmp_path / "old_a0", tmp_path / "new_a0"
    for i, chunk in enumerate((old, new)):
        chunk.mkdir()
        gate.running[chunk] = float(i)
    monkeypatch.setattr(runner.psutil, "virtual_memory", lambda: SimpleNamespace(available=4e9))
    ticks = iter([False, False, True])  # two checks, then the run ends

    class Done:
        def wait(self, seconds):
            return next(ticks)

    events = []
    runner.memory_watchdog(gate, 6.0, events, threading.Lock(), Done())
    # The second critical check falls inside the 60 s back-off: one pause, newest chunk only.
    assert (new / "STOP").exists() and not (old / "STOP").exists()
    assert events == [{"chunk": "new_a0", "available_gb": 4.0}]
