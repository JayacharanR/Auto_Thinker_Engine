"""Adversarial tests of the actual pinned CARLA criteria, without a server.

Install the optional harness with scripts/setup_navigation.py --install first.
Geometry uses real CARLA classes; providers supply controlled scenario state.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def harness(monkeypatch):
    if not Path("outputs/navigation_vendor/carla_garage/.git").exists():
        pytest.skip("Optional pinned navigation harness is not installed")
    monkeypatch.delenv("DATAGEN", raising=False)
    from src.navigation.vendor import activate

    activate()
    import carla
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider as provider
    from srunner.scenariomanager.scenarioatomics import atomic_criteria as criteria

    monkeypatch.setattr(provider, "get_world", lambda: None)
    monkeypatch.setattr(provider, "get_map", lambda: None)
    monkeypatch.setattr(provider, "get_all_actors", lambda: [])
    return carla, provider, criteria


def test_declared_empty_training_background_does_not_spawn_and_canonical_factory_is_restored(
    harness,
    monkeypatch,
):
    import py_trees

    import src.navigation.harness as adapter

    sentinel = object()
    monkeypatch.setattr(adapter, "_pinned_background", lambda *a, **k: sentinel)
    monkeypatch.setitem(adapter.SCENARIO_LOG, "empty_training_background", False)
    assert adapter._background_for_scope(None, None) is sentinel
    monkeypatch.setitem(adapter.SCENARIO_LOG, "empty_training_background", True)
    node = adapter._background_for_scope(None, None)
    assert isinstance(node, adapter.EmptyTrainingBackground)
    assert node.update() == py_trees.common.Status.RUNNING


def test_exported_development_route_replays_same_length(harness, tmp_path):
    import xml.etree.ElementTree as ET

    import numpy as np
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    from src.navigation.routes import create_development_route

    carla = harness[0]
    root = Path.home() / "software/CARLA_0.9.15"
    xodr = root / "CarlaUE4/Content/Carla/Maps/OpenDrive/Town10HD_Opt.xodr"
    if not xodr.exists():
        pytest.skip("Optional simulator maps unavailable")
    route_path = tmp_path / "route.xml"
    manifest = create_development_route(
        root, route_path, town="Town10HD_Opt", minimum=300, maximum=450
    )
    planner = GlobalRoutePlanner(carla.Map("Town10HD_Opt", xodr.read_text()), 1.0)
    endpoints = [
        carla.Location(**{k: float(v) for k, v in node.attrib.items()})
        for node in ET.parse(route_path).findall(".//position")
    ]
    path = planner.trace_route(*endpoints)
    xyz = np.array(
        [[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w, _ in path]
    )
    length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    assert length == pytest.approx(manifest["length_m"], abs=0.001)
    assert 300 <= length <= 450


@pytest.mark.parametrize("red,x,violations", [(True, -5, 0), (True, 1.5, 1), (False, 1.5, 0)])
def test_red_light_checks_stop_line_not_merely_motion(harness, monkeypatch, red, x, violations):
    carla, provider, criteria = harness
    pose = carla.Transform(carla.Location(x=x), carla.Rotation(yaw=0))
    wp = SimpleNamespace(
        road_id=1,
        lane_id=1,
        lane_width=3.5,
        transform=carla.Transform(carla.Location(), carla.Rotation(yaw=0)),
    )
    world_map = SimpleNamespace(get_waypoint=lambda p: wp)
    monkeypatch.setattr(provider, "get_map", lambda: world_map)
    monkeypatch.setattr(provider, "get_transform", lambda actor: pose)
    actor = SimpleNamespace(
        bounding_box=carla.BoundingBox(carla.Location(), carla.Vector3D(1, 1, 1))
    )
    light = SimpleNamespace(
        id=99,
        state=(carla.TrafficLightState.Red if red else carla.TrafficLightState.Green),
        get_transform=lambda: carla.Transform(),
    )
    criterion = criteria.RunningRedLightTest(actor)
    criterion._list_traffic_lights = [(light, carla.Location(), [wp])]
    criterion.update()
    criterion.update()  # A continuous crossing is one event, not an event every tick.
    assert criterion.actual_value == violations
    assert len(criterion.events) == violations


@pytest.mark.parametrize("speed,violations", [(0.0, 0), (0.5, 1)])
def test_stop_requires_actual_stopping_not_time_near_sign(harness, monkeypatch, speed, violations):
    carla, provider, criteria = harness
    actor = SimpleNamespace(get_transform=lambda: carla.Transform())
    stop = SimpleNamespace(id=17, get_transform=lambda: carla.Transform())
    monkeypatch.setattr(provider, "get_velocity", lambda actor: speed)
    criterion = criteria.RunningStopTest(actor)
    criterion._target_stop_sign = stop
    monkeypatch.setattr(criterion, "_get_waypoints", lambda actor: [])
    inside = [True]
    monkeypatch.setattr(criterion, "is_actor_affected_by_stop", lambda wps, stop: inside[0])
    for _ in range(100):
        criterion.update()
    inside[0] = False
    criterion.update()
    assert criterion.actual_value == violations


def test_pinned_native_collision_metric_filters_stationary_ego_and_needs_recorder_audit(
    harness, monkeypatch
):
    carla, provider, criteria = harness
    speed = [0.0]
    monkeypatch.setattr(provider, "get_location", lambda actor: carla.Location())
    monkeypatch.setattr(provider, "get_velocity", lambda actor: speed[0])
    criterion = criteria.CollisionTest(SimpleNamespace())
    event = SimpleNamespace(other_actor=SimpleNamespace(id=55, type_id="vehicle.audi.tt"))
    criterion._count_collisions(event)
    assert criterion.actual_value == 0 and not criterion.events
    speed[0] = 0.5
    criterion._count_collisions(event)
    assert criterion.actual_value == 1 and len(criterion.events) == 1


def test_stationary_agent_cannot_finish_and_shortcut_leaves_route(harness, monkeypatch):
    carla, provider, criteria = harness
    location = [carla.Location(x=0)]
    monkeypatch.setattr(provider, "get_location", lambda actor: location[0])
    route = [
        (carla.Transform(carla.Location(x=x), carla.Rotation(yaw=0)), None)
        for x in range(0, 101, 2)
    ]
    actor = SimpleNamespace()
    completion = criteria.RouteCompletionTest(actor, route)
    in_route = criteria.InRouteTest(actor, route, terminate_on_failure=True)
    for _ in range(100):
        completion.update()
    assert completion.actual_value == 0
    location[0] = carla.Location(x=100)
    in_route.update()
    completion.update()
    assert in_route.test_status == "FAILURE"
    assert completion.test_status != "SUCCESS"


def test_pinned_expert_has_no_forced_throttle_or_actor_deletion(harness):
    # Guards the deliberate vendor adaptation, beyond merely checking config flags.
    import ast

    path = Path("outputs/navigation_vendor/carla_garage/team_code/autopilot.py")
    source = path.read_text()
    assert "control.throttle = 1\n      control.brake = 0" not in source
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "destroy"
        for node in ast.walk(tree)
    )


def test_manager_adapter_matches_actual_tick_signature_and_stop(harness, monkeypatch, tmp_path):
    from leaderboard.scenarios.scenario_manager_local import ScenarioManager

    from src.navigation.harness import StopAwareManager

    manager = StopAwareManager.__new__(StopAwareManager)
    manager.configure_limits(tmp_path, max_steps=1, max_wall=60)
    manager._timestamp_last_run = 0
    manager._running = True

    def tick(self):
        self._timestamp_last_run += 0.05

    monkeypatch.setattr(ScenarioManager, "_tick_scenario", tick)
    manager._tick_scenario()  # Called without arguments by upstream run_scenario.
    assert manager.navigation_ticks == 1
    manager._tick_scenario()
    assert not manager._running
    assert manager.navigation_stop_reason == "profile_step_budget"
    manager._running = True
    monkeypatch.setattr(ScenarioManager, "stop_scenario", lambda self: None)
    manager.stop_scenario()
    assert not manager._running  # Also exits the upstream scenario-builder thread.


def test_camera_track_matches_pinned_validator_enum(harness):
    import yaml
    from leaderboard.autoagents.agent_wrapper_local import validate_sensor_configuration

    from src.navigation.camera_agent import Track
    from src.navigation.sensors import camera_specs

    config = yaml.safe_load(Path("configs/navigation.yaml").read_text())
    # The vendor has two distinct Track enum classes with identical printed names.
    # Only the canonical enum is accepted, even for its local agent wrapper.
    validate_sensor_configuration(camera_specs(config), Track.MAP, "MAP")


def test_each_route_records_separately_without_overwriting_console(harness, monkeypatch, tmp_path):
    from leaderboard.leaderboard_evaluator_local import LeaderboardEvaluator

    from src.navigation.harness import NavigationEvaluator

    evaluator = NavigationEvaluator.__new__(NavigationEvaluator)
    evaluator.manager = SimpleNamespace(navigation_stop_reason=None, start_route=lambda: None)
    args = SimpleNamespace(
        navigation_out=str(tmp_path),
        traffic_manager_seed=42,
        seed_per_route=True,
        record=str(tmp_path),
    )
    console = tmp_path.with_suffix(".log")
    console.write_text("console evidence\n")

    def upstream(self, args, config):
        # Execute the exact filename convention in the pinned local harness.
        Path(f"{args.record}.log").write_bytes(b"CARLA_RECORDER")
        return config.name

    monkeypatch.setattr(LeaderboardEvaluator, "_load_and_run_scenario", upstream)
    for route in (86, 101):
        config = SimpleNamespace(name=f"RouteScenario_{route}", repetition_index=0)
        evaluator._load_and_run_scenario(args, config)
        assert args.record == str(tmp_path)
    assert console.read_text() == "console evidence\n"
    assert len(list((tmp_path / "recordings").glob("*.log"))) == 2
    with pytest.raises(FileExistsError, match="overwrite recorder"):
        evaluator._load_and_run_scenario(args, config)
