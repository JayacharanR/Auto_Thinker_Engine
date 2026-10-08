"""Lifecycle adapter; scoring and scenario behavior remain in the pinned harness."""

import json
import random
import time
from pathlib import Path

import carla
import numpy as np
import torch
from leaderboard.leaderboard_evaluator_local import LeaderboardEvaluator
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.scenarios.scenario_manager_local import ScenarioManager
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from src.navigation.runtime import atomic_json
from src.navigation.scenarios import NAVIGATION_SCENARIOS

_pinned_scenario_classes = RouteScenario.get_all_scenario_classes
_pinned_build_scenarios = RouteScenario.build_scenarios
SCENARIO_LOG = {"path": None}


def _scenario_classes(self):
    classes = _pinned_scenario_classes(self)
    clash = set(classes) & set(NAVIGATION_SCENARIOS)
    if clash:
        raise RuntimeError(f"Navigation scenario names shadow pinned scenarios: {clash}")
    return {**classes, **NAVIGATION_SCENARIOS}


def _build_scenarios(self, ego_vehicle, debug=False):
    # Upstream prints and skips a scenario whose setup fails. An encounter whose
    # declared scenario never started is not that encounter, so record both outcomes.
    pending = list(self.missing_scenario_configurations)
    started = len(self.list_scenarios)
    _pinned_build_scenarios(self, ego_vehicle, debug)
    if SCENARIO_LOG["path"] is None:
        return
    new = self.list_scenarios[started:]
    names = {s.config.name for s in new}
    log = json.loads(SCENARIO_LOG["path"].read_text()) if SCENARIO_LOG["path"].exists() else {}
    for scenario in new:
        log.setdefault("started", []).append(
            {
                "name": scenario.config.name,
                "type": scenario.config.type,
                "ego_light_id": getattr(scenario, "ego_light_id", None),
            }
        )
    for config in pending:
        if config not in self.missing_scenario_configurations and config.name not in names:
            log.setdefault("skipped", []).append({"name": config.name, "type": config.type})
    if new or log.get("skipped"):
        atomic_json(SCENARIO_LOG["path"], log)


RouteScenario.get_all_scenario_classes = _scenario_classes
RouteScenario.build_scenarios = _build_scenarios


class StopAwareManager(ScenarioManager):
    def configure_limits(self, output, max_steps, max_wall):
        self.navigation_output = Path(output)
        self.navigation_max_steps = max_steps
        self.navigation_max_wall = max_wall
        self.navigation_started = time.monotonic()
        self.navigation_ticks = 0
        self.navigation_stop_reason = None

    def start_route(self):
        self.navigation_ticks = 0

    def _tick_scenario(self):
        reason = None
        if (self.navigation_output / "STOP").exists():
            reason = "user_stop"
        elif time.monotonic() - self.navigation_started >= self.navigation_max_wall:
            reason = "wall_budget"
        elif self.navigation_max_steps and self.navigation_ticks >= self.navigation_max_steps:
            reason = "profile_step_budget"
        if reason:
            self.navigation_stop_reason = reason
            atomic_json(self.navigation_output / "stopped.json", {"reason": reason})
            self._running = False
            return
        previous = self._timestamp_last_run
        super()._tick_scenario()
        if self._timestamp_last_run > previous:
            self.navigation_ticks += 1
        if self.navigation_ticks % 20 == 0:
            progress_path = self.navigation_output / "progress.json"
            progress = json.loads(progress_path.read_text())
            for criterion in self.scenario.get_criteria():
                if type(criterion).__name__ == "RouteCompletionTest":
                    progress["route_progress_percent"] = criterion.actual_value
            atomic_json(progress_path, progress)

    def stop_scenario(self):
        # The upstream background-builder thread loops on this flag. On a Python
        # exception the tick loop does not clear it, so cleanup must do so.
        self._running = False
        return super().stop_scenario()


class NavigationEvaluator(LeaderboardEvaluator):
    def __init__(self, args, statistics_manager):
        super().__init__(args, statistics_manager)
        self.manager = StopAwareManager(args.timeout, statistics_manager, args.debug)
        self.manager.configure_limits(args.navigation_out, args.max_steps, args.max_wall)
        self.navigation_halted = False
        self.navigation_seed = args.traffic_manager_seed

    def _load_and_run_scenario(self, args, config):
        # config.index is the position within a routes subset; the XML route id is
        # the stable encounter identity, so seeds and records key on it.
        if self.manager.navigation_stop_reason in ("user_stop", "wall_budget"):
            # Starting further routes would record each one as an instant failure.
            self.navigation_halted = True
            return True
        route_id = config.name.removeprefix("RouteScenario_")
        self.navigation_seed = args.traffic_manager_seed + (
            int(route_id) if args.seed_per_route else 0
        )
        out = Path(args.navigation_out)
        SCENARIO_LOG["path"] = out / "scenario_setup" / f"{route_id}.json"
        atomic_json(
            out / "current_route.json", {"route_id": route_id, "seed": self.navigation_seed}
        )
        self.manager.start_route()
        return super()._load_and_run_scenario(args, config)

    def _setup_simulation(self, args):
        # An owned server is already running. No external scripts, broad process matching,
        # force kills, or arbitrary port fallback are used by this adapter.
        client = carla.Client(args.host, args.port)
        client.set_timeout(120)
        world = client.get_world()
        if world.get_map().name.split("/")[-1] != args.navigation_town:
            # Load before creating Traffic Manager. The original integration
            # reloaded a world while its TM was active and hit a native crash.
            world = client.load_world(args.navigation_town)
        client.set_timeout(args.timeout)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / self.frame_rate
        world.apply_settings(settings)
        manager = client.get_trafficmanager(args.traffic_manager_port)
        manager.set_synchronous_mode(True)
        manager.set_hybrid_physics_mode(True)
        return client, args.timeout, manager, args.traffic_manager_port

    def _load_and_wait_for_world(self, args, town):
        # Every M0 episode owns a fresh server. A required map switch has
        # already happened before Traffic Manager creation in _setup_simulation.
        self.world = self.client.get_world()
        actual = self.world.get_map().name.split("/")[-1]
        if actual != town:
            raise RuntimeError(f"Server map {actual} differs from registered map {town}")
        self.world.reset_all_traffic_lights()
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(self.traffic_manager_port)
        CarlaDataProvider.set_world(self.world)
        seed = self.navigation_seed
        CarlaDataProvider.set_random_seed(seed)
        self.traffic_manager.set_random_device_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        CarlaDataProvider.active_scenarios = []
        self.world.tick()
        signs = []
        for actor in self.world.get_actors():
            if actor.type_id.startswith("traffic."):
                t = actor.get_transform()
                signs.append(
                    {
                        "id": actor.id,
                        "type": actor.type_id,
                        "xyz": [t.location.x, t.location.y, t.location.z],
                        "yaw": t.rotation.yaw,
                    }
                )
        atomic_json(
            Path(args.navigation_out) / "runtime_sign_inventory.json",
            {"map": actual, "signs": signs},
        )
