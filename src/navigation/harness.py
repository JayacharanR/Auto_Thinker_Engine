"""Lifecycle adapter; scoring and scenario behavior remain in the pinned harness."""

import json
import random
import time
from pathlib import Path

import carla
import numpy as np
import torch
from leaderboard.leaderboard_evaluator_local import LeaderboardEvaluator
from leaderboard.scenarios.scenario_manager_local import ScenarioManager
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from src.navigation.runtime import atomic_json


class StopAwareManager(ScenarioManager):
    def configure_limits(self, output, max_steps, max_wall):
        self.navigation_output = Path(output)
        self.navigation_max_steps = max_steps
        self.navigation_max_wall = max_wall
        self.navigation_started = time.monotonic()
        self.navigation_ticks = 0
        self.navigation_stop_reason = None

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
        CarlaDataProvider.set_random_seed(args.traffic_manager_seed)
        self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)
        random.seed(args.traffic_manager_seed)
        np.random.seed(args.traffic_manager_seed)
        torch.manual_seed(args.traffic_manager_seed)
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
