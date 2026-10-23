"""Predeclared encounter setup scenarios, registered alongside the pinned route scenarios.

NavigationSignalHold follows the pinned PriorityAtJunction pattern and its
TrafficLightFreezer atomic. The ego's light is red (other lights in the junction
green) until the ego has stood still near the stop line for `stationary_s`, or
`max_hold_s` has elapsed. A fixed clearance (others yellow, then all red) precedes
the ego's green, which then holds until the ego leaves the junction. The same rule
applies to every driver; the red-light criterion itself is the pinned evaluator's.
"""

from types import SimpleNamespace

import carla
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    AtomicBehavior,
    TrafficLightFreezer,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import RunningRedLightTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    WaitEndIntersection,
)
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.basic_scenario import BasicScenario

RED, YELLOW, GREEN = (
    carla.TrafficLightState.Red,
    carla.TrafficLightState.Yellow,
    carla.TrafficLightState.Green,
)


def pick_ego_light(entry_lane, light_lanes):
    """The single light whose red-light stop lanes (pinned evaluator geometry) include
    the ego's junction-entry lane; anything else is a setup error, never a guess."""
    matches = sorted(light for light, lanes in light_lanes.items() if entry_lane in lanes)
    if len(matches) != 1:
        raise ValueError(f"Entry lane {entry_lane} matches traffic lights {matches}")
    return matches[0]


def parameter(config, name, default):
    value = config.other_parameters.get(name, {}).get("value")
    return default if value is None else float(value)


class EgoHeldNearLine(AtomicBehavior):
    """Succeeds once the ego has been stationary within `near_m` of a point for `seconds`."""

    def __init__(self, ego, location, near_m, seconds, name="EgoHeldNearLine"):
        super().__init__(name)
        self._ego, self._location = ego, location
        self._near_m, self._seconds = near_m, seconds
        self._since = None

    def update(self):
        location = CarlaDataProvider.get_location(self._ego)
        speed = CarlaDataProvider.get_velocity(self._ego)
        if location is None or location.distance(self._location) > self._near_m or speed >= 0.1:
            self._since = None
            return py_trees.common.Status.RUNNING
        now = GameTime.get_time()
        self._since = now if self._since is None else self._since
        if now - self._since >= self._seconds:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class NavigationSignalHold(BasicScenario):
    timeout = 180

    def __init__(
        self, world, ego_vehicles, config, randomize=False, debug_mode=False,
        criteria_enable=True, timeout=180,
    ):  # fmt: skip
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        self._stationary_s = parameter(config, "stationary_s", 3.0)
        self._max_hold_s = parameter(config, "max_hold_s", 45.0)
        self._near_m = parameter(config, "near_m", 20.0)
        self._clearance_s = parameter(config, "clearance_s", 3.0)
        super().__init__(
            "NavigationSignalHold", ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable,
        )  # fmt: skip

    def _initialize_actors(self, config):
        wp = self._map.get_waypoint(config.trigger_points[0].location)
        entry = wp
        while not wp.is_junction:
            nxt = wp.next(1.0)
            if not nxt:
                raise ValueError("No junction ahead of the signal-hold trigger")
            if not nxt[0].is_junction:
                entry = nxt[0]
            wp = nxt[0]
        self._stop_location = entry.transform.location
        self._junction = wp.get_junction()
        # OpenDRIVE light landmarks are missing on many Town03/04 approaches, so the
        # landmark lookup used by PriorityAtJunction skipped 35/100 dev holds. Use the
        # pinned red-light criterion's own lane geometry, as the inventory did.
        helper = SimpleNamespace(
            _map=self._map, rotate_point=lambda p, a: RunningRedLightTest.rotate_point(None, p, a)
        )
        every_light = {x.id: x for x in self._world.get_actors().filter("traffic.traffic_light*")}
        light_lanes = {
            light_id: {
                (w.road_id, w.section_id, w.lane_id)
                for w in RunningRedLightTest.get_traffic_light_waypoints(helper, light)[1]
            }
            for light_id, light in every_light.items()
        }
        ego_light = every_light[
            pick_ego_light((entry.road_id, entry.section_id, entry.lane_id), light_lanes)
        ]
        lights = list(self._world.get_traffic_lights_in_junction(self._junction.id))
        if ego_light.id not in {x.id for x in lights}:
            lights.append(ego_light)
        self.ego_light_id = ego_light.id

        def states(ego, others):
            return {x: ego if x.id == ego_light.id else others for x in lights}

        self._phases = {
            "hold": states(RED, GREEN),
            "clear": states(RED, YELLOW),
            "all_red": states(RED, RED),
            "release": states(GREEN, RED),
        }

    def _create_behavior(self):
        ego = self.ego_vehicles[0]
        hold = py_trees.composites.Parallel(
            "SignalHold", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        hold.add_child(TrafficLightFreezer(self._phases["hold"], duration=self._max_hold_s))
        hold.add_child(EgoHeldNearLine(ego, self._stop_location, self._near_m, self._stationary_s))
        phases = py_trees.composites.Sequence("SignalPhases")
        phases.add_child(hold)
        phases.add_child(TrafficLightFreezer(self._phases["clear"], duration=self._clearance_s))
        phases.add_child(TrafficLightFreezer(self._phases["all_red"], duration=1.0))
        phases.add_child(TrafficLightFreezer(self._phases["release"]))
        root = py_trees.composites.Parallel(
            "NavigationSignalHold", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        root.add_child(WaitEndIntersection(ego, self._junction.id))
        root.add_child(phases)
        return root

    def _create_test_criteria(self):
        return []  # the route already carries the pinned red-light criterion

    def __del__(self):
        self.remove_all_actors()


NAVIGATION_SCENARIOS = {"NavigationSignalHold": NavigationSignalHold}
