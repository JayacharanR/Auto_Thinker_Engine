import json
from types import SimpleNamespace as NS

from src.navigation.native_opportunities import (
    CONTEXT,
    NativeOpportunityMeter,
    install_native_opportunity_hooks,
)


def point(x, y, z=0):
    return NS(x=x, y=y, z=z)


def test_native_returns_unchanged_and_only_native_eligible_rules_are_counted(tmp_path):
    class Red:
        def is_vehicle_crossing_line(self, *args):
            return self.verdict

    class Stop:
        def is_actor_affected_by_stop(self, *args):
            return self.verdict

    waypoint = NS(road_id=5, section_id=0, lane_id=-1, transform=NS(location=point(1, 2)))
    light = NS(id=55, state="Red")
    red, stop = Red(), Stop()
    red.actor = stop.actor = NS(attributes=dict(role_name="hero"))
    red._list_traffic_lights = [(light, None, [waypoint])]
    token = object()
    red.verdict = token
    stop.verdict = True
    install_native_opportunity_hooks(Red, Stop, lambda: 10)
    install_native_opportunity_hooks(Red, Stop, lambda: 99)  # idempotent
    meter = NativeOpportunityMeter(tmp_path / "counts.json")
    CONTEXT["meter"] = meter
    try:
        assert red.is_vehicle_crossing_line(None, [point(0, 2), point(2, 2)]) is token
        assert stop.is_actor_affected_by_stop([waypoint], NS(id=9)) is True
        stop.verdict = False
        assert stop.is_actor_affected_by_stop([waypoint], NS(id=100)) is False
        meter.dump()
        events = json.loads((tmp_path / "counts.json").read_text())["events"]
        assert {(e["kind"], e["actor_id"]) for e in events} == {("red", 55), ("stop", 9)}
        assert all(e["first_frame"] == 10 for e in events)
    finally:
        CONTEXT["meter"] = None


def test_geometry_helpers_without_ego_actor_never_count_as_driving_encounters(tmp_path):
    class Red:
        def is_vehicle_crossing_line(self, *args):
            return False

    class Stop:
        def is_actor_affected_by_stop(self, *args):
            return True

    install_native_opportunity_hooks(Red, Stop, lambda: 0)
    meter = NativeOpportunityMeter(tmp_path / "counts.json")
    CONTEXT["meter"] = meter
    try:
        assert Stop().is_actor_affected_by_stop([], NS(id=1))
        assert not meter.events
    finally:
        CONTEXT["meter"] = None
