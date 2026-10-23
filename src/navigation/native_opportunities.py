"""Scoring-only instrumentation of the pinned native rule eligibility checks.

The original criterion functions run once and their exact return values are
returned. This records eligibility, not visibility, legal understanding or a
new infraction criterion. Camera inference must never import this module.
"""

from pathlib import Path

import numpy as np

from src.navigation.runtime import atomic_json

CONTEXT = {"meter": None}


class NativeOpportunityMeter:
    def __init__(self, path):
        self.path = Path(path)
        self.events = {}
        self.errors = []
        self.last_dump = -100
        self.dump()

    def record(self, kind, actor_id, lane, frame, details=None):
        key = kind, int(actor_id), tuple(int(i) for i in lane)
        new = key not in self.events
        event = self.events.setdefault(
            key,
            dict(
                kind=kind,
                actor_id=int(actor_id),
                lane=list(key[2]),
                first_frame=int(frame),
                last_frame=int(frame),
                native_eligible_calls=0,
                **(details or {}),
            ),
        )
        event["last_frame"] = int(frame)
        event["native_eligible_calls"] += 1
        if new or frame - self.last_dump >= 20:
            self.dump()
            self.last_dump = frame

    def error(self, message):
        if message not in self.errors:
            self.errors.append(message)
            self.dump()

    def dump(self):
        atomic_json(
            self.path,
            dict(
                schema="native-rule-opportunities-v1",
                events=[self.events[key] for key in sorted(self.events)],
                errors=self.errors,
                count_unit="Unique eligible native actor/entry-lane pair in this mission",
                criterion_returns_unchanged=True,
                note="Native geometric eligibility only; no pixel visibility/learned-rule claim",
            ),
        )


def hero(criterion):
    actor = getattr(criterion, "actor", None)
    return actor is not None and getattr(actor, "attributes", {}).get("role_name") in (
        "hero",
        "ego_vehicle",
    )


def position(point):
    return np.array([point.x, point.y, point.z], dtype=float)


def install_native_opportunity_hooks(red_class, stop_class, frame_number):
    """Accept native classes explicitly so passthrough behavior is testable offline."""
    if not getattr(red_class, "_navigation_opportunity_hook", False):
        original_red = red_class.is_vehicle_crossing_line

        def red_crossing(criterion, segment1, segment2):
            verdict = original_red(criterion, segment1, segment2)
            meter = CONTEXT["meter"]
            if meter is not None and hero(criterion):
                try:
                    midpoint = (position(segment2[0]) + position(segment2[1])) / 2
                    matches = []
                    for light, _, waypoints in criterion._list_traffic_lights:
                        if str(light.state) != "Red":
                            continue
                        for waypoint in waypoints:
                            if (
                                np.linalg.norm(position(waypoint.transform.location) - midpoint)
                                < 1e-4
                            ):
                                matches.append((light, waypoint))
                    if len(matches) != 1:
                        meter.error(f"Ambiguous native red stop-line binding:{len(matches)}")
                    else:
                        light, waypoint = matches[0]
                        meter.record(
                            "red",
                            light.id,
                            (waypoint.road_id, waypoint.section_id, waypoint.lane_id),
                            frame_number(),
                            dict(stop_line_xyz=midpoint.tolist()),
                        )
                except Exception as error:
                    meter.error(f"Red opportunity audit failed:{type(error).__name__}:{error}")
            return verdict

        red_class.is_vehicle_crossing_line = red_crossing
        red_class._navigation_opportunity_hook = True
    if not getattr(stop_class, "_navigation_opportunity_hook", False):
        original_stop = stop_class.is_actor_affected_by_stop

        def affected_stop(criterion, waypoints, stop):
            verdict = original_stop(criterion, waypoints, stop)
            meter = CONTEXT["meter"]
            if verdict and meter is not None and hero(criterion):
                try:
                    waypoint = waypoints[0]
                    meter.record(
                        "stop",
                        stop.id,
                        (waypoint.road_id, waypoint.section_id, waypoint.lane_id),
                        frame_number(),
                    )
                except Exception as error:
                    meter.error(f"Stop opportunity audit failed:{type(error).__name__}:{error}")
            return verdict

        stop_class.is_actor_affected_by_stop = affected_stop
        stop_class._navigation_opportunity_hook = True
