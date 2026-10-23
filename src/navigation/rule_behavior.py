"""Scoring-only controlled rule behavior, separate from physical setup validity."""

import numpy as np


def behavior(spec, trace, native_episode, contacts, setup_validated=False):
    scored = [r for r in trace if not r["scripted_setup"]]
    if not scored:
        raise ValueError("An observed post-handoff interval is required")
    treatment = spec["treatment"]
    family = treatment["family"]
    arm = treatment["arm"]
    distance = np.array([r["distance_to_registered_line_m"] for r in scored])
    speeds = np.array([r["speed_mps"] for r in scored])
    times = np.array([r["timestamp"] - scored[0]["timestamp"] for r in scored])
    xy = np.array([r["ego"][:2] for r in scored])
    displacement = float(np.max(np.linalg.norm(xy - xy[0], axis=1)))
    crossings = np.flatnonzero((distance[:-1] > 0) & (distance[1:] <= 0)) + 1
    first_crossing = None if not len(crossings) else int(crossings[0])
    observed_red_cross = any(scored[i]["observed_light"] == "Red" for i in crossings)
    native_red = int(native_episode["red_violations"])
    native_stop = int(native_episode["stop_violations"])
    collision_free = bool(contacts["complete"]) and contacts["ego_contact_starts"] == 0
    arrived = bool(native_episode["completed"])
    common = collision_free and native_red == 0 and native_stop == 0 and not observed_red_cross
    conditions = {}
    if family == "out_of_view_memory":
        if arm == 0:
            conditions = dict(
                full_eight_seconds=bool(times[-1] >= 8 - 1e-5),
                red_hold_displacement_at_most_half_m=displacement <= 0.5,
                speed_below_point_two=bool(speeds.max() < 0.2),
                all_measured_phases_red=all(r["observed_light"] == "Red" for r in scored),
            )
        else:
            within = xy[times <= 8 + 1e-5]
            conditions = dict(
                green_history_departure_at_least_eight_m=bool(
                    len(within) and np.linalg.norm(within - within[0], axis=1).max() >= 8
                )
            )
    elif family == "yellow_distance":
        if arm == 0:
            conditions = dict(
                cleared_during_observed_amber=bool(
                    first_crossing is not None
                    and scored[first_crossing]["observed_light"] == "Yellow"
                    and times[first_crossing] <= 3 + 0.05
                )
            )
        else:
            tail = times >= max(0, times[-1] - 1)
            conditions = dict(
                full_twelve_seconds=bool(times[-1] >= 12 - 1e-5),
                no_registered_line_crossing=not len(crossings),
                stopped_before_line=bool(
                    (distance[tail] >= 0).all() and (speeds[tail] < 0.2).all()
                ),
            )
    elif family == "stop_and_release":
        full = np.array([r["distance_to_registered_line_m"] for r in trace])
        speed = np.array([r["speed_mps"] for r in trace])
        conditions = dict(
            native_stop_compliant=native_stop == 0,
            physical_stop_near_approach=bool(((full >= -0.5) & (full <= 10) & (speed < 0.1)).any()),
            departed=arrived,
        )
    elif family == "empty_light":
        conditions = dict(arrived_after_rule_response=arrived)
        if arm == 0:
            red = [r for r in scored if r["observed_light"] == "Red"]
            conditions["observed_red_stop_near_line"] = any(
                0 <= r["distance_to_registered_line_m"] <= 15 and r["speed_mps"] < 0.1 for r in red
            )
    elif family in ("green_stopped_lead", "green_clearing_cross"):
        conditions = dict(arrived_after_obstacle_response=arrived)
    else:
        raise ValueError("Unknown registered controlled-rule family")
    satisfied = bool(common and all(conditions.values()))
    return dict(
        family=family,
        arm=arm,
        conditions=conditions,
        common_safety_condition=common,
        behavior_condition_satisfied=satisfied,
        setup_validated=setup_validated,
        passed=bool(setup_validated and satisfied),
        all_attempt_contact_starts=contacts["ego_contact_starts"],
        recorder_complete=contacts["complete"],
        native_red=native_red,
        native_stop=native_stop,
        observed_registered_line_red_crossing=bool(observed_red_cross),
        arrived=arrived,
        scored_seconds=float(times[-1]),
        maximum_displacement_m=displacement,
        actual_handoff_speed_mps=float(speeds[0]),
        actual_handoff_distance_m=float(distance[0]),
        first_registered_line_crossing_s=None
        if first_crossing is None
        else float(times[first_crossing]),
        autonomous_route_success_claim=False,
        interpretation="Setup certification and behavior are distinct; "
        "no learned-rule claim from an expert.",
    )
