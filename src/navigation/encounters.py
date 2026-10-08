"""Runtime rule applicability and junction-movement candidates for the encounter catalog.

Applicability reuses the pinned evaluator's own red-light and stop-sign geometry, so
catalog denominators and infraction scoring agree. Candidates are geometry only:
selecting and freezing the 300 development encounters is a separate, outcome-blind step.
"""

from types import SimpleNamespace

import numpy as np

APPROACH_M = 50.0  # target straight approach before the junction entry
MIN_APPROACH_M = 30.0  # shorter approaches cannot give a fair stop/see opportunity
EXIT_M = 25.0
STEP_M = 1.0
MAX_JUNCTION_M = 120.0


def yaw_delta(a, b):
    """Signed heading change in degrees, wrapped to (-180, 180]."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def lane_key(wp):
    return [wp.road_id, wp.section_id, wp.lane_id]


def xyz(location):
    return [round(location.x, 3), round(location.y, 3), round(location.z, 3)]


def path_length(points):
    points = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def walk(wp, step, limit, forward=True, stop_in_junction=None):
    """Follow the first successor/predecessor; stop before (or on leaving) a junction."""
    points, travelled = [wp], 0.0
    while travelled < limit:
        nxt = wp.next(step) if forward else wp.previous(step)
        if not nxt:
            break
        if stop_in_junction is not None and nxt[0].is_junction != stop_in_junction:
            break
        wp = nxt[0]
        points.append(wp)
        travelled += step
    return points, travelled


def light_applicability(world, carla_map):
    from srunner.scenariomanager.scenarioatomics.atomic_criteria import RunningRedLightTest

    def rotate(point, angle):
        return RunningRedLightTest.rotate_point(None, point, angle)

    helper = SimpleNamespace(_map=carla_map, rotate_point=rotate)
    lights = []
    for light in sorted(world.get_actors().filter("traffic.traffic_light*"), key=lambda a: a.id):
        center, wps = RunningRedLightTest.get_traffic_light_waypoints(helper, light)
        lights.append(
            {
                "id": light.id,
                "xyz": xyz(light.get_transform().location),
                "trigger_xyz": xyz(center),
                "group": sorted(x.id for x in light.get_group_traffic_lights()),
                "stop_lanes": [
                    {
                        "lane": lane_key(w),
                        "xyz": xyz(w.transform.location),
                        "in_junction": w.is_junction,
                    }
                    for w in wps
                ],
            }
        )
    return lights


def stop_applicability(world, carla_map, waypoints):
    """Lanes on which the pinned RunningStopTest would select each runtime stop actor."""
    from srunner.scenariomanager.scenarioatomics.atomic_criteria import RunningStopTest

    def inside(point, center, extent, multiplier=1.2):
        return RunningStopTest.point_inside_boundingbox(None, point, center, extent, multiplier)

    helper = SimpleNamespace(
        PROXIMITY_THRESHOLD=RunningStopTest.PROXIMITY_THRESHOLD, point_inside_boundingbox=inside
    )
    steps = int(RunningStopTest.PROXIMITY_THRESHOLD / RunningStopTest.WAYPOINT_STEP)
    positions = np.array([[w.transform.location.x, w.transform.location.y] for w in waypoints])
    stops = []
    for stop in sorted(world.get_actors().filter("traffic.stop"), key=lambda a: a.id):
        center = stop.get_transform().transform(stop.trigger_volume.location)
        near = np.flatnonzero(
            np.hypot(*(positions - [center.x, center.y]).T) <= RunningStopTest.PROXIMITY_THRESHOLD
        )
        lanes = {}
        for index in near:
            wp = waypoints[index]
            wp_list, _ = walk(
                wp, RunningStopTest.WAYPOINT_STEP, steps * RunningStopTest.WAYPOINT_STEP
            )
            if not RunningStopTest.is_actor_affected_by_stop(helper, wp_list, stop):
                continue
            key = tuple(lane_key(wp))
            # Keep the most downstream affected point; the junction entry follows it.
            # OpenDRIVE negative lanes travel with increasing s, positive lanes against it.
            downstream = wp.s if wp.lane_id < 0 else -wp.s
            if key not in lanes or downstream > lanes[key][0]:
                lanes[key] = (downstream, wp)
        stops.append(
            {
                "id": stop.id,
                "xyz": xyz(stop.get_transform().location),
                "trigger_xyz": xyz(center),
                "affected_lanes": [
                    {
                        "lane": list(k),
                        "xyz": xyz(v[1].transform.location),
                        "in_junction": v[1].is_junction,
                        "wp": v[1],
                    }
                    for k, v in sorted(lanes.items())
                ],
            }
        )
    return stops


def junction_entry(wp, limit=30.0):
    """Last non-junction waypoint before the next junction, or None if none is close."""
    if wp.is_junction:
        return None
    points, _ = walk(wp, STEP_M, limit, forward=True, stop_in_junction=False)
    nxt = points[-1].next(STEP_M)
    return points[-1] if nxt and nxt[0].is_junction else None


def movements(carla_map, planner, lights, stops):
    """Every junction movement with an approach, controls and planner validation."""
    import carla

    controls = {}
    for light in lights:
        for lane in light["stop_lanes"]:
            controls.setdefault(tuple(lane["lane"]), []).append(("signal", light["id"]))
    for stop in stops:
        for lane in stop["affected_lanes"]:
            entry = junction_entry(lane.pop("wp"))
            lane["leads_into_junction"] = entry is not None
            if entry is not None:
                controls.setdefault(tuple(lane_key(entry)), []).append(("stop", stop["id"]))
                lane["entry_lane"] = lane_key(entry)

    junctions, result = {}, []
    for wp in carla_map.generate_waypoints(10.0):
        if wp.is_junction:
            junctions.setdefault(wp.get_junction().id, wp.get_junction())
    for junction_id in sorted(junctions):
        pairs = junctions[junction_id].get_waypoints(carla.LaneType.Driving)
        for start, _ in sorted(pairs, key=lambda p: (p[0].road_id, p[0].lane_id)):
            before = start.previous(STEP_M)
            if not before or before[0].is_junction:
                continue
            entry = before[0]
            inside, _ = walk(start, STEP_M, MAX_JUNCTION_M, forward=True, stop_in_junction=True)
            after = inside[-1].next(STEP_M)
            if not after or after[0].is_junction:
                continue
            exit_wp = after[0]
            tail, _ = walk(exit_wp, STEP_M, EXIT_M, forward=True, stop_in_junction=False)
            back, approach = walk(entry, STEP_M, APPROACH_M, forward=False, stop_in_junction=False)
            begin, end = back[-1], tail[-1]
            geometric = [w.transform.location for w in [*reversed(back), *inside, *tail]]
            traced = planner.trace_route(begin.transform.location, end.transform.location)
            traced_xyz = [xyz(w.transform.location) for w, _ in traced]
            options = sorted({o.name for w, o in traced if w.is_junction})
            through = any(w.is_junction and w.get_junction().id == junction_id for w, _ in traced)
            control = controls.get(tuple(lane_key(entry)), [])
            result.append(
                {
                    "junction_id": junction_id,
                    "entry_lane": lane_key(entry),
                    "exit_lane": lane_key(exit_wp),
                    "turn_deg": round(
                        yaw_delta(entry.transform.rotation.yaw, exit_wp.transform.rotation.yaw), 1
                    ),
                    "planner_options": options,
                    "control": sorted({kind for kind, _ in control}) or ["none"],
                    "signal_ids": sorted({i for kind, i in control if kind == "signal"}),
                    "stop_ids": sorted({i for kind, i in control if kind == "stop"}),
                    "start_xyz": xyz(begin.transform.location),
                    "entry_xyz": xyz(entry.transform.location),
                    "exit_xyz": xyz(exit_wp.transform.location),
                    "end_xyz": xyz(end.transform.location),
                    "approach_m": approach,
                    "geometric_length_m": round(path_length([xyz(p) for p in geometric]), 2),
                    "planner_length_m": round(path_length(traced_xyz), 2),
                    "planner_through_junction": through,
                }
            )
    return result


def candidate_valid(movement, tolerance=0.1):
    """Outcome-blind validity: fair approach, and the harness planner takes this movement."""
    g, p = movement["geometric_length_m"], movement["planner_length_m"]
    return (
        movement["approach_m"] >= MIN_APPROACH_M
        and movement["planner_through_junction"]
        and g > 0
        and abs(p - g) <= tolerance * g
    )


def summarize(movement_list):
    valid = [m for m in movement_list if candidate_valid(m)]
    by_control = {}
    for m in valid:
        for kind in m["control"]:
            by_control[kind] = by_control.get(kind, 0) + 1
    return {
        "movements": len(movement_list),
        "valid_movements": len(valid),
        "valid_by_control": dict(sorted(by_control.items())),
        "valid_junctions": len({m["junction_id"] for m in valid}),
        "valid_signal_lights": len({i for m in valid for i in m["signal_ids"]}),
        "valid_stop_actors": len({i for m in valid for i in m["stop_ids"]}),
    }
