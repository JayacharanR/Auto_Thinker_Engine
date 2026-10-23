#!/usr/bin/env python3
"""Prepare outcome-blind controlled rule cases on strict TRAIN geography.

These are setup prototypes until native reference/rendering certification;
they are never replacements for canonical/final benchmark routes.
"""

import argparse
import copy
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.rule_suite import FAMILIES, arms  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def points(route):
    return np.array(
        [
            [
                waypoint.transform.location.x,
                waypoint.transform.location.y,
                waypoint.transform.location.z,
            ]
            for waypoint, _ in route
        ]
    )


def pose(waypoint):
    t = waypoint.transform
    return [
        t.location.x,
        t.location.y,
        t.location.z + 0.5,
        t.rotation.pitch,
        t.rotation.yaw,
        t.rotation.roll,
    ]


def actor_path(route):
    return dict(spawn_pose=pose(route[0][0]), route_xy=points(route)[:, :2].tolist())


def cross_path(ego_route, planner, carla):
    inside = [w for w, _ in ego_route if w.is_junction]
    if not inside:
        raise ValueError("No junction for crossing setup")
    junction = inside[0].get_junction()
    ego_points = points(ego_route)
    choices = []
    for entry, exit in junction.get_waypoints(carla.LaneType.Driving):
        previous = entry.previous(15)
        following = exit.next(20)
        if not previous or not following:
            continue
        route = planner.trace_route(previous[0].transform.location, following[0].transform.location)
        if len(route) < 15 or any(w.is_junction and w.junction_id != junction.id for w, _ in route):
            continue
        xyz = points(route)
        distances = np.linalg.norm(xyz[:, None, :2] - ego_points[None, :, :2], axis=-1)
        vectors = []
        for candidate in (route, ego_route):
            forward = [waypoint.transform.get_forward_vector() for waypoint, _ in candidate]
            vectors.append(np.array([[v.x, v.y] for v in forward]))
        # Merging paths may have their global nearest points on the same outgoing
        # lane. Find the closest transverse interaction inside the junction itself.
        transverse = np.abs(vectors[0] @ vectors[1].T) <= 0.5
        inside_first = np.array([w.is_junction for w, _ in route])
        inside_second = np.array([w.is_junction for w, _ in ego_route])
        distances[~(transverse & inside_first[:, None] & inside_second[None, :])] = np.inf
        cross_index, ego_index = np.unravel_index(distances.argmin(), distances.shape)
        if distances.min() > 1.5:
            continue
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        start = max(0, int(np.searchsorted(arc, arc[cross_index] - 8)))
        if arc[cross_index] - arc[start] < 6:
            continue
        route = route[start:]
        key = (entry.road_id, entry.section_id, entry.lane_id, exit.road_id, exit.lane_id)
        choices.append((key, route, float(distances.min())))
    if not choices:
        raise ValueError("No perpendicular intersecting reactive traffic route")
    key, route, gap = min(choices, key=lambda row: row[0])
    return dict(actor_path(route), selection_lane_key=list(key), intersection_gap_m=gap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/navigation_rule_cases_20261019")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("Preserve prior rule-case geometry/prototypes")
    vendor = activate()
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    catalog_dir = ROOT / "configs/encounters/location_split_v1/train"
    catalog = json.loads((catalog_dir / "catalog.json").read_text())
    guard_path = ROOT / "outputs/navigation_m2_varied_waits_20261018/guard.json"
    guard = json.loads(guard_path.read_text())
    candidates = [
        e
        for e in catalog["encounters"]
        if e["route_id"] in guard["geometry_eligibility"]["kept_route_ids"]
        and set(e["crossed_junction_ids"]) <= set(guard["allowed_junctions"][e["town"]])
    ]
    chosen = {}
    seen_towns = set()
    seen_stops = set()
    selection_maps = {}
    geometry_rejections = []
    for e in sorted(candidates, key=lambda e: e["route_id"]):
        if (
            e["family"] == "signal_stop"
            and len(e["signal_ids_at_inventory"]) == 1
            and e["town"] not in seen_towns
            and len(seen_towns) < 2
        ):
            town = e["town"]
            if town not in selection_maps:
                mapfile = (
                    Path(vendor["carla_root"])
                    / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
                )
                map_ = carla.Map(town, mapfile.read_text())
                selection_maps[town] = GlobalRoutePlanner(map_, 1.0)
            planner = selection_maps[town]
            route_ = planner.trace_route(
                carla.Location(*e["start_xyz"]), carla.Location(*e["end_xyz"])
            )
            try:
                cross_path(route_, planner, carla)
            except ValueError as error:
                geometry_rejections.append(
                    dict(route_id=e["route_id"], town=town, error=str(error))
                )
                continue
            chosen[f"signal{len(seen_towns)}"] = e
            seen_towns.add(e["town"])
        if (
            e["family"] == "stop_sign"
            and (e["town"], e["junction_id"]) not in seen_stops
            and len(seen_stops) < 2
        ):
            chosen[f"stop{len(seen_stops)}"] = e
            seen_stops.add((e["town"], e["junction_id"]))
    if len(chosen) != 4:
        raise ValueError("Two signal towns/two strict stop locations required")
    args.out.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "src/navigation/rule_suite.py",
        ROOT / "src/navigation/rule_prefix.py",
        ROOT / "src/navigation/rule_case_contract.py",
        ROOT / "src/navigation/rule_case_runtime.py",
        ROOT / "reports/navigation_rule_counterfactual_protocol_20261019.md",
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            selection=(
                'First eligible route IDs with a transverse live-traffic path; '
                'two signal towns/two stop locations'
            ),
            geometry_rejections=geometry_rejections,
            chosen=chosen,
            catalog_sha256=digest(catalog_dir / "catalog.json"),
            strict_guard_sha256=digest(guard_path),
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
            state="SETUP_UNCERTIFIED",
            no_student_outcomes=True,
        ),
    )
    maps = {}
    inventory = {}
    cases = []
    errors = []
    for location, entry in chosen.items():
        town = entry["town"]
        if town not in maps:
            mapfile = (
                Path(vendor["carla_root"]) / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
            )
            world_map = carla.Map(town, mapfile.read_text())
            maps[town] = (world_map, GlobalRoutePlanner(world_map, 1.0))
            source = ROOT / catalog["inventory"] / f"{town}.json"
            if digest(source) != catalog["inventory_sha256"][town]:
                raise ValueError("TRAIN physical inventory changed")
            inventory[town] = json.loads(source.read_text())
        world_map, planner = maps[town]
        route = planner.trace_route(
            carla.Location(*entry["start_xyz"]), carla.Location(*entry["end_xyz"])
        )
        xyz = points(route)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        signal = None
        if location.startswith("signal"):
            signal = next(
                row
                for row in inventory[town]["lights"]
                if row["id"] == entry["signal_ids_at_inventory"][0]
            )
            lane = tuple(__import__("ast").literal_eval(entry["movement_key"][2]))
            lines = [row for row in signal["stop_lanes"] if tuple(row["lane"]) == lane]
        else:
            stop = next(
                row
                for row in inventory[town]["stops"]
                if row["id"] in entry["stop_ids_at_inventory"]
            )
            lines = [row for row in stop["affected_lanes"] if row.get("leads_into_junction")]
        if len(lines) != 1:
            raise ValueError("Unambiguous native approach reference required")
        line = lines[0]
        line_wp = world_map.get_waypoint(carla.Location(*line["xyz"]))
        direction = line_wp.transform.get_forward_vector()
        common = dict(
            schema="navigation-physical-rule-case-v1",
            town=town,
            source_catalog_route_id=entry["route_id"],
            source_entry=entry,
            crossed_junction_ids=entry["crossed_junction_ids"],
            strict_allowed_junctions=guard["allowed_junctions"][town],
            location_split_sha256=digest(catalog_dir.parent / "locations.json"),
            strict_guard_sha256=digest(guard_path),
            setup_only_component_test=True,
            autonomous_route_success_claim=False,
            learning_labels_released=False,
            line_xyz=line["xyz"],
            line_heading_xy=[direction.x, direction.y],
            signal_reference=signal,
        )
        families = [f for f in FAMILIES if (f == "stop_and_release") == location.startswith("stop")]
        for family in families:
            actor_paths = {}
            if family == "green_stopped_lead":
                index = int(np.searchsorted(arc, 15.0))
                actor_paths["lead"] = actor_path(route[index:])
            if family == "green_clearing_cross":
                try:
                    actor_paths["crossing"] = cross_path(route, planner, carla)
                except ValueError as error:
                    errors.append(dict(location=location, family=family, error=str(error)))
                    continue
            for treatment in arms(family):
                case = args.out / f"{location}_{family}_a{treatment['arm']}"
                case.mkdir()
                tree = ET.Element("routes")
                node = ET.SubElement(tree, "route", id="0", town=town)
                ET.SubElement(
                    ET.SubElement(node, "weathers"),
                    "weather",
                    route_percentage="0",
                    cloudiness="0",
                    precipitation="0",
                    sun_altitude_angle="70",
                )
                waypoints = ET.SubElement(node, "waypoints")
                for endpoint in (entry["start_xyz"], entry["end_xyz"]):
                    ET.SubElement(waypoints, "position", **dict(zip("xyz", map(str, endpoint))))
                ET.SubElement(node, "scenarios")
                ET.indent(tree)
                route_path = case / "route.xml"
                ET.ElementTree(tree).write(route_path, encoding="utf-8", xml_declaration=True)
                atomic_json(
                    case / "route.json",
                    dict(
                        purpose="Controlled M3 component, not benchmark",
                        town=town,
                        route_sha256=digest(route_path),
                        length_m=float(arc[-1]),
                        junction_ids=entry["crossed_junction_ids"],
                    ),
                )
                spec = dict(
                    copy.deepcopy(common),
                    treatment=treatment,
                    route_sha256=digest(route_path),
                    scripted_actor_paths=actor_paths,
                )
                atomic_json(case / "case.json", spec)
                (case / "case.sha256").write_text(digest(case / "case.json") + "\n")
                cases.append(
                    dict(
                        path=str(case / "case.json"),
                        sha256=digest(case / "case.json"),
                        family=family,
                        location=location,
                        arm=treatment["arm"],
                    )
                )
    atomic_json(
        args.out / "cases.json",
        dict(
            cases=cases,
            geometry_errors=errors,
            setup_certified=False,
            student_tests_started=False,
            expected_case_geometries=24,
            seeds=[61001, 61002, 61003],
            expected_student_tests=72,
        ),
    )
    print(json.dumps(dict(case_geometries=len(cases), geometry_errors=errors)), flush=True)


if __name__ == "__main__":
    main()
