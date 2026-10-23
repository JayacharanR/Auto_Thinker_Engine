#!/usr/bin/env python3
"""Offline TRAIN-only long-route geometry; no simulator, agent or outcome selection."""

import argparse
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.splits import nearest_junction  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    guard_path = (
        ROOT / "outputs/navigation_spatial_data_repaired_20261019/actual_training_guard.json"
    )
    guard = json.loads(guard_path.read_text())
    cases = sorted(
        (ROOT / "outputs/navigation_rule_followthrough_v3_20261020/certified_suite").glob(
            "*/case.json"
        )
    )
    if len(cases) != 24:
        raise ValueError("All current physically certified rule locations must be reserved")
    reserved = {}
    for path in cases:
        case = json.loads(path.read_text())
        reserved.setdefault(case["town"], set()).update(case["crossed_junction_ids"])
    request = dict(
        seed=20261020,
        town_order=sorted(guard["allowed_junctions"]),
        minimum_m=1000,
        maximum_m=1500,
        minimum_endpoint_separation_m=250,
        attempts_per_town=1000,
        maximum_cpu_wall_seconds=360,
        minimum_reserved_junction_clearance_m=1.0,
        guard_sha256=digest(guard_path),
        rule_cases_sha256={str(p): digest(p) for p in cases},
        source_sha256=digest(__file__),
        outcome_selection=False,
        no_native_server=True,
        selection="First seeded exact serialized endpoint pair satisfying every geometry rule",
    )
    atomic_json(args.out / "request.json", request)
    shutil.copyfile(__file__, args.out / "generation_source.py")
    vendor = activate()
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    rng = np.random.default_rng(request["seed"])
    started = time.monotonic()
    rejected = {}
    selected = None
    for town in request["town_order"]:
        path = Path(vendor["carla_root"]) / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
        if digest(path) != guard["geometry_eligibility"]["map_sha256"][town]:
            raise ValueError("Map bytes differ from the actual training guard")
        world_map = carla.Map(town, path.read_text())
        planner = GlobalRoutePlanner(world_map, 1.0)
        allowed = set(guard["allowed_junctions"][town]) - reserved.get(town, set())
        points = sorted(
            (w for w in world_map.generate_waypoints(20) if not w.is_junction),
            key=lambda w: (w.road_id, w.lane_id, w.s),
        )
        forbidden = [
            w
            for w in world_map.generate_waypoints(0.5)
            if w.is_junction and w.junction_id not in allowed
        ]
        forbidden_xyz = [
            [w.transform.location.x, w.transform.location.y, w.transform.location.z]
            for w in forbidden
        ]
        forbidden_ids = [w.junction_id for w in forbidden]
        counts = dict(short_endpoints=0, length=0, crossed_reserved=0, clearance=0, path_error=0)
        for attempt in range(request["attempts_per_town"]):
            if (args.out / "STOP").exists():
                raise InterruptedError("Offline TRAIN geometry STOP")
            if time.monotonic() - started > request["maximum_cpu_wall_seconds"]:
                raise InterruptedError("Offline TRAIN geometry360s cap")
            a, b = rng.choice(len(points), 2, replace=False)
            endpoints = [
                tuple(round(getattr(points[i].transform.location, axis), 5) for axis in "xyz")
                for i in (a, b)
            ]
            if np.linalg.norm(np.array(endpoints[0]) - endpoints[1]) < 250:
                counts["short_endpoints"] += 1
                continue
            try:
                traced = planner.trace_route(*[carla.Location(*p) for p in endpoints])
            except Exception as error:
                counts["path_error"] += 1
                atomic_json(
                    args.out / "last_path_error.json",
                    dict(town=town, attempt=attempt, error=str(error)),
                )
                continue
            xyz = np.array(
                [
                    [w.transform.location.x, w.transform.location.y, w.transform.location.z]
                    for w, _ in traced
                ]
            )
            length = (
                float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0
            )
            if not 1000 <= length <= 1500:
                counts["length"] += 1
                continue
            crossed = {w.junction_id for w, _ in traced if w.is_junction}
            if not crossed <= allowed:
                counts["crossed_reserved"] += 1
                continue
            clearance = nearest_junction(xyz, forbidden_xyz, forbidden_ids)
            if clearance["clearance_m"] < 1:
                counts["clearance"] += 1
                continue
            selected = dict(
                town=town,
                endpoints=endpoints,
                length_m=length,
                junction_ids=sorted(crossed),
                allowed_junctions=sorted(allowed),
                clearance=clearance,
                attempt=attempt,
                xodr_sha256=digest(path),
                dense_waypoints=len(xyz),
            )
            break
        rejected[town] = counts
        atomic_json(args.out / "search.json", dict(rejected=rejected, selected=selected))
        if selected:
            break
    if not selected:
        raise RuntimeError("No strict TRAIN-only long path found; never relax reserved geometry")
    tree = ET.Element("routes")
    route = ET.SubElement(tree, "route", id="0", town=selected["town"])
    weather = ET.SubElement(route, "weathers")
    ET.SubElement(
        weather,
        "weather",
        route_percentage="0",
        cloudiness="0",
        precipitation="0",
        sun_altitude_angle="70",
    )
    waypoints = ET.SubElement(route, "waypoints")
    for x, y, z in selected["endpoints"]:
        ET.SubElement(waypoints, "position", x=f"{x:.5f}", y=f"{y:.5f}", z=f"{z:.5f}")
    ET.SubElement(route, "scenarios")
    ET.indent(tree)
    target = args.out / "route.xml"
    ET.ElementTree(tree).write(target, encoding="utf-8", xml_declaration=True)
    # Reconstruct from the exact serialized XML, using a fresh unmodified planner.
    native = GlobalRoutePlanner(world_map, 1.0)
    vertices = [
        carla.Location(**{axis: float(p.get(axis)) for axis in "xyz"})
        for p in ET.parse(target).findall(".//position")
    ]
    checked = native.trace_route(*vertices)
    check_xyz = np.array(
        [
            [w.transform.location.x, w.transform.location.y, w.transform.location.z]
            for w, _ in checked
        ]
    )
    check_length = float(np.linalg.norm(np.diff(check_xyz, axis=0), axis=1).sum())
    if (
        not np.array_equal(xyz, check_xyz)
        or abs(check_length - selected["length_m"]) > 1e-6
        or {w.junction_id for w, _ in checked if w.is_junction} != set(selected["junction_ids"])
    ):
        raise ValueError("Serialized route changes under native interpolation")
    manifest = dict(
        selected,
        purpose="Strict TRAIN-only engineering integration; not a frozen benchmark",
        route_sha256=digest(target),
        request_sha256=digest(args.out / "request.json"),
        waypoint_count=2,
        seed=request["seed"],
        native_route_interpolation_verified=True,
        native_expert_or_camera_driving_verified=False,
        student_outcomes_used=False,
    )
    atomic_json(target.with_suffix(".json"), manifest)
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            stage="TRAIN-only long route geometry",
            phase="Offline route only; no expert or camera driving yet",
            updated_at=time.time(),
        ),
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
