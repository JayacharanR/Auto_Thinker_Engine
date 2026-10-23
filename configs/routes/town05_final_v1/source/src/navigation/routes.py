"""Deterministic development routes and manifest integrity (never training-town leakage)."""

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from src.navigation.runtime import atomic_json


def signal_inventory(carla_root):
    root = Path(carla_root).expanduser() / "CarlaUE4/Content/Carla/Maps/OpenDrive"
    result = {}
    for town in ("Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"):
        path = root / f"{town}.xodr"
        tree = ET.parse(path)
        signals = tree.findall(".//signals/signal")
        result[town] = {
            "types": dict(Counter(x.get("type") for x in signals)),
            "stop_sign_ids": [x.get("id") for x in signals if x.get("type") == "206"],
            "xodr_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return result


def create_development_route(
    carla_root, destination, town="Town03", seed=20261008, minimum=1000, maximum=1500
):
    """First qualifying seeded start/end pair; selection never uses driving outcomes."""
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    if town.removesuffix("_Opt") == "Town05":
        raise ValueError("M0 development generation cannot use the held-out town")
    path = Path(carla_root).expanduser() / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
    carla_map = carla.Map(town, path.read_text())
    planner = GlobalRoutePlanner(carla_map, 1.0)  # match the pinned harness interpolation
    # OpenDRIVE maps do not include packaged spawn points. Use road waypoints.
    points = sorted(carla_map.generate_waypoints(20), key=lambda w: (w.road_id, w.lane_id, w.s))
    points = [w for w in points if not w.is_junction]
    rng = np.random.default_rng(seed)
    for attempt in range(1000):
        a, b = rng.choice(len(points), 2, replace=False)
        # Plan from the exact coordinates written to XML. The first/last returned
        # path waypoints may lie on adjacent lanes; using those as new endpoints
        # changed a Town10HD mission from 389m to 161m when the harness replanned it.
        endpoints = [
            tuple(round(getattr(points[i].transform.location, axis), 5) for axis in "xyz")
            for i in (a, b)
        ]
        route = planner.trace_route(*[carla.Location(*xyz) for xyz in endpoints])
        xyz = np.array(
            [
                [w.transform.location.x, w.transform.location.y, w.transform.location.z]
                for w, _ in route
            ]
        )
        if len(xyz) < 2:
            continue
        length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        if minimum <= length <= maximum:
            break
    else:
        raise RuntimeError("No development route in requested length interval")
    routes = ET.Element("routes")
    node = ET.SubElement(routes, "route", id="0", town=town)
    weather = ET.SubElement(node, "weathers")
    ET.SubElement(
        weather,
        "weather",
        route_percentage="0",
        cloudiness="0",
        precipitation="0",
        sun_altitude_angle="70",
    )
    waypoints = ET.SubElement(node, "waypoints")
    # This is a single shortest-path mission. Re-interpolating intermediate points
    # near junction boundaries can introduce spurious loops in CARLA's planner.
    for x, y, z in endpoints:
        ET.SubElement(waypoints, "position", x=f"{x:.5f}", y=f"{y:.5f}", z=f"{z:.5f}")
    ET.SubElement(node, "scenarios")
    ET.indent(routes)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace route: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(routes).write(destination, encoding="utf-8", xml_declaration=True)
    manifest = {
        "purpose": "M0 development integration; not a frozen benchmark",
        "town": town,
        "seed": seed,
        "selection_attempt": attempt,
        "length_m": length,
        "route_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "xodr_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "junction_ids": sorted({w.junction_id for w, _ in route if w.is_junction}),
        "waypoint_count": len(endpoints),
        "generation_version": 3,
        "selection_rule": "first seeded pair within length limits, planned from serialized inputs",
    }
    atomic_json(destination.with_suffix(".json"), manifest)
    return manifest


def verify_route(path):
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["route_sha256"]:
        raise ValueError("Route differs from its registered manifest")
    return manifest
