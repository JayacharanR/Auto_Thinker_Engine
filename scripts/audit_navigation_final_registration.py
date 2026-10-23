#!/usr/bin/env python3
"""Independent pinned-harness replay of frozen final geometry, without a simulator."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def main():
    activate()
    import carla
    from leaderboard.utils.route_manipulation import interpolate_trajectory
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

    from scripts.register_navigation_final_suite import registration_opportunities
    from src.navigation.scenarios import pick_ego_light

    root = ROOT / "configs/routes/town05_final_v2"
    suite = json.loads((root / "suite.json").read_text())
    request = json.loads((root / "registration_request.json").read_text())
    if (
        digest(root / "suite.json") != (root / "suite.sha256").read_text().split()[0]
        or digest(root / "Town05_Opt.xml") != suite["route_sha256"]
    ):
        raise ValueError("Frozen suite/XML changed")
    if any(digest(root / "source" / p) != h for p, h in request["source_hashes"].items()):
        raise ValueError("Frozen source archive changed")
    record = json.loads(
        (ROOT / "outputs/navigation_final_geometry_20261018/Town05_Opt.json").read_text()
    )
    if (
        digest(ROOT / "outputs/navigation_final_geometry_20261018/Town05_Opt.json")
        != suite["inventory_sha256"]
    ):
        raise ValueError("Raw final geometry changed")
    xodr = Path.home() / "software/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr"
    world_map = carla.Map("Town05_Opt", xodr.read_text())
    import hashlib

    if hashlib.sha256(world_map.to_opendrive().encode()).hexdigest() != suite["opendrive_sha256"]:
        raise ValueError("Final map changed")
    world = SimpleNamespace(get_map=lambda: world_map)
    light_lanes = {
        light["id"]: {tuple(x["lane"]) for x in light["stop_lanes"]}
        for light in record["lights"]
    }
    rows = []
    for entry, node in zip(suite["routes"], ET.parse(root / "Town05_Opt.xml").findall("route")):
        keypoints = [
            carla.Location(**{k: float(v.attrib[k]) for k in "xyz"})
            for v in node.findall("./waypoints/position")
        ]
        with (
            patch.object(CarlaDataProvider, "get_map", return_value=world_map),
            patch.object(CarlaDataProvider, "get_world", return_value=world),
        ):
            _, route = interpolate_trajectory(keypoints)
        xyz = np.array([[t.location.x, t.location.y, t.location.z] for t, _ in route])
        length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        if abs(length - entry["length_m"]) > 0.001 or not 1000 <= length <= 3000:
            raise ValueError("Pinned harness route length differs")
        trace = [(world_map.get_waypoint(t.location), o) for t, o in route]
        opportunities, _ = registration_opportunities(trace, record)
        if entry["required_stop_actor"] is not None and entry["required_stop_actor"] not in {
            x["actor_id"] for x in opportunities["stop"]
        }:
            raise ValueError("Mandatory stop missing in native interpolation")
        # Exact scenario initialization walk, using source-inventory actor IDs only
        # for geometry validation. No actor identity is an inference input.
        wp = world_map.get_waypoint(carla.Location(*entry["trigger_xyz"]))
        approach = wp
        for _ in range(100):
            if wp.is_junction:
                break
            wp = wp.next(1.0)[0]
            if not wp.is_junction:
                approach = wp
        if not wp.is_junction:
            raise ValueError("Signal trigger never reaches junction")
        light = pick_ego_light(
            (approach.road_id, approach.section_id, approach.lane_id), light_lanes
        )
        if light != entry["held_signal"]["actor_id"]:
            raise ValueError("Scenario would hold a different light")
        rows.append(
            dict(
                route_id=entry["route_id"],
                native_harness_length_m=length,
                required_stop=entry["required_stop_actor"],
                signal_actor_in_inventory=light,
                planned_stop_count=len(opportunities["stop"]),
            )
        )
    if len(rows) != 12 or sum(r["planned_stop_count"] for r in rows) * 3 < 30:
        raise ValueError("Final suite denominator/coverage mismatch")
    result = dict(
        registration_audit_passed=True,
        geometry_only=True,
        policy_loaded=False,
        suite_sha256=digest(root / "suite.json"),
        route_sha256=digest(root / "Town05_Opt.xml"),
        source_sha256=digest(Path(__file__)),
        routes=rows,
        planned_stop_opportunities=sum(r["planned_stop_count"] for r in rows) * 3,
        planned_red_opportunities=36,
        actual_encounters_verified=False,
        runtime_stop_actors=33,
        explicit_mapped_stop_locations=5,
        limitation="No driving test; planned opportunities and setup binding require runtime audit",
    )
    destination = ROOT / "reports/navigation_m6_registration_20261018.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
