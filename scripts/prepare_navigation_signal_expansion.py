#!/usr/bin/env python3
"""Freeze additional TRAIN approaches/distances without opening a CARLA server."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_navigation_signal_pairs import validate_plan  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_data import stop_reference  # noqa: E402
from src.navigation.signal_expansion import duplicate_pose, expansion_entries  # noqa: E402
from src.navigation.signal_learning import junction_roles  # noqa: E402
from src.navigation.splits import nearest_junction  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def prepare(args):
    vendor = activate()
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    if args.plan.exists():
        raise FileExistsError("Fresh expansion plan required")
    original = json.loads((args.base / "plan.json").read_text())
    guard = validate_plan(original)
    catalog = json.loads(Path(original["catalog"]).read_text())
    roles = junction_roles(original["cases"])
    entries = expansion_entries(catalog, guard, roles)
    cases, skipped = [], []
    for town in sorted({e["town"] for e in entries}):
        inventory_path = ROOT / catalog["inventory"] / f"{town}.json"
        if digest(inventory_path) != catalog["inventory_sha256"][town]:
            raise ValueError("Frozen inventory changed")
        inventory = json.loads(inventory_path.read_text())
        xodr = Path(vendor["carla_root"]) / (
            f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town.removesuffix('_Opt')}.xodr"
        )
        world_map = carla.Map(town, xodr.read_text())
        if (
            hashlib.sha256(world_map.to_opendrive().encode()).hexdigest()
            != original["map_sha256"][town]
        ):
            raise ValueError("TRAIN map changed")
        planner = GlobalRoutePlanner(world_map, 1)
        permitted = {j for (t, j), role in roles.items() if t == town and role == "train"}
        reserved = [
            w
            for w in world_map.generate_waypoints(0.5)
            if w.is_junction and w.junction_id not in permitted
        ]
        reserved_xyz = [
            [w.transform.location.x, w.transform.location.y, w.transform.location.z]
            for w in reserved
        ]
        reserved_ids = [w.junction_id for w in reserved]
        for entry in [e for e in entries if e["town"] == town]:
            light = next(
                r for r in inventory["lights"] if r["id"] == entry["signal_ids_at_inventory"][0]
            )
            traced = planner.trace_route(
                carla.Location(*entry["start_xyz"]), carla.Location(*entry["end_xyz"])
            )
            points = np.array(
                [
                    [w.transform.location.x, w.transform.location.y, w.transform.location.z]
                    for w, _ in traced
                ]
            )
            reference = stop_reference(
                points,
                [[w.road_id, w.section_id, w.lane_id] for w, _ in traced],
                light["stop_lanes"],
            )
            if reference is None:
                skipped.append(
                    dict(source_route_id=entry["route_id"], reason="No unique stop-lane reference")
                )
                continue
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            for distance in (32, 12, 2):
                index = max(
                    0,
                    int(np.searchsorted(arc, max(0, reference["arc_m"] - distance), side="right"))
                    - 1,
                )
                wp = traced[index][0]
                t = wp.transform
                clearance = nearest_junction([points[index]], reserved_xyz, reserved_ids)
                if clearance["clearance_m"] < 1 or (
                    wp.is_junction and wp.junction_id not in permitted
                ):
                    skipped.append(
                        dict(
                            source_route_id=entry["route_id"],
                            distance=distance,
                            reason="Reserved-junction spatial clearance",
                            **clearance,
                        )
                    )
                    continue
                case = dict(
                    route_id=10000 + len(cases),
                    source_route_id=entry["route_id"],
                    town=town,
                    junction_id=entry["junction_id"],
                    role="train",
                    ego_blueprint="vehicle.lincoln.mkz_2020",
                    ego_pose=[
                        t.location.x,
                        t.location.y,
                        t.location.z + 0.35,
                        t.rotation.pitch,
                        t.rotation.yaw,
                        t.rotation.roll,
                    ],
                    inventory_light_xyz=light["xyz"],
                    inventory_light_id=light["id"],
                    evaluator_stop_lane=reference["line"],
                    approach_lane=reference["line"]["lane"],
                    reference_distance_m=float(reference["arc_m"] - arc[index]),
                    requested_reference_distance_m=distance,
                    include_yellow=True,
                    background_seed=2026300000 + entry["route_id"] * 100 + distance,
                    reserved_clearance=clearance,
                )
                if duplicate_pose(case, original["cases"] + cases):
                    skipped.append(
                        dict(
                            source_route_id=entry["route_id"],
                            distance=distance,
                            reason="Duplicate existing or clamped pose",
                        )
                    )
                    continue
                cases.append(case)
    plan = {
        **original,
        "kind": "TRAIN-only static signal approach supplement; no new validation",
        "cases": cases,
        "skipped_geometry": skipped,
        "selection": (
            "Original13 TRAIN groups only; first3 light/incoming-lane approaches in route-ID "
            "order;32/12/2m; deduplicate poses"
        ),
        "expansion": dict(
            base_corpus=str(args.base.resolve()),
            base_audit_sha256=digest(args.base / "data_audit.json"),
            base_plan_sha256=digest(args.base / "plan.json"),
            original_groups=[dict(town=t, junction_id=j, role=r) for (t, j), r in roles.items()],
        ),
        "minimum_data_gate": dict(
            towns=4, responsive_junctions=10, supported_pairs=45, multilane_junctions=6
        ),
        "additional_sources": [
            "scripts/prepare_navigation_signal_expansion.py",
            "reports/navigation_m2_signal_expansion_protocol_20261016.md",
        ],
        "maximum_wall_seconds": 900,
    }
    if len(cases) > 117 or len({(c["town"], c["junction_id"]) for c in cases}) != 13:
        raise ValueError("Expected bounded coverage of original13 TRAIN junctions")
    validate_plan(plan)
    atomic_json(args.plan, plan)
    print(
        json.dumps(
            dict(
                plan=str(args.plan),
                sha256=digest(args.plan),
                cases=len(cases),
                towns=dict(Counter(c["town"] for c in cases)),
                approaches=len(
                    {(c["town"], c["junction_id"], tuple(c["approach_lane"])) for c in cases}
                ),
                skipped=len(skipped),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    prepare(parser.parse_args())
