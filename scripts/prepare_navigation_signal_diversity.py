#!/usr/bin/env python3
"""Freeze more eligible TRAIN signal geometry without simulator/image outcomes."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_dense_labels import offline_map  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_data import stop_reference  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    eligibility_path = ROOT / "outputs/navigation_signal_pixel_audit_20261021/request.json"
    eligibility = read(eligibility_path)
    # The original audit archived the actualguard, ruleplan and allowed intersection.
    for file, sha in eligibility["source_hashes"].items():
        if digest(ROOT / file) != sha:
            raise ValueError("Audited geography/source changed")
    guard_path = (
        ROOT / "outputs/navigation_spatial_data_repaired_20261019/actual_training_guard.json"
    )
    guard = read(guard_path)
    catalog_path = ROOT / "configs/encounters/location_split_v1/train/catalog.json"
    catalog = read(catalog_path)
    allowed = eligibility["allowed_junctions"]
    activate()
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    all_cases, skipped, inventories, approaches = [], [], {}, []
    for town in sorted(allowed):
        entries = sorted(
            (
                e
                for e in catalog["encounters"]
                if e["town"] == town
                and e["signal_ids_at_inventory"]
                and e["junction_id"] in allowed[town]
                and set(e["crossed_junction_ids"]) <= set(allowed[town])
                and e["route_id"] in guard["geometry_eligibility"]["kept_route_ids"]
            ),
            key=lambda e: e["route_id"],
        )
        if not entries:
            skipped.append(dict(town=town, reason="No eligible signal route in current TRAIN"))
            continue
        inventory_path = ROOT / catalog["inventory"] / f"{town}.json"
        if digest(inventory_path) != catalog["inventory_sha256"][town]:
            raise ValueError("Original inventory changed")
        inventories[str(inventory_path.relative_to(ROOT))] = digest(inventory_path)
        inventory = read(inventory_path)
        world_map = offline_map(town, guard["geometry_eligibility"]["map_sha256"][town])
        planner = GlobalRoutePlanner(world_map, 1)
        forbidden = [
            w.transform.location
            for w in world_map.generate_waypoints(0.5)
            if w.is_junction and w.junction_id not in allowed[town]
        ]
        tree = cKDTree([[p.x, p.y, p.z] for p in forbidden])
        selected_approaches, positions = {}, set()
        for entry in entries:
            junction = entry["junction_id"]
            if len(selected_approaches.get(junction, set())) >= 2:
                continue
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
                    dict(town=town, route_id=entry["route_id"], reason="No unique evaluator lane")
                )
                continue
            key = (light["id"], tuple(reference["line"]["lane"]))
            if key in selected_approaches.get(junction, set()):
                continue
            selected_approaches.setdefault(junction, set()).add(key)
            approaches.append(
                dict(
                    town=town,
                    junction_id=junction,
                    source_route_id=entry["route_id"],
                    inventory_light_id=light["id"],
                    lane=reference["line"]["lane"],
                )
            )
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            for distance in (40, 20, 6):
                index = max(
                    0,
                    int(np.searchsorted(arc, max(0, reference["arc_m"] - distance), side="right"))
                    - 1,
                )
                wp = traced[index][0]
                t = wp.transform
                pose = [
                    t.location.x,
                    t.location.y,
                    t.location.z + 0.35,
                    t.rotation.pitch,
                    t.rotation.yaw,
                    t.rotation.roll,
                ]
                clearance = float(tree.query(pose[:3])[0])
                reason = None
                if clearance < 1 or (wp.is_junction and wp.junction_id not in allowed[town]):
                    reason = "Reserved actual junction or <1m clearance"
                identity = tuple(round(v, 4) for v in pose)
                if identity in positions:
                    reason = "Duplicate/clamped pose"
                if reason:
                    skipped.append(
                        dict(
                            town=town,
                            source_route_id=entry["route_id"],
                            distance=distance,
                            reason=reason,
                            clearance_m=clearance,
                            actual_junction=wp.junction_id if wp.is_junction else None,
                        )
                    )
                    continue
                positions.add(identity)
                all_cases.append(
                    dict(
                        route_id=20000 + len(all_cases),
                        source_route_id=entry["route_id"],
                        town=town,
                        junction_id=junction,
                        role="train",
                        ego_blueprint="vehicle.lincoln.mkz_2020",
                        ego_pose=pose,
                        inventory_light_xyz=light["xyz"],
                        inventory_light_id=light["id"],
                        evaluator_stop_lane=reference["line"],
                        approach_lane=reference["line"]["lane"],
                        reference_distance_m=float(reference["arc_m"] - arc[index]),
                        requested_reference_distance_m=distance,
                        include_yellow=True,
                        reserved_clearance_m=clearance,
                        background_seed=2026210000 + entry["route_id"] * 100 + distance,
                    )
                )
    weather = dict(
        clear_noon=dict(
            cloudiness=0.0,
            sun_altitude_angle=70.0,
            sun_azimuth_angle=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wetness=0.0,
            fog_density=0.0,
            wind_intensity=0.0,
        ),
        overcast_low_sun=dict(
            cloudiness=80.0,
            sun_altitude_angle=15.0,
            sun_azimuth_angle=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wetness=0.0,
            fog_density=0.0,
            wind_intensity=0.0,
        ),
    )
    tasks = [
        dict(
            task_id=f"{town}_{condition}",
            town=town,
            illumination=condition,
            weather=values,
            cases=[c for c in all_cases if c["town"] == town],
        )
        for town in sorted({c["town"] for c in all_cases})
        for condition, values in weather.items()
    ]
    if (
        len(tasks) > 8
        or not all_cases
        or any(c["town"].removesuffix("_Opt") == "Town05" for c in all_cases)
    ):
        raise ValueError("Fixed TRAIN-only bounded plan required")
    plan = dict(
        schema="navigation-strict-signal-diversity-v1",
        tasks=tasks,
        cases=all_cases,
        approaches=approaches,
        geometry_skips=skipped,
        allowed_junctions=allowed,
        catalog=str(catalog_path),
        catalog_sha256=digest(catalog_path),
        guard=str(guard_path),
        guard_sha256=digest(guard_path),
        eligibility_request=str(eligibility_path),
        eligibility_request_sha256=digest(eligibility_path),
        map_sha256=guard["geometry_eligibility"]["map_sha256"],
        inventory_sha256=inventories,
        cases_per_illumination=len(all_cases),
        planned_cases=sum(len(t["cases"]) for t in tasks),
        maximum_native_seconds=2400,
        child_seconds=300,
        maximum_artifact_bytes=1_000_000_000,
        no_fit=True,
        no_failed_retry=True,
        protocol_sha256=digest(ROOT / "reports/navigation_signal_diversity_protocol_20261021.md"),
        source_sha256=digest(__file__),
    )
    atomic_json(out / "plan.json", plan)
    atomic_json(
        out / "geometry_summary.json",
        dict(
            towns=dict(Counter(c["town"] for c in all_cases)),
            unique_locations=len({(c["town"], c["junction_id"]) for c in all_cases}),
            poses=len(all_cases),
            approaches=len(approaches),
            tasks=len(tasks),
            planned_cases=plan["planned_cases"],
            planned_rgb_images=12 * plan["planned_cases"],
            geometry_skips=skipped,
            plan_sha256=digest(out / "plan.json"),
            no_native_or_model_outcomes=True,
        ),
    )
    print(json.dumps({k: plan[k] for k in ("cases_per_illumination", "planned_cases")}, indent=2))


if __name__ == "__main__":
    main()
