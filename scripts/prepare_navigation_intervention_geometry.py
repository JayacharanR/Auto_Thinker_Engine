#!/usr/bin/env python3
"""Freeze split-safe multi-location action-intervention geometry before outcomes."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_dense_labels import offline_map  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402

SOURCE_IDS = (35, 50, 134, 189, 225, 231, 258, 297, 124, 211)


def read(path):
    return json.loads(Path(path).read_text())


def poses(route):
    return [
        [
            w.transform.location.x,
            w.transform.location.y,
            w.transform.location.z,
            w.transform.rotation.pitch,
            w.transform.rotation.yaw,
            w.transform.rotation.roll,
        ]
        for w, _ in route
    ]


def arc_length(points):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.asarray(points)[:, :2], axis=0), axis=1))]


def first_intersection(ego, other):
    """First geometric crossing with at least60degrees of crossing angle."""
    ego, other = np.asarray(ego), np.asarray(other)
    first_arc, second_arc = arc_length(ego), arc_length(other)
    hits = []
    for i, (a, b) in enumerate(zip(ego[:-1], ego[1:])):
        u = b[:2] - a[:2]
        for j, (c, d) in enumerate(zip(other[:-1], other[1:])):
            v = d[:2] - c[:2]
            denom = np.linalg.norm(u) * np.linalg.norm(v)
            if denom < 1e-9 or abs(u @ v) / denom > 0.5:
                continue
            matrix = np.column_stack([u, -v])
            if abs(np.linalg.det(matrix)) < 1e-9:
                continue
            s, t = np.linalg.solve(matrix, c[:2] - a[:2])
            if 0 <= s <= 1 and 0 <= t <= 1 and abs(a[2] - c[2]) < 2:
                hits.append(
                    (first_arc[i] + s * np.linalg.norm(u), second_arc[j] + t * np.linalg.norm(v))
                )
    return min(hits) if hits else None


def waypoint_key(wp):
    return wp.road_id, wp.section_id, wp.lane_id, wp.s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT / "outputs/navigation_dense_label_collection_20261020/plan.json"
    verified = ROOT / "outputs/navigation_dense_labels_verified_20261020/manifest.json"
    plan, manifest = read(plan_path), read(verified)
    receipt = read(verified.parent / "assessment.json")
    if (
        digest(verified) != receipt["manifest_sha256"]
        or not receipt["every_included_source_independently_verified"]
    ):
        raise ValueError("Only original independently verified source locations")
    accepted = {r["item"]["route_id"] for r in manifest["sources"]}
    if not set(SOURCE_IDS) <= accepted or 6 in SOURCE_IDS:
        raise ValueError("Never add the observed old fixed-branch diagnostic to new training")
    source_rows = {r["route_id"]: r for r in plan["episodes"]}
    guard = read(plan["actual_training_guard"])
    paths = [
        Path(__file__),
        plan_path,
        verified,
        Path(plan["actual_training_guard"]),
        ROOT / "src/navigation/encounters.py",
        ROOT / "scripts/audit_navigation_dense_labels.py",
    ]
    for path in paths:
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(
        args.out / "request.json",
        dict(
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in paths},
            source_ids=SOURCE_IDS,
            source_roles_inherited=True,
            mechanisms=["lead", "crossing", "rear"],
            selection="All8eligibleTRAINsources/2existing calibration locations; geometry only",
            old_route6_branch_diagnostic_excluded=True,
            no_models_or_simulator=True,
        ),
    )
    with BudgetWatch(args.out):
        activate()
        import carla
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        contexts, unavailable = [], []
        for route_id in SOURCE_IDS:
            source = source_rows[route_id]
            town, role, entry = source["town"], source["role"], source["entry"]
            world_map = offline_map(town, guard["geometry_eligibility"]["map_sha256"][town])
            planner = GlobalRoutePlanner(world_map, 1.0)
            route = planner.trace_route(
                carla.Location(*entry["start_xyz"]), carla.Location(*entry["end_xyz"])
            )
            ego = poses(route)
            arc = arc_length(ego)
            allowed = set(guard["allowed_junctions"][town]) - set(
                plan["reserved_rule_junctions"].get(town, [])
            )
            if role == "train":
                for item in plan["episodes"]:
                    if item["town"] == town and item["role"] == "calibration":
                        allowed -= set(item["entry"]["crossed_junction_ids"])
            forbidden = [
                w
                for w in world_map.generate_waypoints(0.5)
                if w.is_junction and w.junction_id not in allowed
            ]
            tree = cKDTree(
                [
                    [w.transform.location.x, w.transform.location.y, w.transform.location.z]
                    for w in forbidden
                ]
            )

            def check_path(points, label, maximum_arc=None):
                values = np.asarray(points)
                use = (
                    values[arc_length(values) <= maximum_arc] if maximum_arc is not None else values
                )
                minimum = float(np.min(tree.query(use[:, :3])[0]))
                if minimum < 1.0:
                    raise ValueError(f"{label} lacks1m reserved geometry clearance:{minimum}")
                for point in use:
                    wp = world_map.get_waypoint(carla.Location(*point[:3]))
                    if wp.is_junction and wp.junction_id not in allowed:
                        raise ValueError(f"{label} entered forbidden junction{wp.junction_id}")
                return minimum

            try:
                clearance = check_path(ego, "ego", maximum_arc=45)
            except ValueError as error:
                for mechanism in ("lead", "crossing", "rear"):
                    unavailable.append(
                        dict(route_id=route_id, role=role, mechanism=mechanism, error=str(error))
                    )
                continue
            common = dict(
                route_id=route_id,
                town=town,
                role=role,
                entry=entry,
                ego_route=ego,
                ego_minimum_reserved_clearance_m=clearance,
                allowed_junctions=sorted(allowed),
                branches=["go", "brake", "wait_go"],
                seed=62600 + route_id,
            )
            for mechanism in ("lead", "crossing", "rear"):
                try:
                    mechanism_ego = ego
                    if mechanism == "lead":
                        index = min(
                            int(np.searchsorted(arc, 18.0 + 6 * (route_id % 2))), len(ego) - 1
                        )
                        other, speed = ego[index:], 0.0
                        check_path(other[:1], "stationary lead")
                        detail = dict(lead_route_arc_m=float(arc[index]))
                    elif mechanism == "rear":
                        # Both actors begin on the original guarded route, rather
                        # than entering a reserved junction behind its old start.
                        requested_gap = 33.0 + 4 * (route_id % 2)
                        index = int(np.searchsorted(arc, requested_gap))
                        if index >= len(ego) - 10:
                            raise ValueError("Original route lacks a bounded rear encounter")
                        mechanism_ego = ego[index:]
                        other = ego
                        speed = 6.0
                        check_path(other, "rear route", maximum_arc=55)
                        check_path(mechanism_ego, "rear ego", maximum_arc=45)
                        detail = dict(
                            initial_rear_gap_requested_m=requested_gap,
                            ego_spawn_original_arc_m=float(arc[index]),
                        )
                    else:
                        junctions = {
                            w.junction_id: w.get_junction() for w, _ in route if w.is_junction
                        }
                        candidates = []
                        for junction_id in sorted(junctions):
                            if junction_id not in allowed:
                                continue
                            pairs = sorted(
                                junctions[junction_id].get_waypoints(carla.LaneType.Driving),
                                key=lambda p: (waypoint_key(p[0]), waypoint_key(p[1])),
                            )
                            for begin, end in pairs:
                                before = sorted(begin.previous(30.0), key=waypoint_key)
                                after = sorted(end.next(20.0), key=waypoint_key)
                                if not before or not after:
                                    continue
                                candidate = poses(
                                    planner.trace_route(
                                        before[0].transform.location, after[0].transform.location
                                    )
                                )
                                if len(candidate) < 3 or arc_length(candidate)[-1] > 160:
                                    continue
                                hit = first_intersection(ego, candidate)
                                if hit is None:
                                    continue
                                other_arc = arc_length(candidate)
                                desired_prefix = 3.0 * (4.0 + 2.0 + 1.2 * (route_id % 2))
                                begin_index = int(
                                    np.searchsorted(other_arc, hit[1] - desired_prefix)
                                )
                                if begin_index < 0 or hit[1] < desired_prefix:
                                    continue
                                cropped = candidate[begin_index:]
                                try:
                                    check_path(cropped, "crossing route", maximum_arc=30)
                                except ValueError:
                                    # Geometry-only selection before any native/model outcome.
                                    continue
                                candidates.append(
                                    (
                                        hit[0],
                                        begin.road_id,
                                        begin.lane_id,
                                        cropped,
                                        hit[1] - other_arc[begin_index],
                                    )
                                )
                        if not candidates:
                            raise ValueError(
                                "No crossing path satisfies original geometry and angle gates"
                            )
                        chosen = min(candidates, key=lambda r: r[:3])
                        other, speed = chosen[3], 3.0
                        # A junction63m away cannot be reached in this8srecord.
                        # Freeze a meaningful near-junction start before execution.
                        spawn_index = int(np.searchsorted(arc, max(0.0, chosen[0] - 20.0)))
                        mechanism_ego = ego[spawn_index:]
                        check_path(mechanism_ego, "crossing ego", maximum_arc=45)
                        detail = dict(
                            ego_arc_at_crossing_m=float(chosen[0] - arc[spawn_index]),
                            ego_spawn_original_arc_m=float(arc[spawn_index]),
                            other_distance_to_crossing_m=float(chosen[4]),
                        )
                    contexts.append(
                        common
                        | dict(
                            context_id=f"r{route_id}_{mechanism}",
                            mechanism=mechanism,
                            ego_route=mechanism_ego,
                            other_route=other,
                            other_target_speed_mps=speed,
                            geometry_detail=detail,
                        )
                    )
                except (ValueError, RuntimeError) as error:
                    unavailable.append(
                        dict(route_id=route_id, role=role, mechanism=mechanism, error=str(error))
                    )
            atomic_json(
                args.out / "progress.json",
                dict(
                    status="running",
                    stage="Intervention geometry",
                    frames=len(contexts) + len(unavailable),
                    target_frames=30,
                    phase="Offline geometry only; no model or simulator outcomes",
                    updated_at=__import__("time").time(),
                ),
            )
        atomic_json(
            args.out / "plan.json",
            dict(
                schema="navigation-action-intervention-geometry-v2",
                contexts=contexts,
                unavailable_geometry=unavailable,
                planned_contexts=30,
                executable_contexts=len(contexts),
                planned_options_per_context=3,
                calibration_route_ids=[124, 211],
                train_route_ids=[35, 50, 134, 189, 225, 231, 258, 297],
                all_source_roles_inherited=True,
                no_native_or_model_outcomes=True,
                original_data_plan=str(plan_path),
                original_data_plan_sha256=digest(plan_path),
            ),
        )
        atomic_json(
            args.out / "progress.json",
            dict(
                status="complete",
                stage="Intervention geometry",
                frames=30,
                target_frames=30,
                phase=f"{len(contexts)}geometries frozen;{len(unavailable)}unavailable retained",
                updated_at=__import__("time").time(),
            ),
        )
        print(json.dumps(dict(executable=len(contexts), unavailable=unavailable), indent=2))


if __name__ == "__main__":
    main()
