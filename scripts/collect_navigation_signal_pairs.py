#!/usr/bin/env python3
"""Freeze/collect/audit bounded static phase pairs on approved TRAIN geometry."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe_navigation_signal_pixels import render_cases  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.rule_geometry import capture_light_geometry  # noqa: E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.signal_data import (  # noqa: E402
    phase_response,
    phase_response_v2,
    select_entries,
    stop_reference,
)
from src.navigation.vendor import activate  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def load_training_world(client, town, map_sha256):
    """A packaged server may open its default map despite its launch URL."""
    client.set_timeout(120)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != town:
        world = client.load_world(town)
    if (
        world.get_map().name.split("/")[-1] != town
        or hashlib.sha256(world.get_map().to_opendrive().encode()).hexdigest() != map_sha256
    ):
        raise ValueError("Wrong live TRAIN map")
    client.set_timeout(10)
    return world


def validate_plan(plan):
    if (
        digest(plan["guard"]) != plan["guard_sha256"]
        or digest(plan["catalog"]) != plan["catalog_sha256"]
    ):
        raise ValueError("Frozen signal-pair sources changed")
    guard = read(plan["guard"])
    if (
        guard["catalog_sha256"] != plan["catalog_sha256"]
        or plan["map_sha256"] != guard["geometry_eligibility"]["map_sha256"]
    ):
        raise ValueError("Plan differs from the original TRAIN geometry guard")
    if digest(guard["internal_manifest"]) != guard["internal_manifest_sha256"]:
        raise ValueError("Original internal split changed")
    for case in plan["cases"]:
        if (
            case["town"].removesuffix("_Opt") == "Town05"
            or case["town"] not in guard["allowed_junctions"]
        ):
            raise ValueError("Only approved TRAIN towns")
        if case["junction_id"] not in guard["allowed_junctions"][case["town"]]:
            raise ValueError("Reserved junction in signal-pair plan")
    if "expansion" in plan:
        from src.navigation.signal_expansion import validate_expansion

        validate_expansion(plan)
    return guard


def prepare(args, vendor):
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    if args.plan.exists():
        raise FileExistsError("Fresh frozen plan required")
    guard, catalog = read(args.guard), read(args.catalog / "catalog.json")
    if (
        digest(args.catalog / "catalog.json") != guard["catalog_sha256"]
        or digest(guard["internal_manifest"]) != guard["internal_manifest_sha256"]
    ):
        raise ValueError("Original guard/catalog/internal manifest changed")
    entries = select_entries(catalog, guard)
    cases, skipped = [], []
    for town in sorted({entry["town"] for entry in entries}):
        path = ROOT / catalog["inventory"] / f"{town}.json"
        if digest(path) != catalog["inventory_sha256"][town]:
            raise ValueError("Signal applicability inventory changed")
        inventory = read(path)
        xodr = (
            Path(vendor["carla_root"])
            / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town.removesuffix('_Opt')}.xodr"
        )
        world_map = carla.Map(town, xodr.read_text())
        if (
            hashlib.sha256(world_map.to_opendrive().encode()).hexdigest()
            != inventory["opendrive_sha256"]
        ):
            raise ValueError("TRAIN map changed")
        planner = GlobalRoutePlanner(world_map, 1)
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
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            reference = stop_reference(
                points,
                [[w.road_id, w.section_id, w.lane_id] for w, _ in traced],
                light["stop_lanes"],
            )
            if not reference:
                skipped.append(
                    {
                        "route_id": entry["route_id"],
                        "reason": "No unambiguous evaluator-line lane on frozen route",
                    }
                )
                continue
            stop_arc, line = reference["arc_m"], reference["line"]
            positions = set()
            for distance in (40, 20, 6):
                index = max(
                    0,
                    int(np.searchsorted(arc, max(0, stop_arc - distance), side="right")) - 1,
                )
                if index in positions:
                    continue  # Clamped-to-start viewpoints are duplicates, not extra data.
                positions.add(index)
                wp = traced[index][0]
                if wp.is_junction and wp.junction_id not in guard["allowed_junctions"][town]:
                    raise ValueError("Static pose enters reserved junction")
                t = wp.transform
                cases.append(
                    {
                        "route_id": len(cases),
                        "source_route_id": entry["route_id"],
                        "town": town,
                        "junction_id": entry["junction_id"],
                        "ego_blueprint": "vehicle.lincoln.mkz_2020",
                        "ego_pose": [
                            t.location.x,
                            t.location.y,
                            t.location.z + 0.35,
                            t.rotation.pitch,
                            t.rotation.yaw,
                            t.rotation.roll,
                        ],
                        "inventory_light_xyz": light["xyz"],
                        "inventory_light_id": light["id"],
                        "evaluator_stop_lane": line,
                        "reference_distance_m": float(stop_arc - arc[index]),
                        "requested_reference_distance_m": distance,
                        "include_yellow": True,
                        "background_seed": 2026101600
                        + entry["route_id"] * 100
                        + distance
                        + args.background_seed_offset,
                    }
                )
    plan = {
        "kind": "Static perception-training pairs; no driving/control/future-action targets",
        "catalog": str((args.catalog / "catalog.json").resolve()),
        "catalog_sha256": digest(args.catalog / "catalog.json"),
        "guard": str(args.guard.resolve()),
        "guard_sha256": digest(args.guard),
        "selection": "First4 distinct junctions per town; stop before memory, route ID order; "
        "poses40/20/6m before evaluator line, deduplicate start clamps",
        "cases": cases,
        "skipped_geometry": skipped,
        "map_sha256": guard["geometry_eligibility"]["map_sha256"],
        "pose_note": "Road pose plus0.35m stationary car height; declared rig, no physics driving",
        "labels": "Red/green/yellow/red-repeat with seeded background phases; training labels only",
        "phase_response_filter": "RG MAE>=2 and>=5*max(RR MAE,.25); R/G/R-repeat dominance>=30. "
        "Chosen from prior TRAIN pixel probe, not benchmark outcomes",
        "phase_filter_version": args.phase_filter_version,
        "minimum_data_gate": {"towns": 5, "responsive_junctions": 10},
        "maximum_wall_seconds": 900,
    }
    if args.phase_filter_version == 2:
        plan["phase_response_filter"] = (
            "RG MAE>=2 and>=5*max(RR MAE,.25); separate red-on and green-on intensity "
            "gains>=20 and max channel change>=25. No absolute green-hue assumption. "
            "Developed on prior TRAIN captures; original v1 data gate remains failed."
        )
    if not cases or len({case["town"] for case in cases}) != 5:
        raise ValueError("Do not freeze an empty/incomplete five-town data plan")
    validate_plan(plan)
    atomic_json(args.plan, plan)
    print(
        json.dumps(
            {
                "plan": str(args.plan),
                "sha256": digest(args.plan),
                "cases": len(cases),
                "towns": dict(Counter(c["town"] for c in cases)),
                "junctions": len({(c["town"], c["junction_id"]) for c in cases}),
                "skipped_geometry": skipped,
            },
            indent=2,
        )
    )


def worker(args, config):
    import carla

    plan = read(args.out / "plan.json")
    guard = validate_plan(plan)
    client = carla.Client("127.0.0.1", config["runtime"]["port"])
    client.set_timeout(10)
    world = load_training_world(client, args.worker, plan["map_sha256"][args.worker])
    geometry = {"town": args.worker, "lights": capture_light_geometry(world)}
    directory = args.out / args.worker
    atomic_json(directory / "geometry.json", geometry)
    cases = []
    for source in [c for c in plan["cases"] if c["town"] == args.worker]:
        case = dict(source)
        candidates = [
            r
            for r in geometry["lights"]
            if np.linalg.norm(np.asarray(r["pose"][:3]) - case["inventory_light_xyz"]) < 0.002
        ]
        if len(candidates) != 1:
            raise ValueError("Inventory lamp lacks unique live physical match")
        lamp = candidates[0]
        if not any(
            line["lane"] == case["evaluator_stop_lane"]["lane"]
            and np.linalg.norm(np.asarray(line["center"]) - case["evaluator_stop_lane"]["xyz"])
            < 0.002
            for line in lamp["evaluator_stop_lines"]
        ):
            raise ValueError("Inventory/live evaluator stop-lane identity changed")
        wp = world.get_map().get_waypoint(carla.Location(*case["ego_pose"][:3]))
        if wp.is_junction and wp.junction_id not in guard["allowed_junctions"][args.worker]:
            raise ValueError("Live pose entered reserved junction")
        case["light"] = {
            "type": "traffic.traffic_light",
            "id": lamp["capture_actor_id"],
            "pose": lamp["pose"],
        }
        cases.append(case)
    atomic_json(directory / "cases.json", cases)
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    world.apply_settings(settings)
    try:
        render_cases(argparse.Namespace(out=directory, geometry=directory), config, world, geometry)
    finally:
        world.apply_settings(original)


def audit(out):
    plan, request = read(out / "plan.json"), read(out / "request.json")
    validate_plan(plan)
    if digest(out / "plan.json") != request["plan_sha256"] or any(
        digest(out / "source" / path) != sha for path, sha in request["source_hashes"].items()
    ):
        raise ValueError("Corpus plan/source archive changed")
    expected = {c["route_id"]: c for c in plan["cases"]}
    labels, image_hashes, seen, responsive = [], {}, set(), set()
    for town in sorted({c["town"] for c in expected.values()}):
        directory = out / town
        a = read(directory / "assessment.json")
        if (
            not a["probe_complete"]
            or digest(directory / "cases.json") != a["source_cases_sha256"]
            or digest(directory / "geometry.json") != a["geometry_sha256"]
        ):
            raise ValueError("Incomplete/changed town capture")
        cases = {c["route_id"]: c for c in read(directory / "cases.json")}
        for row in a["cases"]:
            rid = row["route_id"]
            if (
                rid not in expected
                or expected[rid]["town"] != town
                or row["states"] != ["Red", "Green", "Yellow", "Red"]
                or not row["all_lights_frozen"]
                or row["ego_position_drift_m"] > 1e-3
                or max(row["camera_matrix_max_error"].values()) > 1e-3
            ):
                raise ValueError("Incomplete/invalid phase setup")
            if any(cases[rid][key] != value for key, value in expected[rid].items()):
                raise ValueError("Frozen static pose changed")
            seen.add(rid)
        for rid in cases:
            for state in ("red", "green", "yellow", "red_repeat"):
                for view in ("front", "left", "right"):
                    path = directory / f"route{rid}_{state}_{view}.png"
                    with Image.open(path) as image:
                        if image.size != (512, 288) or image.mode != "RGB":
                            raise ValueError("Wrong native RGB dimensions/encoding")
                        image.load()
                    image_hashes[str(path.relative_to(out))] = digest(path)
        for row in a["housing_pixels"]:
            rid, view = row["route_id"], row["view"]
            box = row["contrast"]["crop_xyxy"]
            crops = {}
            for state in ("red", "green", "yellow", "red_repeat"):
                with Image.open(directory / f"route{rid}_{state}_{view}.png") as image:
                    crops[state] = np.asarray(image.crop(box)).copy()
            validator = (
                phase_response_v2 if plan.get("phase_filter_version", 1) == 2 else phase_response
            )
            result = validator(crops["red"], crops["green"], crops["red_repeat"])
            yellow = crops["yellow"].astype(float)
            yellow_strength = float(
                (np.minimum(yellow[..., 0], yellow[..., 1]) - yellow[..., 2]).max()
            )
            labels.append(
                {
                    "case_id": rid,
                    "town": town,
                    "junction_id": expected[rid]["junction_id"],
                    "view": view,
                    "head": row["head"],
                    "crop_xyxy": box,
                    **result,
                    "yellow_color_supported": bool(
                        result["static_phase_response_supported"] and yellow_strength >= 30
                    ),
                    "yellow_strength": yellow_strength,
                }
            )
            if result["static_phase_response_supported"]:
                responsive.add((town, expected[rid]["junction_id"]))
    gate = (
        len(seen) == len(expected)
        and len({t for t, _ in responsive}) == plan["minimum_data_gate"]["towns"]
        and len(responsive) >= plan["minimum_data_gate"]["responsive_junctions"]
    )
    expansion_coverage = None
    if "expansion" in plan:
        lanes = {}
        for label in labels:
            if label["static_phase_response_supported"]:
                case = expected[label["case_id"]]
                lanes.setdefault((label["town"], label["junction_id"]), set()).add(
                    tuple(case["approach_lane"])
                )
        expansion_coverage = dict(
            responsive_approach_lanes=sum(map(len, lanes.values())),
            multilane_junctions=sum(len(v) >= 2 for v in lanes.values()),
        )
        gate = (
            gate
            and sum(r["static_phase_response_supported"] for r in labels)
            >= plan["minimum_data_gate"]["supported_pairs"]
            and expansion_coverage["multilane_junctions"]
            >= plan["minimum_data_gate"]["multilane_junctions"]
        )
    result = {
        "audit_complete": True,
        "all_cases_complete": seen == set(expected),
        "static_label_data_gate_passed": gate,
        "cases": len(seen),
        "native_rgb_images": len(image_hashes),
        "responsive_junctions": [list(p) for p in sorted(responsive)],
        "supported_head_view_pairs": sum(r["static_phase_response_supported"] for r in labels),
        "yellow_supported_pairs": sum(r["yellow_color_supported"] for r in labels),
        "labels": labels,
        "image_sha256": image_hashes,
        "plan_sha256": digest(out / "plan.json"),
        "scope": "Static TRAIN perception pairs; no driving, action target, "
        "policy performance or original-traffic visibility claim",
        **({"expansion_coverage": expansion_coverage} if expansion_coverage else {}),
    }
    atomic_json(out / "data_audit.json", result)
    return result


def collect(args, vendor, config):
    plan = read(args.plan)
    validate_plan(plan)
    if args.out.exists():
        raise FileExistsError("Fresh corpus output required")
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        check_resources(args.out, config["runtime"]["minimum_free_gb"])
        sources = [
            *sorted((ROOT / "src/navigation").glob("*.py")),
            Path(__file__).resolve(),
            ROOT / "scripts/probe_navigation_signal_pixels.py",
            ROOT / "scripts/audit_navigation_rule_observability.py",
            ROOT / "configs/navigation.yaml",
            *(ROOT / name for name in plan.get("additional_sources", [])),
        ]
        for source in sources:
            target = args.out / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(args.plan, args.out / "plan.json")
        atomic_json(
            args.out / "request.json",
            {
                "plan_sha256": digest(args.plan),
                "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "max_wall": plan["maximum_wall_seconds"],
            },
        )
        started, done, status = time.monotonic(), 0, "failed"
        try:
            for town in sorted({c["town"] for c in plan["cases"]}):
                if (args.out / "STOP").exists():
                    raise InterruptedError("STOP requested")
                directory = args.out / town
                directory.mkdir()
                with CarlaServer(
                    vendor["carla_root"], directory, config["runtime"]["port"], town=town
                ):
                    with (directory / "worker.log").open("x") as log:
                        child = subprocess.Popen(
                            [
                                sys.executable,
                                "-u",
                                str(Path(__file__).resolve()),
                                "collect",
                                "--worker",
                                town,
                                "--plan",
                                str(args.plan),
                                "--out",
                                str(args.out),
                            ],
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        stopping = None
                        while child.poll() is None:
                            if time.monotonic() - started > plan["maximum_wall_seconds"]:
                                (args.out / "STOP").touch()
                            if (args.out / "STOP").exists():
                                (directory / "STOP").touch()
                                stopping = stopping or time.monotonic()
                                if time.monotonic() - stopping > 20:
                                    child.terminate()
                                    child.wait(timeout=20)
                                    raise InterruptedError(
                                        "STOP grace expired; owned child terminated"
                                    )
                            complete_images = len(
                                list(directory.glob("route*_red_repeat_right.png"))
                            )
                            atomic_json(
                                args.out / "progress.json",
                                {
                                    "status": "running",
                                    "stage": "M2 static signal supervision",
                                    "current_chunk": town,
                                    "encounters_done": done + complete_images,
                                    "encounters_total": len(plan["cases"]),
                                    "route_progress_percent": 100
                                    * (done + complete_images)
                                    / len(plan["cases"]),
                                    "updated_at": time.time(),
                                    "note": "Static training poses, not driving encounters",
                                },
                            )
                            time.sleep(1)
                    if child.returncode:
                        raise RuntimeError(f"Static phase worker failed: {town}; see worker.log")
                done += sum(c["town"] == town for c in plan["cases"])
            result = audit(args.out)
            status = "complete"
        finally:
            atomic_json(
                args.out / "progress.json",
                {
                    "status": "stopped" if (args.out / "STOP").exists() else status,
                    "stage": "M2 static signal supervision",
                    "encounters_done": done,
                    "encounters_total": len(plan["cases"]),
                    "updated_at": time.time(),
                    "seconds": time.monotonic() - started,
                },
            )
        print(json.dumps({k: v for k, v in result.items() if k not in ("labels", "image_sha256")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "collect", "audit"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--guard", type=Path, default=ROOT / "reports/navigation_m2_dagger_guard_20261016.json"
    )
    parser.add_argument(
        "--catalog", type=Path, default=ROOT / "configs/encounters/location_split_v1/train"
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--phase-filter-version", type=int, choices=[1, 2], default=1)
    parser.add_argument("--background-seed-offset", type=int, default=0)
    args = parser.parse_args()
    vendor = activate()
    config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
    if args.command == "prepare":
        prepare(args, vendor)
    elif args.command == "audit":
        print(
            json.dumps(
                {k: v for k, v in audit(args.out).items() if k not in ("labels", "image_sha256")}
            )
        )
    elif args.worker:
        worker(args, config)
    else:
        collect(args, vendor, config)


if __name__ == "__main__":
    main()
