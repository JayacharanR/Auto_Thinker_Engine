#!/usr/bin/env python3
"""Town05 geometry-only registration; never loads a policy or produces training data."""

import argparse
import json
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_encounter_catalog import WEATHER  # noqa: E402
from scripts.build_encounter_inventory import inventory_town  # noqa: E402
from src.navigation.catalog import SCENARIO_PARAMETERS  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def registration_opportunities(trace, inventory):
    """Geometry only, with directional lane matching and endpoint censoring."""
    points = np.array(
        [[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w, _ in trace]
    )
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    lanes = np.array([[w.road_id, w.section_id, w.lane_id] for w, _ in trace])
    opportunities = {}
    for kind, actors, lane_field in (
        ("stop", inventory["stops"], "affected_lanes"),
        ("signal", inventory["lights"], "stop_lanes"),
    ):
        found = []
        for actor in actors:
            for lane in actor[lane_field]:
                mask = (lanes == lane["lane"]).all(axis=1)
                distance = np.linalg.norm(points - np.asarray(lane["xyz"]), axis=1)
                indices = np.flatnonzero(
                    mask & (distance <= 2) & (arc >= 20) & (arc <= arc[-1] - 20)
                )
                if len(indices):
                    index = int(indices[np.argmin(distance[indices])])
                    found.append(
                        dict(
                            actor_id=actor["id"],
                            lane=lane["lane"],
                            route_index=index,
                            arc_m=float(arc[index]),
                        )
                    )
                    break
        opportunities[kind] = sorted(found, key=lambda r: (r["arc_m"], r["actor_id"]))
    return opportunities, arc


def match_explicit_stop_signals(world_map, xodr, runtime_stops):
    """Five explicit OpenDRIVE signals; additional runtime stops remain scored."""
    signal_ids = sorted(
        {n.get("id") for n in ET.parse(xodr).findall(".//signals/signal") if n.get("type") == "206"}
    )
    if len(signal_ids) != 5:
        raise ValueError("Approved suite expects five explicit OpenDRIVE stop signals")
    landmarks = world_map.get_all_landmarks_of_type("206")
    matches = []
    for signal_id in signal_ids:
        landmark = next((m for m in landmarks if m.id == signal_id), None)
        if landmark is None:
            raise ValueError("Explicit stop signal has no CARLA landmark")
        p = landmark.transform.location
        xyz = np.array([p.x, p.y, p.z])
        nearby = [s for s in runtime_stops if np.linalg.norm(xyz - np.asarray(s["xyz"])) < 1]
        if len(nearby) != 1:
            raise ValueError("Explicit mapped stop has no unique physical runtime match")
        matches.append(
            dict(
                landmark_id=signal_id,
                actor_id=nearby[0]["id"],
                landmark_xyz=xyz.tolist(),
                runtime_xyz=nearby[0]["xyz"],
            )
        )
    if len({m["actor_id"] for m in matches}) != 5:
        raise ValueError("Mapped stop signals do not identify five distinct runtime locations")
    return matches


def inventory(args):
    if args.out.exists():
        raise FileExistsError("Fresh final-geometry inventory required")
    check_resources(args.out, 25)
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/build_encounter_inventory.py",
        ROOT / "src/navigation/encounters.py",
        ROOT / "src/navigation/runtime.py",
        ROOT / "reports/navigation_m6_registration_protocol_20261018.md",
    ]
    for p in sources:
        target = args.out / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    atomic_json(
        args.out / "request.json",
        dict(
            purpose="Held-out final geometry registration only",
            town="Town05_Opt",
            policy_loaded=False,
            training_data=False,
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        ),
    )
    with CarlaServer(args.carla_root, args.out / "server", 2100, town="Town05_Opt"):
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "inventory-worker",
            "--out",
            str(args.out),
            "--carla-root",
            args.carla_root,
        ]
        with (args.out / "worker.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            stop_started = None
            while child.poll() is None:
                if time.monotonic() - started > 600:
                    (args.out / "STOP").touch()
                if (args.out / "STOP").exists():
                    stop_started = stop_started or time.monotonic()
                    if time.monotonic() - stop_started > 60:
                        child.terminate()
                        child.wait(timeout=30)
                        break
                time.sleep(1)
            if child.returncode:
                raise RuntimeError(f"Geometry worker exited{child.returncode}; preserve attempt")
    result = json.loads((args.out / "Town05_Opt.json").read_text())
    atomic_json(
        args.out / "assessment.json",
        dict(
            geometry_only=True,
            policy_loaded=False,
            runtime_stop_actors=len(result["stops"]),
            runtime_lights=len(result["lights"]),
            inventory_sha256=digest(args.out / "Town05_Opt.json"),
        ),
    )
    print(json.dumps(json.loads((args.out / "assessment.json").read_text()), indent=2))


def freeze(args):
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    if args.out.exists():
        raise FileExistsError("Never overwrite a final suite")
    record = json.loads((args.inventory / "Town05_Opt.json").read_text())
    assessment = json.loads((args.inventory / "assessment.json").read_text())
    inventory_request = json.loads((args.inventory / "request.json").read_text())
    if any(
        digest(args.inventory / "source" / p) != sha
        for p, sha in inventory_request["source_hashes"].items()
    ):
        raise ValueError("Archived inventory source changed")
    if (
        not assessment["geometry_only"]
        or digest(args.inventory / "Town05_Opt.json") != assessment["inventory_sha256"]
    ):
        raise ValueError("Final registration requires verified runtime geometry")
    xodr = Path(args.carla_root).expanduser() / "CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr"
    world_map = carla.Map("Town05_Opt", xodr.read_text())
    import hashlib

    if hashlib.sha256(world_map.to_opendrive().encode()).hexdigest() != record["opendrive_sha256"]:
        raise ValueError("Held-out map changed")
    planner = GlobalRoutePlanner(world_map, 1.0)
    points = sorted(
        (w for w in world_map.generate_waypoints(20) if not w.is_junction),
        key=lambda w: (w.road_id, w.lane_id, w.s),
    )
    rng = np.random.default_rng(20261018)
    mapped_stops = match_explicit_stop_signals(world_map, xodr, record["stops"])
    stop_ids = sorted(s["actor_id"] for s in mapped_stops)
    selected = []
    used = set()
    args.out.mkdir(parents=True)
    sources = [
        Path(__file__).resolve(),
        ROOT / "reports/navigation_m6_registration_protocol_20261018.md",
        ROOT / "configs/navigation.yaml",
        ROOT / "configs/navigation_vendor.json",
        *sorted((ROOT / "src/navigation").glob("*.py")),
    ]
    for path in sources:
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(
        args.out / "registration_request.json",
        dict(
            inventory_sha256=digest(args.inventory / "Town05_Opt.json"),
            registration_source_sha256=digest(Path(__file__)),
            protocol_sha256=digest(
                ROOT / "reports/navigation_m6_registration_protocol_20261018.md"
            ),
            seed=20261018,
            maximum_wall_seconds=900,
            no_model_loaded=True,
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        ),
    )
    deadline = time.monotonic() + 900
    for rid in range(12):
        required = stop_ids[rid // 2] if rid < 10 else None
        for attempt in range(20000):
            if time.monotonic() >= deadline or (args.out / "STOP").exists():
                atomic_json(
                    args.out / "progress.json", dict(status="stopped", routes=len(selected))
                )
                raise InterruptedError("STOP or900s geometry selection cap; no final suite frozen")
            if attempt % 100 == 0:
                atomic_json(
                    args.out / "progress.json",
                    dict(
                        status="running",
                        routes=len(selected),
                        current_required_stop=required,
                        attempt=attempt,
                        updated_at=time.time(),
                    ),
                )
            a, b = map(int, rng.choice(len(points), 2, replace=False))
            endpoints = [
                tuple(round(getattr(points[i].transform.location, k), 5) for k in "xyz")
                for i in (a, b)
            ]
            if tuple(endpoints) in used:
                continue
            trace = planner.trace_route(*[carla.Location(*xyz) for xyz in endpoints])
            if len(trace) < 2:
                continue
            # Length rejection avoids costly per-actor coverage on short routes.
            xyz = np.array(
                [
                    [w.transform.location.x, w.transform.location.y, w.transform.location.z]
                    for w, _ in trace
                ]
            )
            length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
            if not 1000 <= length <= 3000:
                continue
            opportunities, arc = registration_opportunities(trace, record)
            if required is not None and required not in {
                r["actor_id"] for r in opportunities["stop"]
            }:
                continue
            signals = [
                r for r in opportunities["signal"] if r["arc_m"] >= 60 and r["arc_m"] <= length - 30
            ]
            if not signals:
                continue
            signal = signals[0]
            trigger_index = int(np.argmin(np.abs(arc - (signal["arc_m"] - 40))))
            trigger = trace[trigger_index][0]
            if trigger.is_junction:
                continue
            # Require one continuous signal approach, without an intervening junction.
            if any(w.is_junction for w, _ in trace[trigger_index : signal["route_index"]]):
                continue
            used.add(tuple(endpoints))
            selected.append(
                dict(
                    route_id=rid,
                    town="Town05_Opt",
                    endpoints=endpoints,
                    length_m=length,
                    required_stop_actor=required,
                    opportunities=opportunities,
                    held_signal=signal,
                    trigger_xyz=[
                        trigger.transform.location.x,
                        trigger.transform.location.y,
                        trigger.transform.location.z,
                    ],
                    trigger_yaw=trigger.transform.rotation.yaw,
                    selection_attempt=attempt,
                )
            )
            atomic_json(args.out / "selected_geometry.json", dict(routes=selected))
            break
        else:
            raise RuntimeError(f"Cannot satisfy final route{rid}; no suite frozen")
    if sum(len(e["opportunities"]["stop"]) for e in selected) * 3 < 30:
        raise ValueError("Required planned stop coverage was not met")
    root = ET.Element("routes")
    for e in selected:
        node = ET.SubElement(root, "route", id=str(e["route_id"]), town="Town05_Opt")
        ET.SubElement(ET.SubElement(node, "weathers"), "weather", **WEATHER)
        waypoints = ET.SubElement(node, "waypoints")
        for xyz in e["endpoints"]:
            ET.SubElement(waypoints, "position", **dict(zip("xyz", map(str, xyz))))
        scenarios = ET.SubElement(node, "scenarios")
        scenario = ET.SubElement(
            scenarios, "scenario", name=f"FinalHold_{e['route_id']}", type="NavigationSignalHold"
        )
        ET.SubElement(
            scenario,
            "trigger_point",
            **dict(zip("xyz", map(str, e["trigger_xyz"]))),
            yaw=str(e["trigger_yaw"]),
        )
        for k, v in SCENARIO_PARAMETERS["signal_stop"].items():
            ET.SubElement(scenario, k, value=str(v))
    ET.indent(root)
    route_path = args.out / "Town05_Opt.xml"
    ET.ElementTree(root).write(route_path, encoding="utf-8", xml_declaration=True)
    atomic_json(
        route_path.with_suffix(".json"),
        dict(
            purpose="Frozen M6 final custom benchmark",
            town="Town05_Opt",
            route_sha256=digest(route_path),
            lengths_m={str(e["route_id"]): e["length_m"] for e in selected},
            length_m=None,
        ),
    )
    record_out = dict(
        version="town05_final_v1",
        frozen=True,
        role="final_test",
        seed=20261018,
        timing_deviation=(
            "Registered after existing student/component development; "
            "before further fitting or any Town05 student test"
        ),
        traffic_seeds=[3031, 4031, 5031],
        routes=selected,
        inventory_sha256=digest(args.inventory / "Town05_Opt.json"),
        opendrive_sha256=record["opendrive_sha256"],
        explicit_mapped_stop_locations=mapped_stops,
        runtime_stop_actor_count=len(record["stops"]),
        all_runtime_stop_infractions_counted=True,
        route_sha256=digest(route_path),
        registration_source_sha256=digest(Path(__file__)),
        protocol_sha256=digest(ROOT / "reports/navigation_m6_registration_protocol_20261018.md"),
        planned_stop_opportunities=sum(len(e["opportunities"]["stop"]) for e in selected) * 3,
        planned_held_red_opportunities=36,
        actual_encounter_coverage="Must audit at runtime; early failures retained",
        count_all_collisions=True,
        no_model_loaded=True,
        vendor=activate(args.carla_root),
        candidate_registry="Separate immutable checkpoint registrations, never edit this suite",
    )
    atomic_json(args.out / "suite.json", record_out)
    (args.out / "suite.sha256").write_text(digest(args.out / "suite.json") + "  suite.json\n")
    atomic_json(
        args.out / "progress.json", dict(status="complete", routes=12, updated_at=time.time())
    )
    print(
        json.dumps(
            dict(
                suite=str(args.out),
                planned_stop_opportunities=record_out["planned_stop_opportunities"],
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "inventory-worker", "freeze"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--carla-root", default="~/software/CARLA_0.9.15")
    args = parser.parse_args()
    args.out = args.out.resolve()
    activate(args.carla_root)
    if args.command == "inventory-worker":
        inventory_town(2100, "Town05_Opt", args.out)
        return
    if args.command == "freeze":
        freeze(args)
        return
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        inventory(args)


if __name__ == "__main__":
    main()
