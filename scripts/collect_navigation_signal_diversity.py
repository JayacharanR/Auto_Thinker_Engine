#!/usr/bin/env python3
"""One isolated child per frozen TRAINtown/illumination, no failed-case retry."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.native_process_owner import run_native_worker  # noqa:E402
from src.navigation.runtime import (  # noqa:E402
    RunLock,
    atomic_json,
    check_resources,
    resource_shortfall,
)
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def verify_weather(actual, expected, receipt):
    """Keep the exact native mismatch before failing, without retry or tolerance change."""
    observed = {key: float(getattr(actual, key)) for key in expected}
    deltas = {key: observed[key] - value for key, value in expected.items()}
    matches = all(abs(value) <= 0.001 for value in deltas.values())
    atomic_json(receipt, dict(expected=expected, observed=observed, deltas=deltas,
                              tolerance=0.001, matches=matches))
    if not matches:
        raise ValueError("Native weather does not match frozen illumination")


def worker(args):
    import hashlib

    import numpy as np
    import yaml
    from scipy.spatial import cKDTree

    from scripts.probe_navigation_signal_pixels import render_cases
    from src.navigation.native_process_client import connect_owned_engine
    from src.navigation.rule_geometry import capture_light_geometry

    assignment = read(args.task_file)
    request = read(assignment["request"])
    if digest(assignment["request"]) != assignment["request_sha256"]:
        raise ValueError("Frozen parent request changed")
    for file, sha in request["source_hashes"].items():
        if digest(ROOT / file) != sha:
            raise ValueError("Registered source changed before capture")
    plan = read(request["plan"])
    task = next(t for t in plan["tasks"] if t["task_id"] == assignment["task_id"])
    server, client, check = connect_owned_engine(
        args.engine_identity, args.out, maximum_seconds=300
    )
    import carla

    config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != task["town"]:
        world = client.load_world(task["town"])
    check()
    world_map = world.get_map()
    if (
        world_map.name.split("/")[-1] != task["town"]
        or hashlib.sha256(world_map.to_opendrive().encode()).hexdigest()
        != plan["map_sha256"][task["town"]]
    ):
        raise ValueError("Actual TRAINmap mismatch")
    allowed = set(plan["allowed_junctions"][task["town"]])
    forbidden = [
        w.transform.location
        for w in world_map.generate_waypoints(0.5)
        if w.is_junction and w.junction_id not in allowed
    ]
    tree = cKDTree([[p.x, p.y, p.z] for p in forbidden])
    geometry = dict(town=task["town"], lights=capture_light_geometry(world))
    capture = args.out / "capture"
    capture.mkdir()
    (capture / "STOP").symlink_to("../STOP")
    atomic_json(capture / "geometry.json", geometry)
    cases, clearances = [], []
    for source in task["cases"]:
        check()
        case = dict(source)
        matches = [
            light
            for light in geometry["lights"]
            if np.linalg.norm(np.asarray(light["pose"][:3]) - case["inventory_light_xyz"]) < 0.002
        ]
        if len(matches) != 1:
            raise ValueError("No unique physical signal correspondence")
        lamp = matches[0]
        if not any(
            line["lane"] == case["evaluator_stop_lane"]["lane"]
            and np.linalg.norm(np.asarray(line["center"]) - case["evaluator_stop_lane"]["xyz"])
            < 0.002
            for line in lamp["evaluator_stop_lines"]
        ):
            raise ValueError("Original evaluator stop-line identity changed")
        waypoint = world_map.get_waypoint(carla.Location(*case["ego_pose"][:3]))
        clearance = float(tree.query(case["ego_pose"][:3])[0])
        if clearance < 1 or (waypoint.is_junction and waypoint.junction_id not in allowed):
            raise ValueError("Live static pose entered reserved geography")
        clearances.append(
            dict(
                case_id=case["route_id"],
                clearance_m=clearance,
                actual_junction=waypoint.junction_id if waypoint.is_junction else None,
            )
        )
        case["light"] = dict(
            type="traffic.traffic_light", id=lamp["capture_actor_id"], pose=lamp["pose"]
        )
        cases.append(case)
    atomic_json(capture / "cases.json", cases)
    weather = carla.WeatherParameters(**task["weather"])
    world.set_weather(weather)
    actual = world.get_weather()
    verify_weather(actual, task["weather"], capture / "weather_check.json")
    atomic_json(
        capture / "setup.json",
        dict(
            task_id=task["task_id"],
            illumination=task["illumination"],
            weather={key: getattr(actual, key) for key in task["weather"]},
            actual_map_sha256=plan["map_sha256"][task["town"]],
            geometry_clearances=clearances,
            no_dynamic_traffic=True,
            stationary_training_capture_only=True,
        ),
    )
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    world.apply_settings(settings)
    check()
    # Existing exact-frame barrier and atomic per-case receipts; no source changes.
    render_cases(argparse.Namespace(out=capture, geometry=capture), config, world, geometry)
    check()
    atomic_json(
        args.out / "worker_result.json",
        dict(
            complete=True,
            captured_cases=len(cases),
            capture_assessment_sha256=digest(capture / "assessment.json"),
            source_request_sha256=assignment["request_sha256"],
            driving_evidence=False,
        ),
    )
    # Native handles end with this child; parent owns normalSTOPengine cleanup.
    del world_map, world, client, server


def main(args):
    import yaml

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT / "outputs/navigation_signal_diversity_geometry_20261021/plan.json"
    plan = read(plan_path)
    summary = read(plan_path.parent / "geometry_summary.json")
    if digest(plan_path) != summary["plan_sha256"] or plan["planned_cases"] != 200:
        raise ValueError("Exact200case frozen geometry required")
    for file, expected in (
        (plan["catalog"], plan["catalog_sha256"]),
        (plan["guard"], plan["guard_sha256"]),
        (plan["eligibility_request"], plan["eligibility_request_sha256"]),
    ):
        if digest(file) != expected:
            raise ValueError("CurrentTRAINsource/roles changed")
    paths = [
        Path(__file__),
        plan_path,
        ROOT / "reports/navigation_signal_diversity_protocol_20261021.md",
        ROOT / "configs/navigation.yaml",
        ROOT / "configs/navigation_vendor.json",
        *sorted((ROOT / "src/navigation").glob("*.py")),
        ROOT / "scripts/probe_navigation_signal_pixels.py",
        ROOT / "scripts/audit_navigation_rule_observability.py",
        ROOT / "scripts/audit_navigation_dense_labels.py",
        ROOT / "scripts/navigation_label_rig.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for file in paths:
        target = out / "source" / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    request = dict(
        source_hashes=hashes,
        plan=str(plan_path),
        plan_sha256=digest(plan_path),
        planned_cases=200,
        tasks=8,
        native_cap_seconds=2400,
        child_cap_seconds=300,
        artifact_cap_bytes=1_000_000_000,
        all_failures_retained=True,
        no_retry=True,
        no_automatic_training=True,
        native_stability_unqualified=True,
    )
    atomic_json(out / "request.json", request)
    started, status, error, active = time.monotonic(), "running", None, None
    records = []
    (out / "tasks").mkdir()
    (out / "native").mkdir()
    last_bytes = 0.0

    def check():
        nonlocal last_bytes
        if (out / "STOP").exists() or time.monotonic() - started > 2400:
            raise InterruptedError("Ownedstage STOPor2400second capture limit")
        if time.monotonic() - last_bytes > 2:
            last_bytes = time.monotonic()
            if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 1_000_000_000:
                raise InterruptedError("OneGB stage artifact limit")

    def progress(phase):
        captured = sum(r["closed_case_receipts"] for r in records)
        active_cases = len(list((active / "capture/case_metadata").glob("*.json"))) if active else 0
        settled = sum(r["planned_cases"] for r in records)
        atomic_json(
            out / "progress.json",
            dict(
                stage="Strict TRAINsignal diversity",
                status=status,
                error=error,
                frames=settled + active_cases,
                target_frames=200,
                settled_cases=settled,
                captured_case_receipts=captured + active_cases,
                tasks_done=len(records),
                tasks_total=8,
                active_output=str(active) if active else None,
                phase=phase,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        with BudgetWatch(out), RunLock(ROOT / "outputs/navigation/.lock"):
            progress("Frozen200cases; native capture only")
            for task in plan["tasks"]:
                check()
                runtime = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())["runtime"]
                check_resources(out, runtime["minimum_free_gb"])
                shortfall = resource_shortfall(runtime)
                if shortfall:
                    raise RuntimeError(shortfall)
                task_path = out / "tasks" / f"{task['task_id']}.json"
                atomic_json(
                    task_path,
                    dict(
                        task_id=task["task_id"],
                        request=str(out / "request.json"),
                        request_sha256=digest(out / "request.json"),
                    ),
                )
                active = out / "native" / task["task_id"]
                result = run_native_worker(
                    Path("~/software/CARLA_0.9.15").expanduser(),
                    active,
                    task["town"],
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--task-file",
                        str(task_path),
                    ],
                    check,
                    lambda state: progress(
                        f"{task['task_id']} child alive; {state.get('stop_requested')}"
                    ),
                    worker_wall_seconds=300,
                )
                receipts = list((active / "capture/case_metadata").glob("*.json"))
                records.append(
                    dict(
                        task_id=task["task_id"],
                        planned_cases=len(task["cases"]),
                        closed_case_receipts=len(receipts),
                        output=str(active),
                        **result,
                    )
                )
                active = None
                atomic_json(out / "native_results.json", records)
                progress("Task settled; capture receipts need independent audit")
            failed = any(
                r["child_returncode"] != 0 or r["closed_case_receipts"] != r["planned_cases"]
                for r in records
            )
            status = "failed" if failed else "complete"
            atomic_json(
                out / "assessment.json",
                dict(
                    attempts_settled=True,
                    all_workers_successful=not failed,
                    original_cases=200,
                    closed_case_receipts=sum(r["closed_case_receipts"] for r in records),
                    abnormal_native_exits=sum(
                        r["native_returncode"] not in (0, -15) for r in records
                    ),
                    independently_verified=False,
                    native_stability=False,
                    records=records,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress("Collectionterminal; partial/failureevidence retained; auditbeforeanyfit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--engine-identity", type=Path)
    args = parser.parse_args()
    worker(args) if args.worker else main(args)
