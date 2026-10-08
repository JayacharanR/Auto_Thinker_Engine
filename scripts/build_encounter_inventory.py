#!/usr/bin/env python3
"""Bounded per-town runtime inventory of rule applicability and junction movements.

Starts one owned server per training town in turn, loads the map without a Traffic
Manager, and records pinned-evaluator light/stop applicability plus planner-validated
movement candidates. It never drives, never touches Town05 and freezes nothing.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def inventory_town(port, town, output):
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    from src.navigation.encounters import (
        candidate_valid,
        light_applicability,
        movements,
        stop_applicability,
        summarize,
    )

    client = carla.Client("127.0.0.1", port)
    client.set_timeout(120)  # Town10HD_Opt loading exceeded the 20s startup timeout
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != town:
        world = client.load_world(town)  # no Traffic Manager exists in this process
    world.wait_for_tick()
    carla_map = world.get_map()
    actual = carla_map.name.split("/")[-1]
    if actual != town:
        raise RuntimeError(f"Server map {actual} differs from requested {town}")
    started = time.monotonic()
    lights = light_applicability(world, carla_map)
    stops = stop_applicability(world, carla_map, carla_map.generate_waypoints(0.5))
    planner = GlobalRoutePlanner(carla_map, 1.0)  # the harness interpolation step
    found = movements(carla_map, planner, lights, stops)
    for m in found:
        m["valid"] = candidate_valid(m)
    record = {
        "map": actual,
        "opendrive_sha256": hashlib.sha256(carla_map.to_opendrive().encode()).hexdigest(),
        "lights": lights,
        "stops": stops,
        "movements": found,
        "summary": summarize(found),
        "seconds": round(time.monotonic() - started, 1),
    }
    atomic_json(output / f"{town}.json", record)
    print(json.dumps(record["summary"]), flush=True)


def run_child(args, output, town, deadline_s=600):
    """A native client abort stays in the child; the parent still owns server cleanup."""
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", town]
    command += ["--config", args.config, "--carla-root", args.carla_root, "--out", str(output)]
    with (output / town / "worker.log").open("a") as log:
        worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        deadline, stop_started = time.monotonic() + deadline_s, None
        while worker.poll() is None:
            if time.monotonic() >= deadline:
                (output / "STOP").touch()
            if (output / "STOP").exists():
                stop_started = stop_started or time.monotonic()
                if time.monotonic() - stop_started > 60:
                    worker.terminate()  # owned child, after STOP grace period
                    worker.wait(timeout=30)
                    break
            time.sleep(1)
    if worker.returncode:
        raise RuntimeError(f"Inventory child exited {worker.returncode}; see {town}/worker.log")
    return json.loads((output / f"{town}.json").read_text())["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/navigation.yaml")
    parser.add_argument(
        "--carla-root", default=os.environ.get("CARLA_ROOT", "~/software/CARLA_0.9.15")
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--towns", nargs="*", help="Default: all configured training towns")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    vendor = activate(args.carla_root)
    config = yaml.safe_load(Path(args.config).read_text())
    if args.worker:
        if args.worker.removesuffix("_Opt") not in config["data"]["train_towns"]:
            raise ValueError("Inventory is restricted to training towns")
        inventory_town(config["runtime"]["port"], args.worker, Path(args.out))
        return
    towns = args.towns or config["data"]["train_towns"]
    if any(t.removesuffix("_Opt") not in config["data"]["train_towns"] for t in towns):
        raise ValueError("Inventory is restricted to training towns; Town05 stays held out")
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory; attempts must not be overwritten")
    runtime = config["runtime"]
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        check_resources(output, runtime["minimum_free_gb"])
        source = output / "source"
        source.mkdir(parents=True)
        for path in (ROOT / "src/navigation/encounters.py", Path(__file__).resolve()):
            shutil.copyfile(path, source / path.name)
        progress = {"vendor": vendor, "towns": {}, "status": "running"}
        atomic_json(output / "progress.json", progress)
        for town in towns:
            town = f"{town.removesuffix('_Opt')}_Opt"
            if (output / "STOP").exists():
                progress["status"] = "stopped"
                break
            try:
                with CarlaServer(args.carla_root, output / town, runtime["port"], town=town):
                    progress["towns"][town] = run_child(args, output, town)
            except Exception as exc:  # preserve and continue with the next town
                progress["towns"][town] = {"error": f"{type(exc).__name__}: {exc}"}
            atomic_json(output / "progress.json", progress)
            print(town, json.dumps(progress["towns"][town]), flush=True)
        else:
            progress["status"] = "complete"
        atomic_json(output / "progress.json", progress)


if __name__ == "__main__":
    main()
