#!/usr/bin/env python3
"""Bounded training-town static light-housing/evaluator-line inventory; no driving."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_navigation_signal_pairs import load_training_world  # noqa: E402
from src.navigation.rule_geometry import capture_light_geometry  # noqa: E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def worker(args, config):
    import carla

    client = carla.Client("127.0.0.1", config["runtime"]["port"])
    client.set_timeout(10)
    plan = json.loads(
        (ROOT / "reports/navigation_m2_signal_pairs_plan_v3_20261016.json").read_text()
    )
    world = load_training_world(client, args.town, plan["map_sha256"][args.town])
    started = time.monotonic()
    geometry = {
        "schema": "training-rule-geometry-v1",
        "town": args.town,
        "opendrive_sha256": hashlib.sha256(world.get_map().to_opendrive().encode()).hexdigest(),
        "lights": capture_light_geometry(world),
        "housing_coordinates": "world; get_light_boxes plus identity get_world_vertices",
        "line_semantics": "pinned RunningRedLightTest crossing lines, not painted-line truth",
        "visibility": "unknown: geometry alone does not establish occlusion, facing, or legibility",
        "seconds": time.monotonic() - started,
        "source_reference": "https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Util/BoundingBoxCalculator.cpp",
    }
    if args.include_level_boxes:

        def box_record(box):
            return {
                "center": [box.location.x, box.location.y, box.location.z],
                "extent": [box.extent.x, box.extent.y, box.extent.z],
                "rotation": [box.rotation.pitch, box.rotation.yaw, box.rotation.roll],
                "vertices": [[v.x, v.y, v.z] for v in box.get_world_vertices(carla.Transform())],
            }

        geometry["level_traffic_light_boxes"] = [
            box_record(box) for box in world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        ]
        geometry["environment_traffic_lights"] = [
            {"id": obj.id, "name": obj.name, "box_world": box_record(obj.bounding_box)}
            for obj in world.get_environment_objects(carla.CityObjectLabel.TrafficLight)
        ]
    atomic_json(args.out / "geometry.json", geometry)
    if args.pixel_case_id is not None:
        from src.navigation.signal_label_probe import capture_label_pixels

        if args.pixel_case_id == -1:
            manifest = json.loads((args.mining_root / "manifest.json").read_text())
            cases = sorted(
                {
                    r["sample"]["case_id"]
                    for r in manifest["rows"]
                    if r["sample"]["town"] == args.town
                }
            )
            completed = []
            for case_id in cases:
                if (args.out / "STOP").exists():
                    raise InterruptedError("STOP requested")
                capture_label_pixels(
                    world,
                    config,
                    args.corpus,
                    case_id,
                    args.out / f"case{case_id}",
                    stop=args.out / "STOP",
                )
                completed.append(case_id)
                atomic_json(
                    args.out / "progress.json",
                    {
                        "status": "running",
                        "stage": "M2 semantic label verification",
                        "images_done": 6 * len(completed),
                        "images_total": 6 * len(cases),
                        "phase": "TRAIN-only registered labels",
                        "updated_at": time.time(),
                    },
                )
            atomic_json(
                args.out / "pixel_batch.json",
                {
                    "town": args.town,
                    "cases": completed,
                    "mining_manifest_sha256": hashlib.sha256(
                        (args.mining_root / "manifest.json").read_bytes()
                    ).hexdigest(),
                },
            )
        else:
            capture_label_pixels(world, config, args.corpus, args.pixel_case_id, args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--town", default="Town03_Opt")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/navigation.yaml")
    parser.add_argument("--max-wall", type=int, default=240)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--include-level-boxes", action="store_true")
    parser.add_argument("--pixel-case-id", type=int)
    parser.add_argument(
        "--mining-root",
        type=Path,
        default=ROOT / "outputs/navigation_m2_signal_hard_negatives_20261016",
    )
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "outputs/navigation_m2_signal_corpus_v2_20261016"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.town.removesuffix("_Opt") not in config["data"]["train_towns"]:
        raise ValueError("Only approved training towns; Town05 stays held out")
    if not 0 < args.max_wall <= 300:
        raise ValueError("Inventory is capped at five minutes")
    args.out = args.out.resolve()
    vendor = activate()
    if args.worker:
        worker(args, config)
        return
    if args.out.exists():
        raise FileExistsError("Fresh output required")
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        check_resources(args.out, config["runtime"]["minimum_free_gb"])
        hashes = {}
        for path in [
            Path(__file__).resolve(),
            ROOT / "src/navigation/rule_geometry.py",
            ROOT / "src/navigation/runtime.py",
            ROOT / "src/navigation/signal_data.py",
            ROOT / "scripts/collect_navigation_signal_pairs.py",
            ROOT / "src/navigation/signal_label_probe.py",
            ROOT / "scripts/probe_navigation_signal_pixels.py",
            ROOT / "reports/navigation_m2_signal_pairs_plan_v3_20261016.json",
            ROOT / "configs/navigation_vendor.json",
            args.config.resolve(),
        ]:
            relative = path.relative_to(ROOT)
            destination = args.out / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        atomic_json(
            args.out / "request.json",
            {
                "town": args.town,
                "include_level_boxes": args.include_level_boxes,
                "pixel_case_id": args.pixel_case_id,
                "max_wall": args.max_wall,
                "vendor": vendor,
                "source_hashes": hashes,
                "kind": "static training geometry; no policy or benchmark evaluation",
            },
        )
        atomic_json(
            args.out / "progress.json",
            {"status": "running", "stage": "M2 rule geometry", "updated_at": time.time()},
        )
        started, child = time.monotonic(), None
        status = "failed"
        try:
            with CarlaServer(
                vendor["carla_root"], args.out, config["runtime"]["port"], town=args.town
            ):
                with (args.out / "worker.log").open("x") as log:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--out",
                            str(args.out),
                            "--town",
                            args.town,
                            "--config",
                            str(args.config),
                            *(["--include-level-boxes"] if args.include_level_boxes else []),
                            *(
                                [
                                    "--pixel-case-id",
                                    str(args.pixel_case_id),
                                    "--corpus",
                                    str(args.corpus.resolve()),
                                ]
                                if args.pixel_case_id is not None
                                else []
                            ),
                            *(
                                ["--mining-root", str(args.mining_root.resolve())]
                                if args.pixel_case_id == -1
                                else []
                            ),
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    stopping = None
                    while child.poll() is None:
                        if time.monotonic() - started >= args.max_wall:
                            (args.out / "STOP").touch()
                        if (args.out / "STOP").exists():
                            stopping = stopping or time.monotonic()
                            if time.monotonic() - stopping >= 15:
                                child.terminate()  # owned child after STOP grace, never force kill
                                child.wait(timeout=20)
                                break
                        time.sleep(1)
                status = (
                    "stopped"
                    if (args.out / "STOP").exists()
                    else "complete"
                    if child.returncode == 0
                    else "failed"
                )
        finally:
            atomic_json(
                args.out / "progress.json",
                {
                    "status": status,
                    "stage": "M2 rule geometry",
                    "seconds": time.monotonic() - started,
                    "updated_at": time.time(),
                },
            )
        if status != "complete":
            raise RuntimeError(f"Geometry capture {status}; see {args.out}")
        geometry = json.loads((args.out / "geometry.json").read_text())
        print(
            json.dumps(
                {
                    "status": status,
                    "lights": len(geometry["lights"]),
                    "boxes": sum(len(r["boxes_world"]) for r in geometry["lights"]),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
