#!/usr/bin/env python3
"""Red/green/red stationary renders at audited TRAIN poses; no driving or score."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_rule_observability import (  # noqa: E402
    read,
    representative,
    validate_training_data,
    verify_sources,
)
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.rule_geometry import (  # noqa: E402
    camera_matrix,
    ego_pose_from_facts,
    match_light,
    project_rule_sample,
)
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.sensors import CameraRig, calibrations  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def select_cases(audit_root, config):
    audit = read(audit_root / "assessment.json")
    if (
        not audit["audit_complete"]
        or digest(audit_root / "samples.jsonl") != audit["provenance"]["samples_sha256"]
    ):
        raise ValueError("Completed, unchanged training geometry audit required")
    verify_sources(audit_root, audit["provenance"])
    groups = defaultdict(list)
    for row in map(json.loads, (audit_root / "samples.jsonl").read_text().splitlines()):
        if row["signal"] is not None:
            groups[row["route_id"]].append(row)
    assessment_path = ROOT / "reports/navigation_m2_dagger_assessment_20261016.json"
    if digest(assessment_path) != audit["provenance"]["assessment_sha256"]:
        raise ValueError("Original correction assessment changed")
    assessment = read(assessment_path)
    cases = []
    for rid, rows in sorted(groups.items()):
        row = representative(rows)
        data = Path(row["shard"]).parent
        meta = validate_training_data(data, assessment)
        with zipfile.ZipFile(row["shard"]) as archive:
            targets = json.loads(archive.read(f"{row['record']['prefix']}/targets.json"))
            policy = json.loads(archive.read(f"{row['record']['prefix']}/policy.json"))
        if meta["town"] != "Town03_Opt" or targets["frame"] != row["frame"]:
            raise ValueError("Only the audited Town03 internal-TRAIN cases allowed")
        expected = json.loads(json.dumps({k: asdict(v) for k, v in calibrations(config).items()}))
        if policy["calibration"] != expected:
            raise ValueError("Recorded/configured camera rig differs")
        facts = targets["facts"]
        actor = next(a for a in facts["world"]["actors"] if a["id"] == facts["applicable_light_id"])
        hero = next(a for a in facts["world"]["actors"] if a["role"] == "hero")
        cases.append(
            {
                "route_id": rid,
                "original_frame": row["frame"],
                "town": meta["town"],
                "ego_pose": ego_pose_from_facts(facts),
                "ego_blueprint": hero["type"],
                "light": actor,
                "source_shard": row["shard"],
                "source_record": row["record"],
                "recorded_distance_m": row["next_light_distance_m"],
            }
        )
    if not cases:
        raise ValueError("No applicable-light TRAIN cases")
    return cases


def contrast(first, second, repeat, bbox):
    height, width = first.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
    x1, y1 = min(width, math.ceil(x1)), min(height, math.ceil(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    a, b, c = (im[y0:y1, x0:x1].astype(float) for im in (first, second, repeat))
    difference, baseline = np.abs(a - b), np.abs(a - c)
    return {
        "crop_xyxy": [x0, y0, x1, y1],
        "crop_size": [x1 - x0, y1 - y0],
        "red_green_mae": float(difference.mean()),
        "red_repeat_mae": float(baseline.mean()),
        "red_green_max_channel": float(difference.max()),
        "red_repeat_max_channel": float(baseline.max()),
        "red_green_pixels_change_gt10": int((difference.max(axis=2) > 10).sum()),
        "red_repeat_pixels_change_gt10": int((baseline.max(axis=2) > 10).sum()),
        "green_minus_red_mean_rgb": (b - a).mean(axis=(0, 1)).tolist(),
    }


def make_crop_sheet(rows, out):
    selected = [r for r in rows if r["contrast"] is not None]
    canvas = Image.new("RGB", (700, 110 * len(selected) + 40), "#101820")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 10),
        "STATIC TRAIN PIXEL PROBE | RED / GREEN / RED repeat | original scene traffic omitted",
        fill="white",
    )
    for i, row in enumerate(selected):
        result = row["contrast"]
        y = 40 + 110 * i
        label = (
            f"route {row['route_id']} {row['view']} head {row['head']} "
            f"{result['crop_size']}px RG={result['red_green_mae']:.2f} "
            f"RR={result['red_repeat_mae']:.2f}"
        )
        draw.text((10, y), label, fill="white")
        for col, state in enumerate(("red", "green", "red_repeat")):
            with Image.open(out / f"route{row['route_id']}_{state}_{row['view']}.png") as im:
                crop = im.crop(result["crop_xyxy"])
                crop.thumbnail((150, 80))
                size = (
                    max(1, int(crop.width * min(150 / crop.width, 75 / crop.height))),
                    max(1, int(crop.height * min(150 / crop.width, 75 / crop.height))),
                )
                crop = crop.resize(size, Image.Resampling.NEAREST)
                canvas.paste(crop, (10 + col * 220, y + 25))
    canvas.save(out / "signal_crop_pairs.png")


def capture_static_frame(world, rig, stop, settle_ticks):
    """Advance a frozen scene; drain the exact requested frame after GPU latency.

    This probe has no moving actors or policy. Two further ticks let the GPU
    produce the requested image without waiting for a current-frame callback
    that may require a subsequent render tick. No latest-frame substitution.
    """
    if settle_ticks < 1 or settle_ticks + 2 >= 64:
        raise ValueError("Static capture must fit the camera queue capacity")
    frame = None
    for _ in range(settle_ticks):
        if stop.exists():
            raise InterruptedError("STOP requested")
        frame = world.tick(10)
    for _ in range(2):
        if stop.exists():
            raise InterruptedError("STOP requested")
        world.tick(10)
    return frame, rig.barrier.collect(frame, timeout=10)


def worker(args, config):
    import carla

    client = carla.Client("127.0.0.1", config["runtime"]["port"])
    client.set_timeout(10)
    world = client.get_world()
    geometry = read(args.geometry / "geometry.json")
    if world.get_map().name.split("/")[-1] != "Town03_Opt":
        raise ValueError("Unexpected map")
    if (
        hashlib.sha256(world.get_map().to_opendrive().encode()).hexdigest()
        != geometry["opendrive_sha256"]
    ):
        raise ValueError("Training map changed")
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    world.apply_settings(settings)
    try:
        render_cases(args, config, world, geometry)
    finally:
        # Restore ticking even on setup failure, before the owner's graceful exit.
        world.apply_settings(original)


def render_cases(args, config, world, geometry):
    import carla

    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    for light in lights:
        light.freeze(True)
    actors = [
        {
            "actor": light,
            "type": light.type_id,
            "pose": [
                light.get_location().x,
                light.get_location().y,
                light.get_location().z,
                light.get_transform().rotation.pitch,
                light.get_transform().rotation.yaw,
                light.get_transform().rotation.roll,
            ],
        }
        for light in lights
    ]
    rows, case_results = [], []
    for case in read(args.out / "cases.json"):
        if (args.out / "STOP").exists():
            raise InterruptedError("STOP requested")
        identity = match_light(case["light"], geometry)
        matching = [
            a["actor"]
            for a in actors
            if match_light(a, geometry)["capture_actor_id"] == identity["capture_actor_id"]
        ]
        if len(matching) != 1:
            raise ValueError("Static render lamp identity is ambiguous")
        light = matching[0]
        background = []
        if "background_seed" in case:
            generator = np.random.default_rng(case["background_seed"])
            for actor in sorted(actors, key=lambda a: tuple(a["pose"])):
                state = int(generator.integers(3))
                actor["actor"].set_state(
                    [
                        carla.TrafficLightState.Red,
                        carla.TrafficLightState.Green,
                        carla.TrafficLightState.Yellow,
                    ][state]
                )
                background.append(
                    {"actor_pose": actor["pose"], "state": ["Red", "Green", "Yellow"][state]}
                )
        original_state = light.get_state()
        p = case["ego_pose"]
        pose = carla.Transform(
            carla.Location(*p[:3]), carla.Rotation(pitch=p[3], yaw=p[4], roll=p[5])
        )
        bp = world.get_blueprint_library().find(case["ego_blueprint"])
        bp.set_attribute("role_name", "hero")
        # Settled physical poses can fail the spawn overlap test against asphalt.
        # Spawn above it, disable physics, then place at the exact recorded pose.
        spawn_pose = carla.Transform(carla.Location(p[0], p[1], p[2] + 3), pose.rotation)
        hero = world.spawn_actor(bp, spawn_pose)
        rig = None
        try:
            hero.set_simulate_physics(False)
            hero.set_transform(pose)
            rig = CameraRig(world, hero, config)

            capture_static_frame(world, rig, args.out / "STOP", 20)
            calibration_errors = {}
            for sensor, view in zip(rig.sensors, rig.calibration):
                actual = np.asarray(sensor.get_transform().get_matrix())
                calibration_errors[view] = float(
                    np.max(np.abs(actual - camera_matrix(p, rig.calibration[view])))
                )
            if max(calibration_errors.values()) > 1e-3:
                raise ValueError("Actual attached camera transform disagrees with projection")
            images, states, frames, phase_lights = {}, [], [], []
            sequence = [
                ("red", carla.TrafficLightState.Red),
                ("green", carla.TrafficLightState.Green),
            ]
            if case.get("include_yellow"):
                sequence.append(("yellow", carla.TrafficLightState.Yellow))
            sequence.append(("red_repeat", carla.TrafficLightState.Red))
            for label, state in sequence:
                light.set_state(state)
                frame, packet = capture_static_frame(world, rig, args.out / "STOP", 4)
                states.append(str(light.get_state()))
                frames.append(frame)
                phase_lights.append(
                    {
                        "phase": label,
                        "light_states": {str(lamp.id): str(lamp.get_state()) for lamp in lights},
                    }
                )
                images[label] = {view: packet[view][2] for view in rig.calibration}
                for view, rgb in images[label].items():
                    Image.fromarray(rgb).save(
                        args.out / f"route{case['route_id']}_{label}_{view}.png"
                    )
            if states != [str(state) for _, state in sequence]:
                raise ValueError("Requested signal intervention did not occur")
            if any(
                len({phase["light_states"][str(lamp.id)] for phase in phase_lights}) != 1
                for lamp in lights
                if lamp.id != light.id
            ):
                raise ValueError("A non-target lamp changed during the phase intervention")
            actual_pose = hero.get_transform()
            drift = np.linalg.norm(
                np.array([actual_pose.location.x, actual_pose.location.y, actual_pose.location.z])
                - p[:3]
            )
            if drift > 1e-3:
                raise ValueError("Stationary ego pose drifted")
            facts = {
                "world": {
                    "frame": frames[-1],
                    "actors": [
                        {"id": hero.id, "role": "hero", "type": hero.type_id, "pose": p},
                        {
                            "id": light.id,
                            "role": "",
                            "type": "traffic.traffic_light",
                            "pose": case["light"]["pose"],
                            "signal": "Red",
                        },
                    ],
                },
                "ego_pose_world": [*p[:3], math.radians(p[4])],
                "applicable_light_id": light.id,
                "next_light_state": "Red",
            }
            projected = project_rule_sample(facts, rig.calibration, geometry)
            for view, data in projected["views"].items():
                for head, box in enumerate(data["boxes"]):
                    if box["in_image"]:
                        rows.append(
                            {
                                "route_id": case["route_id"],
                                "view": view,
                                "head": head,
                                "projection": box,
                                "contrast": contrast(
                                    images["red"][view],
                                    images["green"][view],
                                    images["red_repeat"][view],
                                    box["bbox_xyxy"],
                                ),
                            }
                        )
            case_results.append(
                {
                    "route_id": case["route_id"],
                    "frames": frames,
                    "extra_render_ticks_after_capture_frame": 2,
                    "states": states,
                    "ego_position_drift_m": float(drift),
                    "camera_matrix_max_error": calibration_errors,
                    "all_lights_frozen": all(lamp.is_frozen() for lamp in lights),
                    "background_lights": background,
                    "phase_lights": phase_lights,
                }
            )
            atomic_json(
                args.out / "case_metadata" / f"{case['route_id']}.json",
                {
                    "case": case,
                    "setup": case_results[-1],
                    "housing_pixels": [row for row in rows if row["route_id"] == case["route_id"]],
                    "image_sha256": {
                        path.name: digest(path)
                        for path in sorted(args.out.glob(f"route{case['route_id']}_*.png"))
                    },
                },
            )
        finally:
            light.set_state(original_state)
            if rig:
                rig.close()
            hero.destroy()
    make_crop_sheet(rows, args.out)
    atomic_json(
        args.out / "assessment.json",
        {
            "probe_complete": True,
            "scope": "Static TRAIN renders; original traffic omitted, no driving/policy inference",
            "cases": case_results,
            "housing_pixels": rows,
            "limitations": [
                "Pixel response does not certify original episode visibility or legibility",
                "Background temporal rendering/exposure variation measured by red-repeat control",
                "No new state label is approved by this probe",
            ],
            "source_cases_sha256": digest(args.out / "cases.json"),
            "geometry_sha256": digest(args.geometry / "geometry.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
    vendor = activate()
    if args.worker:
        worker(args, config)
        return
    cases = select_cases(args.audit, config)
    if args.out.exists():
        raise FileExistsError("Fresh output required")
    if (
        digest(args.geometry / "geometry.json")
        != read(args.audit / "assessment.json")["provenance"]["geometry_sha256"]
    ):
        raise ValueError("Audited geometry changed")
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        check_resources(args.out, config["runtime"]["minimum_free_gb"])
        hashes = {}
        for source in [
            *sorted((ROOT / "src/navigation").glob("*.py")),
            Path(__file__).resolve(),
            ROOT / "scripts/audit_navigation_rule_observability.py",
            ROOT / "configs/navigation.yaml",
        ]:
            target = args.out / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[str(source.relative_to(ROOT))] = digest(source)
        atomic_json(args.out / "cases.json", cases)
        atomic_json(
            args.out / "request.json",
            {
                "kind": "Static paired TRAIN pixel audit, no model inference/training",
                "max_wall": 240,
                "audit_sha256": digest(args.audit / "assessment.json"),
                "source_hashes": hashes,
            },
        )
        atomic_json(
            args.out / "progress.json",
            {
                "status": "running",
                "stage": "M2 static signal pixel probe",
                "updated_at": time.time(),
            },
        )
        started, status = time.monotonic(), "failed"
        try:
            with CarlaServer(
                vendor["carla_root"], args.out, config["runtime"]["port"], town="Town03_Opt"
            ):
                with (args.out / "worker.log").open("x") as log:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--audit",
                            str(args.audit),
                            "--geometry",
                            str(args.geometry),
                            "--out",
                            str(args.out),
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    stopping = None
                    while child.poll() is None:
                        if time.monotonic() - started >= 240:
                            (args.out / "STOP").touch()
                        if (args.out / "STOP").exists():
                            stopping = stopping or time.monotonic()
                            if time.monotonic() - stopping > 15:
                                child.terminate()  # owned worker after STOP grace, never force kill
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
        except BaseException as exc:
            status = "failed"
            atomic_json(
                args.out / "failure.json",
                {"error": f"{type(exc).__name__}: {exc}", "phase": "owned probe/server lifecycle"},
            )
            raise
        finally:
            atomic_json(
                args.out / "progress.json",
                {
                    "status": status,
                    "stage": "M2 static signal pixel probe",
                    "updated_at": time.time(),
                    "seconds": time.monotonic() - started,
                },
            )
        if status != "complete":
            raise RuntimeError(f"Pixel probe {status}; see {args.out}/worker.log")
        print(
            json.dumps(
                {"status": status, "cases": len(cases), "seconds": time.monotonic() - started}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
