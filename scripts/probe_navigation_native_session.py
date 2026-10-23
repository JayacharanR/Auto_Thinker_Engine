#!/usr/bin/env python3
"""Two stationary camera lifecycle checks; all observations excluded from fitting."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_dense_labels import actual_geography  # noqa:E402
from scripts.collect_navigation_branches import selected_entry  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.label_collection import validate_label_guard  # noqa:E402
from src.navigation.native_cleanup import cleanup_owned_collection  # noqa:E402
from src.navigation.native_session import owned_native_session  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.replay import capture_world_state  # noqa:E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.sensors import CameraRig  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def exercise(server, client, out, entry, config, check):
    """All CARLA handles stay in this frame; return JSON-serializable values only."""
    import carla

    world = client.load_world(entry["town"], reset_settings=True)
    if world.get_map().name.rsplit("/", 1)[-1] != entry["town"]:
        raise ValueError("Wrong town for native lifecycle instrument")
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 5
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    actors, rig, recording = [], None, False
    try:
        bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
        bp.set_attribute("role_name", "hero")
        pose = world.get_map().get_waypoint(carla.Location(*entry["start_xyz"])).transform
        pose.location.z += 0.5
        ego = world.try_spawn_actor(bp, pose)
        if ego is None:
            raise ValueError("Original TRAINinstrument spawn unavailable; no replacement")
        actors.append(ego)
        ego.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(20):
            check()
            world.tick()
        rig = CameraRig(world, ego, config)
        for _ in range(10):
            check()
            world.tick()
        client.start_recorder(str(out / "native.log"), True)
        recording = True
        frames, poses, pixels = [], [], []
        for _ in range(20):
            check()
            frame = world.tick()
            values = rig.barrier.collect(frame)
            if set(values) != {"front", "left", "right"}:
                raise ValueError("Three same-frame cameras required")
            frames.append(frame)
            poses.append(capture_world_state(world))
            pixels.append(
                {view: hashlib.sha256(v[2].tobytes()).hexdigest() for view, v in values.items()}
            )
        client.stop_recorder()
        recording = False
        world.tick()
        native = recorder_contacts(out / "native.log")
        if not native["complete"]:
            raise ValueError("Closed instrument recorder incomplete")
        atomic_json(out / "world_trace.json", poses)
        atomic_json(out / "pixel_sha256.json", pixels)
        result = dict(
            frames=frames,
            pixel_frames=len(pixels),
            contacts=native,
            instrument_only_excluded_from_fitting=True,
        )
        atomic_json(out / "capture.json", result)
        return result
    except BaseException as error:
        atomic_json(out / "primary_failure.json", dict(error=str(error), at=time.time()))
        raise
    finally:
        cleanup_owned_collection(server, client, world, rig, actors, out, recording)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    entry, _ = selected_entry()
    plan_path = ROOT / "outputs/navigation_dense_label_collection_20261020/plan.json"
    plan = read(plan_path)
    validate_label_guard(plan_path, plan["catalog"], entry["town"], "6")
    paths = [
        Path(__file__),
        plan_path,
        ROOT / "configs/navigation.yaml",
        ROOT / "reports/navigation_native_session_protocol_20261020.md",
        *[
            ROOT / "src/navigation" / name
            for name in (
                "native_session.py",
                "native_cleanup.py",
                "runtime.py",
                "sensors.py",
                "replay.py",
                "contracts.py",
            )
        ],
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for p in paths:
        dst = args.out / "source" / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    atomic_json(
        args.out / "request.json",
        dict(
            source_hashes=hashes,
            cycles=2,
            measured_ticks_per_cycle=20,
            maximum_wall_seconds=300,
            maximum_artifact_bytes=20_000_000,
            no_model_or_training_data=True,
            release_client_before_server=True,
            entry=entry,
        ),
    )
    started = time.monotonic()
    status = "running"
    records = []

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 300:
            raise InterruptedError("Owned instrument STOPor300scap")
        if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > 20_000_000:
            raise InterruptedError("Instrument20MBcap")

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Native callback lifecycle instrument",
                frames=len(records),
                target_frames=2,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 10)
            vendor = activate()
            config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
            progress("Two scoped braked-camera cycles; no dataset or driving-policy comparison")
            for index in range(2):
                check()
                out = args.out / f"cycle_{index}"
                out.mkdir()
                (out / "STOP").symlink_to("../STOP")
                server = CarlaServer(vendor["carla_root"], out, town=entry["town"])
                with owned_native_session(server) as (owner, client):
                    result = exercise(owner, client, out, entry, config, check)
                # Client proxy is still a Python variable; its native payload was
                # explicitly released by the context before server.close.
                result["server_returncode"] = server.process.returncode
                result["client_release_before_server_close"] = True
                state = read(out / "world_trace.json")
                item = next(r for r in plan["episodes"] if r["route_id"] == 6)
                result["actual_geography"] = actual_geography(state, item, plan)
                atomic_json(out / "result.json", result)
                records.append(result)
                progress("Native teardown completed; retain actual server exit code")
            passed = all(r["server_returncode"] in (0, -15) for r in records)
            atomic_json(
                args.out / "assessment.json",
                dict(
                    records=records,
                    native_client_abort_absent=True,
                    normal_owned_server_exits=passed,
                    infrastructure_check_passed=passed,
                    no_model_or_dataset_or_driving_qualification=True,
                ),
            )
            status = "complete"
            progress("Two instrument cycles finished; inspect both native exits")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
