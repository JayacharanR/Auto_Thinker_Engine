#!/usr/bin/env python3
"""Bounded TRAIN-only RGB versus RGB+instance/depth label collection profile."""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.apply_navigation_compact_generation import live_dependencies  # noqa:E402
from scripts.collect_navigation_branches import selected_entry  # noqa:E402
from scripts.navigation_label_rig import (  # noqa:E402
    LabelRig,
    LabelShards,
    decode_depth,
    decode_instances,
)
from src.navigation.collection import RouteProjector, training_position  # noqa:E402
from src.navigation.controller import track  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.replay import capture_world_state  # noqa:E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    entry, _ = selected_entry()
    guard_path = (
        ROOT / "outputs/navigation_spatial_data_repaired_20261019/actual_training_guard.json"
    )
    guard = read(guard_path)
    if not set(entry["crossed_junction_ids"]) <= set(guard["allowed_junctions"][entry["town"]]):
        raise ValueError("Existing diagnostic location must remain strict TRAIN")
    config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
    paths = [
        Path(__file__),
        ROOT / "scripts/navigation_label_rig.py",
        ROOT / "scripts/collect_navigation_branches.py",
        ROOT / "configs/navigation.yaml",
        ROOT / "reports/navigation_label_rig_profile_protocol_20261020.md",
        *[
            ROOT / "src/navigation" / name
            for name in (
                "sensors.py",
                "collection.py",
                "controller.py",
                "runtime.py",
                "replay.py",
                "storage_budget.py",
                "recorder_collisions.py",
            )
        ],
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    atomic_json(
        args.out / "request.json",
        dict(
            after=str(args.after.resolve()),
            entry=entry,
            guard_sha256=digest(guard_path),
            source_hashes=hashes,
            modes=["rgb_only", "rgb_instance_depth"],
            ordering="Fixed baseline then label rig, no repetition",
            physics_hz=20,
            measured_ticks_per_mode=120,
            warmup_ticks=10,
            stored_timestamps_per_mode=60,
            storage_hz=10,
            maximum_wall_seconds=600,
            maximum_artifact_bytes=350_000_000,
            teacher_control="Declared numerical route tracker at2m/s; no imitation teacher labels",
            traffic="One stationary lead at25m; no TrafficManager background",
            purpose="Label/throughput/storage instrumentation, not learned driving or benchmark",
            new_labels_not_added_to_training=True,
            actor_id_instance_id_equivalence_claim=False,
            future_label_efficiency_requires_all_supervised_timestamps_counted=True,
        ),
    )
    for path in paths:
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    status, done, active, started = "running", 0, None, None

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="TRAIN label-rig profile",
                frames=done,
                target_frames=240,
                active_output=str(active) if active else None,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists():
            raise InterruptedError("Owned label-rig profile STOP")
        if started is not None and time.monotonic() - started > 600:
            raise InterruptedError("Label-rig600s wall cap")
        if shutil.disk_usage(args.out).free < 10e9 + 40e6:
            raise InterruptedError("Approved reserve reached in label-rig profile")
        if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > 350_000_000:
            raise InterruptedError("Label-rig350MBartifact cap reached; preserve all bytes")

    try:
        with BudgetWatch(args.out):
            while True:
                check()
                path = args.after / "progress.json"
                if path.exists() and read(path)["status"] not in ("running", "starting"):
                    if read(path)["status"] not in ("complete", "failed"):
                        raise InterruptedError(
                            "Prior owner stopped; do not start another native stage"
                        )
                    # Explicitly allow a failed instrumentation predecessor to terminate;
                    # it remains failed, with no retry or reclassification by this profile.
                    if not live_dependencies([args.after]):
                        break
                progress("Waiting for original TRAINdrive owner to terminate; no duplicate server")
                time.sleep(3)
            if any(digest(ROOT / p) != h for p, h in hashes.items()):
                raise ValueError("Registered label-rig source/config changed while queued")
            with RunLock(ROOT / "outputs/.navigation.lock"):
                check_resources(args.out, 10)
                vendor = activate()
                import carla
                from agents.navigation.global_route_planner import GlobalRoutePlanner

                enum = {name: int(value) for name, value in carla.CityObjectLabel.names.items()}
                if enum.get("Car") != 14 or enum.get("TrafficLight") != 7:
                    raise ValueError("Pinned0.9.15runtime semantic enum changed")
                atomic_json(args.out / "semantic_enum.json", enum)
                results = []
                started = time.monotonic()
                with CarlaServer(vendor["carla_root"], args.out, town=entry["town"]) as client:
                    for mode in ("rgb_only", "rgb_instance_depth"):
                        check()
                        active = args.out / mode
                        active.mkdir()
                        world = client.load_world(entry["town"], reset_settings=True)
                        if world.get_map().name.rsplit("/", 1)[-1] != entry["town"]:
                            raise ValueError("Wrong native map for label instrumentation")
                        settings = world.get_settings()
                        settings.synchronous_mode = True
                        settings.fixed_delta_seconds = 0.05
                        settings.substepping = True
                        settings.max_substep_delta_time = 0.01
                        settings.max_substeps = 5
                        world.apply_settings(settings)
                        world.set_weather(carla.WeatherParameters.ClearNoon)
                        planner = GlobalRoutePlanner(world.get_map(), 1.0)
                        route = planner.trace_route(
                            carla.Location(*entry["start_xyz"]), carla.Location(*entry["end_xyz"])
                        )
                        xyz = np.array(
                            [[w.transform.location.x, w.transform.location.y] for w, _ in route]
                        )
                        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
                        lead_index = int(np.searchsorted(arc, 25.0))
                        actors = []
                        rig = None
                        writer = None
                        recording = False
                        timing = []
                        events = []
                        perception_stats = []
                        world_rows = []
                        actions = []
                        try:
                            for index, model, role in (
                                (0, "vehicle.lincoln.mkz_2020", "hero"),
                                (lead_index, "vehicle.lincoln.mkz_2017", "profile_stationary_lead"),
                            ):
                                transform = route[index][0].transform
                                transform.location.z += 0.5
                                bp = world.get_blueprint_library().find(model)
                                bp.set_attribute("role_name", role)
                                actor = world.try_spawn_actor(bp, transform)
                                if actor is None:
                                    raise RuntimeError(f"Label profile {role} spawn failed")
                                actor.apply_control(carla.VehicleControl(brake=1.0))
                                actors.append(actor)
                            ego, lead = actors
                            for _ in range(20):
                                world.tick()
                            collision = world.spawn_actor(
                                world.get_blueprint_library().find("sensor.other.collision"),
                                carla.Transform(),
                                attach_to=ego,
                            )
                            actors.append(collision)
                            collision.listen(
                                lambda event: events.append(
                                    dict(frame=event.frame, other_actor=event.other_actor.id)
                                )
                            )
                            rig = LabelRig(world, ego, config, labels=mode != "rgb_only")
                            atomic_json(active / "sensors.json", rig.specification)
                            for _ in range(10):
                                world.tick()
                            writer = LabelShards(active / "shards", labels=mode != "rgb_only")
                            projector = RouteProjector(xyz)
                            client.start_recorder(str(active / "native.log"), True)
                            recording = True
                            run_started = time.monotonic()
                            maximum_alignment_position = maximum_alignment_angle = 0.0
                            for tick in range(120):
                                check()
                                before = time.monotonic()
                                frame = world.tick()
                                rgb, labels, timestamp, poses = rig.collect(frame)
                                pose = training_position(
                                    world.get_map(),
                                    ego.get_transform(),
                                    guard["allowed_junctions"][entry["town"]],
                                    frame,
                                    active / "partition_violation.json",
                                )
                                yaw = math.radians(pose.rotation.yaw)
                                nav = projector.project(
                                    np.array([pose.location.x, pose.location.y]), yaw
                                )
                                command = track(
                                    nav, np.zeros(8), ego.get_velocity().length(), 2.0, 0.0
                                )
                                chosen = (command.steer, command.throttle, command.brake)
                                if events:
                                    chosen = (0.0, 0.0, 1.0)
                                for name, value in poses.items():
                                    reference = np.asarray(poses[name.split("_", 1)[0] + "_rgb"])
                                    value = np.asarray(value)
                                    maximum_alignment_position = max(
                                        maximum_alignment_position,
                                        float(np.linalg.norm(value[:3] - reference[:3])),
                                    )
                                    maximum_alignment_angle = max(
                                        maximum_alignment_angle,
                                        float(
                                            np.abs(
                                                (value[3:] - reference[3:] + 180) % 360 - 180
                                            ).max()
                                        ),
                                    )
                                if (
                                    maximum_alignment_position > 0.001
                                    or maximum_alignment_angle > 0.01
                                ):
                                    raise ValueError(
                                        "RGB and privileged labels are not rigidly aligned"
                                    )
                                if tick % 2 == 0:
                                    writer.append(frame, timestamp, rgb, labels)
                                    if labels:
                                        stats = {}
                                        for view in ("front", "left", "right"):
                                            semantic, _ = decode_instances(
                                                labels[f"{view}_instance"]
                                            )
                                            depth = decode_depth(labels[f"{view}_depth"])
                                            ids, counts = np.unique(semantic, return_counts=True)
                                            vehicle = semantic == enum["Car"]
                                            stats[view] = dict(
                                                semantic_counts={
                                                    int(k): int(v) for k, v in zip(ids, counts)
                                                },
                                                car_pixels=int(vehicle.sum()),
                                                median_car_depth_m=float(np.median(depth[vehicle]))
                                                if vehicle.any()
                                                else None,
                                            )
                                        perception_stats.append(dict(frame=frame, views=stats))
                                world_rows.append(capture_world_state(world))
                                actions.append(
                                    dict(
                                        frame=frame,
                                        control=list(chosen),
                                        previous_applied=[
                                            ego.get_control().steer,
                                            ego.get_control().throttle,
                                            ego.get_control().brake,
                                        ],
                                        sensor_poses=poses,
                                    )
                                )
                                ego.apply_control(
                                    carla.VehicleControl(
                                        steer=chosen[0], throttle=chosen[1], brake=chosen[2]
                                    )
                                )
                                lead.apply_control(carla.VehicleControl(brake=1.0))
                                timing.append(time.monotonic() - before)
                                done += 1
                                if tick % 10 == 0:
                                    progress(f"{mode}: same native20Hz/10Hzstored label profile")
                            elapsed = time.monotonic() - run_started
                            writer.close()
                            client.stop_recorder()
                            recording = False
                            world.tick()
                            deadline, previous_identity = time.monotonic() + 5, None
                            while True:
                                contacts = recorder_contacts(active / "native.log")
                                stat = (active / "native.log").stat()
                                identity = stat.st_size, stat.st_mtime_ns
                                if (
                                    contacts["complete"]
                                    and contacts["frames"] >= 120
                                    and identity == previous_identity
                                ):
                                    break
                                if time.monotonic() >= deadline:
                                    raise RuntimeError(
                                        "Label-profile native recorder did not close/flush"
                                    )
                                previous_identity = identity
                                time.sleep(0.05)
                            for name, rows in (
                                ("world_trace", world_rows),
                                ("actions", actions),
                                ("label_statistics", perception_stats),
                            ):
                                with (active / f"{name}.jsonl").open("x") as stream:
                                    stream.writelines(json.dumps(row) + "\n" for row in rows)
                            result = dict(
                                mode=mode,
                                physics_ticks=120,
                                stored_timestamps=writer.count,
                                wall_seconds=elapsed,
                                physics_frames_per_second=120 / elapsed,
                                tick_p95_ms=float(np.percentile(timing, 95) * 1000),
                                raw_array_bytes=writer.raw_bytes,
                                committed_shards=writer.shards,
                                compressed_shard_bytes=sum(s["bytes"] for s in writer.shards),
                                maximum_cross_modality_position_error_m=maximum_alignment_position,
                                maximum_cross_modality_angle_error_deg=maximum_alignment_angle,
                                all_pixel_roundtrips_verified=True,
                                all_frames_barrier_aligned=True,
                                native_contacts=contacts,
                                collision_sensor_events=events,
                                labels_not_used_by_controller=True,
                                controller_no_supervised_imitation_claim=True,
                            )
                            atomic_json(active / "assessment.json", result)
                            results.append(result)
                        finally:
                            if recording:
                                client.stop_recorder()
                            if writer:
                                writer.close()
                            if rig:
                                rig.close()
                            for actor in reversed(actors):
                                if actor.is_alive:
                                    if actor.type_id.startswith("sensor."):
                                        actor.stop()
                                    actor.destroy()
                            try:
                                settings = world.get_settings()
                                settings.synchronous_mode = False
                                settings.fixed_delta_seconds = None
                                world.apply_settings(settings)
                            except Exception as error:
                                atomic_json(active / "cleanup_error.json", dict(error=str(error)))
                total = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
                if total > 350_000_000:
                    raise RuntimeError(
                        "Measured profile exceeded registered350MBcap; preserve all bytes"
                    )
                atomic_json(
                    args.out / "assessment.json",
                    dict(
                        results=results,
                        artifact_bytes=total,
                        training_or_benchmark_modified=False,
                        policy_observations_modified=False,
                        limitations="Single TRAINlocation/fixed order/slow2m/s, "
                        "no diverse-traffic, learned-driving or general-storage claim. "
                        "Native label enum verified; "
                        "instance render IDs are not actor IDs.",
                    ),
                )
                status = "complete"
                progress("Label throughput/storage measured; no training or promotion")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
