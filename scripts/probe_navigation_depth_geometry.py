#!/usr/bin/env python3
"""Verify metric depth against fixed known native boxes at one guarded TRAIN site."""

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

from scripts.audit_navigation_dense_labels import actual_geography  # noqa: E402
from scripts.collect_navigation_branches import selected_entry  # noqa: E402
from scripts.navigation_label_rig import (  # noqa: E402
    LabelRig,
    LabelShards,
    decode_depth,
    decode_instances,  # noqa: E402
)
from src.navigation.collection import training_position  # noqa: E402
from src.navigation.contracts import VIEWS  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.label_collection import validate_label_guard  # noqa: E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa: E402
from src.navigation.replay import capture_world_state  # noqa: E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def pixel_points(depth, mask, calibration, radial=False):
    """Unreal camera coordinates, with each original pixel sampled at its centre."""
    ys, xs = np.nonzero(mask)
    ray = np.stack(
        [
            np.ones(len(xs)),
            (xs + 0.5 - calibration.width / 2) / calibration.intrinsic[0, 0],
            -(ys + 0.5 - calibration.height / 2) / calibration.intrinsic[1, 1],
        ],
        axis=1,
    )
    if radial:
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    return ray * depth[mask, None]


def outside_box(points_camera, camera_to_world, box_to_world, extent, padding=0.30):
    homogeneous = np.column_stack([points_camera, np.ones(len(points_camera))])
    local = homogeneous @ (np.linalg.inv(box_to_world) @ camera_to_world).T
    return np.linalg.norm(np.maximum(abs(local[:, :3]) - extent - padding, 0), axis=1)


def identities(raw):
    semantic, pair = decode_instances(raw)
    return semantic, pair[..., 0].astype(np.uint32) + 256 * pair[..., 1].astype(np.uint32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    entry, _ = selected_entry()
    plan_path = ROOT / "outputs/navigation_dense_label_collection_20261020/plan.json"
    plan = read(plan_path)
    guard = validate_label_guard(plan_path, plan["catalog"], entry["town"], "6")
    item = next(r for r in plan["episodes"] if r["route_id"] == 6)
    config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
    sources = [
        Path(__file__),
        plan_path,
        ROOT / "reports/navigation_depth_geometry_protocol_20261020.md",
        ROOT / "configs/navigation.yaml",
        ROOT / "scripts/navigation_label_rig.py",
        ROOT / "scripts/audit_navigation_dense_labels.py",
        *[
            ROOT / "src/navigation" / name
            for name in (
                "runtime.py",
                "sensors.py",
                "collection.py",
                "label_collection.py",
                "replay.py",
                "recorder_collisions.py",
                "storage_budget.py",
            )
        ],
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    cases = [
        dict(view=view, bearing_deg=bearing, axis_displacement_m=25)
        for view in VIEWS
        for bearing in (0, 40)
    ]
    atomic_json(
        args.out / "request.json",
        dict(
            entry=entry,
            cases=cases,
            maximum_wall_seconds=300,
            maximum_artifact_bytes=100_000_000,
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
            purpose="Depth instrumentation only; excluded from all model fitting",
            inference_inputs_changed=False,
            padding_m=0.30,
            axial_inside_required=0.95,
            off_axis_radial_outside_required=0.80,
        ),
    )
    status, completed, started = "running", 0, time.monotonic()

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="TRAIN metric depth convention",
                frames=completed,
                target_frames=6,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 300:
            raise InterruptedError("Owned STOP or depth probe300s execution cap")
        size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
        if size > 100_000_000:
            raise InterruptedError("Depth probe100MB artifact cap")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 10)
            vendor = activate()
            import carla

            progress("Starting one guarded TRAIN instrumentation server")
            with CarlaServer(vendor["carla_root"], args.out, town=entry["town"]) as client:
                check()
                world = client.load_world(entry["town"], reset_settings=True)
                if world.get_map().name.rsplit("/", 1)[-1] != entry["town"]:
                    raise ValueError("Wrong physical town")
                settings = world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                settings.substepping = True
                settings.max_substep_delta_time = 0.01
                settings.max_substeps = 5
                world.apply_settings(settings)
                world.set_weather(carla.WeatherParameters.ClearNoon)
                bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
                bp.set_attribute("role_name", "hero")
                start = world.get_map().get_waypoint(carla.Location(*entry["start_xyz"])).transform
                start.location.z += 0.5
                actors, rig, writer, recording = [], None, None, False
                results, world_rows = [], []
                try:
                    ego = world.try_spawn_actor(bp, start)
                    if ego is None:
                        raise RuntimeError("Depth probe ego spawn failed")
                    actors.append(ego)
                    ego.apply_control(carla.VehicleControl(brake=1))
                    for _ in range(30):
                        check()
                        world.tick()
                    ego.set_simulate_physics(False)
                    training_position(
                        world.get_map(),
                        ego.get_transform(),
                        guard["allowed_junctions"][entry["town"]],
                        world.get_snapshot().frame,
                        args.out / "partition_violation.json",
                    )
                    world_rows.append(capture_world_state(world))
                    actual_geography(world_rows, item, plan)
                    bp.set_attribute("role_name", "depth_instrument_target")
                    high = ego.get_transform()
                    high.location.z += 60
                    target = world.try_spawn_actor(bp, high)
                    if target is None:
                        raise RuntimeError("Depth probe target spawn failed")
                    actors.append(target)
                    target.set_simulate_physics(False)
                    rig = LabelRig(world, ego, config, labels=True)
                    writer = LabelShards(args.out / "shards", labels=True)
                    atomic_json(args.out / "sensors.json", rig.specification)
                    client.start_recorder(str(args.out / "native.log"), True)
                    recording = True

                    def capture():
                        for _ in range(5):
                            check()
                            frame = world.tick()
                            rgb, labels, timestamp, poses = rig.collect(frame)
                        writer.append(frame, timestamp, rgb, labels)
                        world_rows.append(capture_world_state(world))
                        return frame, labels, poses

                    for case in cases:
                        check()
                        target.set_transform(high)
                        baseline_frame, baseline, _ = capture()
                        view = case["view"]
                        calibration = rig.calibration[view]
                        bearing = math.radians(case["bearing_deg"])
                        yaw = math.radians(calibration.pitch_yaw_roll[1])
                        offset = np.array([25.0, 25 * math.tan(bearing)])
                        local = np.array(calibration.xyz)
                        local[:2] += (
                            offset
                            @ np.array(
                                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
                            ).T
                        )
                        local[2] = 0.5
                        target_xyz = np.asarray(ego.get_transform().get_matrix()) @ np.r_[local, 1]
                        pose = carla.Transform(
                            carla.Location(*map(float, target_xyz[:3])),
                            carla.Rotation(
                                yaw=ego.get_transform().rotation.yaw + calibration.pitch_yaw_roll[1]
                            ),
                        )
                        target.set_transform(pose)
                        frame, labels, sensor_poses = capture()
                        before_semantic, before_ids = identities(baseline[view + "_instance"])
                        semantic, ids = identities(labels[view + "_instance"])
                        before_vehicle = np.isin(before_semantic, [14, 15, 16, 17])
                        vehicle = np.isin(semantic, [14, 15, 16, 17])
                        new_ids = set(map(int, np.unique(ids[vehicle]))) - set(
                            map(int, np.unique(before_ids[before_vehicle]))
                        )
                        if len(new_ids) != 1:
                            raise ValueError(f"Expected one isolated new render actor: {new_ids}")
                        mask = vehicle & np.isin(ids, list(new_ids))
                        if mask.sum() < 100:
                            raise ValueError("Insufficient visible target pixels; no replacement")
                        measured = sensor_poses[view + "_depth"]
                        camera = carla.Transform(
                            carla.Location(*measured[:3]),
                            carla.Rotation(pitch=measured[3], yaw=measured[4], roll=measured[5]),
                        )
                        bb = target.bounding_box
                        box_to_world = np.asarray(target.get_transform().get_matrix()) @ np.asarray(
                            carla.Transform(bb.location, bb.rotation).get_matrix()
                        )
                        extent = np.array([bb.extent.x, bb.extent.y, bb.extent.z])
                        depth = decode_depth(labels[view + "_depth"])
                        hypotheses = {}
                        for name, radial in (("axial", False), ("radial", True)):
                            points = pixel_points(depth, mask, calibration, radial=radial)
                            gap = outside_box(
                                points, np.asarray(camera.get_matrix()), box_to_world, extent
                            )
                            hypotheses[name] = dict(
                                inside_fraction=float(np.mean(gap < 1e-6)),
                                median_outside_m=float(np.median(gap)),
                                p95_outside_m=float(np.percentile(gap, 95)),
                            )
                        result = dict(
                            case=case,
                            frame=frame,
                            baseline_frame=baseline_frame,
                            new_render_id=next(iter(new_ids)),
                            pixels=int(mask.sum()),
                            sensor_pose=measured,
                            sensor_poses=sensor_poses,
                            box_to_world=box_to_world.tolist(),
                            extent=extent.tolist(),
                            hypotheses=hypotheses,
                        )
                        results.append(result)
                        atomic_json(args.out / f"case_{completed:02d}.json", result)
                        completed += 1
                        progress("Fixed native target placements; both hypotheses retained")
                    writer.close()
                    writer = None
                    client.stop_recorder()
                    recording = False
                    world.tick()
                    # The recorder is closed before independent decoding; keep every contact.
                    contacts = recorder_contacts(args.out / "native.log")
                    if not contacts["complete"]:
                        raise ValueError("Incomplete native diagnostic recording")
                    geography = actual_geography(world_rows, item, plan)
                    atomic_json(args.out / "world_trace.json", world_rows)
                    confirmed = all(
                        r["hypotheses"]["axial"]["inside_fraction"] >= 0.95
                        and (
                            r["case"]["bearing_deg"] == 0
                            or r["hypotheses"]["radial"]["inside_fraction"] <= 0.20
                        )
                        for r in results
                    )
                    atomic_json(
                        args.out / "assessment.json",
                        dict(
                            cases=results,
                            convention_check_passed=confirmed,
                            axial_depth_confirmed_for_this_installation=confirmed,
                            padded_box_tolerance_m=0.30,
                            contacts=contacts,
                            geography=geography,
                            source_and_labels_excluded_from_fitting=True,
                            no_perception_or_driving_qualification=True,
                        ),
                    )
                finally:
                    if writer is not None:
                        writer.close()
                    if recording:
                        client.stop_recorder()
                    if rig is not None:
                        rig.close()
                    for actor in reversed(actors):
                        if actor.is_alive:
                            actor.destroy()
            status = "complete"
            progress("All six fixed placements assessed; instrument evidence only")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
