#!/usr/bin/env python3
"""Independently verify dense label pixels, original RGB, splits and native controls."""

import argparse
import io
import json
import shutil
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.navigation_label_rig import decode_depth, decode_instances, digest_array  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa:E402
from src.navigation.label_collection import validate_label_guard  # noqa:E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=5)
def offline_map(town, expected_hash):
    from src.navigation.vendor import activate

    vendor = activate()
    import carla

    path = Path(vendor["carla_root"]) / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
    if digest(path) != expected_hash:
        raise ValueError("Original strict-training map bytes changed")
    return carla.Map(town, path.read_text())


def actual_geography(world, item, plan):
    from src.navigation.vendor import activate

    activate()
    import carla
    from scipy.spatial import cKDTree

    guard, town = read(plan["actual_training_guard"]), item["town"]
    world_map = offline_map(town, guard["geometry_eligibility"]["map_sha256"][town])
    allowed = set(guard["allowed_junctions"][town]) - set(
        plan["reserved_rule_junctions"].get(town, [])
    )
    if item["role"] == "train":
        for entry in plan["episodes"]:
            if entry["town"] == town and entry["role"] == "calibration":
                allowed -= set(entry["entry"]["crossed_junction_ids"])
    seen, positions = set(), []
    for row in world:
        hero = next(a for a in row["actors"] if a["role"] == "hero")
        positions.append(hero["pose"][:3])
        waypoint = world_map.get_waypoint(carla.Location(*hero["pose"][:3]))
        if waypoint.is_junction:
            seen.add(waypoint.junction_id)
            if waypoint.junction_id not in allowed:
                raise ValueError(
                    "Actual source pose entered original VAL/rule/new-calibration geography"
                )
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
    minimum = float(np.min(tree.query(positions)[0]))
    if minimum < 1.0:
        raise ValueError(f"Actual trajectory lacks1m reserved-junction clearance:{minimum}")
    return dict(
        observed_junctions=sorted(seen),
        all_actual_world_poses_checked=len(world),
        minimum_reserved_geometry_clearance_m=minimum,
        role=item["role"],
    )


def audit_episode(item, output, plan_path, check):
    plan = read(plan_path)
    validate_label_guard(plan_path, plan["catalog"], item["town"], str(item["route_id"]))
    episodes = list((output / "episodes").iterdir())
    native_paths = list((output / "recordings").glob("*.log"))
    if len(episodes) != 1 or len(native_paths) != 1:
        raise ValueError("One original physical episode and closed native recording required")
    episode = episodes[0]
    profile = read(episode / "label_profile.json")
    metadata = read(episode / "data/episode.json")
    if (
        metadata["dense_label_role"] != item["role"]
        or metadata["label_plan_sha256"] != digest(plan_path)
        or profile["label_role"] != item["role"]
        or metadata["town"] != item["town"]
    ):
        raise ValueError("Dense label role/geography identity changed")
    manifest = read(output / "manifest.json")
    if (
        manifest["student"]
        or manifest["final_benchmark"]
        or manifest["rule_test"]
        or not manifest["training_collection"]
        or not manifest["collection"]["dense_labels"]
    ):
        raise ValueError("Dense labels must remain separate privileged TRAINcollection")
    world = [json.loads(line) for line in (episode / "world_trace.jsonl").read_text().splitlines()]
    geography = actual_geography(world, item, plan)
    hero = next(a["id"] for a in world[0]["actors"] if a["role"] == "hero")
    aligned, receipt = align_world_frames(vehicle_frames(native_paths[0], hero), world)
    contacts = recorder_contacts(native_paths[0])
    if not contacts["complete"]:
        raise ValueError("Closed native recorder incomplete")
    references = {}
    controls = 0
    for shard in sorted((episode / "data").glob("shard_*.zip")):
        check()
        with zipfile.ZipFile(shard) as archive:
            for row in shard_records(shard):
                packet = load_policy_sample(shard, row)  # Verify every original RGBpixel hash.
                target = json.loads(archive.read(row["prefix"] + "/targets.json"))
                ack = target["facts"]["actuator_feedback_frame"]
                if (
                    ack != packet.frame + 1
                    or ack not in aligned
                    or not np.allclose(
                        aligned[ack]["control"], target["executed_control"], atol=1e-5, rtol=0
                    )
                ):
                    raise ValueError(
                        "Original executed control lacks exact next-tick native acknowledgment"
                    )
                if packet.frame in references:
                    raise ValueError("Duplicate original RGBframe")
                references[packet.frame] = (shard, row, packet.timestamp)
                controls += 1
    label_frames = []
    class_counts = {}
    label_shas = {}
    label_sources = []
    expected = {f"rgb_{v}" for v in ("front", "left", "right")} | {
        f"{v}_{kind}" for v in ("front", "left", "right") for kind in ("instance", "depth")
    }
    for shard in sorted((episode / "labels").glob("*.zip")):
        check()
        label_shas[str(shard)] = digest(shard)
        with zipfile.ZipFile(shard) as archive:
            for name in sorted(n for n in archive.namelist() if n.endswith("metadata.json")):
                meta = json.loads(archive.read(name))
                frame = meta["frame"]
                if (
                    set(meta["pixel_sha256"]) != expected
                    or frame not in references
                    or meta["privileged_labels_not_policy_inputs"] is not True
                ):
                    raise ValueError("Missing original RGBframe or changed privileged-label fields")
                rgb_shard, rgb_row, timestamp = references[frame]
                if abs(meta["timestamp"] - timestamp) > 1e-6:
                    raise ValueError("Original RGBand label stored timestamps differ")
                original = load_policy_sample(rgb_shard, rgb_row)
                prefix = name.rsplit("/", 1)[0]
                arrays = {}
                for field, want in meta["pixel_sha256"].items():
                    a = np.load(
                        io.BytesIO(archive.read(prefix + "/" + field + ".npy")), allow_pickle=False
                    )
                    channels = 3 if field.startswith("rgb_") else 4
                    if (
                        a.dtype != np.uint8
                        or a.shape != (288, 512, channels)
                        or digest_array(a) != want
                    ):
                        raise ValueError("Raw label array geometry/dtype/hash differs")
                    arrays[field] = a
                for view in ("front", "left", "right"):
                    if not np.array_equal(arrays["rgb_" + view], original.rgb[view]):
                        raise ValueError("Label RGBdiffers from independent original policy shard")
                    semantic, _ = decode_instances(arrays[view + "_instance"])
                    ids, counts = np.unique(semantic, return_counts=True)
                    if not set(map(int, ids)) <= set(range(29)):
                        raise ValueError("Unexpected semantic labels for pinned CARLA0.9.15")
                    for k, v in zip(ids, counts):
                        class_counts[int(k)] = class_counts.get(int(k), 0) + int(v)
                    depth = decode_depth(arrays[view + "_depth"])
                    if not np.isfinite(depth).all() or depth.min() < 0 or depth.max() > 1000:
                        raise ValueError("Invalid decoded depth")
                label_frames.append(frame)
                label_sources.append(
                    dict(
                        frame=frame,
                        timestamp=timestamp,
                        label_shard=str(shard),
                        label_prefix=prefix,
                        rgb_shard=str(rgb_shard),
                        rgb_prefix=rgb_row["prefix"],
                    )
                )
    if (
        len(label_frames) != profile["label_timestamps"]
        or len(label_frames) < 2
        or len(set(label_frames)) != len(label_frames)
        or any(b != a + 10 for a, b in zip(label_frames, label_frames[1:]))
    ):
        raise ValueError("Dense labels must be unique contiguous2Hz and match profile counts")
    if (
        profile["maximum_position_error_m"] > 0.002
        or profile["maximum_angle_error_degrees"] > 0.05
        or not profile["all_raw_pixel_roundtrips_verified"]
    ):
        raise ValueError("Recorded camera/label rigid alignment failed")
    return dict(
        item=item,
        output=str(output),
        rgb_timestamps=controls,
        label_timestamps=len(label_frames),
        zero_next_tick_control_differences=True,
        alignment=receipt,
        all_label_RGB_matches_original=True,
        all_pixel_hashes_verified=True,
        label_sources=label_sources,
        semantic_pixel_counts=class_counts,
        original_source_shards_sha256={
            str(p): digest(p) for p in (episode / "data").glob("shard_*.zip")
        },
        label_shards_sha256=label_shas,
        world_trace_sha256=digest(episode / "world_trace.jsonl"),
        native_contacts=contacts,
        label_profile_sha256=digest(episode / "label_profile.json"),
        frame_pose_receipt_sha256=digest(episode / "label_alignment.jsonl"),
        independently_verified_actual_geography=geography,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--continuation", type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    source_files = [
        Path(__file__),
        ROOT / "scripts/navigation_label_rig.py",
        ROOT / "src/navigation/dataset.py",
        ROOT / "src/navigation/label_collection.py",
        ROOT / "src/navigation/recorder_actions.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    for source in source_files:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            root=str(args.root.resolve()),
            source_hashes=hashes,
            plan_sha256=digest(args.root / "plan.json"),
            no_training=True,
            maximum_compute_seconds=600,
            continuation=str(args.continuation.resolve()) if args.continuation else None,
        ),
    )
    status, done, started = "running", 0, None

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Dense-label independent data audit",
                frames=done,
                target_frames=13,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists():
            raise InterruptedError("Dense-label independent audit STOP")
        if started is not None and time.monotonic() - started > 600:
            raise InterruptedError("Dense-label audit600s cap")

    try:
        with BudgetWatch(args.out):
            owner = args.continuation or args.root
            while read(owner / "progress.json")["status"] in (
                "running",
                "prepared",
                "starting",
            ):
                check()
                progress("Waiting for terminal data generation; no GPUor simulator")
                time.sleep(3)
            if read(owner / "progress.json")["status"] != "complete":
                raise RuntimeError(
                    "Collection incomplete; preserve planned denominator and inspect first"
                )
            if any(digest(ROOT / p) != h for p, h in hashes.items()):
                raise ValueError("Queued auditor source changed")
            started = time.monotonic()
            plan = read(args.root / "plan.json")
            attempted = read(owner / "assessment.json")["results"]
            if [row["item"] for row in attempted] != plan["episodes"] or (
                args.continuation
                and read(args.continuation / "request.json")["plan_sha256"]
                != digest(args.root / "plan.json")
            ):
                raise ValueError(
                    "Every original source must appear once in unchanged planned order"
                )
            results = []
            errors = []
            failed_sources = []
            all_contacts = []
            for row in attempted:
                item = row["item"]
                check()
                progress(
                    f"Verify every label/RGB/native acknowledgment for route{item['route_id']}"
                )
                output = Path(row["output"])
                for recording in sorted((output / "recordings").glob("*.log")):
                    all_contacts.append(recorder_contacts(recording))
                if row["label_timestamps"] == 0:
                    failed_sources.append(
                        dict(
                            item=item,
                            output=str(output),
                            no_dense_labels_eligible=True,
                            partition_violation=read(output / "collection_partition_violation.json")
                            if (output / "collection_partition_violation.json").exists()
                            else None,
                            native_summary=row["native_summary"],
                        )
                    )
                    done += 1
                    continue
                try:
                    results.append(audit_episode(item, output, args.root / "plan.json", check))
                except Exception as error:
                    errors.append(dict(route_id=item["route_id"], error=str(error)))
                done += 1
            atomic_json(
                args.out / "assessment.json",
                dict(
                    results=results,
                    errors=errors,
                    failed_sources=failed_sources,
                    all_planned_episodes=len(results) == 13 and not errors and not failed_sources,
                    all_nonempty_sources_verified=not errors and bool(results),
                    original_planned_denominator=13,
                    total_rgb_timestamps=sum(r["rgb_timestamps"] for r in results),
                    total_label_timestamps=sum(r["label_timestamps"] for r in results),
                    all_attempt_native_recorders=all_contacts,
                    unfiltered_contact_starts=sum(r["ego_contact_starts"] for r in all_contacts),
                    calibration_and_train_locations_predeclared=True,
                    training_started=False,
                    driving_or_calibration_qualification=False,
                    collection_plan_sha256=digest(args.root / "plan.json"),
                ),
            )
            status = (
                "complete" if not errors and len(results) + len(failed_sources) == 13 else "failed"
            )
            progress("Every original data source assessed; failures remain failures")
            if errors:
                raise RuntimeError(
                    "Independent dense-label audit errors; inspect retained assessment"
                )
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
