#!/usr/bin/env python3
"""Inventory strict-TRAIN factual prediction sources, including noisy/failed actions."""

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import shard_records  # noqa:E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--guard",
        type=Path,
        default=ROOT / "outputs/navigation_m2_varied_waits_20261018/guard.json",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve every source-inventory attempt")
    args.out.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "src/navigation/recorder_actions.py",
        ROOT / "src/navigation/recorder_collisions.py",
        ROOT / "reports/navigation_dynamics_data_protocol_20261019.md",
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    spatial_path = ROOT / "outputs/navigation_spatial_data_20261019/manifest.json"
    spatial = json.loads(spatial_path.read_text())
    guard_path = args.guard.resolve()
    guard = json.loads(guard_path.read_text())
    forbidden_data = {e["data"] for e in spatial["episodes"] if e["role"] == "validation"}
    original_train = {e["data"] for e in spatial["episodes"] if e["role"] == "train"}
    candidates = {Path(e["data"]).resolve() for e in spatial["episodes"]}
    for root in [
        "outputs/navigation_m1_20k_v2_20261015",
        "outputs/navigation_m2_dagger_pilot_20261016",
    ]:
        candidates.update(p.parent.resolve() for p in (ROOT / root).rglob("data/episode.json"))
    atomic_json(
        args.out / "request.json",
        dict(
            candidates=[str(p) for p in sorted(candidates)],
            original_manifest_sha256=digest(spatial_path),
            strict_guard_sha256=digest(guard_path),
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
            no_model_updates=True,
            no_evaluation_outcomes_used=True,
        ),
    )
    vendor = activate()
    import carla

    maps = {}
    accepted, excluded = [], []
    for i, data in enumerate(sorted(candidates)):
        try:
            if str(data) in forbidden_data:
                raise ValueError("Protected original driving-validation episode")
            metadata = json.loads((data / "episode.json").read_text())
            town = metadata["town"]
            allowed = set(guard["allowed_junctions"][town])
            if metadata.get("partition") != "train" or town.removesuffix("_Opt") == "Town05":
                raise ValueError("Not strict TRAIN")
            if not set(metadata["junction_ids"]) <= allowed:
                raise ValueError("Reserved driving/component geography")
            if town not in maps:
                xodr = (
                    Path(vendor["carla_root"])
                    / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
                )
                maps[town] = carla.Map(town, xodr.read_text())
            world_path = data.parent / "world_trace.jsonl"
            world = [json.loads(line) for line in world_path.read_text().splitlines()]
            hero_ids = set()
            for row in world:
                hero = [a for a in row["actors"] if a["role"] == "hero"]
                if len(hero) != 1:
                    raise ValueError("Ambiguous hero")
                hero_ids.add(hero[0]["id"])
                wp = maps[town].get_waypoint(carla.Location(*hero[0]["pose"][:3]))
                if wp.is_junction and wp.junction_id not in allowed:
                    raise ValueError(
                        f"Actual trajectory entered reserved junction:{wp.junction_id}"
                    )
            if len(hero_ids) != 1:
                raise ValueError("Hero identity changed")
            run = data.parents[2]
            recording = run / "recordings" / f"route_{metadata['route_id']}_rep0.log"
            if not recording.exists():
                recording = run.with_suffix(".log")
            native = vehicle_frames(recording, next(iter(hero_ids)))
            aligned, alignment = align_world_frames(native, world)
            contacts = recorder_contacts(recording)
            if not contacts["complete"]:
                raise ValueError("Partial recorder cannot establish ordered control horizon")
            labels, shard_info, interventions = [], [], []
            ack_differences = []
            for shard in sorted(data.glob("shard_*.zip")):
                shard_info.append(dict(path=str(shard), sha256=digest(shard)))
                with zipfile.ZipFile(shard) as archive:
                    for record in shard_records(shard):
                        target = json.loads(archive.read(record["prefix"] + "/targets.json"))
                        actual = aligned[target["facts"]["actuator_feedback_frame"]]["control"]
                        if not np.allclose(actual, target["executed_control"], atol=1e-5, rtol=0):
                            ack_differences.append(record["frame"])
                        labels.append(record["frame"])
                        if target["intervention"] and target["intervention"] != "steering_noise":
                            interventions.append(
                                dict(frame=record["frame"], reason=target["intervention"])
                            )
            if not labels or ack_differences:
                raise ValueError(f"Missing samples or control differences:{ack_differences[:3]}")
            if any(b != a + 2 for a, b in zip(labels, labels[1:])):
                raise ValueError("Committed RGB samples have a10Hz gap")
            outcome = json.loads((data / "outcome.json").read_text())
            accepted.append(
                dict(
                    data=str(data),
                    metadata=metadata,
                    samples=len(labels),
                    first_rgb_frame=labels[0],
                    last_rgb_frame=labels[-1],
                    source_was_imitation_training=str(data) in original_train,
                    mode=metadata["mode"],
                    world_trace_sha256=digest(world_path),
                    recording=str(recording),
                    recording_sha256=digest(recording),
                    alignment=alignment,
                    executed_acknowledgments_match=True,
                    contacts=contacts,
                    interventions=interventions,
                    shards=shard_info,
                    historical_outcome=outcome,
                )
            )
        except Exception as error:
            excluded.append(dict(data=str(data), reason=str(error)))
        atomic_json(
            args.out / "progress.json",
            dict(
                status="running",
                stage="M5 strict factual source inventory",
                frames=i + 1,
                target_frames=len(candidates),
                updated_at=time.time(),
            ),
        )
    locations = sorted(
        {(e["metadata"]["town"], j) for e in accepted for j in e["metadata"]["junction_ids"]}
    )
    held = set()
    for town in sorted(maps):
        group = [key for key in locations if key[0] == town]
        ordered = sorted(
            group, key=lambda key: hashlib.sha256(f"62019:{key[0]}:{key[1]}".encode()).hexdigest()
        )
        held.update(ordered[: math.ceil(len(group) * 0.2)])
    for e in accepted:
        group = {(e["metadata"]["town"], j) for j in e["metadata"]["junction_ids"]}
        if group & held and not group <= held:
            raise ValueError("Prediction source spans train and dynamics-validation locations")
        e["dynamics_role"] = "validation" if group <= held else "train"
    result = dict(
        status="complete",
        accepted=accepted,
        excluded=excluded,
        dynamics_validation_locations=sorted(held),
        train_video_encoder_exposure_disclosed=True,
        model_updates=0,
        total_samples=sum(e["samples"] for e in accepted),
        additional_non_imitation_samples=sum(
            e["samples"] for e in accepted if not e["source_was_imitation_training"]
        ),
        request_sha256=digest(args.out / "request.json"),
    )
    atomic_json(args.out / "manifest.json", result)
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            stage="M5 strict factual source inventory",
            frames=len(candidates),
            target_frames=len(candidates),
            updated_at=time.time(),
        ),
    )
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("accepted", "excluded")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
