#!/usr/bin/env python3
"""Prepare factual controls/masks, with future targets isolated from policy inputs."""

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import POLICY_KEYS, shard_records  # noqa:E402
from src.navigation.dynamics_data import factual_window  # noqa:E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve every factual-window attempt")
    sources = json.loads((args.sources / "manifest.json").read_text())
    args.out.mkdir(parents=True)
    files = [
        Path(__file__),
        ROOT / "src/navigation/dynamics_data.py",
        ROOT / "src/navigation/recorder_actions.py",
        ROOT / "reports/navigation_dynamics_data_protocol_20261019.md",
    ]
    for source in files:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            source_manifest_sha256=digest(args.sources / "manifest.json"),
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in files},
            stride_timestamps=5,
            minimum_past_timestamps=8,
            models_trained=0,
        ),
    )
    windows, episodes, offset = [], [], 0
    for episode_id, source in enumerate(sources["accepted"]):
        if (args.out / "STOP").exists():
            raise InterruptedError("Factual window preparation STOP")
        data = Path(source["data"])
        if digest(source["recording"]) != source["recording_sha256"]:
            raise ValueError("Factual recorder changed")
        world = [
            json.loads(line)
            for line in (data.parent / "world_trace.jsonl").read_text().splitlines()
        ]
        native = vehicle_frames(source["recording"], source["alignment"]["hero_id"])
        aligned, alignment = align_world_frames(native, world)
        world_by_frame = {r["frame"]: r for r in world}
        policy, interventions = [], []
        for shard in source["shards"]:
            if digest(shard["path"]) != shard["sha256"]:
                raise ValueError("Factual source shard changed")
            with zipfile.ZipFile(shard["path"]) as archive:
                for record in shard_records(shard["path"]):
                    p = json.loads(archive.read(record["prefix"] + "/policy.json"))
                    if set(p) != POLICY_KEYS:
                        raise ValueError("Policy inputs contain unapproved fields")
                    policy.append(p)
        interventions.extend(r["frame"] - 1 for r in source["interventions"])
        dagger = data.parent / "dagger_trace.jsonl"
        if dagger.exists():
            for row in map(json.loads, dagger.read_text().splitlines()):
                if row["intervention"]:
                    interventions.append(row["frame"])
        frames = np.array([p["frame"] for p in policy])
        navigation = np.array([p["navigation_waypoints"] for p in policy], dtype=np.float32)
        ego = np.array([p["ego_motion"] for p in policy], dtype=np.float32)
        positions = np.array(
            [
                next(
                    a["pose"][:3]
                    for a in world_by_frame[p["frame"]]["actors"]
                    if a["role"] == "hero"
                )
                for p in policy
            ]
        )
        contact_frames = [
            world[0]["frame"] + c["frame"] - 1 - alignment["offset"]
            for c in source["contacts"]["raw_samples"]
        ]
        record = source["historical_outcome"].get("evaluator_record")
        red_zero = bool(
            record and "crash" not in record["status"].lower() and record["red_violations"] == 0
        )
        first_safety = min(interventions) if interventions else None
        first_window = len(windows)
        for index in range(7, len(policy), 5):
            row = factual_window(
                index,
                frames,
                navigation,
                ego,
                positions,
                aligned,
                contact_frames,
                first_safety_frame=first_safety,
                native_red_zero_observed=red_zero,
            )
            if not row["latent_valid"].any():
                continue
            row.update(
                input_index=offset + index,
                future_indices=row["future_indices"] + offset,
                episode_id=episode_id,
                is_validation=source["dynamics_role"] == "validation",
            )
            windows.append(row)
        episodes.append(
            dict(
                data=str(data),
                start=offset,
                end=offset + len(policy),
                samples=len(policy),
                dynamics_role=source["dynamics_role"],
                metadata=source["metadata"],
                source_shards=source["shards"],
                first_safety_world_frame=first_safety,
                window_start=first_window,
                window_end=len(windows),
            )
        )
        offset += len(policy)
        atomic_json(
            args.out / "progress.json",
            dict(
                status="running",
                stage="M5 factual target windows",
                frames=episode_id + 1,
                target_frames=len(sources["accepted"]),
                updated_at=time.time(),
            ),
        )
    arrays = {key: np.stack([row[key] for row in windows]) for key in windows[0]}
    required = sum(a.nbytes for a in arrays.values())
    if shutil.disk_usage(args.out).free < 25e9 + required + 1e6:
        raise RuntimeError("Factual arrays would breach25GB reserve")
    np.savez(args.out / "windows.npz", **arrays)
    manifest = dict(
        episodes=episodes,
        samples=offset,
        windows=len(windows),
        train_windows=int((~arrays["is_validation"]).sum()),
        validation_windows=int(arrays["is_validation"].sum()),
        collision_positive_horizons=int(
            ((arrays["outcomes"][..., 0] > 0) & (arrays["cost_valid"][..., 0] > 0)).sum()
        ),
        red_positive_horizons=0,
        red_limitation=(
            "No positive red-crossing labels. "
            "This corpus alone cannot qualify a learned red-risk head."
        ),
        window_arrays_sha256=digest(args.out / "windows.npz"),
        request_sha256=digest(args.out / "request.json"),
        no_model_fitting=True,
    )
    atomic_json(args.out / "manifest.json", manifest)
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            stage="M5 factual target windows",
            frames=len(episodes),
            target_frames=len(episodes),
            updated_at=time.time(),
        ),
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
