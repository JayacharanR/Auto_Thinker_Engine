#!/usr/bin/env python3
"""Persistent causal features from verified factual options; no fit or deployment."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.contracts import VIEWS  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa:E402
from src.navigation.first_event_data import first_contact_targets, latent_target_mask  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def save(path, **arrays):
    with path.with_suffix(".part").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(".part").replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / "outputs/navigation_interventions_verified_20261021/manifest.json"
    manifest = read(manifest_path)
    receipt = read(manifest_path.parent / "assessment.json")
    if receipt["manifest_sha256"] != digest(manifest_path) or manifest["accepted"] != 68:
        raise ValueError("Only the audited68recording corpus")
    encoder_path = ROOT / (
        "outputs/navigation_repaired_study_v4_20261020/jepa_pretraining/encoder_descriptor.json"
    )
    detector_path = ROOT / "outputs/navigation_models/navigation_tracking_fpn_20261019.json"
    paths = [
        Path(__file__),
        manifest_path,
        encoder_path,
        detector_path,
        ROOT / "reports/navigation_first_event_cache_protocol_20261021.md",
        *[
            ROOT / "src/navigation" / name
            for name in (
                "dataset.py",
                "contracts.py",
                "grid_encoders.py",
                "rectangular_jepa.py",
                "road_user_observations.py",
                "road_user_fpn.py",
                "vehicle_perception.py",
                "camera_ground.py",
                "first_event_data.py",
                "storage_budget.py",
            )
        ],
        ROOT / "src/jepa/encoder.py",
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
            manifest_sha256=digest(manifest_path),
            encoder_descriptor=read(encoder_path),
            detector_descriptor=read(detector_path),
            target_episodes=68,
            detector_past_timestamps=544,
            maximum_wall_seconds=1200,
            maximum_artifact_bytes=250_000_000,
            no_fitting_or_native_driving=True,
            encoder_forward="BF16;FP16exports",
            detector_forward="OriginalFP32",
        ),
    )
    started, done, status = time.monotonic(), 0, "running"
    records = []

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 1200:
            raise InterruptedError("Owned cache STOPor1200scap")
        if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > 250_000_000:
            raise InterruptedError("Cache250MBartifact cap")

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Shared causal intervention features",
                frames=done,
                target_frames=68,
                phase=phase,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 10)
            import torch
            from PIL import Image

            from src.navigation.grid_encoders import FrozenGridEncoder
            from src.navigation.road_user_observations import FrozenRoadUserObservations

            torch.set_num_threads(2)
            encoder = FrozenGridEncoder(read(encoder_path)).cuda()
            detector = FrozenRoadUserObservations(detector_path, "cuda")
            for episode in manifest["episodes"]:
                check()
                for p, sha in episode["source_hashes"].items():
                    if digest(p) != sha:
                        raise ValueError("Verified native/plan/geometry source changed")
                packets = []
                for shard, sha in sorted(episode["RGB_shards_sha256"].items()):
                    check()
                    if digest(shard) != sha:
                        raise ValueError("Verified raw RGBsource changed")
                    packets.extend(load_policy_sample(shard, r) for r in shard_records(shard))
                if (
                    len(packets) != 80
                    or packets[40].frame != episode["original_plan"]["branch_frame"]
                ):
                    raise ValueError("Exact causal branch RGB identity required")
                # All forwards take policy packets/RGB only, never targets/world traces.
                observed = [detector.observe(p) for p in packets[33:41]]
                rgb = np.stack(
                    [
                        np.stack(
                            [
                                np.asarray(
                                    Image.fromarray(p.rgb[v]).resize(
                                        (256, 144), Image.Resampling.BILINEAR
                                    )
                                ).transpose(2, 0, 1)
                                for v in VIEWS
                            ]
                        )
                        for p in packets
                    ]
                )
                target_indices = [40, 45, 50, 55, 60, 65, 70, 75]
                indices = np.array([np.arange(i - 7, i + 1) for i in target_indices])
                video = torch.from_numpy(rgb[indices].copy()).cuda().permute(0, 2, 1, 3, 4, 5)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = encoder.video_features(video).reshape(8, 48, 384).half().cpu().numpy()
                destination = args.out / "episodes" / episode["case"]["case_id"]
                destination.mkdir(parents=True)
                save(
                    destination / "inputs.npz",
                    current_tokens=tokens[0],
                    road_users=np.stack([o["values"] for o in observed]),
                    presence=np.stack([o["presence"] for o in observed]),
                    past_ego=np.stack([p.ego_motion for p in packets[33:41]]),
                    timestamps=np.array([p.timestamp for p in packets[33:41]]),
                    frames=np.array([p.frame for p in packets[33:41]]),
                    navigation=packets[40].navigation_waypoints,
                    ego=packets[40].ego_motion,
                    own_controls=np.array(
                        episode["original_plan"]["actual_previous_controls"], dtype=np.float32
                    ),
                )
                save(destination / "latent_targets.npz", tokens=tokens[1:])
                atomic_json(
                    destination / "observations.json",
                    [
                        dict(
                            frame=o["frame"],
                            timestamp=o["timestamp"],
                            observations=o["observations"],
                            unknown_ground_contacts=o["unknown_ground_contacts"],
                        )
                        for o in observed
                    ],
                )
                # Attach outcome labels only AFTER committing all pixel-derived features.
                plan = episode["original_plan"]
                selected = next(
                    p
                    for p in plan["slate"]
                    if p["path"] == plan["selected_path"] and p["speed"] == plan["selected_speed"]
                )
                support = episode["prefix_support"]
                known = int(support.get("matching_commands_before_safety", 0))
                contact = support.get("first_contact_tick")
                event, valid = first_contact_targets(known, contact)
                save(
                    destination / "targets.npz",
                    event=event,
                    valid=valid,
                    latent_valid=latent_target_mask(known, contact),
                )
                save(
                    destination / "proposals.npz",
                    ordered_controls=np.array(selected["ordered_controls"], np.float32),
                    all_controls=np.array(
                        [p["ordered_controls"] for p in plan["slate"]], np.float32
                    ),
                    all_forecasts=np.array(
                        [p["forecast_states"] for p in plan["slate"]], np.float32
                    ),
                )
                files = {p.name: digest(p) for p in destination.iterdir() if p.is_file()}
                records.append(
                    dict(
                        case=episode["case"],
                        source_role=episode["source_role"],
                        context_id=episode["context_id"],
                        mechanism=episode["mechanism"],
                        town=episode["town"],
                        route_id=episode["route_id"],
                        output=str(destination),
                        files_sha256=files,
                        input_pixel_features_saved_before_targets=True,
                        observed_hazard_ticks=int(valid.sum()),
                        positive_first_event=int(event.sum()),
                        latent_horizons_valid=latent_target_mask(known, contact).tolist(),
                        detector_boxes=sum(int(o["presence"].sum()) for o in observed),
                        unknown_boxes=sum(o["unknown_ground_contacts"] for o in observed),
                    )
                )
                atomic_json(args.out / "records.json", records)
                done += 1
                progress("Same complete RGB-road-user history forCV/RSSM/JEPA; no training")
            if any(digest(ROOT / p) != sha for p, sha in hashes.items()):
                raise ValueError("Registered feature source changed during cache")
            atomic_json(
                args.out / "assessment.json",
                dict(
                    episodes=68,
                    by_role={
                        role: dict(
                            episodes=sum(r["source_role"] == role for r in records),
                            positive_first_events=sum(
                                r["positive_first_event"]
                                for r in records
                                if r["source_role"] == role
                            ),
                        )
                        for role in ("train", "calibration")
                    },
                    no_model_fit_or_driving_promotion=True,
                    all_eighth_latent_targets_unknown=True,
                    records_sha256=digest(args.out / "records.json"),
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status = "complete"
            progress("All68causal inputs/labels cached; native qualification remains open")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
