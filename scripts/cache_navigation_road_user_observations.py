#!/usr/bin/env python3
"""Cache a common RGB-only pack, retaining every unlocated detector observation."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_navigation_dense_perception import (  # noqa:E402
    detector_prediction,
    original_packet,
)
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.road_user_observations import (  # noqa:E402
    FIELDS,
    SCHEMA,
    FrozenRoadUserObservations,
    observation_pack,
)
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def raw_rows(rows):
    return [{key: row[key] for key in ("view", "box", "label", "score")} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    data = ROOT / "outputs/navigation_dense_labels_verified_20261020"
    diagnostic = ROOT / "outputs/navigation_dense_perception_diagnosis_20261020"
    manifest, receipt = read(data / "manifest.json"), read(data / "assessment.json")
    assessment, original_request = (
        read(diagnostic / "assessment.json"),
        read(diagnostic / "request.json"),
    )
    descriptor = ROOT / "outputs/navigation_models/navigation_tracking_fpn_20261019.json"
    if (
        digest(data / "manifest.json") != receipt["manifest_sha256"]
        or not receipt["every_included_source_independently_verified"]
        or digest(diagnostic / "predictions.jsonl") != assessment["prediction_sha256"]
        or digest(descriptor)
        != original_request["source_hashes"][str(descriptor.relative_to(ROOT))]
        or sum(r["label_timestamps"] for r in manifest["sources"]) != 429
    ):
        raise ValueError(
            "Unchanged verified11source data and original calibration predictions required"
        )
    predictions = [
        json.loads(line) for line in (diagnostic / "predictions.jsonl").read_text().splitlines()
    ]
    reused = {(r["route_id"], r["frame"]): raw_rows(r["prediction"]["raw"]) for r in predictions}
    paths = [
        Path(__file__),
        descriptor,
        data / "manifest.json",
        ROOT / "reports/navigation_shared_observations_protocol_20261020.md",
        ROOT / "scripts/assess_navigation_dense_perception.py",
        *[
            ROOT / "src/navigation" / name
            for name in (
                "road_user_observations.py",
                "road_user_fpn.py",
                "vehicle_perception.py",
                "camera_ground.py",
                "contracts.py",
                "dataset.py",
                "storage_budget.py",
            )
        ],
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for path in paths:
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(
        args.out / "request.json",
        dict(
            schema=SCHEMA,
            fields=FIELDS,
            source_hashes=hashes,
            target_timestamps=429,
            train_timestamps=351,
            calibration_timestamps=78,
            no_model_updates=True,
            original_prediction_sha256=assessment["prediction_sha256"],
            maximum_wall_seconds=900,
            maximum_artifact_bytes=200_000_000,
            equal_future_availability=["constant_velocity", "rssm", "jepa"],
        ),
    )
    status, done, started = "running", 0, time.monotonic()
    records, pending, chunks = [], [], []
    wrapper = None
    parity = None

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Shared RGB road-user cache",
                frames=done,
                target_frames=429,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 900:
            raise InterruptedError("Own STOP or shared-cache900s cap")
        if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > 200_000_000:
            raise InterruptedError("Shared-cache200MB artifact cap")

    def commit():
        if not pending:
            return
        path = args.out / f"features_{len(chunks):04d}.npz"
        part = path.with_suffix(".npz.part")
        with part.open("xb") as stream:
            np.savez_compressed(
                stream,
                values=np.stack([p["values"] for p in pending]),
                presence=np.stack([p["presence"] for p in pending]),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, path)
        chunks.append(dict(path=str(path), sha256=digest(path), timestamps=len(pending)))
        pending.clear()
        atomic_json(args.out / "chunks.json", chunks)

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 10)
            import torch

            torch.set_num_threads(2)
            progress("Verify source pixels; retain unknown observations, no labels in forward")
            for source in manifest["sources"]:
                for path, sha in source["original_source_shards_sha256"].items():
                    check()
                    if digest(path) != sha:
                        raise ValueError("Verified original policyRGBshard changed")
                for reference in source["label_sources"]:
                    check()
                    packet = original_packet(reference)
                    role = source["item"]["role"]
                    if role == "calibration":
                        key = (source["item"]["route_id"], packet.frame)
                        pack = observation_pack(packet, reused[key])
                        origin = "original_raw_calibration_predictions_reused"
                    elif role == "train":
                        if wrapper is None:
                            wrapper = FrozenRoadUserObservations(descriptor, "cuda")
                            for _ in range(3):
                                wrapper.observe(packet)
                            old = detector_prediction(wrapper.detector, packet)
                            first = observation_pack(packet, raw_rows(old["raw"]))
                            pack = wrapper.observe(packet)
                            parity = dict(
                                values_bitwise_equal=np.array_equal(
                                    first["values"], pack["values"]
                                ),
                                presence_equal=np.array_equal(first["presence"], pack["presence"]),
                                observations_equal=first["observations"] == pack["observations"],
                                frame=packet.frame,
                                route_id=source["item"]["route_id"],
                            )
                            atomic_json(args.out / "original_forward_parity.json", parity)
                            if not all(
                                parity[k]
                                for k in (
                                    "values_bitwise_equal",
                                    "presence_equal",
                                    "observations_equal",
                                )
                            ):
                                raise ValueError("Raw original/new FPN observation parity failed")
                        else:
                            pack = wrapper.observe(packet)
                        origin = "same_frozen_FP32_FPN"
                    else:
                        raise ValueError("Only original TRAIN/calibration roles")
                    metadata = dict(
                        role=role,
                        town=source["item"]["town"],
                        route_id=source["item"]["route_id"],
                        frame=packet.frame,
                        timestamp=packet.timestamp,
                        rgb_shard=reference["rgb_shard"],
                        rgb_prefix=reference["rgb_prefix"],
                        origin=origin,
                        chunk_index=len(chunks),
                        offset=len(pending),
                        observations=pack["observations"],
                        unknown_ground_contacts=pack["unknown_ground_contacts"],
                        detections=int(pack["presence"].sum()),
                    )
                    with (args.out / "observations.jsonl").open("a") as log:
                        log.write(json.dumps(metadata, allow_nan=False) + "\n")
                    records.append(metadata)
                    pending.append(pack)
                    done += 1
                    if len(pending) == 64:
                        commit()
                    progress(
                        "All accepted boxes, explicit presence and approximate-ground validity"
                    )
            commit()
            if any(digest(ROOT / p) != sha for p, sha in hashes.items()):
                raise ValueError("Source bytes changed during registered cache generation")
            stats = {
                role: dict(
                    timestamps=sum(r["role"] == role for r in records),
                    detections=sum(r["detections"] for r in records if r["role"] == role),
                    unknown_ground_contacts=sum(
                        r["unknown_ground_contacts"] for r in records if r["role"] == role
                    ),
                )
                for role in ("train", "calibration")
            }
            if stats["calibration"]["detections"] != assessment["counts"]["raw"]["predictions"]:
                raise ValueError("Calibration cache must preserve every original raw accepted box")
            atomic_json(
                args.out / "assessment.json",
                dict(
                    stats=stats,
                    schema=SCHEMA,
                    fields=FIELDS,
                    first_train_forward_parity=parity,
                    chunk_sha256={r["path"]: r["sha256"] for r in chunks},
                    metadata_sha256=digest(args.out / "observations.jsonl"),
                    no_label_arrays_loaded=True,
                    no_model_updates=True,
                    cv_rssm_jepa_adoption_pending=True,
                    no_driving_qualification=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status = "complete"
            progress("Shared observations complete; no model fit or planner deployment")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
