#!/usr/bin/env python3
"""Score a frozen RGB detector against separately read same-frame setup actors."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.contracts import SensorPacket  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.sensors import calibrations  # noqa:E402
from src.navigation.vehicle_perception import FrozenRoadUsers  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fpn", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve prior perception diagnostic")
    descriptor = ROOT / "outputs/navigation_models/navigation_tracking_detector_20261019.json"
    if args.fpn:
        descriptor = ROOT / "outputs/navigation_models/navigation_tracking_fpn_20261019.json"
    sources = [
        Path(__file__),
        ROOT / "src/navigation/vehicle_perception.py",
        ROOT / "src/navigation/camera_ground.py",
        ROOT / "reports/navigation_cv_perception_protocol_20261019.md",
        ROOT / "src/navigation/road_user_fpn.py",
        ROOT / "reports/navigation_cv_fpn_protocol_20261019.md",
        descriptor,
    ]
    for path in sources:
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(
        args.out / "request.json",
        dict(
            root=str(args.root.resolve()),
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
            device=args.device,
            cuda_warmups=3 if args.device == "cuda" else 0,
            fixed_origin_match_gate_m=10.0,
            geometry_reference_radius_m=60.0,
            scope="Separate M5 diagnostic; not visibility-certified recall/AP or training",
        ),
    )
    torch.set_num_threads(2)
    if args.fpn:
        from src.navigation.road_user_fpn import FrozenRoadUsersFPN

        detector = FrozenRoadUsersFPN(descriptor, device=args.device)
    else:
        detector = FrozenRoadUsers(descriptor, device=args.device)
    calibration = calibrations(yaml.safe_load((ROOT / "configs/navigation.yaml").read_text()))
    rows = []
    for output in sorted(args.root.iterdir()):
        if not output.is_dir() or not (output / "manifest.json").exists():
            continue
        episode = next((output / "episodes").iterdir())
        world = {
            r["frame"]: r
            for r in map(json.loads, (episode / "world_trace.jsonl").read_text().splitlines())
        }
        trace = {
            r["frame"]: r
            for r in map(
                json.loads, (episode / "controlled_rule/setup_trace.jsonl").read_text().splitlines()
            )
        }
        manifest = json.loads((episode / "controlled_rule/raw_rgb_manifest.json").read_text())
        for sample in manifest["samples"]:
            if (args.out / "STOP").exists():
                atomic_json(
                    args.out / "progress.json",
                    dict(
                        status="stopped",
                        stage="M5 frozen vehicle perception",
                        frames=len(rows),
                        target_frames=165,
                        updated_at=time.time(),
                    ),
                )
                return
            frame, timestamp = sample["frame"], sample["timestamp"]
            state = trace[frame]
            yaw = np.deg2rad(state["ego"][3])
            packet = SensorPacket(
                frame,
                timestamp,
                {v: np.asarray(Image.open(p).convert("RGB")) for v, p in sample["images"].items()},
                calibration,
                np.array([[5.0, 0.0], [10.0, 0.0]], dtype=np.float32),
                np.array([state["speed_mps"], yaw, 0.0], dtype=np.float32),
            )
            if args.device == "cuda":
                if not rows:
                    for _ in range(3):
                        detector.detect(packet)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.monotonic()
            predicted, diagnostics = detector.detect(packet)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            # Reference is read AFTER inference and never enters the packet.
            actors = [
                a
                for a in world[frame]["actors"]
                if a["type"].startswith(("vehicle.", "walker.")) and a["role"] != "hero"
            ]
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            references = []
            for actor in actors:
                local = (np.array(actor["pose"][:2]) - state["ego"][:2]) @ rotation
                if np.linalg.norm(local) <= 60:
                    references.append(
                        dict(id=actor["id"], role=actor["role"], origin_xy=local.tolist())
                    )
            errors, matches = [], []
            if predicted and references:
                distances = np.linalg.norm(
                    np.array([r["origin_xy"] for r in references])[:, None]
                    - np.array([r["position_xy"] for r in predicted])[None],
                    axis=-1,
                )
                for i, j in zip(*linear_sum_assignment(distances)):
                    if distances[i, j] <= 10:
                        errors.append(float(distances[i, j]))
                        matches.append(
                            dict(
                                reference_index=int(i),
                                prediction_index=int(j),
                                origin_error_m=float(distances[i, j]),
                            )
                        )
            rows.append(
                dict(
                    case=output.name,
                    frame=frame,
                    timestamp=timestamp,
                    rgb_sha256={v: digest(p) for v, p in sample["images"].items()},
                    predictions=[
                        dict(
                            r,
                            position_xy=np.asarray(r["position_xy"]).tolist(),
                            sigma_m=float(r["sigma_m"]),
                        )
                        for r in predicted
                    ],
                    diagnostics=diagnostics,
                    references=references,
                    matches=matches,
                    elapsed_s=elapsed,
                    nearby_actor_count=len(references),
                    matched_actor_count=len(matches),
                    unmatched_predictions=len(predicted) - len(matches),
                )
            )
            with (args.out / "predictions.jsonl").open("a") as stream:
                stream.write(json.dumps(rows[-1]) + "\n")
            atomic_json(
                args.out / "progress.json",
                dict(
                    status="running",
                    stage="M5 frozen vehicle perception",
                    frames=len(rows),
                    target_frames=165,
                    updated_at=time.time(),
                ),
            )
    errors = [m["origin_error_m"] for r in rows for m in r["matches"]]
    nearby = sum(r["nearby_actor_count"] for r in rows)
    matched = sum(r["matched_actor_count"] for r in rows)
    result = dict(
        status="complete",
        frames=len(rows),
        nearby_actor_instances=nearby,
        matched_origin_instances=matched,
        geometric_availability=matched / nearby if nearby else None,
        matched_origin_error_median_m=float(np.median(errors)) if errors else None,
        matched_origin_error_p95_m=float(np.percentile(errors, 95)) if errors else None,
        detector_p95_ms=float(np.percentile([r["elapsed_s"] for r in rows], 95) * 1000),
        device=args.device,
        unmatched_prediction_instances=sum(r["unmatched_predictions"] for r in rows),
        local_carla_training_labels=0,
        visibility_certified=False,
        planner_qualified=False,
        request_sha256=digest(args.out / "request.json"),
        rows=rows,
    )
    result[f"detector_{args.device}_p95_ms"] = result["detector_p95_ms"]
    if args.device == "cuda":
        result["peak_allocated_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    atomic_json(args.out / "assessment.json", result)
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            stage="M5 frozen vehicle perception",
            frames=len(rows),
            target_frames=165,
            updated_at=time.time(),
        ),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
