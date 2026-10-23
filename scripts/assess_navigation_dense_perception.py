#!/usr/bin/env python3
"""Frozen RGB detector versus visible instance masks on fixed TRAIN calibration data."""

import argparse
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.navigation_label_rig import decode_depth, decode_instances, digest_array  # noqa:E402
from src.navigation.camera_ground import ground_contact, mount_rotation  # noqa:E402
from src.navigation.contracts import VIEWS  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402

COCO_GROUP = {
    1: "person",
    2: "two_wheeler",
    3: "vehicle",
    4: "two_wheeler",
    6: "vehicle",
    8: "vehicle",
}
SEMANTIC_GROUP = {
    12: "person",
    13: "person",
    14: "vehicle",
    15: "vehicle",
    16: "vehicle",
    17: "vehicle",
    18: "two_wheeler",
    19: "two_wheeler",
}


def read(path):
    return json.loads(Path(path).read_text())


def box_iou(first, second):
    first = np.asarray(first, float).reshape(-1, 4)
    second = np.asarray(second, float).reshape(-1, 4)
    lower = np.maximum(first[:, None, :2], second[None, :, :2])
    upper = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.maximum(0, upper - lower).prod(-1)
    a = np.maximum(0, first[:, 2:] - first[:, :2]).prod(-1)
    b = np.maximum(0, second[:, 2:] - second[:, :2]).prod(-1)
    return intersection / np.maximum(a[:, None] + b[None] - intersection, 1e-12)


def match_boxes(references, predictions):
    if not references or not predictions:
        return []
    overlap = box_iou([r["box"] for r in references], [p["box"] for p in predictions])
    groups = np.array([[a["group"] == b["group"] for b in predictions] for a in references])
    accepted = (overlap >= 0.3) & groups
    # The unit reward dominates the <=100/image detector's IoU tiebreak. This
    # maximizes eligible one-to-one matches, rather than greedy largest IoU.
    cost = -accepted.astype(float) * (1 + overlap / 1000)
    rows, columns = linear_sum_assignment(cost)
    return [
        dict(reference=int(i), prediction=int(j), iou=float(overlap[i, j]))
        for i, j in zip(rows, columns)
        if accepted[i, j]
    ]


def lift_pixel(u, v, depth, calibration, radial=False):
    ray = np.array(
        [
            1.0,
            (u - calibration.width / 2) / calibration.intrinsic[0, 0],
            -(v - calibration.height / 2) / calibration.intrinsic[1, 1],
        ]
    )
    if radial:
        ray /= np.linalg.norm(ray)
    return np.array(calibration.xyz) + depth * ray @ mount_rotation(calibration).T


def visible_objects(raw, depth):
    semantic, instance_pair = decode_instances(raw)
    identities = instance_pair[..., 0].astype(np.uint32) + 256 * instance_pair[..., 1].astype(
        np.uint32
    )
    included = []
    tiny = []
    for group in sorted(set(SEMANTIC_GROUP.values())):
        valid = np.isin(semantic, [k for k, v in SEMANTIC_GROUP.items() if v == group])
        for identity in np.unique(identities[valid]):
            mask = valid & (identities == identity)
            ys, xs = np.nonzero(mask)
            pixels = len(xs)
            if not pixels:
                continue
            foot_y = int(ys.max())
            foot_x = float(np.median(xs[ys == foot_y]))
            depths = depth[mask]
            # The median can fall in a disconnected silhouette hole; use an
            # actually observed bottom-row pixel nearest that median instead.
            bottom = xs[ys == foot_y]
            foot_x = int(bottom[np.argmin(abs(bottom - foot_x))])
            foot_depth = float(depth[foot_y, foot_x])
            row = dict(
                group=group,
                render_actor_low16=int(identity),
                visible_pixels=pixels,
                semantic_classes=sorted(map(int, np.unique(semantic[mask]))),
                box=[int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                median_encoded_depth_m=float(np.median(depths)),
                observed_bottom_pixel=[foot_x, foot_y],
                bottom_pixel_depth_m=foot_depth,
                image_truncated=bool(
                    xs.min() == 0 or xs.max() == 511 or ys.min() == 0 or ys.max() == 287
                ),
            )
            (included if pixels >= 8 else tiny).append(row)
    return included, tiny


def original_packet(reference):
    matches = [
        r for r in shard_records(reference["rgb_shard"]) if r["prefix"] == reference["rgb_prefix"]
    ]
    if len(matches) != 1:
        raise ValueError("Exact independently verified original RGBrecord required")
    packet = load_policy_sample(reference["rgb_shard"], matches[0])
    if packet.frame != reference["frame"]:
        raise ValueError("Original label/RGBframe identity changed")
    return packet


def label_arrays(reference):
    with zipfile.ZipFile(reference["label_shard"]) as archive:
        prefix = reference["label_prefix"]
        metadata = json.loads(archive.read(prefix + "/metadata.json"))
        values = {
            k: np.load(io.BytesIO(archive.read(prefix + "/" + k + ".npy")), allow_pickle=False)
            for k in metadata["pixel_sha256"]
        }
    if metadata["frame"] != reference["frame"] or any(
        digest_array(a) != metadata["pixel_sha256"][k] for k, a in values.items()
    ):
        raise ValueError("Verified raw label frame or pixels changed")
    return values


def detector_prediction(detector, packet):
    capture = []

    def observe(module, inputs, outputs):
        capture.extend([{k: v.detach().cpu().numpy() for k, v in row.items()} for row in outputs])

    handle = detector.model.register_forward_hook(observe)
    try:
        fused, diagnostics = detector.detect(
            packet
        )  # Original RGB-only production method unchanged.
    finally:
        handle.remove()
    if len(capture) != 3:
        raise ValueError("One raw prediction per original RGBview required")
    raw = []
    projectable = []
    for view, row in zip(VIEWS, capture):
        for box, score, label in zip(row["boxes"], row["scores"], row["labels"]):
            if int(label) not in COCO_GROUP or score < 0.35:
                continue
            value = dict(
                view=view,
                box=box.tolist(),
                score=float(score),
                label=int(label),
                group=COCO_GROUP[int(label)],
            )
            raw.append(value)
            xy = ground_contact(box, packet.calibration[view])
            if xy is not None:
                projectable.append(dict(value, position_xy=xy.tolist()))
    fused = [
        {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items()}
        | dict(group=COCO_GROUP[row["label"]])
        for row in fused
    ]
    return dict(raw=raw, projectable=projectable, fused=fused, production_diagnostics=diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = read(args.data / "manifest.json")
    receipt = read(args.data / "assessment.json")
    if (
        digest(args.data / "manifest.json") != receipt["manifest_sha256"]
        or not receipt["every_included_source_independently_verified"]
    ):
        raise ValueError("Only the separate verified11source manifest may supply calibration data")
    sources = [r for r in manifest["sources"] if r["item"]["role"] == "calibration"]
    if len(sources) != 2 or sum(r["label_timestamps"] for r in sources) != 78:
        raise ValueError("Use both original predeclared calibration sources and every78timestamp")
    descriptor = ROOT / "outputs/navigation_models/navigation_tracking_fpn_20261019.json"
    files = [
        Path(__file__),
        ROOT / "scripts/navigation_label_rig.py",
        descriptor,
        ROOT / "reports/navigation_dense_perception_diagnosis_protocol_20261020.md",
        *[
            ROOT / "src/navigation" / name
            for name in (
                "camera_ground.py",
                "road_user_fpn.py",
                "vehicle_perception.py",
                "dataset.py",
                "contracts.py",
            )
        ],
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    for p in files:
        target = args.out / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    atomic_json(
        args.out / "request.json",
        dict(
            data_manifest_sha256=digest(args.data / "manifest.json"),
            source_hashes=hashes,
            role="calibration",
            timestamps=78,
            views=234,
            source_locations=2,
            detector_unchanged=True,
            score_threshold=0.35,
            IoU_gate=0.3,
            minimum_visible_instance_pixels=8,
            no_threshold_sweep=True,
            no_updates=True,
            maximum_wall_seconds=900,
            threads=2,
            groups=COCO_GROUP,
            semantic_groups=SEMANTIC_GROUP,
            scope=(
                "Post-hoc engineering diagnosis, not a final benchmark or isolated model comparison"
            ),
        ),
    )
    started = time.monotonic()
    done = 0
    status = "running"

    def check():
        if (args.out / "STOP").exists():
            raise InterruptedError("Dense perception diagnostic STOP")
        if time.monotonic() - started > 900:
            raise InterruptedError("Dense perception diagnostic900s cap")

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M5 visible-instance perception audit",
                frames=done,
                target_frames=78,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 10)
            import torch

            from src.navigation.road_user_fpn import FrozenRoadUsersFPN

            torch.set_num_threads(2)
            detector = FrozenRoadUsersFPN(descriptor, "cuda")
            records = []
            for source in sources:
                for path, sha in (
                    source["label_shards_sha256"] | source["original_source_shards_sha256"]
                ).items():
                    check()
                    if digest(path) != sha:
                        raise ValueError("Independently verified input shard changed")
                for reference in source["label_sources"]:
                    check()
                    packet = original_packet(reference)
                    if done == 0:
                        for _ in range(3):
                            detector.detect(packet)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    before = time.monotonic()
                    prediction = detector_prediction(detector, packet)
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - before
                    # Commit RGB-only predictions BEFORE loading any instance/depth target.
                    with (args.out / "predictions.jsonl").open("a") as log:
                        log.write(
                            json.dumps(
                                dict(
                                    route_id=source["item"]["route_id"],
                                    frame=packet.frame,
                                    prediction=prediction,
                                    inference_seconds=elapsed,
                                ),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    arrays = label_arrays(reference)
                    row = dict(
                        route_id=source["item"]["route_id"],
                        town=source["item"]["town"],
                        frame=packet.frame,
                        inference_seconds=elapsed,
                        views={},
                        stage_object_groups={},
                        depth_ground_diagnostics=[],
                    )
                    ids_all = set()
                    ids_stage = {name: set() for name in ("raw", "projectable", "fused")}
                    for view in VIEWS:
                        if not np.array_equal(arrays["rgb_" + view], packet.rgb[view]):
                            raise ValueError(
                                "Calibration labels do not correspond to original input pixels"
                            )
                        depth = decode_depth(arrays[view + "_depth"])
                        objects, tiny = visible_objects(arrays[view + "_instance"], depth)
                        ids_all.update((r["group"], r["render_actor_low16"]) for r in objects)
                        evaluations = {}
                        for name in ("raw", "projectable", "fused"):
                            predicted = [p for p in prediction[name] if p["view"] == view]
                            matched = match_boxes(objects, predicted)
                            ids_stage[name].update(
                                (
                                    objects[m["reference"]]["group"],
                                    objects[m["reference"]]["render_actor_low16"],
                                )
                                for m in matched
                            )
                            evaluations[name] = dict(
                                predictions=len(predicted),
                                matches=matched,
                                matched=len(matched),
                                unmatched_predictions=len(predicted) - len(matched),
                            )
                        row["views"][view] = dict(
                            visible_objects=objects, tiny_objects=tiny, stages=evaluations
                        )
                        calibration = packet.calibration[view]
                        # Both interpretations are retained; no unverified claim
                        # that a visible surface is the centre of a full actor.
                        for obj in objects:
                            d = obj["bottom_pixel_depth_m"]
                            if not 0 < d < 60:
                                continue
                            x, y = obj["observed_bottom_pixel"]
                            flat = ground_contact(obj["box"], calibration)
                            if flat is None:
                                continue
                            axial = lift_pixel(x + 0.5, y + 0.5, d, calibration)
                            radial = lift_pixel(x + 0.5, y + 0.5, d, calibration, radial=True)
                            row["depth_ground_diagnostics"].append(
                                dict(
                                    view=view,
                                    group=obj["group"],
                                    encoded_bottom_depth_m=d,
                                    truncated=obj["image_truncated"],
                                    axial_surface_height_m=float(axial[2]),
                                    radial_surface_height_m=float(radial[2]),
                                    flat_to_axial_visible_surface_xy_m=float(
                                        np.linalg.norm(flat - axial[:2])
                                    ),
                                    flat_to_radial_visible_surface_xy_m=float(
                                        np.linalg.norm(flat - radial[:2])
                                    ),
                                )
                            )
                    row["visible_object_groups"] = len(ids_all)
                    row["stage_object_groups"] = {k: len(v) for k, v in ids_stage.items()}
                    records.append(row)
                    done += 1
                    with (args.out / "scored_records.jsonl").open("a") as log:
                        log.write(json.dumps(row, allow_nan=False) + "\n")
                    progress(
                        "Fixed RGBdetector; independently read visible masks/depth for scoring only"
                    )
            counts = {}
            for stage in ("raw", "projectable", "fused"):
                counts[stage] = dict(
                    visible_instance_views=sum(
                        len(v["visible_objects"]) for r in records for v in r["views"].values()
                    ),
                    matched_instance_views=sum(
                        v["stages"][stage]["matched"] for r in records for v in r["views"].values()
                    ),
                    predictions=sum(
                        v["stages"][stage]["predictions"]
                        for r in records
                        for v in r["views"].values()
                    ),
                    unmatched_predictions=sum(
                        v["stages"][stage]["unmatched_predictions"]
                        for r in records
                        for v in r["views"].values()
                    ),
                    visible_object_groups=sum(r["visible_object_groups"] for r in records),
                    matched_object_groups=sum(r["stage_object_groups"][stage] for r in records),
                )
            geometry = [d for r in records for d in r["depth_ground_diagnostics"]]
            diagnostic = (
                {
                    key: dict(
                        count=len(geometry),
                        median=float(np.median([r[key] for r in geometry])),
                        p95=float(np.percentile([r[key] for r in geometry], 95)),
                    )
                    for key in (
                        "axial_surface_height_m",
                        "radial_surface_height_m",
                        "flat_to_axial_visible_surface_xy_m",
                        "flat_to_radial_visible_surface_xy_m",
                    )
                }
                if geometry
                else {}
            )
            atomic_json(
                args.out / "assessment.json",
                dict(
                    counts=counts,
                    depth_geometry=diagnostic,
                    images=done * 3,
                    timestamps=done,
                    independent_locations=2,
                    tiny_instance_views=sum(
                        len(v["tiny_objects"]) for r in records for v in r["views"].values()
                    ),
                    mean_detector_ms=float(
                        np.mean([r["inference_seconds"] for r in records]) * 1000
                    ),
                    p95_detector_ms=float(
                        np.percentile([r["inference_seconds"] for r in records], 95) * 1000
                    ),
                    peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                    prediction_sha256=digest(args.out / "predictions.jsonl"),
                    scored_records_sha256=digest(args.out / "scored_records.jsonl"),
                    perception_or_planner_qualified=False,
                    no_model_updates=True,
                    limitations=[
                        "Visible-mask boxes are occluded-surface boxes, not amodal actor boxes.",
                        "Low16 Unreal actor IDs may alias; no identity-certified tracking.",
                        "Repeated snapshots at two familiar TRAIN locations; correlated.",
                        "Unmatched predictions use included >=8px masks, not exhaustive object AP.",
                        "Geometry compares visible surfaces, not actor-centre 3D accuracy.",
                        "No signal applicability, risk calibration or autonomous driving result.",
                    ],
                ),
            )
            status = "complete"
            progress("All fixed calibration frames assessed; no tuning or promotion")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
