#!/usr/bin/env python3
"""Cheap TRAIN-only encoded-depth baselines from frozen RGB box observations."""

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_navigation_dense_perception import (  # noqa:E402
    COCO_GROUP,
    label_arrays,
    match_boxes,
    original_packet,
    visible_objects,
)
from scripts.navigation_label_rig import decode_depth  # noqa:E402
from src.navigation.camera_ground import mount_rotation  # noqa:E402
from src.navigation.contracts import VIEWS  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.road_user_observations import observation_pack  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def geometry_features(values, presence):
    """No target-derived fields: frozen RGB observation values and view index only."""
    rows = []
    for view_index in range(3):
        for value in values[view_index, presence[view_index]]:
            width, height = value[9] - value[7], value[10] - value[8]
            bottom = max(float(value[10]) - 0.5, 1e-4)
            rows.append(
                np.r_[
                    value,
                    np.log(max(width, 1e-4)),
                    np.log(max(height, 1e-4)),
                    np.log(bottom),
                    np.eye(3)[view_index],
                ]
            )
    return np.asarray(rows, dtype=float).reshape(-1, 26)


def encoded_flat_depth(packet, obj):
    if obj["flat_ground_xy"] is None:
        return None
    cal = packet.calibration[obj["view"]]
    x0, _, x1, y1 = obj["box"]
    direction = (
        np.array(
            [
                1.0,
                ((x0 + x1) * 0.5 - cal.width / 2) / cal.intrinsic[0, 0],
                -(y1 - cal.height / 2) / cal.intrinsic[1, 1],
            ]
        )
        @ mount_rotation(cal).T
    )
    return float(-cal.xyz[2] / direction[2])


def pair_targets(reference, packet, observations, features, role, check):
    check()
    arrays = label_arrays(reference)
    result, unsupported = [], []
    offset = 0
    for view in VIEWS:
        if not np.array_equal(arrays["rgb_" + view], packet.rgb[view]):
            raise ValueError("Depth targets must match exact original policyRGBpixels")
        objects, _ = visible_objects(
            arrays[view + "_instance"], decode_depth(arrays[view + "_depth"])
        )
        rows = [o for o in observations if o["view"] == view]
        predicted = [r | dict(group=COCO_GROUP[r["label"]]) for r in rows]
        for match in match_boxes(objects, predicted):
            obj, index = objects[match["reference"]], offset + match["prediction"]
            target = obj["bottom_pixel_depth_m"]
            metadata = dict(
                frame=packet.frame,
                view=view,
                prediction_index=index,
                group=obj["group"],
                target_depth_m=target,
                original_projectable=rows[match["prediction"]]["flat_ground_xy"] is not None,
            )
            if not np.isfinite(target) or not 0 < target <= 1000:
                unsupported.append(metadata)
            else:
                result.append(
                    dict(
                        metadata=metadata,
                        features=features[index],
                        log_depth=float(np.log(target)),
                        role=role,
                    )
                )
        offset += len(rows)
    return result, unsupported


def metrics(rows, name, known=None):
    selected = [
        r
        for r in rows
        if (known is None or r["original_projectable"] == known)
        and r["predictions"][name] is not None
    ]
    if not selected:
        return dict(count=0)
    target = np.array([r["target_depth_m"] for r in selected])
    predicted = np.array([r["predictions"][name] for r in selected])
    error = abs(target - predicted)
    return dict(
        count=len(selected),
        mean_absolute_log_error=float(np.mean(abs(np.log(target) - np.log(predicted)))),
        mae_m=float(error.mean()),
        median_absolute_error_m=float(np.median(error)),
        p95_absolute_error_m=float(np.percentile(error, 95)),
        mean_absolute_relative_error=float(np.mean(error / target)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    cache = ROOT / "outputs/navigation_shared_road_users_20261020"
    data = ROOT / "outputs/navigation_dense_labels_verified_20261020"
    manifest, receipt = read(data / "manifest.json"), read(data / "assessment.json")
    cached = read(cache / "assessment.json")
    if (
        digest(data / "manifest.json") != receipt["manifest_sha256"]
        or not receipt["every_included_source_independently_verified"]
        or digest(cache / "observations.jsonl") != cached["metadata_sha256"]
        or read(cache / "progress.json")["status"] != "complete"
    ):
        raise ValueError("Complete verified data/cache required")
    paths = [
        Path(__file__),
        ROOT / "reports/navigation_box_depth_protocol_20261020.md",
        ROOT / "src/navigation/road_user_observations.py",
        ROOT / "scripts/assess_navigation_dense_perception.py",
        ROOT / "scripts/navigation_label_rig.py",
        data / "manifest.json",
        cache / "assessment.json",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for source in paths:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            source_hashes=hashes,
            supervised_timestamps=429,
            train_timestamps=351,
            calibration_timestamps=78,
            targets="Encoded bottom visible-pixel camera depth, not actor centre",
            ridge_alpha=1.0,
            boosting=dict(iterations=200, learning_rate=0.05, leaves=15, L2=10, seed=42),
            maximum_wall_seconds=900,
            maximum_artifact_bytes=100_000_000,
            native_depth_confirmation="failed0observed/6planned; no rerun",
            no_automatic_deployment=True,
        ),
    )
    status, done, started = "running", 0, time.monotonic()

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Encoded box-depth baselines",
                frames=done,
                target_frames=507,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 900:
            raise InterruptedError("Owned STOP or box-depth900s cap")
        if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > 100_000_000:
            raise InterruptedError("Box-depth100MBcap")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 10)
            import joblib
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from threadpoolctl import threadpool_limits

            metadata = [
                json.loads(line) for line in (cache / "observations.jsonl").read_text().splitlines()
            ]
            by_key = {(r["route_id"], r["frame"]): r for r in metadata}
            chunks = {}
            for path, sha in cached["chunk_sha256"].items():
                check()
                if digest(path) != sha:
                    raise ValueError("Cached observation values changed")
                with np.load(path, allow_pickle=False) as z:
                    chunks[int(Path(path).stem.rsplit("_", 1)[1])] = {
                        k: z[k].copy() for k in z.files
                    }
            frames, train, unsupported = [], [], []
            for source in manifest["sources"]:
                for path, sha in (
                    source["original_source_shards_sha256"] | source["label_shards_sha256"]
                ).items():
                    check()
                    if digest(path) != sha:
                        raise ValueError("Original verified source pixels changed")
                role, route = source["item"]["role"], source["item"]["route_id"]
                for reference in source["label_sources"]:
                    check()
                    record = by_key[(route, reference["frame"])]
                    packet = original_packet(reference)
                    sanitized = [
                        {k: o[k] for k in ("view", "label", "score", "box")}
                        for o in record["observations"]
                    ]
                    pack = observation_pack(packet, sanitized)
                    chunk = chunks[record["chunk_index"]]
                    if (
                        record["role"] != role
                        or not np.array_equal(pack["values"], chunk["values"][record["offset"]])
                        or not np.array_equal(pack["presence"], chunk["presence"][record["offset"]])
                        or pack["observations"] != record["observations"]
                    ):
                        raise ValueError("Role or independent full observation readback mismatch")
                    features = geometry_features(pack["values"], pack["presence"])
                    frame = dict(
                        reference=reference,
                        packet=packet,
                        role=role,
                        route=route,
                        record=record,
                        features=features,
                        flat=[encoded_flat_depth(packet, o) for o in pack["observations"]],
                    )
                    # RGB arrays occupy~570MB if retained for all429frames; calibration
                    # targets are reopened only after every prediction is committed.
                    if role == "train":
                        pairs, invalid = pair_targets(
                            reference, packet, pack["observations"], features, role, check
                        )
                        for pair in pairs:
                            pair["source_route"] = route
                        train.extend(pairs)
                        unsupported.extend(invalid)
                    frame.pop("packet")
                    frames.append(frame)
                    done += 1
                    progress("Verify all cached values; build only TRAIN supervised pairs")
            counts = Counter(r["source_route"] for r in train)
            if len(counts) != 9:
                raise ValueError("All nine registered TRAINsources must supply supervised pairs")
            features = np.stack([r["features"] for r in train])
            targets = np.array([r["log_depth"] for r in train])
            weight = np.array([1 / counts[r["source_route"]] for r in train])
            weight *= len(weight) / weight.sum()
            ordering = np.argsort(targets)
            median = float(
                targets[ordering[np.searchsorted(np.cumsum(weight[ordering]), weight.sum() / 2)]]
            )
            models = dict(
                ridge=make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                boosting=HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=10,
                    random_state=42,
                    early_stopping=False,
                ),
            )
            progress("Two fixed CPUfits; no sweep or continuation")
            with threadpool_limits(limits=2):
                models["ridge"].fit(features, targets, ridge__sample_weight=weight)
                check()
                models["boosting"].fit(features, targets, sample_weight=weight)
            for name, model in models.items():
                joblib.dump(model, args.out / (name + ".joblib"))
            atomic_json(
                args.out / "train_pair_counts.json",
                dict(
                    per_source=counts,
                    pairs=len(train),
                    weighted_log_median=median,
                    groups=dict(Counter(r["metadata"]["group"] for r in train)),
                ),
            )
            for frame in frames:
                check()
                x = frame["features"]
                frame["predictions"] = dict(
                    constant=np.full(len(x), np.exp(median)).tolist(), flat=frame["flat"]
                )
                with threadpool_limits(limits=2):
                    for name, model in models.items():
                        frame["predictions"][name] = (
                            np.exp(np.clip(model.predict(x), np.log(0.01), np.log(1000))).tolist()
                            if len(x)
                            else []
                        )
                with (args.out / "geometry_predictions.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            dict(
                                route_id=frame["route"],
                                role=frame["role"],
                                frame=frame["reference"]["frame"],
                                predictions=frame["predictions"],
                            ),
                            allow_nan=False,
                        )
                        + "\n"
                    )
            # All calibration box predictions are now immutable before target reads.
            scored = []
            for frame in frames:
                if frame["role"] != "calibration":
                    continue
                check()
                packet = original_packet(frame["reference"])
                pairs, invalid = pair_targets(
                    frame["reference"],
                    packet,
                    frame["record"]["observations"],
                    frame["features"],
                    "calibration",
                    check,
                )
                unsupported.extend(invalid)
                for pair in pairs:
                    index = pair["metadata"]["prediction_index"]
                    scored.append(
                        pair["metadata"]
                        | dict(
                            route_id=frame["route"],
                            predictions={
                                name: p[index] for name, p in frame["predictions"].items()
                            },
                        )
                    )
                done += 1
                progress("Score all precommitted calibration predictions; no calibration fitting")
            names = ("constant", "flat", "ridge", "boosting")
            per_location = {
                str(route): {
                    name: dict(
                        all=metrics([r for r in scored if r["route_id"] == route], name),
                        projectable=metrics(
                            [r for r in scored if r["route_id"] == route], name, True
                        ),
                        unprojectable=metrics(
                            [r for r in scored if r["route_id"] == route], name, False
                        ),
                    )
                    for name in names
                }
                for route in sorted({r["route_id"] for r in scored})
            }
            macro = {
                name: {
                    subset: float(
                        np.mean(
                            [
                                r[name][subset]["mean_absolute_log_error"]
                                for r in per_location.values()
                                if r[name][subset]["count"]
                            ]
                        )
                    )
                    for subset in (
                        ("projectable",)
                        if name == "flat"
                        else ("all", "projectable", "unprojectable")
                    )
                }
                for name in names
            }
            best = min(("ridge", "boosting"), key=lambda name: macro[name]["all"])
            protected = macro[best]["projectable"] <= 0.95 * macro["flat"]["projectable"] and all(
                r[best]["all"]["mean_absolute_log_error"]
                < r["constant"]["all"]["mean_absolute_log_error"]
                for r in per_location.values()
            )
            atomic_json(args.out / "scored_pairs.json", scored)
            atomic_json(args.out / "unsupported_targets.json", unsupported)
            if any(digest(ROOT / p) != sha for p, sha in hashes.items()):
                raise ValueError("Registered source changed during depth diagnostic")
            atomic_json(
                args.out / "assessment.json",
                dict(
                    train_pairs=len(train),
                    train_sources=len(counts),
                    calibration_pairs=len(scored),
                    calibration_locations=len(per_location),
                    metrics={name: metrics(scored, name) for name in names},
                    per_location=per_location,
                    macro_absolute_log_error=macro,
                    diagnostic_best=best,
                    selected_offline=best if protected else None,
                    selection_protection_passed=protected,
                    unsupported_targets=len(unsupported),
                    all429_cached_packets_independently_reproduced=True,
                    prediction_sha256=digest(args.out / "geometry_predictions.jsonl"),
                    no_driving_qualification=True,
                    no_runtime_model_changed=True,
                    physical_depth_convention_native_confirmation=False,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status = "complete"
            progress("Fixed box-depth diagnostic complete; no runtime deployment")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
