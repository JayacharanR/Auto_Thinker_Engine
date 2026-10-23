#!/usr/bin/env python3
"""Diagnose ego response timing on TRAIN towns without reading dynamics VAL."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def fit(x, y):
    return lsq_linear(x, y, bounds=([0, 0, 0, -2], [20, 30, 5, 2])).x


def statistics(values):
    values = np.asarray(values, float)
    return dict(count=len(values), mean=float(values.mean()),
                p50=float(np.median(values)), p95=float(np.quantile(values, .95)),
                maximum=float(values.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "outputs/navigation_dynamics_sources_repaired_20261019/manifest.json"
    manifest = json.loads(source.read_text())
    entries = [e for e in manifest["accepted"] if e["dynamics_role"] == "train"]
    if len(entries) != 52:
        raise ValueError("The exact52registered TRAIN sources are required")
    paths = [Path(__file__), ROOT / "src/navigation/recorder_actions.py",
             ROOT / "reports/navigation_actuator_timing_protocol_20261020.md"]
    for path in paths:
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(args.out / "request.json", dict(
        source_manifest_sha256=digest(source),
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in paths},
        training_episodes=52, validation_episodes_opened=0,
        lags_ticks=list(range(9)), internal_cv="Leave-one-training-town-out",
    ))
    groups, design, target, pose_errors, source_rows = [], [[] for _ in range(9)], [], [], []
    regime = {"throttle": [], "brake": [], "coast": []}
    for entry in entries:
        if (args.out / "STOP").exists():
            raise InterruptedError("Actuator diagnosis STOP")
        path = Path(entry["data"]).parent / "world_trace.jsonl"
        if digest(path) != entry["world_trace_sha256"]:
            raise ValueError("TRAIN world source changed")
        if digest(entry["recording"]) != entry["recording_sha256"]:
            raise ValueError("TRAIN native source changed")
        world = [json.loads(row) for row in path.read_text().splitlines()]
        aligned, alignment = align_world_frames(
            vehicle_frames(entry["recording"], entry["alignment"]["hero_id"]), world
        )
        frames = np.array([w["frame"] for w in world])
        heroes = [next(a for a in w["actors"] if a["role"] == "hero") for w in world]
        velocity = np.array([a["velocity"][:2] for a in heroes])
        xy = np.array([a["pose"][:2] for a in heroes])
        speed = np.linalg.norm(velocity, axis=1)
        controls = np.array([aligned[int(f) + 1]["control"] for f in frames[:-1]])
        supported = np.array([not aligned[int(f) + 1]["handbrake"] and
                              aligned[int(f) + 1]["gear"] >= 0 for f in frames[:-1]])
        eligible = ((np.arange(len(controls)) >= 28) & (speed[:-1] > .5)
                    & (speed[1:] > .5) & supported)
        for contact in entry["contacts"]["raw_samples"]:
            frame = frames[0] + contact["frame"] - 1 - alignment["offset"]
            eligible &= ~((frames[:-1] >= frame - 4) & (frames[:-1] <= frame + 10))
        ids = np.flatnonzero(eligible)
        accel = np.diff(speed) / .05
        target.extend(accel[ids].tolist())
        groups.extend([entry["metadata"]["town"]] * len(ids))
        for lag in range(9):
            c = controls[ids - lag]
            design[lag].append(np.column_stack([c[:, 1], -c[:, 2], -speed[ids],
                                                np.ones(len(ids))]))
        position_speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / .05
        pose_errors.extend(np.abs(position_speed[ids] - (speed[ids] + speed[ids + 1]) / 2))
        selected = controls[ids]
        for key, mask in (
            ("throttle", selected[:, 1] > .3), ("brake", selected[:, 2] > .8),
            ("coast", (selected[:, 1] < .01) & (selected[:, 2] < .01)),
        ):
            regime[key].extend(accel[ids][mask].tolist())
        source_rows.append(dict(data=entry["data"], town=entry["metadata"]["town"],
                                samples=len(ids), world_trace_sha256=digest(path),
                                recording_sha256=entry["recording_sha256"]))
    y, groups = np.asarray(target), np.asarray(groups)
    rows = []
    for lag in range(9):
        x = np.concatenate(design[lag])
        parameters = fit(x, y)
        folds = []
        errors = np.full(len(y), np.nan)
        for town in sorted(set(groups)):
            held = groups == town
            estimate = fit(x[~held], y[~held])
            error = x[held] @ estimate - y[held]
            errors[held] = error
            folds.append(dict(town=town, samples=int(held.sum()),
                              acceleration_mae_mps2=float(np.abs(error).mean())))
        if not np.isfinite(errors).all():
            raise ValueError("All TRAIN cross-validation transitions need held-town predictions")
        rows.append(dict(lag_ticks=lag, lag_seconds=lag * .05,
                         parameters=parameters.tolist(), folds=folds,
                         fit_acceleration_mae_mps2=float(np.abs(x @ parameters - y).mean()),
                         cv_acceleration_mae_mps2=float(np.abs(errors).mean()),
                         cv_absolute_error=statistics(np.abs(errors))))
    result = dict(status="complete", rows=rows, source_episodes=source_rows,
                  samples=len(y), snapshot_pose_speed_difference=statistics(pose_errors),
                  acceleration_absolute=statistics(np.abs(y)),
                  acceleration_over8_mps2=int((np.abs(y) > 8).sum()),
                  control_regimes={k: statistics(v) if v else None for k, v in regime.items()},
                  validation_sources_opened=0, deployed=False, planner_qualified=False)
    atomic_json(args.out / "assessment.json", result)
    print(json.dumps(dict(samples=len(y), rows=rows, velocity_pose=result[
        "snapshot_pose_speed_difference"], validation_sources_opened=0)), flush=True)


if __name__ == "__main__":
    main()
