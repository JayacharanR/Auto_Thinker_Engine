#!/usr/bin/env python3
"""TRAIN-only constant/action/CV calibration, with held-out predictions saved first."""

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402
from src.navigation.vehicle_perception import CameraVehicleTracker  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def load_inputs(record):
    root = Path(record["output"])
    for name, sha in record["files_sha256"].items():
        if digest(root / name) != sha:
            raise ValueError("Frozen feature file changed")
    with np.load(root / "inputs.npz") as data:
        inputs = {k: v.copy() for k, v in data.items()}
    with np.load(root / "proposals.npz") as data:
        plans = {k: v.copy() for k, v in data.items()}
    return inputs, plans, read(root / "observations.json")


def load_targets(record):
    with np.load(Path(record["output"]) / "targets.npz") as data:
        return {k: v.copy() for k, v in data.items()}


@dataclass
class Odometry:
    timestamp: float
    ego_motion: np.ndarray

    def validate(self):
        if (
            self.ego_motion.shape != (3,)
            or not np.isfinite(np.r_[self.timestamp, self.ego_motion]).all()
        ):
            raise ValueError("Declared finite ego odometry only")


def cv_proximity(inputs, observations, forecasts):
    tracker, tracks = CameraVehicleTracker(), []
    for motion, timestamp, row in zip(inputs["past_ego"], inputs["timestamps"], observations):
        located = [
            dict(
                position_xy=np.asarray(o["flat_ground_xy"]),
                score=o["score"],
                label=o["label"],
                sigma_m=0.5 + 0.2 * np.linalg.norm(o["flat_ground_xy"]),
            )
            for o in row["observations"]
            if o["flat_ground_xy"] is not None
        ]
        fused = []
        for observation in sorted(located, key=lambda o: -o["score"]):
            if not any(
                np.linalg.norm(observation["position_xy"] - p["position_xy"]) < 2.0 for p in fused
            ):
                fused.append(observation)
        tracks = tracker.step(Odometry(float(timestamp), motion), fused)
    times = np.arange(1, 81) * 0.05
    score = np.zeros((len(forecasts), 80))
    for index, forecast in enumerate(forecasts):
        xy, yaw = forecast[:, :2], forecast[:, 2]
        for track in tracks:
            delta = track.position_xy[None] + times[:, None] * track.velocity_xy[None] - xy
            sigma = track.position_sigma_m + (times + track.age_seconds) * track.velocity_sigma_mps
            longitudinal = delta[:, 0] * np.cos(yaw) + delta[:, 1] * np.sin(yaw)
            lateral = -delta[:, 0] * np.sin(yaw) + delta[:, 1] * np.cos(yaw)
            radius = np.hypot(track.half_length, track.half_width)
            gap_x = np.maximum(0, abs(longitudinal) - 2.5 - radius)
            gap_y = np.maximum(0, abs(lateral) - 1.1 - radius)
            proximity = np.exp(-0.5 * (gap_x**2 + gap_y**2) / np.maximum(sigma**2, 1e-6))
            score[index] = np.maximum(score[index], proximity)
    return np.maximum.accumulate(score, axis=1), len(tracks)


def cumulative_labels(target):
    event, valid = target["event"], target["valid"]
    cumulative = np.cumsum(event) > 0
    observed = valid | cumulative
    return cumulative.astype(float), observed


def sequence_nll(cumulative, target):
    survival = 1 - np.clip(np.asarray(cumulative), 0, 1 - 1e-9)
    previous = np.r_[1.0, survival[:-1]]
    hazard = np.clip(1 - survival / previous, 1e-9, 1 - 1e-9)
    event, valid = target["event"], target["valid"]
    return float((-(event * np.log(hazard) + (1 - event) * np.log1p(-hazard)) * valid).sum())


def metrics(prediction, targets):
    labels = np.array([cumulative_labels(t)[0][-1] for t in targets])
    known = np.array([cumulative_labels(t)[1][-1] for t in targets])
    if not known.all():
        raise ValueError("Do not turn an unknown endpoint into a negative")
    final = np.asarray(prediction)[:, -1]
    return dict(
        examples=len(targets),
        positives=int(labels.sum()),
        endpoint_brier=float(np.mean((final - labels) ** 2)),
        sequence_nll=float(np.mean([sequence_nll(p, t) for p, t in zip(prediction, targets)])),
        misses=int(((final < 0.5) & (labels > 0)).sum()),
        false_alarms=int(((final >= 0.5) & (labels == 0)).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if read(args.features / "progress.json")["status"] != "complete":
        raise ValueError("Wait for all frozen features")
    records = read(args.features / "records.json")
    train = np.array([r["source_role"] == "train" for r in records])
    with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
        from scipy.optimize import minimize
        from scipy.special import expit, logit
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        raw, action_features, track_counts, selected_indices = [], [], [], []
        for row in records:
            inputs, plans, observations = load_inputs(row)
            risk, count = cv_proximity(inputs, observations, plans["all_forecasts"])
            selected = np.flatnonzero(
                np.all(plans["all_controls"] == plans["ordered_controls"][None], axis=(1, 2))
            )
            if len(selected) != 1:
                raise ValueError("Selected prospectively frozen option must be unique")
            selected_indices.append(int(selected[0]))
            raw.append(risk)
            track_counts.append(count)
            action_features.append(
                np.c_[plans["all_controls"].mean(1), np.full(15, inputs["ego"][0])]
            )
        raw, action_features = np.array(raw), np.array(action_features)
        selected = np.array(selected_indices)
        train_targets = [load_targets(records[i]) for i in np.flatnonzero(train)]
        train_labels = np.array([cumulative_labels(t)[0] for t in train_targets])
        prior = sum(t["event"].sum() for t in train_targets) / sum(
            t["valid"].sum() for t in train_targets
        )
        constant = 1 - (1 - prior) ** np.arange(1, 81)
        x = logit(np.clip(raw[train, selected[train]][:, 9::10], 1e-6, 1 - 1e-6)).ravel()
        y = train_labels[:, 9::10].ravel()

        def objective(parameters):
            z = parameters[0] * x + parameters[1]
            loss = np.mean(np.logaddexp(0, z) - y * z) + 1e-3 * parameters[0] ** 2
            error = expit(z) - y
            gradient = [np.mean(error * x) + 0.002 * parameters[0], np.mean(error)]
            return loss, np.array(gradient)

        fit = minimize(
            objective, [1.0, -3.0], jac=True, method="L-BFGS-B", bounds=[(0, 20), (-20, 20)]
        )
        if not fit.success:
            raise RuntimeError("FixedTRAINCVcalibration failed:" + fit.message)
        calibrated = expit(fit.x[0] * logit(np.clip(raw, 1e-6, 1 - 1e-6)) + fit.x[1])
        action_x = action_features[train, selected[train]]
        scaler = StandardScaler().fit(action_x)
        action_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(
            scaler.transform(action_x), train_labels[:, -1]
        )
        action_probability = action_model.predict_proba(
            scaler.transform(action_features.reshape(-1, 4))
        )[:, 1].reshape(len(records), 15)
        np.savez_compressed(
            args.out / "predictions.npz",
            raw_cv=raw,
            calibrated_cv=calibrated,
            constant=np.tile(constant, (len(records), 1)),
            action_endpoint=action_probability,
            selected_indices=selected,
        )
        # Commit all calibration predictions before opening calibration targets.
        atomic_json(
            args.out / "frozen_fit.json",
            dict(
                prior_per_tick=float(prior),
                cv_slope_intercept=fit.x.tolist(),
                action_coefficients=action_model.coef_.tolist(),
                action_intercept=action_model.intercept_.tolist(),
                action_scale=scaler.scale_.tolist(),
                action_mean=scaler.mean_.tolist(),
                fitted_only_on_TRAIN=True,
                predictions_sha256=digest(args.out / "predictions.npz"),
            ),
        )
        targets = [load_targets(r) for r in records]
        result = {}
        for role, mask in (("train", train), ("calibration", ~train)):
            indices = np.flatnonzero(mask)
            actual = [targets[i] for i in indices]
            labels = np.array([cumulative_labels(t)[0][-1] for t in actual])
            probability = action_probability[indices, selected[indices]]
            result[role] = dict(
                constant=metrics(np.tile(constant, (len(indices), 1)), actual),
                cv=metrics(calibrated[indices, selected[indices]], actual),
                raw_cv=metrics(raw[indices, selected[indices]], actual),
                action_only=dict(
                    endpoint_brier=float(np.mean((probability - labels) ** 2)),
                    misses=int(((probability < 0.5) & (labels > 0)).sum()),
                    false_alarms=int(((probability >= 0.5) & (labels == 0)).sum()),
                ),
            )
        atomic_json(
            args.out / "assessment.json",
            dict(
                metrics=result,
                final_tracks_per_episode=track_counts,
                no_tracks=sum(c == 0 for c in track_counts),
                raw_cv_is_not_probability=True,
                features_records_sha256=digest(args.features / "records.json"),
                no_fit_or_driving_promotion=True,
            ),
        )
        shutil.copyfile(__file__, args.out / "baseline_source.py")
        atomic_json(
            args.out / "progress.json",
            dict(
                status="complete",
                stage="First-event simple baselines",
                frames=68,
                target_frames=68,
                phase="TRAINfrozen before calibration targets; no planner qualification",
                updated_at=time.time(),
            ),
        )
        print(json.dumps(result["calibration"]), flush=True)


if __name__ == "__main__":
    main()
