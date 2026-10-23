#!/usr/bin/env python3
"""Compare the frozen expert's ego model with the initial fitted bicycle."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.ego_kinematics import EgoKinematics  # noqa: E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def forecast_vendor(model, state, controls):
    location = np.array([state[0], state[1], 0.0])
    heading, speed = np.array([state[2]]), np.array([state[3]])
    result = []
    for command in controls:
        location, heading, speed = model.forecast_ego_vehicle(location, heading, speed, command)
        yaw = float(np.asarray(heading).reshape(-1)[0])
        velocity = float(np.asarray(speed).reshape(-1)[0])
        result.append([location[0], location[1], yaw, velocity])
        heading, speed = np.array([yaw]), np.array([velocity])
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    vendor = activate()
    from config import GlobalConfig
    from kinematic_bicycle_model import KinematicBicycleModel

    dynamics = ROOT / "outputs/navigation_dynamics_sources_repaired_20261019/manifest.json"
    entries = json.loads(dynamics.read_text())["accepted"]
    parameters = ROOT / "outputs/navigation_ego_kinematics_20261020/parameters.json"
    protocol = ROOT / "reports/navigation_vendor_ego_protocol_20261020.md"
    source_paths = [
        Path(__file__), protocol, ROOT / "src/navigation/ego_kinematics.py",
        ROOT / "src/navigation/recorder_actions.py",
        Path(vendor["root"]) / "team_code/config.py",
        Path(vendor["root"]) / "team_code/kinematic_bicycle_model.py",
    ]
    for path in source_paths:
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(args.out / "request.json", dict(
        manifest_sha256=digest(dynamics), parameters_sha256=digest(parameters),
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        vendor=vendor, no_parameter_fitting=True,
        fixed_predictors=["initial_bicycle", "vendor_lag0", "vendor_lag2"],
        lag_decision_train_only="outputs/navigation_actuator_timing_20261020/assessment.json",
        lag_diagnosis_sha256=digest(ROOT /
            "outputs/navigation_actuator_timing_20261020/assessment.json"),
    ))
    original = EgoKinematics(json.loads(parameters.read_text()))
    model = KinematicBicycleModel(GlobalConfig())
    errors, velocities, roles, episode_ids, starts = [], [], [], [], []
    fractional, all_commands = 0, 0
    for episode_id, entry in enumerate(entries):
        if (args.out / "STOP").exists():
            raise InterruptedError("Frozen expert ego-model assessment STOP")
        trace = Path(entry["data"]).parent / "world_trace.jsonl"
        if digest(trace) != entry["world_trace_sha256"]:
            raise ValueError("Factual world trace changed")
        if digest(entry["recording"]) != entry["recording_sha256"]:
            raise ValueError("Factual native recording changed")
        world = [json.loads(row) for row in trace.read_text().splitlines()]
        raw = vehicle_frames(entry["recording"], entry["alignment"]["hero_id"])
        native, alignment = align_world_frames(raw, world)
        heroes = [next(a for a in w["actors"] if a["role"] == "hero") for w in world]
        state = np.array([[a["pose"][0], a["pose"][1], np.deg2rad(a["pose"][4]),
                           np.linalg.norm(a["velocity"][:2])] for a in heroes])
        frames = [w["frame"] for w in world]
        controls = np.array([native[f + 1]["control"] for f in frames[:-1]])
        supported = np.array([not native[f + 1]["handbrake"] and native[f + 1]["gear"] >= 0
                              for f in frames[:-1]])
        fractional += int(((controls[:, 2] > 0) & (controls[:, 2] < .99)).sum())
        all_commands += len(controls)
        for start in range(0, len(controls) - 79, 10):
            if not supported[start:start + 80].all():
                continue
            offset = alignment["offset"] + start
            if offset < 1:
                raise ValueError("Cannot guess the two past acknowledged actions")
            history = raw[offset - 1:offset + 1]
            if len(history) != 2 or any("control" not in r for r in history):
                raise ValueError("Actual two-command history is missing")
            actions = controls[start:start + 80]
            delayed = np.vstack([[r["control"] for r in history], actions[:-2]])
            forecasts = [original.rollout(state[start], actions),
                         forecast_vendor(model, state[start], actions),
                         forecast_vendor(model, state[start], delayed)]
            truth = state[start + 1:start + 81]
            errors.append([np.linalg.norm(f[9::10, :2] - truth[9::10, :2], axis=-1)
                           for f in forecasts])
            velocities.append([np.abs(f[9::10, 3] - truth[9::10, 3]) for f in forecasts])
            roles.append(entry["dynamics_role"] == "validation")
            episode_ids.append(episode_id)
            starts.append(start)
    errors, velocities, roles = np.asarray(errors), np.asarray(velocities), np.asarray(roles)
    np.savez(args.out / "window_errors.npz", position_m=errors, speed_mps=velocities,
             validation=roles, source_episode=np.asarray(episode_ids), start=np.asarray(starts))
    rows = []
    for role in ("train", "validation"):
        selected = roles == (role == "validation")
        for index, name in enumerate(("initial_bicycle", "vendor_lag0", "vendor_lag2")):
            rows.append(dict(role=role, predictor=name, windows=int(selected.sum()),
                position_mae_m_by_horizon=errors[selected, index].mean(0).tolist(),
                position_p95_m_by_horizon=np.quantile(
                    errors[selected, index], .95, axis=0).tolist(),
                speed_mae_mps_by_horizon=velocities[selected, index].mean(0).tolist()))
    atomic_json(args.out / "assessment.json", dict(
        status="complete", rows=rows, window_arrays_sha256=digest(args.out / "window_errors.npz"),
        fractional_brake_commands=fractional, commands=all_commands,
        no_factual_contact_or_assisted_windows_removed=True,
        no_parameter_fitting=True, no_validation_lag_search=True, planner_qualified=False,
        note="Frozen expert coefficients are external prior knowledge, not learned on this corpus. "
        "Factual errors alone cannot qualify candidate-action or reactive planning."))
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
