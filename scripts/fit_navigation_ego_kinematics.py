#!/usr/bin/env python3
"""Fit shared actuator constants on TRAIN only; validate factual forecasts separately."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, lsq_linear

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.ego_kinematics import EgoKinematics  # noqa:E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve every fitted physics baseline")
    data = ROOT / "outputs/navigation_dynamics_sources_repaired_20261019/manifest.json"
    manifest = json.loads(data.read_text())
    sources = [
        Path(__file__),
        ROOT / "src/navigation/ego_kinematics.py",
        ROOT / "src/navigation/recorder_actions.py",
        ROOT / "reports/navigation_ego_dynamics_protocol_20261020.md",
    ]
    args.out.mkdir(parents=True)
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            source_manifest_sha256=digest(data),
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
            training_only_fitted_constants=True,
            no_validation_search=True,
            no_driving_or_collision_headline_exclusions=True,
        ),
    )
    episodes = []
    design = []
    accelerations = []
    steer_inputs = []
    yaw_targets = []
    for e in manifest["accepted"]:
        if (args.out / "STOP").exists():
            raise InterruptedError("Ego identification STOP")
        raw = Path(e["data"])
        trace = raw.parent / "world_trace.jsonl"
        if (
            digest(trace) != e["world_trace_sha256"]
            or digest(e["recording"]) != e["recording_sha256"]
        ):
            raise ValueError("Factual physics source changed")
        world = [json.loads(s) for s in trace.read_text().splitlines()]
        native, alignment = align_world_frames(
            vehicle_frames(e["recording"], e["alignment"]["hero_id"]), world
        )
        frames = np.array([w["frame"] for w in world])
        heroes = [next(a for a in w["actors"] if a["role"] == "hero") for w in world]
        state = np.array(
            [
                [
                    a["pose"][0],
                    a["pose"][1],
                    np.deg2rad(a["pose"][4]),
                    np.linalg.norm(a["velocity"][:2]),
                ]
                for a in heroes
            ]
        )
        controls = np.array([native[int(frame) + 1]["control"] for frame in frames[:-1]])
        supported = np.array(
            [
                not native[int(frame) + 1]["handbrake"] and native[int(frame) + 1]["gear"] >= 0
                for frame in frames[:-1]
            ]
        )
        collision_frames = np.array(
            [
                world[0]["frame"] + c["frame"] - 1 - alignment["offset"]
                for c in e["contacts"]["raw_samples"]
            ],
            dtype=np.int64,
        )
        eligible = (
            (np.arange(len(controls)) >= 20)
            & (state[:-1, 3] > 0.5)
            & (state[1:, 3] > 0.5)
            & supported
        )
        for contact in collision_frames:
            eligible &= ~((frames[:-1] >= contact - 4) & (frames[:-1] <= contact + 10))
        x = np.column_stack(
            [controls[:, 1], -controls[:, 2], -state[:-1, 3], np.ones(len(controls))]
        )
        accel = np.diff(state[:, 3]) / 0.05
        yawrate = ((np.diff(state[:, 2]) + np.pi) % (2 * np.pi) - np.pi) / 0.05
        if e["dynamics_role"] == "train":
            design.append(x[eligible])
            accelerations.append(accel[eligible])
            steer_inputs.append(
                np.column_stack(
                    [controls[eligible, 0], 0.5 * (state[:-1, 3] + state[1:, 3])[eligible]]
                )
            )
            yaw_targets.append(yawrate[eligible])
        episodes.append(
            dict(
                data=e["data"],
                role=e["dynamics_role"],
                state=state,
                controls=controls,
                supported=supported,
                eligible=eligible,
                design=x,
                accel=accel,
                yawrate=yawrate,
            )
        )
    x = np.concatenate(design)
    y = np.concatenate(accelerations)
    acceleration = lsq_linear(x, y, bounds=([0, 0, 0, -2], [20, 30, 5, 2]))
    steering = np.concatenate(steer_inputs)
    rates = np.concatenate(yaw_targets)
    angle = least_squares(
        lambda a: steering[:, 1] / 2.8 * np.tan(steering[:, 0] * a[0]) - rates,
        [0.7],
        bounds=([0.3], [1.3]),
    )
    params = dict(
        zip(("throttle_accel", "brake_decel", "drag", "accel_bias"), map(float, acceleration.x))
    )
    params.update(maximum_steer_angle_rad=float(angle.x[0]), wheelbase_m=2.8)
    model = EgoKinematics(params)
    atomic_json(args.out / "parameters.json", params)
    diagnostics = []
    for role in ("train", "validation"):
        group = [e for e in episodes if e["role"] == role]
        acceleration_errors = []
        angle_errors = []
        position_errors = []
        speed_errors = []
        windows = 0
        for e in group:
            eligible = e["eligible"]
            acceleration_errors.extend(
                (e["design"][eligible] @ acceleration.x - e["accel"][eligible]).tolist()
            )
            predicted_rate = (
                0.5
                * (e["state"][:-1, 3] + e["state"][1:, 3])
                / 2.8
                * np.tan(e["controls"][:, 0] * params["maximum_steer_angle_rad"])
            )
            angle_errors.extend((predicted_rate[eligible] - e["yawrate"][eligible]).tolist())
            for start in range(0, len(e["controls"]) - 79, 10):
                if not e["supported"][start : start + 80].all():
                    continue
                forecast = model.rollout(e["state"][start], e["controls"][start : start + 80])
                truth = e["state"][start + 1 : start + 81]
                position_errors.append(
                    np.linalg.norm(forecast[9::10, :2] - truth[9::10, :2], axis=-1)
                )
                speed_errors.append(np.abs(forecast[9::10, 3] - truth[9::10, 3]))
                windows += 1
        diagnostics.append(
            dict(
                role=role,
                episodes=len(group),
                eligible_one_step_samples=len(acceleration_errors),
                acceleration_mae_mps2=float(np.abs(acceleration_errors).mean()),
                yaw_rate_mae_radps=float(np.abs(angle_errors).mean()),
                factual_four_second_windows=windows,
                open_loop_position_mae_m_by_horizon=np.mean(position_errors, axis=0).tolist(),
                open_loop_position_p95_m_by_horizon=np.percentile(
                    position_errors, 95, axis=0
                ).tolist(),
                open_loop_speed_mae_mps_by_horizon=np.mean(speed_errors, axis=0).tolist(),
                includes_contacts_and_assisted_factual_forecasts=True,
            )
        )
    atomic_json(
        args.out / "assessment.json",
        dict(
            status="complete",
            parameters=params,
            parameters_sha256=digest(args.out / "parameters.json"),
            diagnostics=diagnostics,
            current_inputs_and_ordered_controls_only=True,
            train_fit_samples=len(y),
            validation_used_for_fitting=False,
            planner_qualified=False,
            note="Factual forecast errors do not prove counterfactual validity "
            "or collision calibration.",
        ),
    )
    print(
        json.dumps(dict(parameters=params, diagnostics=diagnostics, planner_qualified=False)),
        flush=True,
    )


if __name__ == "__main__":
    main()
