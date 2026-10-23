#!/usr/bin/env python3
"""Export the frozen expert coefficients and verify the pure inference equations."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.frozen_ego import FrozenExpertEgo, parameter_hash  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Retain every prior frozen numerical export")
    vendor = activate()
    from config import GlobalConfig
    from kinematic_bicycle_model import KinematicBicycleModel

    config = GlobalConfig()
    names = ("time_step", "front_wheel_base", "rear_wheel_base", "steering_gain",
             "throttle_values", "brake_values", "throttle_threshold_during_forecasting")
    parameters = {}
    for name in names:
        value = getattr(config, name)
        parameters[name] = value.tolist() if isinstance(value, np.ndarray) else float(value)
    descriptor = dict(
        schema="navigation-frozen-expert-ego-v1", lag_ticks=2,
        parameters=parameters, parameters_sha256=parameter_hash(parameters), vendor=vendor,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in (
            Path(vendor["root"]) / "team_code/config.py",
            Path(vendor["root"]) / "team_code/kinematic_bicycle_model.py",
            ROOT / "src/navigation/frozen_ego.py", Path(__file__),
        )},
        lag_chosen_using_train_only=True,
        training_timing_diagnosis_sha256=digest(ROOT /
            "outputs/navigation_actuator_timing_20261020/assessment.json"),
        factual_assessment_sha256=digest(ROOT /
            "outputs/navigation_vendor_ego_20261020/assessment.json"),
        planner_qualified=False,
        limitations=["Any positive brake treated as full braking", "Throttle<.3 holds speed",
                     "No interaction/contact physics", "Factual error is not native qualification"],
    )
    pure, reference = FrozenExpertEgo(descriptor), KinematicBicycleModel(config)
    rng = np.random.default_rng(62020)
    differences = []
    for index in range(256):
        state = np.r_[rng.uniform(-30, 30, 2), rng.uniform(-np.pi, np.pi), rng.uniform(0, 8)]
        command = np.r_[rng.uniform(-1, 1), rng.uniform(0, 1),
                        0 if index % 2 else rng.uniform(0, 1)]
        a = pure.step(state, command)
        location, heading, speed = reference.forecast_ego_vehicle(
            np.r_[state[:2], 0], np.array([state[2]]), np.array([state[3]]), command)
        b = np.r_[location[:2], np.asarray(heading).reshape(-1)[0],
                  np.asarray(speed).reshape(-1)[0]]
        differences.append(float(np.abs(a - b).max()))
    if max(differences) > 1e-10:
        raise ValueError("Pure inference equations differ from the frozen expert")
    descriptor["numerical_compatibility"] = dict(samples=256,
        maximum_absolute_difference=max(differences), no_fitted_parameters=True)
    atomic_json(args.out, descriptor)
    print(json.dumps(descriptor["numerical_compatibility"]), flush=True)


if __name__ == "__main__":
    main()
