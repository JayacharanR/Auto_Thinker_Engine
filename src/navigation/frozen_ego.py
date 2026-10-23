"""Pure numerical export of the pinned expert ego model; no simulator inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np


def parameter_hash(parameters):
    return hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bounded_controls(values, shape):
    values = np.asarray(values, float)
    if (values.shape != shape or not np.isfinite(values).all()
            or (np.abs(values[..., 0]) > 1).any()
            or (values[..., 1:] < 0).any() or (values[..., 1:] > 1).any()):
        raise ValueError("Explicit finite bounded steer/throttle/brake commands required")
    return values


class FrozenExpertEgo:
    """Uses only current ego pose/speed, previous commands and proposed controls.

    The vendor polynomial treats any positive brake as full braking and holds
    speed below its throttle threshold. Those limitations remain unchanged.
    Better factual error is not physical counterfactual qualification.
    """

    def __init__(self, descriptor):
        if isinstance(descriptor, (str, Path)):
            descriptor = json.loads(Path(descriptor).read_text())
        parameters = descriptor["parameters"]
        names = {
            "time_step", "front_wheel_base", "rear_wheel_base", "steering_gain",
            "throttle_values", "brake_values", "throttle_threshold_during_forecasting",
        }
        if (
            descriptor.get("schema") != "navigation-frozen-expert-ego-v1"
            or descriptor.get("lag_ticks") != 2
            or set(parameters) != names
            or parameter_hash(parameters) != descriptor.get("parameters_sha256")
            or parameters["time_step"] != .05
            or np.shape(parameters["throttle_values"]) != (8,)
            or np.shape(parameters["brake_values"]) != (7,)
            or parameters["rear_wheel_base"] <= 0
            or parameters["front_wheel_base"] + parameters["rear_wheel_base"] <= 0
        ):
            raise ValueError(
                "Exact source-identified frozen expert parameters and TRAIN lag required"
            )
        self.parameters = {
            key: np.asarray(value, float) if isinstance(value, list) else float(value)
            for key, value in parameters.items()
        }
        values = np.concatenate([np.atleast_1d(v) for v in self.parameters.values()])
        if not np.isfinite(values).all():
            raise ValueError("Frozen model parameters must be finite")
        self.descriptor = dict(descriptor)
        self.lag_ticks = 2

    def step(self, current_state, executed_control):
        state = np.asarray(current_state, float)
        command = bounded_controls(executed_control, (3,))
        if state.shape != (4,) or not np.isfinite(state).all() or state[3] < 0:
            raise ValueError("Only finite current ego x/y/yaw/speed may initialize propagation")
        p = self.parameters
        steer, throttle, brake = command
        x, y, yaw, speed = state
        beta = np.arctan(p["rear_wheel_base"] / (
            p["front_wheel_base"] + p["rear_wheel_base"]) * np.tan(p["steering_gain"] * steer))
        next_state = [
            x + speed * np.cos(yaw + beta) * .05,
            y + speed * np.sin(yaw + beta) * .05,
            yaw + speed / p["rear_wheel_base"] * np.sin(beta) * .05,
        ]
        kph = speed * 3.6
        if brake:
            next_speed = np.dot(kph ** np.arange(1, 8), p["brake_values"]) / 3.6
        elif throttle < p["throttle_threshold_during_forecasting"]:
            next_speed = speed
        else:
            values = [kph, kph**2, throttle, throttle**2, kph * throttle,
                      kph * throttle**2, kph**2 * throttle, kph**2 * throttle**2]
            next_speed = np.dot(values, p["throttle_values"]) / 3.6
        result = np.r_[next_state, max(0, next_speed)]
        if not np.isfinite(result).all():
            raise FloatingPointError("Ego model left its finite operating range")
        return result

    def rollout(self, current_state, ordered_controls, previous_controls):
        controls = bounded_controls(ordered_controls, (80, 3))
        history = bounded_controls(previous_controls, (2, 3))
        delayed = np.vstack([history, controls[:-2]])
        state = np.asarray(current_state, float).copy()
        rows = []
        for command in delayed:
            state = self.step(state, command)
            rows.append(state.copy())
        return np.stack(rows)
