"""Shared candidate ego model: current odometry and ordered controls only."""

import numpy as np


class EgoKinematics:
    def __init__(self, parameters):
        names = (
            "throttle_accel",
            "brake_decel",
            "drag",
            "accel_bias",
            "maximum_steer_angle_rad",
            "wheelbase_m",
        )
        if set(parameters) != set(names) or not np.isfinite(list(parameters.values())).all():
            raise ValueError("Explicit finite fitted ego-actuation parameters required")
        self.parameters = dict(parameters)
        if (
            min(parameters["throttle_accel"], parameters["brake_decel"], parameters["drag"]) < 0
            or not 0.3 <= parameters["maximum_steer_angle_rad"] <= 1.3
            or parameters["wheelbase_m"] != 2.8
        ):
            raise ValueError("Declared kinematic identification bounds changed")

    def step(self, state, control, dt=0.05):
        # State x/y/yaw/speed is initialized from current approved inputs and
        # advanced internally, never supplied from a future simulator state.
        state = np.asarray(state, float)
        control = np.asarray(control, float)
        if (
            state.shape != (4,)
            or control.shape != (3,)
            or dt != 0.05
            or not np.isfinite(np.r_[state, control]).all()
            or state[3] < 0
            or not -1 <= control[0] <= 1
            or (control[1:] < 0).any()
            or (control[1:] > 1).any()
        ):
            raise ValueError("Finite current state and bounded20Hz actuator command required")
        p = self.parameters
        accel = (
            p["throttle_accel"] * control[1]
            - p["brake_decel"] * control[2]
            - p["drag"] * state[3]
            + p["accel_bias"]
        )
        speed = max(0.0, state[3] + dt * accel)
        average = 0.5 * (state[3] + speed)
        rate = average / p["wheelbase_m"] * np.tan(control[0] * p["maximum_steer_angle_rad"])
        heading = state[2] + dt * rate / 2
        return np.array(
            [
                state[0] + dt * average * np.cos(heading),
                state[1] + dt * average * np.sin(heading),
                state[2] + dt * rate,
                speed,
            ]
        )

    def rollout(self, current_state, ordered_controls):
        controls = np.asarray(ordered_controls, float)
        if controls.shape != (80, 3):
            raise ValueError("Complete4second80control horizon required")
        state = np.asarray(current_state, float).copy()
        states = []
        for control in controls:
            state = self.step(state, control)
            states.append(state.copy())
        return np.stack(states)
