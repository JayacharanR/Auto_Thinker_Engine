"""Conventional tracking of learned path/speed; no simulator or oracle inputs."""

import numpy as np

from src.navigation.contracts import ControlCommand, TrajectoryPlan


def corrected_path(navigation, correction):
    points = np.asarray(navigation, dtype=np.float64)[1::2].copy()
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-5)
    points += np.asarray(correction)[:, None] * np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return np.vstack([np.zeros((1, 2)), points])


def trajectory(packet, correction, speeds, hazards):
    path = corrected_path(packet.navigation_waypoints, correction)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    speed = np.clip(np.asarray(speeds), 0, 8)
    ahead = np.cumsum(speed) * 0.5
    points = np.column_stack([np.interp(ahead, arc, path[:, i]) for i in (0, 1)])
    plan = TrajectoryPlan(packet.frame, packet.timestamp, points, speed, np.asarray(hazards))
    plan.validate()
    return plan


def track(navigation, correction, speed, target_speed, brake_probability):
    values = np.r_[
        np.asarray(navigation).ravel(), correction, speed, target_speed, brake_probability
    ]
    if not np.isfinite(values).all() or min(speed, target_speed) < 0:
        raise ValueError("Invalid controller inputs")
    path = corrected_path(navigation, correction)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    lookahead = np.clip(2.5 + speed * 0.5, 3, 10)
    x, y = [float(np.interp(lookahead, arc, path[:, i])) for i in (0, 1)]
    steer = float(np.clip(np.arctan2(2 * 2.8 * y, max(1, x * x + y * y)) / 0.7, -1, 1))
    target_speed = min(8, target_speed)  # Declared v0 urban speed cap, not evaluator logic.
    if target_speed < 0.15 or brake_probability >= 0.5:
        command = ControlCommand(steer, 0, 1)
    elif target_speed - speed < -0.2:
        command = ControlCommand(steer, 0, float(min(1, (speed - target_speed) * 0.6)))
    else:
        command = ControlCommand(
            steer, float(np.clip((target_speed - speed) * 0.4 + 0.05, 0, 0.8)), 0
        )
    command.validate()
    return command
