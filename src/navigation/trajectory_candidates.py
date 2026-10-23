"""Shared15 camera-planning candidates; no oracle/teacher or outcome-dependent edits."""

from dataclasses import dataclass

import numpy as np

from src.navigation.controller import corrected_path

PATH_OFFSETS = (-0.75, 0.0, 0.75)
SPEED_PROFILES = ("stop", "slow", "cruise", "faster", "wait_then_go")
TIMES = np.arange(0.5, 4.01, 0.5)


@dataclass(frozen=True)
class Candidate:
    path_index: int
    speed_index: int
    correction: np.ndarray
    speeds: np.ndarray
    waypoints: np.ndarray
    times: np.ndarray
    route_progress_m: np.ndarray
    actual_speed_mps: float


def ordered_projection(points, navigation):
    route = np.vstack([np.zeros((1, 2)), navigation])
    delta = np.diff(route, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    arc = np.r_[0, np.cumsum(lengths)]
    previous = 0
    progress = []
    for point in points:
        start = max(0, previous - 1)
        stop = min(len(delta), previous + 8)
        relative = point - route[start:stop]
        ratio = np.clip(
            (relative * delta[start:stop]).sum(1) / np.maximum(lengths[start:stop] ** 2, 1e-8), 0, 1
        )
        nearest = route[start:stop] + ratio[:, None] * delta[start:stop]
        local = int(np.argmin(np.linalg.norm(nearest - point, axis=1)))
        index = start + local
        previous = max(previous, index)
        progress.append(arc[index] + ratio[local] * lengths[index])
    return np.array(progress)


def candidates(navigation, correction, cruise_speeds, actual_speed):
    navigation = np.asarray(navigation, float)
    correction = np.asarray(correction, float)
    cruise_speeds = np.asarray(cruise_speeds, float)
    if (
        navigation.shape != (16, 2)
        or correction.shape != (8,)
        or cruise_speeds.shape != (8,)
        or not np.isfinite(np.r_[navigation.ravel(), correction, cruise_speeds, actual_speed]).all()
        or actual_speed < 0
    ):
        raise ValueError("Finite declared route/camera proposals and actual ego speed required")
    cruise = np.clip(cruise_speeds, 0, 8)
    profiles = [
        np.zeros(8),
        cruise * 0.4,
        cruise,
        np.minimum(8, cruise * 1.2),
        np.r_[np.zeros(2), cruise[2:]],
    ]
    output = []
    for path_index, offset in enumerate(PATH_OFFSETS):
        shifted = np.clip(correction + offset, -2, 2)
        path = corrected_path(navigation, shifted)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        for speed_index, profile in enumerate(profiles):
            # Candidate speeds obey the same acceleration/deceleration envelope
            # regardless of predictor; hard-stop means physically bounded braking.
            speed = actual_speed
            executed = []
            distance = []
            travel = 0.0
            stopped_seconds = 0.0
            for desired in profile:
                if SPEED_PROFILES[speed_index] == "wait_then_go":
                    desired = 0.0 if stopped_seconds < 1.0 else desired
                # The8m/s cap is a desired speed, not an instantaneous change
                # in measured overspeed. Preserve bounded braking above the cap.
                next_speed = np.clip(desired, max(0, speed - 3.5 * 0.5), speed + 2.5 * 0.5)
                travel += (speed + next_speed) * 0.25
                distance.append(travel)
                executed.append(next_speed)
                if next_speed < 0.1:
                    stopped_seconds += 0.5
                speed = next_speed
            xy = np.column_stack([np.interp(distance, arc, path[:, i]) for i in (0, 1)])
            output.append(
                Candidate(
                    path_index,
                    speed_index,
                    shifted.copy(),
                    np.array(executed),
                    xy,
                    TIMES.copy(),
                    ordered_projection(xy, navigation),
                    float(actual_speed),
                )
            )
    return output


def common_cost(candidate, collision_probability, red_probability, progress_m=None):
    collision = np.asarray(collision_probability, float)
    red = np.asarray(red_probability, float)
    if (
        collision.shape != (8,)
        or red.shape != (8,)
        or not np.isfinite(np.r_[collision, red]).all()
        or (collision < 0).any()
        or (collision > 1).any()
        or (red < 0).any()
        or (red > 1).any()
    ):
        raise ValueError("Eight finite shared-horizon risk scores required; calibrate separately")
    progress = float(candidate.route_progress_m[-1]) if progress_m is None else float(progress_m)
    if not np.isfinite(progress):
        raise ValueError("Finite progress required")
    acceleration = np.diff(np.r_[candidate.actual_speed_mps, candidate.speeds]) / 0.5
    return float(
        1000 * collision.max()
        + 300 * red.max()
        - progress
        + 0.2 * np.square(acceleration).mean()
        + 0.1 * np.square(candidate.correction).mean()
    )
