"""Shared camera-derived candidate commands; no oracle or future-state inputs."""

from dataclasses import dataclass

import numpy as np

from src.navigation.controller import corrected_path, track
from src.navigation.frozen_ego import bounded_controls
from src.navigation.trajectory_candidates import Candidate, candidates, ordered_projection


@dataclass(frozen=True)
class ControlCandidate:
    desired: Candidate
    realized: Candidate
    ordered_controls: np.ndarray
    forecast_states: np.ndarray


def future_vertices(path, state, previous_segment):
    """Current synthetic position can only advance through the ordered path."""
    delta = np.diff(path, axis=0)
    starts = path[previous_segment:-1]
    directions = delta[previous_segment:]
    ratios = np.clip(((state[:2] - starts) * directions).sum(1) /
                     np.maximum(np.square(directions).sum(1), 1e-8), 0, 1)
    nearest = starts + ratios[:, None] * directions
    index = previous_segment + int(np.argmin(np.linalg.norm(nearest - state[:2], axis=1)))
    vertices = path[np.minimum(np.arange(index + 1, index + 9), len(path) - 1)]
    cosine, sine = np.cos(state[2]), np.sin(state[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    local = (vertices - state[:2]) @ rotation
    # track() selects odd-index vertices, reproducing the exact same polyline.
    return np.repeat(local, 2, axis=0), index


def candidate_commands(navigation, desired, previous_controls, ego_model):
    navigation = np.asarray(navigation, float)
    history = bounded_controls(previous_controls, (2, 3)).copy()
    if navigation.shape != (16, 2) or not np.isfinite(navigation).all():
        raise ValueError("Current declared16-point navigation required")
    if desired.speeds.shape != (8,) or not np.isfinite(desired.speeds).all():
        raise ValueError("Registered eight-horizon desired speed profile required")
    path = corrected_path(navigation, desired.correction)
    state = np.array([0.0, 0.0, 0.0, desired.actual_speed_mps])
    actions, forecasts, segment = [], [], 0
    for tick in range(80):
        local, segment = future_vertices(path, state, segment)
        control = track(local, np.zeros(8), float(state[3]),
                        float(desired.speeds[tick // 10]), 0.0)
        command = np.array([control.steer, control.throttle, control.brake])
        # The accepted command acts only after the measured two-tick delay.
        state = ego_model.step(state, history[0])
        history = np.vstack([history[1:], command])
        actions.append(command)
        forecasts.append(state.copy())
    forecasts, actions = np.stack(forecasts), np.stack(actions)
    selected = forecasts[9::10]
    realized = Candidate(
        desired.path_index, desired.speed_index, desired.correction.copy(),
        selected[:, 3].copy(), selected[:, :2].copy(), desired.times.copy(),
        ordered_projection(selected[:, :2], navigation), desired.actual_speed_mps,
    )
    return ControlCandidate(desired, realized, actions, forecasts)


def control_candidates(navigation, correction, cruise_speeds, actual_speed,
                       previous_controls, ego_model):
    """All predictors score these same15 plans and the same ordered commands."""
    return [candidate_commands(navigation, proposal, previous_controls, ego_model)
            for proposal in candidates(navigation, correction, cruise_speeds, actual_speed)]
