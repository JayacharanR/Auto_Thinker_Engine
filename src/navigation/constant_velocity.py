"""Camera-derived track forecasting baseline, never simulator actor state."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerceivedTrack:
    id: int
    position_xy: np.ndarray
    velocity_xy: np.ndarray
    half_length: float = 2.5
    half_width: float = 1.1
    position_sigma_m: float = 0.5
    velocity_sigma_mps: float = 1.0
    confidence: float = 1.0
    age_seconds: float = 0.0

    def validate(self):
        if (
            np.shape(self.position_xy) != (2,)
            or np.shape(self.velocity_xy) != (2,)
            or not np.isfinite(
                np.r_[
                    self.position_xy,
                    self.velocity_xy,
                    self.half_length,
                    self.half_width,
                    self.position_sigma_m,
                    self.velocity_sigma_mps,
                    self.confidence,
                    self.age_seconds,
                ]
            ).all()
            or min(self.half_length, self.half_width) <= 0
            or min(self.position_sigma_m, self.velocity_sigma_mps, self.age_seconds) < 0
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Finite camera-derived track geometry/uncertainty required")


def forecast(track, times):
    track.validate()
    times = np.asarray(times, float)
    if times.shape != (8,) or (times <= 0).any() or not np.isfinite(times).all():
        raise ValueError("Positive shared eight-step future required")
    positions = track.position_xy[None] + times[:, None] * track.velocity_xy[None]
    sigma = track.position_sigma_m + (times + track.age_seconds) * track.velocity_sigma_mps
    return positions, sigma


def collision_risk(candidate, tracks, ego_half_length=2.5, ego_half_width=1.1):
    """Conservative Gaussian proximity score; calibrate before calling a probability."""
    risks = []
    direction = np.diff(np.vstack([np.zeros((1, 2)), candidate.waypoints]), axis=0)
    heading = np.arctan2(direction[:, 1], direction[:, 0])
    for index in range(1, len(heading)):
        if np.linalg.norm(direction[index]) < 1e-6:
            heading[index] = heading[index - 1]
    for track in tracks:
        predicted, sigma = forecast(track, candidate.times)
        delta = predicted - candidate.waypoints
        longitudinal = delta[:, 0] * np.cos(heading) + delta[:, 1] * np.sin(heading)
        lateral = -delta[:, 0] * np.sin(heading) + delta[:, 1] * np.cos(heading)
        # Unknown vehicle orientation gets a circular track envelope, rather
        # than silently assuming crossing actors are aligned with ego.
        radius = float(np.hypot(track.half_length, track.half_width))
        gap_x = np.maximum(0, np.abs(longitudinal) - ego_half_length - radius)
        gap_y = np.maximum(0, np.abs(lateral) - ego_half_width - radius)
        score = np.exp(-0.5 * (gap_x**2 + gap_y**2) / np.maximum(sigma**2, 1e-6))
        risks.append(score)
    return np.max(risks, axis=0) if risks else np.zeros(8)
