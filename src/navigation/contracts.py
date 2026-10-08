"""The camera-policy boundary contains arrays and values, never CARLA objects."""

from dataclasses import dataclass

import numpy as np

VIEWS = ("front", "left", "right")


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    fov: float
    # CARLA coordinates: x forward, y right, z up; rotations in degrees.
    xyz: tuple[float, float, float]
    pitch_yaw_roll: tuple[float, float, float]

    @property
    def intrinsic(self) -> np.ndarray:
        focal = self.width / (2 * np.tan(np.deg2rad(self.fov) / 2))
        return np.array(
            [[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]], dtype=np.float32
        )


@dataclass(frozen=True)
class SensorPacket:
    frame: int
    timestamp: float
    rgb: dict[str, np.ndarray]
    calibration: dict[str, CameraCalibration]
    navigation_waypoints: np.ndarray  # (N, 2), metres in the ego frame
    ego_motion: np.ndarray  # speed m/s, heading radians, yaw rate rad/s

    def validate(self) -> None:
        if self.frame < 0 or not np.isfinite(self.timestamp):
            raise ValueError("Invalid frame/timestamp")
        if set(self.rgb) != set(VIEWS) or set(self.calibration) != set(VIEWS):
            raise ValueError("Exactly front, left and right RGB views are required")
        for view in VIEWS:
            a, c = self.rgb[view], self.calibration[view]
            if a.dtype != np.uint8 or a.shape != (c.height, c.width, 3):
                raise ValueError(f"Invalid {view} RGB geometry/dtype")
        if self.navigation_waypoints.ndim != 2 or self.navigation_waypoints.shape[1] != 2:
            raise ValueError("Navigation must be an (N,2) ego-frame array")
        if not len(self.navigation_waypoints) or not np.isfinite(self.navigation_waypoints).all():
            raise ValueError("Navigation is empty or non-finite")
        if self.ego_motion.shape != (3,) or not np.isfinite(self.ego_motion).all():
            raise ValueError("Invalid ego motion")


@dataclass(frozen=True)
class TrainingTargets:
    """Collection-side labels; this class is forbidden at the policy boundary."""

    frame: int
    expert_control: tuple[float, float, float]  # steer, throttle, brake
    executed_control: tuple[float, float, float]
    intervention: str | None
    facts: dict


@dataclass(frozen=True)
class ControlCommand:
    steer: float
    throttle: float
    brake: float

    def validate(self) -> None:
        if not np.isfinite([self.steer, self.throttle, self.brake]).all():
            raise ValueError("Non-finite controls")
        if not (-1 <= self.steer <= 1 and 0 <= self.throttle <= 1 and 0 <= self.brake <= 1):
            raise ValueError("Controls outside actuator bounds")
        if self.throttle > 0 and self.brake > 0:
            raise ValueError("Conflicting throttle and brake")


@dataclass(frozen=True)
class TrajectoryPlan:
    frame: int
    timestamp: float
    waypoints: np.ndarray  # (8,2)
    speeds: np.ndarray  # (8,)
    risks: np.ndarray

    def validate(self) -> None:
        if self.waypoints.shape != (8, 2) or self.speeds.shape != (8,):
            raise ValueError("Expected eight path/speed waypoints")
        for value in (self.waypoints, self.speeds, self.risks):
            if not np.isfinite(value).all():
                raise ValueError("Non-finite plan")
        if (self.speeds < 0).any():
            raise ValueError("Negative target speed")
