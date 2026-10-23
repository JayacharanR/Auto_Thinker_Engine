"""Scripted setup for controlled rule tests; never an evaluated driving policy.

Geometry is privileged SETUP information. Once handed over, this controller
returns None forever and must never alter the evaluated agent's controls.
"""

from dataclasses import dataclass

import numpy as np

from src.navigation.controller import track


@dataclass(frozen=True)
class PrefixSpec:
    mode: str = "none"
    speed_mps: float = 6.0
    handoff_distance_m: float = 3.0
    stationary_seconds: float = 1.0
    maximum_seconds: float = 30.0

    def validate(self):
        if self.mode not in ("none", "stop_before_line", "cruise_to_handoff"):
            raise ValueError("Unknown controlled-test prefix")
        numbers = [
            self.speed_mps,
            self.handoff_distance_m,
            self.stationary_seconds,
            self.maximum_seconds,
        ]
        if (
            not np.isfinite(numbers).all()
            or not 0 < self.speed_mps <= 6
            or not 2 <= self.handoff_distance_m <= 40
            or not 0 <= self.stationary_seconds <= 3
            or not 0 < self.maximum_seconds <= 45
        ):
            raise ValueError("Controlled-test prefix is outside its registered bounds")


class RulePrefix:
    def __init__(self, spec):
        spec.validate()
        self.spec = spec
        self.started = self.stationary_since = self.handoff_time = None
        self.last_timestamp = None

    def step(self, timestamp, navigation, speed, distance_to_line):
        if not np.isfinite([timestamp, speed, distance_to_line]).all() or speed < 0:
            raise ValueError("Finite physical prefix measurements required")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Controlled setup requires increasing timestamps")
        self.last_timestamp = timestamp
        self.started = timestamp if self.started is None else self.started
        # No measured line distance, light state, or speed can change an agent
        # command after handover. The caller uses the original proposed control.
        if self.handoff_time is not None:
            return None
        if self.spec.mode == "none":
            self.handoff_time = timestamp
            return None
        if timestamp - self.started > self.spec.maximum_seconds:
            raise RuntimeError("Scripted setup timed out; do not silently force a handover")
        if self.spec.mode == "cruise_to_handoff":
            if distance_to_line <= self.spec.handoff_distance_m:
                self.handoff_time = timestamp
                return None
            target = self.spec.speed_mps
        else:
            remaining = distance_to_line - self.spec.handoff_distance_m
            if distance_to_line < 1.0 or remaining < -0.75:
                raise RuntimeError("Scripted stopping setup overshot its registered pose")
            at_pose = abs(remaining) <= 0.5 and speed < 0.1
            self.stationary_since = (
                (timestamp if self.stationary_since is None else self.stationary_since)
                if at_pose
                else None
            )
            if (
                self.stationary_since is not None
                and timestamp - self.stationary_since >= self.spec.stationary_seconds
            ):
                self.handoff_time = timestamp
                return None
            target = min(self.spec.speed_mps, float(np.sqrt(2 * 2.5 * max(0, remaining))))
        return track(navigation, np.zeros(8), speed, target, 0.0)

    def state(self, timestamp):
        return dict(
            scripted_setup=self.handoff_time is None,
            handoff_time=self.handoff_time,
            seconds_since_handoff=None
            if self.handoff_time is None
            else timestamp - self.handoff_time,
            privileged_setup_geometry=True,
            control_after_handoff="evaluated agent proposal, unchanged",
        )
