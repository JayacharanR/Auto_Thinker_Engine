"""
Independent Safety Supervisor and Action Shield (P5).

An independent safety supervisor sits between the learned Dreamer policy and the
vehicle actuation. It enforces action bounds, steering rate / jerk limits,
TTC-based emergency braking, lane-boundary departure guards, and an
anti-stall crawl watchdog to prevent stationary freezing on clear roadways.

Action space semantics (CarDreamer continuous Box(-1, 1, shape=(2,))):
    action[0]: Acceleration / throttle_brake (positive = throttle, negative = brake)
    action[1]: Steering angle (positive = right, negative = left)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SafetyIntervention:
    """Record of a single safety supervisor override."""

    step: int
    original_action: List[float]
    safe_action: List[float]
    reasons: List[str]
    ttc: Optional[float] = None
    lateral_deviation: Optional[float] = None
    speed_kmh: Optional[float] = None


class SafetySupervisor:
    """
    Action shield and emergency intervention watchdog.

    Args:
        max_steer_rate: Maximum steering change per simulation step.
        max_throttle_rate: Maximum throttle/acceleration increase per step.
        ttc_threshold_sec: Time-to-collision threshold for emergency brake.
        max_lateral_deviation_m: Maximum allowable distance from waypoint line.
        anti_stall_enabled: Whether to nudge vehicle forward when stalled on clear roads.
        stall_timeout_steps: Steps of near-zero speed before anti-stall activates.
        crawl_acceleration: Forward acceleration applied during anti-stall crawl.
        enabled: Master enable switch.
    """

    def __init__(
        self,
        max_steer_rate: float = 0.3,
        max_throttle_rate: float = 0.5,
        ttc_threshold_sec: float = 1.2,
        max_lateral_deviation_m: float = 1.8,
        anti_stall_enabled: bool = True,
        stall_timeout_steps: int = 15,
        min_crawl_steps: int = 35,
        crawl_acceleration: float = 0.35,
        enabled: bool = True,
    ):
        self.max_steer_rate = float(max_steer_rate)
        self.max_throttle_rate = float(max_throttle_rate)
        self.ttc_threshold_sec = float(ttc_threshold_sec)
        self.max_lateral_deviation_m = float(max_lateral_deviation_m)
        self.anti_stall_enabled = bool(anti_stall_enabled)
        self.stall_timeout_steps = int(stall_timeout_steps)
        self.min_crawl_steps = int(min_crawl_steps)
        self.crawl_acceleration = float(crawl_acceleration)
        self.enabled = bool(enabled)

        self._prev_action: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._stalled_steps: int = 0
        self._crawl_active_steps: int = 0
        self.interventions: List[SafetyIntervention] = []

    def reset(self) -> None:
        """Reset internal temporal state at episode boundary."""
        self._prev_action = None
        self._step_count = 0
        self._stalled_steps = 0
        self._crawl_active_steps = 0
        self.interventions = []

    def filter_action(
        self,
        action: np.ndarray | List[float],
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Shield the policy action and enforce safety constraints.

        Args:
            action: Policy output [acc, steer] in [-1, 1].
            telemetry: Current vehicle state (TTC, speed, waypoint deviation, etc.).

        Returns:
            Tuple of (safe_action, intervention_info).
        """
        self._step_count += 1
        telemetry = telemetry or {}

        # 1. Sanitize input array: [acc, steer]
        action_arr = np.asarray(action, dtype=np.float32).copy()
        if action_arr.shape == () or action_arr.ndim == 0:
            action_arr = np.array([float(action_arr), 0.0], dtype=np.float32)
        elif len(action_arr) == 1:
            action_arr = np.array([float(action_arr[0]), 0.0], dtype=np.float32)
        elif len(action_arr) > 2:
            action_arr = action_arr[:2]

        orig_acc = float(action_arr[0])
        orig_steer = float(action_arr[1])
        reasons: List[str] = []

        if not self.enabled:
            safe_act = np.clip(action_arr, -1.0, 1.0)
            return safe_act, {"intervened": False, "reasons": []}

        # Check for NaN / Inf from model policy
        if np.isnan(action_arr).any() or np.isinf(action_arr).any():
            reasons.append("nan_inf_policy_failure")
            action_arr = np.array([-1.0, 0.0], dtype=np.float32)  # Emergency full brake, center steer

        acc = float(action_arr[0])
        steer = float(action_arr[1])

        # 2. Hard Action Space Clipping
        clipped_acc = float(np.clip(acc, -1.0, 1.0))
        clipped_steer = float(np.clip(steer, -1.0, 1.0))
        if clipped_acc != acc or clipped_steer != steer:
            reasons.append("action_bound_clip")
            acc, steer = clipped_acc, clipped_steer

        # 3. Rate & Jerk Limiting
        if self._prev_action is not None:
            prev_acc = float(self._prev_action[0])
            prev_steer = float(self._prev_action[1])

            # Steer rate limiting (action[1])
            steer_delta = steer - prev_steer
            if abs(steer_delta) > self.max_steer_rate:
                steer = prev_steer + np.sign(steer_delta) * self.max_steer_rate
                reasons.append("steering_rate_limit")

            # Throttle / Brake rate limiting (action[0]) (allow fast emergency braking, rate-limit acceleration)
            acc_delta = acc - prev_acc
            if acc_delta > self.max_throttle_rate:
                acc = prev_acc + self.max_throttle_rate
                reasons.append("throttle_rate_limit")

        # Telemetry metrics
        ttc = telemetry.get("ttc", telemetry.get("reward/time_to_collision", float("inf")))
        speed_norm = telemetry.get("speed_norm", telemetry.get("speed", 0.0))
        speed_kmh = float(speed_norm) * 3.6
        lateral_dev = telemetry.get("wpt_dis", telemetry.get("reward/lane_keeping", 0.0))

        # 4. TTC Collision Guard (Emergency Brake)
        if isinstance(ttc, (int, float)) and 0.0 < ttc < self.ttc_threshold_sec:
            if acc > -0.8:
                acc = -1.0  # Full brake
                reasons.append("emergency_brake_ttc")

        # 5. Lane Boundary / Drivable Area Guard
        if isinstance(lateral_dev, (int, float)) and abs(lateral_dev) > self.max_lateral_deviation_m:
            if acc > 0.0:
                acc = -0.3  # gentle deceleration
                reasons.append("lane_departure_slowdown")

        # 6. Traffic Light & Intersection Guard
        traffic_light = telemetry.get("traffic_light_state", None)
        if traffic_light in ("RED", "YELLOW_RED") or telemetry.get("red_light_hazard", False):
            if acc > -0.8:
                acc = -1.0  # Full stop at red light
                reasons.append("red_light_stop")

        # 7. Anti-Stall Crawl Watchdog with Hysteresis
        if self.anti_stall_enabled:
            is_stopped = speed_kmh < 0.8
            is_clear = (np.isinf(ttc) or ttc > 2.5) and ("red_light_stop" not in reasons)
            is_in_lane = isinstance(lateral_dev, (int, float)) and abs(lateral_dev) < self.max_lateral_deviation_m

            if not is_clear or not is_in_lane:
                self._crawl_active_steps = 0
                self._stalled_steps = 0
            else:
                if self._crawl_active_steps > 0:
                    self._crawl_active_steps += 1
                    # Yield crawl if policy commands forward acceleration proactively
                    if orig_acc > 0.1:
                        self._crawl_active_steps = 0
                        self._stalled_steps = 0
                    elif self._crawl_active_steps <= self.min_crawl_steps or speed_kmh < 12.0:
                        if acc < self.crawl_acceleration:
                            acc = self.crawl_acceleration
                            reasons.append("anti_stall_crawl")
                    else:
                        self._crawl_active_steps = 0
                        self._stalled_steps = 0
                else:
                    if is_stopped:
                        self._stalled_steps += 1
                        if self._stalled_steps >= self.stall_timeout_steps:
                            self._crawl_active_steps = 1
                            if acc < self.crawl_acceleration:
                                acc = self.crawl_acceleration
                                reasons.append("anti_stall_crawl")
                    else:
                        self._stalled_steps = 0

        safe_action = np.array([acc, steer], dtype=np.float32)
        self._prev_action = safe_action.copy()

        intervened = len(reasons) > 0
        intervention_info = {
            "intervened": intervened,
            "reasons": reasons,
            "original_action": [orig_acc, orig_steer],
            "safe_action": [float(acc), float(steer)],
        }

        if intervened:
            record = SafetyIntervention(
                step=self._step_count,
                original_action=[orig_acc, orig_steer],
                safe_action=[float(acc), float(steer)],
                reasons=reasons,
                ttc=float(ttc) if isinstance(ttc, (int, float)) and not np.isinf(ttc) else None,
                lateral_deviation=float(lateral_dev) if isinstance(lateral_dev, (int, float)) else None,
                speed_kmh=round(speed_kmh, 1),
            )
            self.interventions.append(record)

        return safe_action, intervention_info

    def get_intervention_summary(self) -> Dict[str, Any]:
        """Aggregate intervention statistics for reporting."""
        total = len(self.interventions)
        reason_counts: Dict[str, int] = {}
        for iv in self.interventions:
            for r in iv.reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        return {
            "total_interventions": total,
            "total_steps": self._step_count,
            "intervention_rate": round(float(total / max(1, self._step_count)), 4),
            "reasons": reason_counts,
        }
