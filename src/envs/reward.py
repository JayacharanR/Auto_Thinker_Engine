"""
EPDMS-inspired reward function for CARLA/CarDreamer driving tasks.

This module implements a config-driven reward function combining five terms:
1. Collision penalty (terminal)
2. Time-to-collision proxy
3. Lane-keeping (lateral deviation)
4. Progress (forward along route)
5. Traffic-rule compliance

CRITICAL: This reward function is written ONCE (Phase 1) and reused
UNCHANGED in Phase 3. If it changes, the three-arm comparison is
no longer controlled. All term weights are loaded from YAML config.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class RewardTermResult:
    """Result from a single reward term computation."""

    value: float
    raw_value: float  # Unweighted value (for logging)
    terminal: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class RewardResult:
    """Aggregated reward result with per-term breakdown."""

    total: float
    terms: dict[str, RewardTermResult] = field(default_factory=dict)
    terminal: bool = False

    def to_log_dict(self) -> dict[str, float]:
        """Convert to flat dict for logging."""
        d = {"reward/total": self.total}
        for name, term in self.terms.items():
            d[f"reward/{name}"] = term.raw_value
            d[f"reward/{name}_weighted"] = term.value
        return d


class RewardFunction:
    """
    Config-driven reward function for CARLA driving tasks.

    Each reward term is independently weighted and can be enabled/disabled.
    Term weights are loaded from the YAML config file and MUST NOT be
    modified between Phase 1 and Phase 3.

    Args:
        config: Reward section of the YAML config. Expected structure:
            collision:
              weight: float
              terminal: bool
            time_to_collision:
              weight: float
              safe_threshold: float
              decay: str
            lane_keeping:
              weight: float
              max_deviation: float
            progress:
              weight: float
              scale: float
            traffic_rules:
              weight: float
              stop_sign_dwell: float
              red_light_penalty: float
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Ensure all required reward terms are present in config."""
        required_terms = [
            "collision",
            "time_to_collision",
            "lane_keeping",
            "progress",
            "traffic_rules",
        ]
        for term in required_terms:
            if term not in self.config:
                raise ValueError(
                    f"Missing reward term '{term}' in config. "
                    "All five EPDMS-inspired terms are required."
                )
            if "weight" not in self.config[term]:
                raise ValueError(f"Reward term '{term}' missing 'weight' in config.")

    def compute(self, obs: dict[str, Any]) -> RewardResult:
        """
        Compute the total reward from environment observations.

        Args:
            obs: Dictionary of observations from the CARLA environment.
                Expected keys (provided by CarDreamer's Observer-Handler):
                - 'collision': bool — whether a collision occurred this step
                - 'velocity': np.ndarray — ego vehicle velocity [vx, vy, vz]
                - 'distance_to_leading': float — distance to vehicle ahead (meters)
                - 'relative_velocity': float — closing speed to leading vehicle
                - 'lateral_deviation': float — deviation from lane center (meters)
                - 'route_progress': float — progress along planned route (meters)
                - 'traffic_light_state': str — 'red', 'green', 'yellow', or 'none'
                - 'at_stop_sign': bool — whether at a stop sign
                - 'stop_sign_wait_time': float — seconds spent waiting at stop sign

        Returns:
            RewardResult with total reward, per-term breakdown, and terminal flag.
        """
        result = RewardResult(total=0.0)

        # 1. Collision penalty
        collision_result = self._compute_collision(obs)
        result.terms["collision"] = collision_result
        result.total += collision_result.value
        if collision_result.terminal:
            result.terminal = True

        # 2. Time-to-collision proxy
        ttc_result = self._compute_ttc(obs)
        result.terms["time_to_collision"] = ttc_result
        result.total += ttc_result.value

        # 3. Lane keeping
        lane_result = self._compute_lane_keeping(obs)
        result.terms["lane_keeping"] = lane_result
        result.total += lane_result.value

        # 4. Progress
        progress_result = self._compute_progress(obs)
        result.terms["progress"] = progress_result
        result.total += progress_result.value

        # 5. Traffic rules
        traffic_result = self._compute_traffic_rules(obs)
        result.terms["traffic_rules"] = traffic_result
        result.total += traffic_result.value

        return result

    def _compute_collision(self, obs: dict[str, Any]) -> RewardTermResult:
        """Collision: large negative penalty, terminates episode."""
        cfg = self.config["collision"]
        collided = obs.get("collision", False)

        if collided:
            return RewardTermResult(
                value=cfg["weight"],
                raw_value=1.0,
                terminal=cfg.get("terminal", True),
                info={"collided": True},
            )
        return RewardTermResult(value=0.0, raw_value=0.0)

    def _compute_ttc(self, obs: dict[str, Any]) -> RewardTermResult:
        """
        Time-to-collision proxy.

        Penalizes when closing distance to leading vehicle falls below
        a safe threshold. Uses a configurable decay function.
        """
        cfg = self.config["time_to_collision"]
        distance = obs.get("distance_to_leading", float("inf"))
        relative_vel = obs.get("relative_velocity", 0.0)

        # Compute TTC (time to collision)
        if relative_vel > 0.1:  # Approaching the vehicle ahead
            ttc = distance / relative_vel
        else:
            ttc = float("inf")  # Not approaching or moving away

        safe_threshold = cfg.get("safe_threshold", 3.0)

        if ttc >= safe_threshold or ttc == float("inf"):
            return RewardTermResult(value=0.0, raw_value=ttc, info={"ttc": ttc})

        # Penalize based on decay function
        decay = cfg.get("decay", "linear")
        if decay == "linear":
            penalty = 1.0 - (ttc / safe_threshold)  # 0 at threshold, 1 at ttc=0
        elif decay == "exponential":
            penalty = np.exp(-ttc / safe_threshold)
        else:
            penalty = 1.0 - (ttc / safe_threshold)

        reward = cfg["weight"] * penalty
        return RewardTermResult(
            value=reward, raw_value=penalty, info={"ttc": ttc, "penalty": penalty}
        )

    def _compute_lane_keeping(self, obs: dict[str, Any]) -> RewardTermResult:
        """
        Lane keeping: penalize lateral deviation from centerline.

        Deviation is clamped to [0, max_deviation] and normalized.
        """
        cfg = self.config["lane_keeping"]
        deviation = abs(obs.get("lateral_deviation", 0.0))
        max_dev = cfg.get("max_deviation", 2.0)

        # Clamp and normalize
        normalized = min(deviation, max_dev) / max_dev  # 0 = on center, 1 = at max

        reward = cfg["weight"] * normalized
        return RewardTermResult(
            value=reward,
            raw_value=deviation,
            info={"lateral_deviation": deviation, "normalized": normalized},
        )

    def _compute_progress(self, obs: dict[str, Any]) -> RewardTermResult:
        """
        Progress: reward forward movement along the planned route.

        This is a per-step distance delta, scaled by the config scale factor.
        """
        cfg = self.config["progress"]
        progress = obs.get("route_progress", 0.0)  # meters traveled this step
        scale = cfg.get("scale", 0.01)

        reward = cfg["weight"] * progress * scale
        return RewardTermResult(
            value=reward, raw_value=progress, info={"progress_meters": progress}
        )

    def _compute_traffic_rules(self, obs: dict[str, Any]) -> RewardTermResult:
        """
        Traffic rule compliance.

        Handles stop signs (dwell time check) and traffic lights.
        """
        cfg = self.config["traffic_rules"]
        penalty = 0.0
        info: dict[str, Any] = {}

        # Red light violation
        light_state = obs.get("traffic_light_state", "none")
        if light_state == "red":
            # Check if vehicle is moving through red light
            velocity = obs.get("velocity", np.zeros(3))
            speed = np.linalg.norm(velocity)
            if speed > 0.5:  # Moving through red light
                red_penalty = cfg.get("red_light_penalty", -5.0)
                penalty += abs(red_penalty)  # penalty is accumulated as positive
                info["red_light_violation"] = True

        # Stop sign compliance
        at_stop_sign = obs.get("at_stop_sign", False)
        if at_stop_sign:
            wait_time = obs.get("stop_sign_wait_time", 0.0)
            required_dwell = cfg.get("stop_sign_dwell", 2.0)
            velocity = obs.get("velocity", np.zeros(3))
            speed = np.linalg.norm(velocity)

            if speed > 0.5 and wait_time < required_dwell:
                # Moving through stop sign without waiting
                penalty += 1.0
                info["stop_sign_violation"] = True
                info["wait_time"] = wait_time
            else:
                info["stop_sign_compliant"] = True

        reward = cfg["weight"] * penalty if penalty > 0 else 0.0
        return RewardTermResult(value=reward, raw_value=penalty, info=info)


def create_reward_function(config_path: str) -> RewardFunction:
    """
    Create a RewardFunction from a YAML config file.

    Args:
        config_path: Path to YAML config containing a 'reward' section.

    Returns:
        Configured RewardFunction instance.
    """
    import yaml

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    if "reward" not in full_config:
        raise ValueError(f"Config file {config_path} missing 'reward' section.")

    return RewardFunction(full_config["reward"])
