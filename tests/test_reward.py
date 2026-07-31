"""
Unit tests for the reward function.

Tests each reward term independently and validates config-driven behavior.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.reward import RewardFunction, RewardResult


@pytest.fixture
def reward_config():
    return {
        "collision": {"weight": -10.0, "terminal": True},
        "time_to_collision": {"weight": -1.0, "safe_threshold": 3.0, "decay": "linear"},
        "lane_keeping": {"weight": -0.5, "max_deviation": 2.0, "normalization": "clamp"},
        "progress": {"weight": 1.0, "scale": 0.01},
        "traffic_rules": {"weight": -2.0, "stop_sign_dwell": 2.0, "red_light_penalty": -5.0},
    }


@pytest.fixture
def reward_fn(reward_config):
    return RewardFunction(reward_config)


def make_obs(**overrides):
    """Create a default observation dict with optional overrides."""
    obs = {
        "collision": False,
        "velocity": np.array([5.0, 0.0, 0.0]),
        "distance_to_leading": 50.0,
        "relative_velocity": 0.0,
        "lateral_deviation": 0.0,
        "route_progress": 1.0,
        "traffic_light_state": "green",
        "at_stop_sign": False,
        "stop_sign_wait_time": 0.0,
    }
    obs.update(overrides)
    return obs


class TestCollisionReward:
    def test_no_collision(self, reward_fn):
        result = reward_fn.compute(make_obs(collision=False))
        assert result.terms["collision"].value == 0.0
        assert not result.terminal

    def test_collision_penalty(self, reward_fn):
        result = reward_fn.compute(make_obs(collision=True))
        assert result.terms["collision"].value == -10.0
        assert result.terminal

    def test_collision_terminates(self, reward_fn):
        result = reward_fn.compute(make_obs(collision=True))
        assert result.terminal is True


class TestTTCReward:
    def test_safe_distance(self, reward_fn):
        result = reward_fn.compute(make_obs(distance_to_leading=100.0, relative_velocity=0.0))
        assert result.terms["time_to_collision"].value == 0.0

    def test_dangerous_approach(self, reward_fn):
        # TTC = 1.0s (distance=5, relative_vel=5), threshold=3.0s
        result = reward_fn.compute(make_obs(distance_to_leading=5.0, relative_velocity=5.0))
        assert result.terms["time_to_collision"].value < 0  # Should be penalized

    def test_not_approaching(self, reward_fn):
        result = reward_fn.compute(make_obs(distance_to_leading=5.0, relative_velocity=-2.0))
        assert result.terms["time_to_collision"].value == 0.0  # Moving away


class TestLaneKeeping:
    def test_on_center(self, reward_fn):
        result = reward_fn.compute(make_obs(lateral_deviation=0.0))
        assert result.terms["lane_keeping"].value == 0.0

    def test_small_deviation(self, reward_fn):
        result = reward_fn.compute(make_obs(lateral_deviation=1.0))
        # weight=-0.5, deviation=1.0, max=2.0 → -0.5 * 0.5 = -0.25
        assert result.terms["lane_keeping"].value == pytest.approx(-0.25)

    def test_max_deviation(self, reward_fn):
        result = reward_fn.compute(make_obs(lateral_deviation=5.0))  # Clamped to 2.0
        assert result.terms["lane_keeping"].value == pytest.approx(-0.5)


class TestProgress:
    def test_moving_forward(self, reward_fn):
        result = reward_fn.compute(make_obs(route_progress=10.0))
        # weight=1.0, scale=0.01, progress=10 → 0.1
        assert result.terms["progress"].value == pytest.approx(0.1)

    def test_stationary(self, reward_fn):
        result = reward_fn.compute(make_obs(route_progress=0.0))
        assert result.terms["progress"].value == 0.0


class TestTrafficRules:
    def test_green_light(self, reward_fn):
        result = reward_fn.compute(make_obs(traffic_light_state="green"))
        assert result.terms["traffic_rules"].value == 0.0

    def test_red_light_violation(self, reward_fn):
        result = reward_fn.compute(
            make_obs(traffic_light_state="red", velocity=np.array([5.0, 0.0, 0.0]))
        )
        assert result.terms["traffic_rules"].value < 0

    def test_red_light_stopped(self, reward_fn):
        result = reward_fn.compute(
            make_obs(traffic_light_state="red", velocity=np.array([0.0, 0.0, 0.0]))
        )
        assert result.terms["traffic_rules"].value == 0.0


class TestRewardAggregation:
    def test_log_dict(self, reward_fn):
        result = reward_fn.compute(make_obs())
        log_dict = result.to_log_dict()
        assert "reward/total" in log_dict
        assert "reward/collision" in log_dict

    def test_total_is_sum_of_terms(self, reward_fn):
        result = reward_fn.compute(make_obs())
        term_sum = sum(t.value for t in result.terms.values())
        assert result.total == pytest.approx(term_sum)


class TestConfigValidation:
    def test_missing_term_raises(self):
        incomplete = {"collision": {"weight": -10.0}}
        with pytest.raises(ValueError, match="Missing reward term"):
            RewardFunction(incomplete)

    def test_missing_weight_raises(self, reward_config):
        bad = dict(reward_config)
        bad["collision"] = {"terminal": True}  # No weight
        with pytest.raises(ValueError, match="missing 'weight'"):
            RewardFunction(bad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
