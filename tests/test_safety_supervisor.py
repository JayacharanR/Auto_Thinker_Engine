"""
Unit tests for independent SafetySupervisor (P5 Action Shield).
Action contract: action[0] is acceleration, action[1] is steering.
"""

import numpy as np
import pytest

from src.safety.supervisor import SafetySupervisor


class TestSafetySupervisor:
    """Test suite for safety shielding and emergency intervention."""

    def test_safe_action_passthrough(self):
        """Safe actions within limits should pass through without modification."""
        supervisor = SafetySupervisor()
        action = np.array([0.5, 0.1], dtype=np.float32)  # acc=0.5, steer=0.1
        safe_action, info = supervisor.filter_action(action, telemetry={"ttc": 10.0, "wpt_dis": 0.2})

        assert not info["intervened"]
        assert len(info["reasons"]) == 0
        np.testing.assert_allclose(safe_action, action, atol=1e-5)

    def test_action_bounds_clipping(self):
        """Out-of-bounds actions should be clamped to [-1, 1]."""
        supervisor = SafetySupervisor()
        action = np.array([2.5, -1.8], dtype=np.float32)
        safe_action, info = supervisor.filter_action(action)

        assert info["intervened"]
        assert "action_bound_clip" in info["reasons"]
        assert safe_action[0] == 1.0   # acc clipped to 1.0
        assert safe_action[1] == -1.0  # steer clipped to -1.0

    def test_steering_rate_limiting(self):
        """Jerk-inducing steering changes must be rate-limited."""
        supervisor = SafetySupervisor(max_steer_rate=0.2)
        # First step sets baseline acc=0.5, steer=0.0
        supervisor.filter_action(np.array([0.5, 0.0]))

        # Second step requests sudden full lock right steer=1.0
        safe_action, info = supervisor.filter_action(np.array([0.5, 1.0]))

        assert info["intervened"]
        assert "steering_rate_limit" in info["reasons"]
        assert safe_action[1] == pytest.approx(0.2, abs=1e-4)

    def test_ttc_emergency_brake(self):
        """Low time-to-collision must trigger full emergency braking."""
        supervisor = SafetySupervisor(ttc_threshold_sec=1.5)
        action = np.array([0.8, 0.0])  # Policy trying to accelerate forward
        safe_action, info = supervisor.filter_action(action, telemetry={"ttc": 0.9, "speed_norm": 10.0})

        assert info["intervened"]
        assert "emergency_brake_ttc" in info["reasons"]
        assert safe_action[0] == -1.0  # Full brake (acc = -1.0)

    def test_lane_departure_slowdown(self):
        """Severe lateral deviation must prevent positive acceleration."""
        supervisor = SafetySupervisor(max_lateral_deviation_m=1.8)
        action = np.array([0.7, 0.0])
        safe_action, info = supervisor.filter_action(action, telemetry={"wpt_dis": 2.2})

        assert info["intervened"]
        assert "lane_departure_slowdown" in info["reasons"]
        assert safe_action[0] <= 0.0

    def test_nan_inf_policy_failure_fallback(self):
        """Corrupted policy output (NaN/Inf) must fail safe to a stop."""
        supervisor = SafetySupervisor()
        action = np.array([float("nan"), float("inf")])
        safe_action, info = supervisor.filter_action(action)

        assert info["intervened"]
        assert "nan_inf_policy_failure" in info["reasons"]
        assert safe_action[0] == -1.0  # Full brake
        assert safe_action[1] == 0.0   # Center steer
        assert not np.isnan(safe_action).any()

    def test_intervention_tracking_and_reset(self):
        """Interventions should accumulate and clear on reset."""
        supervisor = SafetySupervisor(ttc_threshold_sec=1.5)
        supervisor.filter_action(np.array([0.8, 0.0]), telemetry={"ttc": 0.5})
        supervisor.filter_action(np.array([0.8, 0.0]), telemetry={"ttc": 0.4})

        assert len(supervisor.interventions) == 2
        assert supervisor.interventions[0].step == 1
        assert supervisor.interventions[1].step == 2

        supervisor.reset()
        assert len(supervisor.interventions) == 0
        assert supervisor._prev_action is None

    def test_anti_stall_triggers_on_clear_road(self):
        """Vehicle stalled on clear road should be nudged into crawl speed."""
        supervisor = SafetySupervisor(stall_timeout_steps=5, crawl_acceleration=0.35)
        # Policy outputs full brake acc=-1.0, vehicle speed is 0.0, road is clear (ttc=inf)
        for _ in range(4):
            safe_act, info = supervisor.filter_action(
                np.array([-1.0, 0.0]),
                telemetry={"speed_norm": 0.0, "ttc": float("inf"), "wpt_dis": 0.2},
            )
            # Timeout not reached yet (step < 5)
            assert not any("anti_stall_crawl" in r for r in info["reasons"])

        # 5th step triggers anti-stall crawl
        safe_act, info = supervisor.filter_action(
            np.array([-1.0, 0.0]),
            telemetry={"speed_norm": 0.0, "ttc": float("inf"), "wpt_dis": 0.2},
        )
        assert "anti_stall_crawl" in info["reasons"]
        assert safe_act[0] == pytest.approx(0.35, abs=1e-4)

    def test_anti_stall_inhibited_by_low_ttc(self):
        """Anti-stall must NEVER trigger if there is an obstacle in front."""
        supervisor = SafetySupervisor(stall_timeout_steps=3, crawl_acceleration=0.35)
        for _ in range(5):
            safe_act, info = supervisor.filter_action(
                np.array([-1.0, 0.0]),
                telemetry={"speed_norm": 0.0, "ttc": 1.0, "wpt_dis": 0.2},
            )
            # Low TTC must trigger emergency brake, NOT anti-stall
            assert "anti_stall_crawl" not in info["reasons"]
            assert safe_act[0] == -1.0

    def test_anti_stall_hysteresis_maintains_crawl(self):
        """Crawl must persist across min_crawl_steps even as speed rises."""
        supervisor = SafetySupervisor(stall_timeout_steps=3, min_crawl_steps=10, crawl_acceleration=0.35)
        # 3 stationary steps to trigger crawl
        for _ in range(3):
            supervisor.filter_action(
                np.array([-1.0, 0.0]),
                telemetry={"speed_norm": 0.0, "ttc": float("inf"), "wpt_dis": 0.1},
            )

        # In steps 4-10, vehicle is moving (speed_kmh = 5.0, so speed_norm ~ 1.38)
        # Policy still commands -1.0 (brake lock), but supervisor should hold crawl
        for _ in range(7):
            safe_act, info = supervisor.filter_action(
                np.array([-1.0, 0.0]),
                telemetry={"speed_norm": 1.38, "ttc": float("inf"), "wpt_dis": 0.1},
            )
            assert "anti_stall_crawl" in info["reasons"]
            assert safe_act[0] == pytest.approx(0.35, abs=1e-4)

    def test_red_light_emergency_stop(self):
        """Red light state in telemetry forces emergency braking."""
        supervisor = SafetySupervisor()
        safe_act, info = supervisor.filter_action(
            np.array([0.8, 0.0]),  # Policy wants forward acceleration
            telemetry={"traffic_light_state": "RED", "speed_norm": 5.0, "ttc": float("inf")},
        )
        assert "red_light_stop" in info["reasons"]
        assert safe_act[0] == -1.0

    def test_intervention_summary_reporting(self):
        """Intervention summary should aggregate counts and rates correctly."""
        supervisor = SafetySupervisor()
        supervisor.filter_action(np.array([0.5, 0.0]), telemetry={"ttc": 0.5})  # TTC brake
        supervisor.filter_action(np.array([0.5, 0.0]), telemetry={"traffic_light_state": "RED"})  # Red light
        supervisor.filter_action(np.array([0.2, 0.0]), telemetry={"ttc": float("inf")})  # Normal step

        summary = supervisor.get_intervention_summary()
        assert summary["total_steps"] == 3
        assert summary["total_interventions"] == 2
        assert summary["intervention_rate"] == pytest.approx(0.6667, abs=1e-3)
        assert "emergency_brake_ttc" in summary["reasons"]
        assert "red_light_stop" in summary["reasons"]


