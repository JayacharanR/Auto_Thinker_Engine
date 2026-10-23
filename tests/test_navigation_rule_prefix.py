import numpy as np
import pytest

from src.navigation.rule_prefix import PrefixSpec, RulePrefix


def navigation():
    return np.column_stack([np.arange(2, 34, 2), np.zeros(16)])


def test_no_prefix_means_unmodified_agent_control_from_first_tick():
    prefix = RulePrefix(PrefixSpec())
    assert prefix.step(0, navigation(), 0, 50) is None
    assert prefix.handoff_time == 0
    assert prefix.step(0.05, navigation(), 0, -5) is None
    assert not prefix.state(0.05)["scripted_setup"]


def test_memory_setup_waits_at_registered_pose_then_never_overrides_again():
    prefix = RulePrefix(PrefixSpec(mode="stop_before_line", stationary_seconds=0.1))
    assert prefix.step(0, navigation(), 2, 15).throttle > 0
    assert prefix.step(1, navigation(), 0, 3).brake == 1
    assert prefix.step(1.05, navigation(), 0, 3).brake == 1
    assert prefix.step(1.15, navigation(), 0, 3) is None
    # New privileged measurements cannot trigger another scripted action.
    assert prefix.step(1.2, navigation(), 8, -100) is None


def test_rolling_setup_hands_over_without_secret_braking_or_forced_go():
    prefix = RulePrefix(PrefixSpec(mode="cruise_to_handoff", speed_mps=3, handoff_distance_m=3))
    assert prefix.step(0, navigation(), 3, 20).brake == 0
    assert prefix.step(1, navigation(), 3, 3) is None
    assert prefix.state(1)["control_after_handoff"] == "evaluated agent proposal, unchanged"


def test_invalid_setups_are_failures_instead_of_hidden_assistance():
    with pytest.raises(ValueError):
        RulePrefix(PrefixSpec(speed_mps=20))
    prefix = RulePrefix(PrefixSpec(mode="stop_before_line", maximum_seconds=2))
    with pytest.raises(RuntimeError, match="overshot"):
        prefix.step(0, navigation(), 6, 1)
    prefix = RulePrefix(PrefixSpec(mode="stop_before_line", maximum_seconds=2))
    prefix.step(0, navigation(), 0, 15)
    with pytest.raises(RuntimeError, match="timed out"):
        prefix.step(3, navigation(), 0, 15)
