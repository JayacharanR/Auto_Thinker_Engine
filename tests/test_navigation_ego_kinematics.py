import numpy as np
import pytest

from src.navigation.ego_kinematics import EgoKinematics


def physics():
    return EgoKinematics(
        dict(
            throttle_accel=7.0,
            brake_decel=5.0,
            drag=0.1,
            accel_bias=-1.0,
            maximum_steer_angle_rad=0.65,
            wheelbase_m=2.8,
        )
    )


def test_ego_forecasts_respect_action_order_and_cannot_see_future_controls():
    model = physics()
    first = np.tile([0.0, 0.8, 0.0], (80, 1))
    changed = first.copy()
    changed[40:] = [0.0, 0.0, 1.0]
    a = model.rollout([0.0, 0.0, 0.0, 2.0], first)
    b = model.rollout([0.0, 0.0, 0.0, 2.0], changed)
    assert np.array_equal(a[:40], b[:40])
    assert a[-1, 0] > b[-1, 0] and (b[:, 3] >= 0).all()
    reversed_order = model.rollout([0.0, 0.0, 0.0, 2.0], changed[::-1])
    assert not np.allclose(b[-1, :2], reversed_order[-1, :2])


def test_turning_and_braking_use_only_current_state_and_bounded_actuators():
    model = physics()
    next_state = model.step([0.0, 0.0, 0.0, 3.0], [0.5, 0.0, 1.0])
    assert 0 < next_state[3] < 3 and next_state[1] > 0 and next_state[2] > 0
    stopped = model.rollout([0.0, 0.0, 0.0, 0.1], np.tile([0.0, 0.0, 1.0], (80, 1)))
    assert (stopped[:, 3] == 0).all()
    with pytest.raises(ValueError, match="bounded20Hz"):
        model.step([0.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0])
