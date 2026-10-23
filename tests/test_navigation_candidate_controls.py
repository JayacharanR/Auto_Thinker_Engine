from pathlib import Path

import numpy as np
import pytest

from src.navigation.candidate_controls import control_candidates
from src.navigation.controller import track
from src.navigation.frozen_ego import FrozenExpertEgo

ROOT = Path(__file__).resolve().parents[1]


def setup():
    model = FrozenExpertEgo(ROOT / "outputs/navigation_models/frozen_expert_ego_20261020.json")
    navigation = np.column_stack([np.arange(2, 34, 2), np.zeros(16)])
    previous = np.array([[0, .4, 0], [0, .3, 0]])
    return model, navigation, previous


def test_commands_share_tracker_and_forecast_uses_only_explicit_past_actions():
    model, navigation, previous = setup()
    values = control_candidates(navigation, np.zeros(8), np.full(8, 6.), 4., previous, model)
    assert len(values) == 15
    for plan in values:
        first = track(navigation, plan.desired.correction, 4., plan.desired.speeds[0], 0.)
        assert np.allclose(plan.ordered_controls[0], [first.steer, first.throttle, first.brake])
        recomputed = model.rollout([0, 0, 0, 4], plan.ordered_controls, previous)
        assert np.array_equal(recomputed, plan.forecast_states)
        assert np.array_equal(plan.realized.waypoints, recomputed[9::10, :2])
        assert plan.ordered_controls.shape == (80, 3)
        assert np.isfinite(plan.ordered_controls).all()
        assert (np.abs(plan.ordered_controls[:, 0]) <= 1).all()
        assert ((plan.ordered_controls[:, 1:] >= 0) &
                (plan.ordered_controls[:, 1:] <= 1)).all()


def test_actual_actuator_history_matters_and_unknown_or_future_history_is_rejected():
    model, navigation, previous = setup()
    a = control_candidates(navigation, np.zeros(8), np.full(8, 5.), 4., previous, model)[2]
    braking = np.array([[0, 0, 1], [0, 0, 1]])
    b = control_candidates(navigation, np.zeros(8), np.full(8, 5.), 4., braking, model)[2]
    assert not np.allclose(a.forecast_states[:2], b.forecast_states[:2])
    with pytest.raises(ValueError, match="Explicit finite bounded"):
        control_candidates(navigation, np.zeros(8), np.full(8, 5.), 4., None, model)
    with pytest.raises(TypeError):
        model.rollout([0, 0, 0, 4], a.ordered_controls, previous, future_states=np.zeros((80, 4)))
