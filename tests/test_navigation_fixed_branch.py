import numpy as np
import pytest

from scripts.collect_navigation_branches import frozen_branch_plan


class SmallEgo:
    def step(self, state, control):
        result = state.copy()
        result[0] += .05 * state[3]
        result[3] = max(0, state[3] + .05 * (control[1] - control[2]))
        return result


def test_only_registered_center_option_executes_and_all_plans_are_immutable():
    nav = np.column_stack([np.arange(1, 17) * 2, np.zeros(16)])
    for name, speed in [("go", 2), ("brake", 0), ("wait_go", 4)]:
        slate, selected = frozen_branch_plan(nav, 3., np.zeros((2, 3)), name, SmallEgo())
        assert len(slate) == 15
        assert selected.desired.path_index == 1 and selected.desired.speed_index == speed
        assert selected.ordered_controls.shape == (80, 3)
        with pytest.raises(ValueError):
            selected.ordered_controls[0] = 0
    with pytest.raises(ValueError, match="registered"):
        frozen_branch_plan(nav, 3., np.zeros((2, 3)), "tuned_after_result", SmallEgo())
