import numpy as np
import pytest

from scripts.audit_navigation_fixed_branches import matching_prefix


def test_early_privileged_braking_cannot_count_as_a_factual_safe_plan():
    planned = np.zeros((80, 3))
    with pytest.raises(ValueError, match="preceded"):
        matching_prefix(planned, planned, [None] * 20 + ["stop"] * 60, None)
    with pytest.raises(ValueError, match="preceded"):
        matching_prefix(planned, planned, [None] * 20 + ["stop"] * 60, 30)


def test_only_the_executed_prefix_supports_collision_and_tail_is_explicitly_unknown():
    planned = np.zeros((80, 3))
    applied = planned.copy()
    applied[31:, 2] = 1
    intervention = [None] * 31 + ["post_contact_stop"] * 49
    result = matching_prefix(planned, applied, intervention, 30)
    assert result["matching_commands_before_safety"] == 31
    assert result["future_latent_progress_after_intervention_unknown"]
    assert result["absorbing_first_collision_supported"]
    applied[5, 1] = .8
    with pytest.raises(ValueError, match="differ before"):
        matching_prefix(planned, applied, intervention, 30)
