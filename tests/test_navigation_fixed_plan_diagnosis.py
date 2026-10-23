"""Observed first-event labels, censored tails and strictly past-only RGB inputs."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.diagnose_navigation_fixed_plans import (
    first_collision_labels,
    past_context,
    score_hazards,
)


def test_first_collision_is_absorbing_after_a_matching_action_prefix():
    labels, valid = first_collision_labels(
        dict(matching_commands_before_safety=26, first_contact_tick=26)
    )
    assert np.array_equal(labels, [0, 0, 1, 1, 1, 1, 1, 1]) and valid.all()
    scored = score_hazards(np.zeros(8), labels, valid)
    assert scored["missed_positive_at_half"] == 6 and scored["sum_squared_error"] == 6


def test_censored_no_event_tail_is_unknown_and_nan_is_masked_before_arithmetic():
    labels, valid = first_collision_labels(
        dict(matching_commands_before_safety=35, first_contact_tick=None)
    )
    assert valid.tolist() == [True, True, True, False, False, False, False, False]
    assert np.isnan(labels[3:]).all()
    assert score_hazards(np.ones(8), labels, valid)["sum_squared_error"] == 3
    for hit in (0, 36, -1, True):
        with pytest.raises(ValueError, match="First contact"):
            first_collision_labels(dict(matching_commands_before_safety=35, first_contact_tick=hit))


def test_future_rgb_is_never_loaded_and_context_ends_at_decision(tmp_path, monkeypatch):
    import src.navigation.dataset as dataset

    data = tmp_path / "data"
    data.mkdir()
    (data / "shard_0000.zip").write_bytes(b"synthetic metadata fixture only")
    monkeypatch.setattr(
        dataset, "shard_records", lambda _: [dict(frame=f) for f in range(0, 100, 2)]
    )
    loaded = []

    def load(shard, row):
        assert row["frame"] <= 40, "Future RGB entered prediction input"
        loaded.append(row["frame"])
        return SimpleNamespace(frame=row["frame"])

    monkeypatch.setattr(dataset, "load_policy_sample", load)
    context, sources = past_context(tmp_path, 40)
    assert [p.frame for p in context] == list(range(26, 42, 2))
    assert max(loaded) == 40 and len(sources) == 1
    with pytest.raises(ValueError, match="ending at decision"):
        past_context(tmp_path, 41)
