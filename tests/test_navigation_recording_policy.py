from copy import deepcopy

import pytest

from src.navigation.recording_policy import frozen_showcase_policy, preview_enabled


def test_same_six_predeclared_final_videos_without_outcome_filtering():
    policy = frozen_showcase_policy()
    expert = dict(final_benchmark=True, final_registration=dict(recording_policy=policy))
    student = dict(expert, student=dict(architecture="camera"))
    selected = []
    for kind, manifest in (("expert", expert), ("student", student)):
        for seed in (3031, 4031, 5031):
            for route_id in range(12):
                current = dict(route_id=str(route_id), seed=seed)
                a = preview_enabled(manifest, current)
                b = preview_enabled(dict(manifest, collisions=99, completed=False), current)
                assert a == b
                if a:
                    selected.append((kind, seed, route_id))
    assert len(selected) == 6
    assert {row[2] for row in selected} == {0, 4, 8}


def test_rule_training_and_existing_nonfinal_videos_remain_full():
    for fields in ({}, dict(rule_test={"family": "memory"}), dict(training_collection=True)):
        assert preview_enabled(fields, dict(route_id="9", seed=61001))
    final = dict(final_benchmark=True)
    with pytest.raises(ValueError, match="preview selection"):
        preview_enabled(final, dict(route_id="9", seed=3031))


def test_modified_selection_cannot_follow_a_final_outcome():
    policy = deepcopy(frozen_showcase_policy())
    policy["full_preview_route_ids"].append(9)
    with pytest.raises(ValueError, match="frozen approved"):
        preview_enabled(dict(final_benchmark=True,
                             final_registration=dict(recording_policy=policy)),
                        dict(route_id="9", seed=3031))
