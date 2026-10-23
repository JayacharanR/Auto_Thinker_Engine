"""Continuous-episode evidence cannot be inferred from family names or partial waits."""

import copy

import pytest

from src.navigation.approach_data import approach_coverage, reserve_components, select_approaches
from src.navigation.collection import (
    empty_training_background,
    training_position,
    training_signal_wait,
)
from src.navigation.policy_worker import PrivilegeGuard


def rows():
    result = []
    for i in range(100):
        result.append(
            dict(
                frame=2 * i,
                timestamp=i / 10,
                light_id=7,
                light_distance=max(0, 60 - i),
                light_state="Red" if i < 60 else "Green",
                light_hazard=20 <= i < 60,
                teacher_speed=0 if 20 <= i < 60 else 5,
                ego_speed=0 if 30 <= i < 63 else 3,
            )
        )
    return result


def test_only_contiguous_moving_approach_stationary_wait_and_recorded_release_pass():
    full = rows()
    result = approach_coverage(full)
    assert result["continuous_signal_sequence"] and result["post_release_s"] >= 3
    assert not result["visibility_certified"] and not result["state_labels_released"]
    for partial in (full[30:], full[:60], full[:70]):
        assert not approach_coverage(partial)["continuous_signal_sequence"]
    missing = full[:40] + full[41:]
    with pytest.raises(ValueError, match="contiguous"):
        approach_coverage(missing)


def test_vehicle_stop_wrong_actor_and_motion_through_wait_do_not_pass_as_signal_sequence():
    full = rows()
    bad = copy.deepcopy(full)
    for r in bad:
        r["light_hazard"] = False
    assert not approach_coverage(bad)["continuous_signal_sequence"]
    bad = copy.deepcopy(full)
    for r in bad[60:]:
        r["light_id"] = 8
    assert not approach_coverage(bad)["continuous_signal_sequence"]
    bad = copy.deepcopy(full)
    bad[40]["ego_speed"] = 2
    assert not approach_coverage(bad)["continuous_signal_sequence"]


def test_selector_excludes_reserved_junctions_other_families_and_duplicate_locations():
    catalog = dict(
        role="train",
        encounters=[
            dict(
                route_id=i,
                town="Town03_Opt",
                family="signal_stop",
                crossed_junction_ids=[j],
                junction_id=j,
                signal_ids_at_inventory=[7],
            )
            for i, j in enumerate([655, 238, 238, 356])
        ],
    )
    selected = select_approaches(catalog, {"Town03_Opt": [238, 356]}, [0, 1, 2, 3])
    assert [e["route_id"] for e in selected] == [1, 3]
    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("src.navigation.approach_data")


def test_component_reservations_only_remove_from_driving_train():
    cases = [dict(town="Town03_Opt", junction_id=i) for i in range(10)]
    allowed, reserved = reserve_components(
        dict(allowed_junctions={"Town03_Opt": list(range(10))}), cases
    )
    assert not set(allowed["Town03_Opt"]) & set(reserved["Town03_Opt"])
    assert set(allowed["Town03_Opt"]) | set(reserved["Town03_Opt"]) == set(range(10))


def test_no_lead_curriculum_cannot_change_student_or_unguarded_evaluation_background():
    assert not empty_training_background({})
    allowed = dict(
        collection=dict(
            background="empty",
            mode="expert",
            learning_guard=dict(kind="continuous-approach-train-v1"),
        )
    )
    assert empty_training_background(allowed)
    for bad in (
        dict(collection=dict(background="empty", mode="expert")),
        dict(allowed, student=dict(checkpoint="camera.pt")),
    ):
        with pytest.raises(ValueError, match="TRAIN collection"):
            empty_training_background(bad)


def test_training_wait_override_rejects_evaluation_and_invalid_durations():
    assert training_signal_wait({}) is None
    scope = dict(
        background="empty", mode="expert", learning_guard=dict(kind="continuous-approach-train-v1")
    )
    for value in (3, 7, 11):
        assert (
            training_signal_wait(dict(collection=dict(scope, signal_stationary_seconds=value)))
            == value
        )
    for value in (0, 16, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            training_signal_wait(dict(collection=dict(scope, signal_stationary_seconds=value)))
    with pytest.raises(ValueError):
        training_signal_wait(
            dict(collection=dict(scope, background="canonical", signal_stationary_seconds=7))
        )


def test_location_guard_accepts_train_and_records_unsaved_tick_violation(tmp_path):
    import json
    from types import SimpleNamespace

    pose = SimpleNamespace(location=SimpleNamespace(x=1, y=2, z=3))
    wp = SimpleNamespace(is_junction=True, junction_id=655, road_id=1, lane_id=-1)
    world_map = SimpleNamespace(get_waypoint=lambda _: wp)
    assert training_position(world_map, pose, {655}, 1001) is pose
    violation = tmp_path / "violation.json"
    with pytest.raises(RuntimeError, match="held-out"):
        training_position(world_map, pose, {238}, 1001, violation)
    assert json.loads(violation.read_text())["frame"] == 1001
