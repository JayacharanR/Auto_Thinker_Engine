"""Outcome-blind TRAIN selection and conservative static pixel evidence."""

import numpy as np
import pytest

from src.navigation.policy_worker import PrivilegeGuard
from src.navigation.signal_data import (
    phase_response,
    phase_response_v2,
    select_entries,
    stop_reference,
)


def entry(rid, junction, family="signal_stop", town="Town03_Opt"):
    return {
        "route_id": rid,
        "junction_id": junction,
        "crossed_junction_ids": [junction],
        "signal_ids_at_inventory": [rid + 100],
        "family": family,
        "town": town,
    }


def test_signal_selection_is_by_junction_geometry_and_never_driving_outcome():
    entries = [entry(3, 1, "signal_release_memory"), entry(7, 2), entry(6, 1), entry(2, 9)]
    guard = {
        "allowed_junctions": {"Town03_Opt": [1, 2]},
        "geometry_eligibility": {"kept_route_ids": [3, 6, 7]},
    }
    catalog = {"role": "train", "encounters": entries}
    picks = select_entries(catalog, guard, 2)
    assert [e["route_id"] for e in picks] == [6, 7]
    other = {**catalog, "encounters": [{**e, "success": False} for e in reversed(entries)]}
    assert [e["route_id"] for e in select_entries(other, guard, 2)] == [6, 7]
    with pytest.raises(ValueError, match="Training catalog"):
        select_entries({**catalog, "role": "heldout_dev"}, guard)


def test_signal_collection_refuses_reserved_and_final_town_locations():
    guard = {
        "allowed_junctions": {"Town03_Opt": [1]},
        "geometry_eligibility": {"kept_route_ids": [1]},
    }
    with pytest.raises(ValueError, match="reserved"):
        select_entries({"role": "train", "encounters": [entry(1, 2)]}, guard)
    with pytest.raises(ValueError, match="Town05"):
        select_entries({"role": "train", "encounters": [entry(1, 1, town="Town05_Opt")]}, guard)


def test_static_phase_response_accepts_repeated_color_switch_without_claiming_original_visibility():
    red = np.zeros((5, 3, 3), np.uint8)
    green = red.copy()
    red[1, 1], green[3, 1] = [250, 20, 20], [20, 250, 20]
    result = phase_response(red, green, red.copy())
    assert result["static_phase_response_supported"] is True
    assert result["original_scene_visibility_valid"] is None
    assert result["red_repeat_mae"] == 0


@pytest.mark.parametrize("grey", [0, 80, 220])
def test_back_facing_or_unresponsive_housing_cannot_receive_visible_state_label(grey):
    image = np.full((4, 2, 3), grey, np.uint8)
    assert not phase_response(image, image.copy(), image.copy())["static_phase_response_supported"]


def test_unstable_repeat_control_rejects_a_color_response_and_geometry_oracle_stays_blocked():
    red = np.full((5, 3, 3), [200, 10, 10], np.uint8)
    green = np.full((5, 3, 3), [10, 200, 10], np.uint8)
    assert not phase_response(red, green, green.copy())["static_phase_response_supported"]
    with pytest.raises(ValueError, match="RGB crops"):
        phase_response(red, green[:3], red)
    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("src.navigation.signal_data")


def test_stop_line_projection_spans_a_lane_to_junction_road_id_change():
    points = [[0, 0, 0], [8, 0, 0], [10.5, 0, 0], [12, 0, 0]]
    keys = [(1, 0, -1), (1, 0, -1), (99, 0, -1), (99, 0, -1)]
    line = {"lane": [1, 0, -1], "xyz": [10, 0, 0]}
    result = stop_reference(points, keys, [line])
    assert result["arc_m"] == pytest.approx(10) and result["gap_m"] == 0
    assert stop_reference(points, keys, [{**line, "xyz": [10, 3.5, 0]}]) is None


def test_static_collection_loads_requested_map_before_accepting_its_hash():
    import hashlib
    from types import SimpleNamespace

    from scripts.collect_navigation_signal_pairs import load_training_world

    def world(name):
        return SimpleNamespace(
            get_map=lambda: SimpleNamespace(name=name, to_opendrive=lambda: name)
        )

    loaded = []
    client = SimpleNamespace(
        set_timeout=lambda timeout: None,
        get_world=lambda: world("Town03_Opt"),
        load_world=lambda town: loaded.append(town) or world(town),
    )
    expected = hashlib.sha256(b"Town01_Opt").hexdigest()
    assert load_training_world(client, "Town01_Opt", expected).get_map().name == "Town01_Opt"
    assert loaded == ["Town01_Opt"]
    with pytest.raises(ValueError, match="Wrong live"):
        load_training_world(client, "Town01_Opt", "wrong hash")


def test_phase_v2_accepts_near_white_green_bulb_with_opposite_local_changes():
    red = np.full((5, 3, 3), 20, np.uint8)
    green = red.copy()
    red[1, 1], green[3, 1] = [255, 205, 160], [236, 255, 236]
    assert not phase_response(red, green, red.copy())["static_phase_response_supported"]
    result = phase_response_v2(red, green, red.copy())
    assert result["static_phase_response_supported"] and result["validator_version"] == 2
    assert result["original_scene_visibility_valid"] is None
    # Preserve the predeclared old rule and its failure, not an alias to v2.
    assert not phase_response(red, green, red.copy())["static_phase_response_supported"]


def test_phase_v2_rejects_global_brightness_drift_and_unstable_repeats():
    red = np.full((5, 3, 3), 50, np.uint8)
    green = np.full((5, 3, 3), 100, np.uint8)
    assert not phase_response_v2(red, green, red.copy())["static_phase_response_supported"]
    red[1, 1], green[3, 1] = [250, 20, 20], [20, 250, 20]
    assert not phase_response_v2(red, green, green.copy())["static_phase_response_supported"]


def test_expansion_keeps_original_train_groups_and_deduplicates_incoming_approaches():
    from src.navigation.signal_expansion import expansion_entries

    rows = [
        dict(
            entry(i, 1 if i < 5 else 2),
            signal_ids_at_inventory=[100 + (i % 2)],
            movement_key=["Town03_Opt", "junction", str(i % 2), str(i)],
        )
        for i in range(6)
    ]
    guard = dict(
        allowed_junctions={"Town03_Opt": [1, 2]},
        geometry_eligibility=dict(kept_route_ids=list(range(6))),
    )
    roles = {("Town03_Opt", 1): "train", ("Town03_Opt", 2): "validation"}
    catalog = dict(role="train", encounters=rows)
    assert [e["route_id"] for e in expansion_entries(catalog, guard, roles)] == [0, 1]
    assert [e["route_id"] for e in expansion_entries(catalog, guard, roles, 1)] == [0]
    changed = dict(catalog, encounters=[dict(e, success=False) for e in reversed(rows)])
    assert expansion_entries(changed, guard, roles) == [
        dict(rows[0], success=False),
        dict(rows[1], success=False),
    ]


def test_expansion_duplicate_pose_handles_yaw_wrap_without_merging_distinct_views():
    from src.navigation.signal_expansion import duplicate_pose

    old = dict(town="Town01_Opt", ego_pose=[1, 2, 0.35, 0, 359.99, 0])
    new = dict(old, ego_pose=[1.1, 2, 0.35, 0, -0.01, 0])
    assert duplicate_pose(new, [old])
    assert not duplicate_pose(dict(new, ego_pose=[1.6, 2, 0.35, 0, -0.01, 0]), [old])
    assert not duplicate_pose(dict(new, town="Town05_Opt"), [old])
    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("src.navigation.signal_expansion")


def test_expansion_validates_original_roles_instead_of_resplitting_subset(tmp_path):
    import json

    from src.navigation.dagger import digest
    from src.navigation.signal_expansion import validate_expansion
    from src.navigation.signal_learning import junction_roles

    cases = [
        dict(town="Town03_Opt", junction_id=j, route_id=j, ego_pose=[j * 10, 0, 0.35, 0, 0, 0])
        for j in range(6)
    ]
    (tmp_path / "plan.json").write_text(json.dumps(dict(cases=cases)))
    (tmp_path / "data_audit.json").write_text("{}")
    roles = junction_roles(cases)
    train = next(j for (t, j), r in roles.items() if r == "train")
    val = next(j for (t, j), r in roles.items() if r == "validation")
    case = dict(
        cases[train], role="train", route_id=10000, ego_pose=[train * 10 + 3, 0, 0.35, 0, 0, 0]
    )
    plan = dict(
        cases=[case],
        expansion=dict(
            base_corpus=str(tmp_path),
            base_audit_sha256=digest(tmp_path / "data_audit.json"),
            base_plan_sha256=digest(tmp_path / "plan.json"),
            original_groups=[dict(town=t, junction_id=j, role=r) for (t, j), r in roles.items()],
        ),
    )
    assert validate_expansion(plan) == roles  # A one-group TRAIN subset is never resplit.
    with pytest.raises(ValueError, match="original TRAIN"):
        validate_expansion(dict(plan, cases=[dict(case, junction_id=val)]))
    with pytest.raises(ValueError, match="identity collides"):
        validate_expansion(dict(plan, cases=[dict(case, route_id=0)]))
    with pytest.raises(ValueError, match="source changed"):
        validate_expansion(dict(plan, expansion=dict(plan["expansion"], base_audit_sha256="bad")))
