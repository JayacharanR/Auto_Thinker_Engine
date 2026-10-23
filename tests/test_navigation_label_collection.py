"""Training-only labels cannot leak into inference or reassign reserved geography."""

import json

import pytest

from src.navigation.dagger import digest
from src.navigation.label_collection import require_label_scope, validate_label_guard
from src.navigation.policy_worker import PrivilegeGuard


def test_dense_label_scope_requires_guarded_canonical_expert_and_no_evaluation_inputs():
    collection = dict(mode="expert", learning_guard=dict(kind="dense-label-train-v1"))
    require_label_scope("expert", collection, None, None, None, False, None)
    for changed in (
        dict(command="student"),
        dict(student={"checkpoint": "fixture"}),
        dict(final="fixture"),
        dict(rule="fixture"),
        dict(empty=True),
        dict(gap=8),
        dict(collection=dict(mode="noisy")),
    ):
        values = dict(
            command="expert",
            collection=collection,
            student=None,
            final=None,
            rule=None,
            empty=False,
            gap=None,
        )
        with pytest.raises(ValueError, match="guarded canonical"):
            require_label_scope(**dict(values, **changed))
    for name in (
        "src.navigation.label_collection",
        "src.navigation.labeled_collection_agent",
        "scripts.navigation_label_rig",
    ):
        with pytest.raises(ImportError, match="Forbidden"):
            PrivilegeGuard().find_spec(name)


def test_dense_label_plan_preserves_original_and_rule_reservations_and_calibration_split(tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    entry = dict(route_id=1, town="Town01_Opt", crossed_junction_ids=[2])
    other = dict(route_id=2, town="Town01_Opt", crossed_junction_ids=[3])
    (catalog / "catalog.json").write_text(json.dumps(dict(encounters=[entry, other])))
    guard = tmp_path / "guard.json"
    guard.write_text(
        json.dumps(
            dict(
                allowed_junctions={"Town01_Opt": [2, 3, 4]},
                geometry_eligibility=dict(kept_route_ids=[1, 2]),
            )
        )
    )
    case = tmp_path / "case.json"
    case.write_text(json.dumps(dict(town="Town01_Opt", crossed_junction_ids=[4])))
    plan = dict(
        schema="navigation-dense-label-collection-v1",
        catalog_sha256=digest(catalog / "catalog.json"),
        actual_training_guard=str(guard),
        actual_training_guard_sha256=digest(guard),
        rule_case_sha256={str(case): digest(case)},
        reserved_rule_junctions={"Town01_Opt": [4]},
        episodes=[
            dict(route_id=1, town="Town01_Opt", role="train", entry=entry),
            dict(route_id=2, town="Town01_Opt", role="calibration", entry=other),
        ],
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    checked = validate_label_guard(path, catalog, "Town01_Opt", "1")
    assert checked["allowed_junctions"]["Town01_Opt"] == [2, 3] and checked["label_role"] == "train"
    assert validate_label_guard(path, catalog, "Town01_Opt", "2")["label_role"] == "calibration"
    with pytest.raises(ValueError, match="mix"):
        validate_label_guard(path, catalog, "Town01_Opt", "1,2")
    with pytest.raises(ValueError, match="Unregistered"):
        validate_label_guard(path, catalog, "Town01_Opt", "3")
    changed = dict(plan, reserved_rule_junctions={"Town01_Opt": []})
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="reservations"):
        validate_label_guard(path, catalog, "Town01_Opt", "1")
