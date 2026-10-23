"""Strict identity/scope for physical component tests, outside benchmark catalogs."""

import json
from pathlib import Path

from src.navigation.dagger import digest
from src.navigation.rule_suite import arms

ROOT = Path(__file__).resolve().parents[2]


def verify_rule_case(path, route, subset, seed, collection, final, overrides, max_steps):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Controlled-rule case changed")
    if (
        collection
        or final
        or overrides
        or max_steps
        or subset not in ("", "0")
        or seed not in (61001, 61002, 61003)
    ):
        raise ValueError(
            "Rule tests are separate controlled cases, with no collection/final/override"
        )
    declared = arms(spec["treatment"]["family"])[spec["treatment"]["arm"]]
    if spec["treatment"] != declared:
        raise ValueError("Rule treatment differs from its frozen pair contract")
    if spec["schema"] != "navigation-physical-rule-case-v1":
        raise ValueError("Unknown rule-test schema")
    if (
        not spec["setup_only_component_test"]
        or spec["autonomous_route_success_claim"]
        or spec["learning_labels_released"]
    ):
        raise ValueError("Component assistance cannot become route success or training labels")
    route = Path(route).resolve()
    if (
        route != path.parent / "route.xml"
        or digest(route) != spec["route_sha256"]
        or spec["town"].removesuffix("_Opt") == "Town05"
    ):
        raise ValueError("Only the exact separate TRAIN component route is allowed")
    assignments = ROOT / "configs/encounters/location_split_v1/locations.json"
    if digest(assignments) != spec["location_split_sha256"]:
        raise ValueError("Rule case location split changed")
    global_allowed = set(json.loads(assignments.read_text())["assignments"][spec["town"]]["train"])
    allowed = set(spec["strict_allowed_junctions"])
    if not allowed <= global_allowed or not set(spec["crossed_junction_ids"]) <= allowed:
        raise ValueError("Component route enters reserved geography")
    guarded = ROOT / "outputs/navigation_m2_varied_waits_20261018/guard.json"
    if digest(guarded) != spec["strict_guard_sha256"]:
        raise ValueError("Component reservation guard changed")
    guard = json.loads(guarded.read_text())
    if allowed != set(guard["allowed_junctions"][spec["town"]]):
        raise ValueError("Component route claims a forbidden training location")
    return dict(spec, case_path=str(path), case_sha256=digest(path))
