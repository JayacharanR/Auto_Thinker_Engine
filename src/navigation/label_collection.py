"""Collection-only fixed source/geography contract for dense TRAIN camera labels."""

import json
from pathlib import Path

from src.navigation.dagger import digest


def validate_label_guard(path, catalog, town, subset):
    path, catalog = Path(path), Path(catalog)
    plan = json.loads(path.read_text())
    if plan.get("schema") != "navigation-dense-label-collection-v1" or not subset:
        raise ValueError("Explicit dense-label TRAIN plan and route IDs required")
    if (
        digest(catalog / "catalog.json") != plan["catalog_sha256"]
        or digest(plan["actual_training_guard"]) != plan["actual_training_guard_sha256"]
    ):
        raise ValueError("Dense-label source catalog or original actual-geography guard changed")
    guard = json.loads(Path(plan["actual_training_guard"]).read_text())
    if town.removesuffix("_Opt") == "Town05" or town not in guard["allowed_junctions"]:
        raise ValueError("Dense labels may only use the original strict TRAINtowns")
    reserved_actual = {}
    for source, sha in plan["rule_case_sha256"].items():
        if digest(source) != sha:
            raise ValueError("Reserved original rule case identity changed")
        case = json.loads(Path(source).read_text())
        reserved_actual.setdefault(case["town"], set()).update(case["crossed_junction_ids"])
    if {town: sorted(ids) for town, ids in reserved_actual.items()} != plan[
        "reserved_rule_junctions"
    ]:
        raise ValueError("Do not remove declared rule-location reservations")
    seen = {"train": {}, "calibration": {}}
    for item in plan["episodes"]:
        seen[item["role"]].setdefault(item["town"], set()).update(
            item["entry"]["crossed_junction_ids"]
        )
    if any(seen["train"].get(t, set()) & ids for t, ids in seen["calibration"].items()):
        raise ValueError("New TRAINand calibration locations overlap")
    entries = {e["route_id"]: e for e in plan["episodes"]}
    catalog_entries = {
        e["route_id"]: e for e in json.loads((catalog / "catalog.json").read_text())["encounters"]
    }
    requested = []
    for rid in map(int, subset.split(",")):
        if rid not in entries:
            raise ValueError("Unregistered dense-label collection route")
        item = entries[rid]
        entry = catalog_entries[rid]
        reserved = set(plan["reserved_rule_junctions"].get(town, []))
        allowed = set(guard["allowed_junctions"][town]) - reserved
        if (
            entry["town"] != town
            or item["town"] != town
            or item["role"] not in ("train", "calibration")
            or item["entry"] != entry
            or rid not in guard["geometry_eligibility"]["kept_route_ids"]
            or not set(entry["crossed_junction_ids"]) <= allowed
        ):
            raise ValueError(
                "Dense-label route touches heldout, original VALor reserved rule geography"
            )
        requested.append(item)
    if len({e["role"] for e in requested}) != 1:
        raise ValueError("Do not mix TRAINand calibration roles in a native invocation")
    # Runtime trajectory check remains stricter than merely checking the route manifest.
    permitted = {
        t: sorted(set(ids) - set(plan["reserved_rule_junctions"].get(t, [])))
        for t, ids in guard["allowed_junctions"].items()
    }
    return dict(
        kind="dense-label-train-v1",
        guard_sha256=digest(path),
        allowed_junctions=permitted,
        label_role=requested[0]["role"],
        label_hz=2,
        physics_hz=20,
        label_plan=str(path.resolve()),
        source_plan_sha256=digest(path),
    )


def require_label_scope(command, collection, student, final, rule, empty, gap):
    if (
        command != "expert"
        or not collection
        or student
        or final
        or rule
        or empty
        or gap is not None
        or collection.get("mode") != "expert"
        or (collection.get("learning_guard") or {}).get("kind") != "dense-label-train-v1"
    ):
        raise ValueError(
            "Dense-label sensors require explicit guarded canonical expert TRAINcollection"
        )
