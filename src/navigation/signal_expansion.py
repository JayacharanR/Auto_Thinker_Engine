"""Training-only approach selection; an expansion never resamples the old split."""

import json
from pathlib import Path

import numpy as np

from src.navigation.dagger import digest


def expansion_entries(catalog, guard, roles, maximum_approaches=3):
    if catalog["role"] != "train" or maximum_approaches < 1:
        raise ValueError("TRAIN catalog and positive approach limit required")
    kept = set(guard["geometry_eligibility"]["kept_route_ids"])
    chosen, approaches = [], {}
    for entry in sorted(catalog["encounters"], key=lambda e: e["route_id"]):
        town, junction = entry["town"], entry["junction_id"]
        if town.removesuffix("_Opt") == "Town05":
            raise ValueError("Town05 remains held out")
        if entry["route_id"] not in kept or roles.get((town, junction)) != "train":
            continue
        if not set(entry["crossed_junction_ids"]) <= set(guard["allowed_junctions"][town]):
            raise ValueError("Expansion route crosses a reserved junction")
        if any(roles.get((town, j)) == "validation" for j in entry["crossed_junction_ids"]):
            continue
        if len(entry["signal_ids_at_inventory"]) != 1:
            continue
        # The incoming lane distinguishes approaches; outgoing turn and episode
        # family do not create a new visual approach to the same stop line.
        key = entry["signal_ids_at_inventory"][0], entry["movement_key"][2]
        visited = approaches.setdefault((town, junction), set())
        if key not in visited and len(visited) < maximum_approaches:
            chosen.append(entry)
            visited.add(key)
    return chosen


def duplicate_pose(case, existing, distance=0.5):
    p = np.asarray(case["ego_pose"], float)
    for other in existing:
        if other["town"] != case["town"]:
            continue
        q = np.asarray(other["ego_pose"], float)
        yaw = abs((p[4] - q[4] + 180) % 360 - 180)
        if np.linalg.norm(p[:3] - q[:3]) < distance and yaw < 0.05:
            return True
    return False


def validate_expansion(plan):
    """Bind TRAIN supplement to the original roles, corpus and native rig."""
    from src.navigation.signal_learning import junction_roles

    spec = plan["expansion"]
    base = Path(spec["base_corpus"])
    for filename, key in [
        ("data_audit.json", "base_audit_sha256"),
        ("plan.json", "base_plan_sha256"),
    ]:
        if digest(base / filename) != spec[key]:
            raise ValueError("Original expansion source changed")
    original = json.loads((base / "plan.json").read_text())
    roles = junction_roles(original["cases"])
    expected = [{"town": t, "junction_id": j, "role": r} for (t, j), r in roles.items()]
    if spec["original_groups"] != expected:
        raise ValueError("Original component split changed")
    seen = set()
    old_ids = {c["route_id"] for c in original["cases"]}
    for case in plan["cases"]:
        if case.get("role") != "train" or roles.get((case["town"], case["junction_id"])) != "train":
            raise ValueError("Expansion may contain only original TRAIN groups")
        if case["route_id"] in seen or case["route_id"] in old_ids:
            raise ValueError("Expansion case identity collides")
        seen.add(case["route_id"])
        if duplicate_pose(case, original["cases"]):
            raise ValueError("Expansion duplicates an original pose")
    return roles
