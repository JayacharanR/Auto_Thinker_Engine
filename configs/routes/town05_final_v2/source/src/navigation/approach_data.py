"""TRAIN-only continuous-approach scope and temporal-coverage checks."""

import json
from pathlib import Path

from src.navigation.dagger import digest, validate_guard
from src.navigation.signal_learning import junction_roles


def reserve_components(base, cases):
    reserved = {}
    for (town, junction), role in junction_roles(cases).items():
        if role == "validation":
            reserved.setdefault(town, []).append(junction)
    return {
        town: sorted(set(junctions) - set(reserved.get(town, [])))
        for town, junctions in base["allowed_junctions"].items()
    }, {town: sorted(ids) for town, ids in reserved.items()}


def select_approaches(catalog, allowed, eligible, per_town=2):
    if catalog.get("role") != "train" or per_town < 1:
        raise ValueError("Approved TRAIN catalog and positive per-town count required")
    selected, visited = [], {}
    for entry in sorted(catalog["encounters"], key=lambda e: (e["town"], e["route_id"])):
        town = entry["town"]
        if town.removesuffix("_Opt") == "Town05":
            raise ValueError("Town05 cannot enter approach collection")
        if (
            entry["family"] != "signal_stop"
            or entry["route_id"] not in eligible
            or not set(entry["crossed_junction_ids"]) <= set(allowed.get(town, []))
            or len(entry["signal_ids_at_inventory"]) != 1
        ):
            continue
        used = visited.setdefault(town, set())
        if entry["junction_id"] not in used and len(used) < per_town:
            selected.append(entry)
            used.add(entry["junction_id"])
    return selected


def validate_approach_guard(path, catalog, town, subset):
    guard = json.loads(Path(path).read_text())
    if guard["kind"] != "continuous-approach-train-v1":
        raise ValueError("Unrecognized approach guard")
    if digest(guard["base_guard"]) != guard["base_guard_sha256"]:
        raise ValueError("Original driving split guard changed")
    base = validate_guard(guard["base_guard"], catalog, town, subset)
    component = Path(guard["component_plan"])
    if digest(component) != guard["component_plan_sha256"]:
        raise ValueError("Component split plan changed")
    allowed, reserved = reserve_components(base, json.loads(component.read_text())["cases"])
    if allowed != guard["allowed_junctions"] or reserved != guard["reserved_component_junctions"]:
        raise ValueError("Approach guard changes validation reservations")
    entries = {
        e["route_id"]: e
        for e in json.loads((Path(catalog) / "catalog.json").read_text())["encounters"]
    }
    for rid in map(int, subset.split(",")):
        entry = entries[rid]
        if rid not in guard["geometry_eligibility"]["kept_route_ids"] or not set(
            entry["crossed_junction_ids"]
        ) <= set(allowed.get(town, [])):
            raise ValueError("Approach route touches reserved component geography")
    return dict(guard, guard_sha256=digest(path))


def approach_coverage(rows):
    """Descriptive, training-data eligibility; never redefines driving evaluation."""
    if not rows:
        return dict(continuous_signal_sequence=False, reason="no committed timestamps")
    for a, b in zip(rows, rows[1:]):
        if b["frame"] != a["frame"] + 2 or abs(b["timestamp"] - a["timestamp"] - 0.1) > 1e-5:
            raise ValueError("Approach audit requires contiguous10Hz samples")
    # Seek the first actual signal-hazard stop with motion before it. A queued
    # vehicle stop without a light hazard is not relabelled as a signal stop.
    stop = next(
        (
            i
            for i, r in enumerate(rows)
            if r["light_hazard"] and r["teacher_speed"] < 0.1 and r["ego_speed"] < 0.1
        ),
        None,
    )
    result = dict(
        continuous_signal_sequence=False,
        samples=len(rows),
        stop_index=stop,
        left_censored=stop == 0,
        distance_semantics="Teacher planner/API reference, not evaluator crossing line",
    )
    if stop is None:
        return dict(result, reason="no actual stationary teacher signal stop")
    signal = rows[stop]["light_id"]
    before = rows[:stop]
    first = next(
        (r for r in rows if r["light_id"] == signal and r["light_distance"] is not None), None
    )
    moving = [
        r
        for r in before
        if r["light_id"] == signal and r["ego_speed"] >= 1 and r["teacher_speed"] >= 1
    ]
    onset = next(
        (
            i
            for i in range(stop + 1)
            if rows[i]["light_id"] == signal
            and rows[i]["light_hazard"]
            and rows[i]["teacher_speed"] < 0.1
        ),
        stop,
    )
    release = next(
        (
            i
            for i in range(stop + 1, len(rows))
            if rows[i]["light_id"] == signal
            and rows[i]["light_state"] == "Green"
            and rows[i]["teacher_speed"] >= 1
        ),
        None,
    )
    waited = 0 if release is None else rows[release]["timestamp"] - rows[stop]["timestamp"]
    after = 0 if release is None else rows[-1]["timestamp"] - rows[release]["timestamp"]
    moved_after = release is not None and any(r["ego_speed"] >= 1 for r in rows[release:])
    stationary_wait = release is not None and all(
        r["ego_speed"] < 0.2 and r["light_id"] == signal for r in rows[stop:release]
    )
    start_distance = first["light_distance"] if first else None
    passed = bool(
        signal is not None
        and start_distance is not None
        and start_distance >= 40
        and len(moving) >= 10
        and onset > 0
        and release is not None
        and waited >= 1
        and after >= 3
        and moved_after
        and stationary_wait
    )
    result["left_censored"] = onset == 0
    return dict(
        result,
        continuous_signal_sequence=passed,
        signal_id=signal,
        first_signal_distance_m=start_distance,
        moving_pre_stop_samples=len(moving),
        onset_index=onset,
        release_index=release,
        actual_wait_s=waited,
        post_release_s=after,
        actual_motion_after_release=bool(moved_after),
        stationary_during_wait=bool(stationary_wait),
        reason="coverage passes" if passed else "incomplete approach/wait/release coverage",
        visibility_certified=False,
        state_labels_released=False,
    )
