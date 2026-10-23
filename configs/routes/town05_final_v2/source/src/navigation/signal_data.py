"""TRAIN-only static signal-supervision selection; no driving or action targets."""

import numpy as np


def stop_reference(points, lane_keys, stop_lanes):
    """Project a line onto segments leaving its approach lane, across road-ID changes."""
    points = np.asarray(points, float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(lengths)]
    candidates = []
    for line in stop_lanes:
        center = np.asarray(line["xyz"])
        for index, length in enumerate(lengths):
            if tuple(lane_keys[index]) != tuple(line["lane"]) or length < 1e-6:
                continue
            delta = points[index + 1] - points[index]
            ratio = np.clip(np.dot(center - points[index], delta) / length**2, 0, 1)
            gap = np.linalg.norm(points[index] + ratio * delta - center)
            if gap < 1:
                candidates.append(
                    {"gap_m": float(gap), "arc_m": float(arc[index] + ratio * length), "line": line}
                )
    return min(candidates, key=lambda row: row["gap_m"]) if candidates else None


def select_entries(catalog, guard, maximum_per_town=4):
    if catalog["role"] != "train" or maximum_per_town < 1:
        raise ValueError("Training catalog and positive per-town limit required")
    kept = set(guard["geometry_eligibility"]["kept_route_ids"])
    picked, junctions = [], {}
    for entry in sorted(
        catalog["encounters"],
        key=lambda e: (e["town"], e["family"] != "signal_stop", e["route_id"]),
    ):
        town = entry["town"]
        if town.removesuffix("_Opt") == "Town05":
            raise ValueError("Town05 remains held out")
        if (
            entry["family"] not in ("signal_stop", "signal_release_memory")
            or entry["route_id"] not in kept
        ):
            continue
        if not set(entry["crossed_junction_ids"]) <= set(guard["allowed_junctions"][town]):
            raise ValueError("Signal collection entry overlaps a reserved junction")
        if len(entry["signal_ids_at_inventory"]) != 1:
            continue  # Ambiguous labels cannot supply a physical intervention target.
        visited = junctions.setdefault(town, set())
        if entry["junction_id"] not in visited and len(visited) < maximum_per_town:
            picked.append(entry)
            visited.add(entry["junction_id"])
    return picked


def phase_response(red, green, red_repeat):
    """Conservative static RGB-response heuristic, not original-scene visibility.

    Cutoffs were chosen on the earlier four TRAIN probes, before this corpus.
    Empty/back-facing/weak responses remain unlabelled, never negative state truth.
    """
    if (
        red.shape != green.shape
        or red.shape != red_repeat.shape
        or red.ndim != 3
        or red.shape[2] != 3
        or not red.size
    ):
        raise ValueError("Three nonempty equally shaped RGB crops required")
    a, b, c = (image.astype(np.float32) for image in (red, green, red_repeat))
    changed, noise = float(np.abs(a - b).mean()), float(np.abs(a - c).mean())
    red_strength = float((a[..., 0] - a[..., 1]).max())
    green_strength = float((b[..., 1] - b[..., 0]).max())
    repeat_red_strength = float((c[..., 0] - c[..., 1]).max())
    supported = (
        changed >= 2
        and changed >= 5 * max(noise, 0.25)
        and min(red_strength, green_strength, repeat_red_strength) >= 30
    )
    return {
        "static_phase_response_supported": bool(supported),
        "red_green_mae": changed,
        "red_repeat_mae": noise,
        "red_green_dominance": [red_strength, green_strength, repeat_red_strength],
        "original_scene_visibility_valid": None,
    }


def phase_response_v2(red, green, red_repeat):
    """Phase-on/off evidence without assuming a fixed rendered green chroma.

    Require separate red-on and green-on intensity changes plus a stable repeat.
    A global brightness shift alone cannot satisfy both opposite-signed changes.
    This version does not retroactively change the original corpus data gate.
    """
    baseline = phase_response(red, green, red_repeat)
    a, b = red.astype(np.float32), green.astype(np.float32)
    red_on = float((a - b).mean(axis=2).max())
    green_on = float((b - a).mean(axis=2).max())
    changed, noise = baseline["red_green_mae"], baseline["red_repeat_mae"]
    supported = (
        changed >= 2
        and changed >= 5 * max(noise, 0.25)
        and min(red_on, green_on) >= 20
        and float(np.abs(a - b).max()) >= 25
    )
    return {
        **baseline,
        "static_phase_response_supported": bool(supported),
        "red_on_gain": red_on,
        "green_on_gain": green_on,
        "validator_version": 2,
    }
