"""Outcome-blind, seeded selection of the development encounter catalog.

Selection sees only inventory geometry and controls, never a driving result.
Each family is filled round-robin over its strata (towns, stop actors or turn
types), preferring movements, lights and stops not yet used. A movement is reused
only when its stratum is exhausted; every reuse is a separate route with its own
traffic seed, and analysis clusters by location.
"""

from collections import Counter

import numpy as np

FAMILY_SPECS = {
    "signal_stop": {"scenario": "NavigationSignalHold", "strata": "town", "distinct": "light"},
    "signal_release_memory": {
        "scenario": "NavigationSignalHold",
        "strata": "town",
        "distinct": "light",
    },
    "stop_sign": {"scenario": None, "strata": "stop", "distinct": None},
    "crossing_yield": {"scenario": "crossing", "strata": "town_turn", "distinct": "junction"},
    "following_stopped_vehicle": {
        "scenario": "HardBreakRoute",
        "strata": "town",
        "distinct": "junction",
    },
    "route_junction": {"scenario": None, "strata": "town_turn", "distinct": "junction"},
}

SCENARIO_PARAMETERS = {
    # A queue at the held red stopped the pilot ego 25.6 m from the line, outside a 20 m
    # zone, so the light stayed red for the full maximum; 40 m covers a short queue.
    "signal_stop": {"stationary_s": 3.0, "max_hold_s": 45.0, "near_m": 40.0, "clearance_s": 3.0},
    "signal_release_memory": {
        "stationary_s": 10.0,
        "max_hold_s": 30.0,
        "near_m": 20.0,
        "clearance_s": 3.0,
    },
    "SignalizedJunctionLeftTurn": {"flow_speed": 10.0, "source_dist_interval": [25.0, 50.0]},
    "NonSignalizedJunctionLeftTurn": {"flow_speed": 8.3, "source_dist_interval": [15.0, 30.0]},
    "NonSignalizedJunctionRightTurn": {"flow_speed": 8.3, "source_dist_interval": [15.0, 30.0]},
}

RELEASE_START_M = 12.0  # signal_release_memory starts this far before the junction entry
HARD_BREAK_TRIGGER_M = 20.0  # along the approach, leaving room for a lead vehicle
FOLLOW_MIN_APPROACH_M = 45.0


def movement_key(m):
    return (m["town"], m["junction_id"], tuple(m["entry_lane"]), tuple(m["exit_lane"]))


def turn(m):
    return next((o for o in m["planner_options"] if o != "LANEFOLLOW"), "STRAIGHT")


def crossing_scenario(m):
    """Pinned junction-flow scenario applicable to this movement, if any."""
    if "signal" in m["control"] and turn(m) == "LEFT" and m.get("has_opposite_entry"):
        return "SignalizedJunctionLeftTurn"
    if m["control"] == ["none"] and turn(m) == "LEFT" and m.get("has_opposite_entry"):
        return "NonSignalizedJunctionLeftTurn"
    if m["control"] == ["none"] and turn(m) == "RIGHT" and m.get("has_left_entry"):
        return "NonSignalizedJunctionRightTurn"
    return None


def family_pool(family, movements):
    usable = [m for m in movements if m["valid"] and not m.get("excluded")]
    if family in ("signal_stop", "signal_release_memory"):
        return [m for m in usable if "signal" in m["control"]]
    if family == "stop_sign":
        return [m for m in usable if "stop" in m["control"]]
    if family == "crossing_yield":
        return [m for m in usable if crossing_scenario(m)]
    if family == "following_stopped_vehicle":
        return [m for m in usable if m["approach_m"] >= FOLLOW_MIN_APPROACH_M]
    return usable


def stratum(kind, m):
    if kind == "town":
        return (m["town"],)
    if kind == "stop":
        return (m["town"], min(m["stop_ids"]))
    if kind == "town_turn":
        return (m["town"], crossing_scenario(m) or turn(m))
    raise ValueError(kind)


def distinct_value(kind, m):
    if kind == "light":
        return (m["town"], tuple(m["signal_ids"]))
    if kind == "junction":
        return (m["town"], m["junction_id"])
    return None


def select_family(family, movements, count, rng, used_elsewhere):
    spec = FAMILY_SPECS[family]
    groups = {}
    for m in family_pool(family, movements):
        groups.setdefault(stratum(spec["strata"], m), []).append(m)
    if not groups:
        raise ValueError(f"No candidate movements for {family}")
    for key in groups:  # seeded order within each stratum
        groups[key] = [groups[key][i] for i in rng.permutation(len(groups[key]))]
    order = sorted(groups)
    order = [order[i] for i in rng.permutation(len(order))]
    picks, uses, distinct = [], Counter(), Counter()
    while len(picks) < count:
        for key in order:
            if len(picks) == count:
                break
            best = min(
                groups[key],
                key=lambda m: (
                    uses[movement_key(m)],
                    distinct[distinct_value(spec["distinct"], m)] if spec["distinct"] else 0,
                    used_elsewhere[movement_key(m)],
                ),
            )
            occurrence = uses[movement_key(best)]
            uses[movement_key(best)] += 1
            if spec["distinct"]:
                distinct[distinct_value(spec["distinct"], best)] += 1
            picks.append({"family": family, "movement": best, "occurrence": occurrence})
    return picks


def select_catalog(movements, families, per_family, seed):
    """Deterministic for a given inventory, family list, count and seed."""
    rng = np.random.default_rng(seed)
    used, catalog = Counter(), []
    for family in families:
        picks = select_family(family, movements, per_family, rng, used)
        for pick in picks:
            used[movement_key(pick["movement"])] += 1
            m = pick["movement"]
            scenario = FAMILY_SPECS[family]["scenario"]
            pick["scenario"] = crossing_scenario(m) if scenario == "crossing" else scenario
            pick["parameters"] = SCENARIO_PARAMETERS.get(
                family, SCENARIO_PARAMETERS.get(pick["scenario"], {})
            )
        catalog.extend(picks)
    return catalog


def coverage(catalog):
    """Counts reported with the frozen catalog; locations are the clustering unit."""
    out = {}
    for family in dict.fromkeys(p["family"] for p in catalog):
        picks = [p for p in catalog if p["family"] == family]
        out[family] = {
            "encounters": len(picks),
            "distinct_movements": len({movement_key(p["movement"]) for p in picks}),
            "distinct_junctions": len(
                {(p["movement"]["town"], p["movement"]["junction_id"]) for p in picks}
            ),
            "reused_movements": sum(p["occurrence"] > 0 for p in picks),
            "towns": dict(Counter(p["movement"]["town"] for p in picks)),
            "turns": dict(Counter(turn(p["movement"]) for p in picks)),
            "scenarios": dict(Counter(str(p["scenario"]) for p in picks)),
        }
    return out
