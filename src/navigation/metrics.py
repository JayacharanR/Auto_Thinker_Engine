"""Honest headline metrics over the existing harness's raw episode records."""

import math


def execution_status(records, crashed, stop_reason, profile_steps=0):
    """Distinguish a finished driving trial from an evaluator/setup failure."""
    if crashed or not records:
        return "failed"
    if any("couldn't be set up" in r.get("status", "") for r in records):
        return "failed"
    if stop_reason == "user_stop":
        return "stopped"
    if profile_steps:
        return "profile_complete" if stop_reason == "profile_step_budget" else "failed"
    # A blocked/crashed driver is still a recorded policy failure, not a successful
    # route. The summary retains its failure even though evaluation is finished.
    return "evaluation_complete" if stop_reason is None else "stopped"


def wilson_interval(successes, trials, z=1.959963984540054):
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Expected 0 <= successes <= trials, trials > 0")
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def summarize_records(records, execution_failure=False):
    episodes = []
    for record in records:
        infractions = record.get("infractions", {})
        collisions = sum(
            len(infractions.get(k, []))
            for k in ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")
        )
        red = len(infractions.get("red_light", []))
        stop = len(infractions.get("stop_infraction", []))
        completion = record.get("scores", {}).get("score_route", 0.0)
        # This harness calls an infraction-free arrival "Perfect"; "Completed"
        # denotes arrival with infractions. Runtime failures always take precedence.
        completed = (
            record.get("status") in ("Completed", "Perfect")
            and completion >= 99.99
            and not execution_failure
        )
        # Any infraction prevents the stricter safe-and-compliant headline.
        safe = completed and not any(infractions.values())
        episodes.append(
            {
                "route_id": record.get("route_id"),
                "status": record.get("status"),
                "completion_percent": completion,
                "completed": completed,
                "safe_completed": safe,
                "collisions": collisions,
                "red_violations": red,
                "stop_violations": stop,
                "infractions": infractions,
                "meta": record.get("meta", {}),
            }
        )
    n = len(episodes)
    successes = sum(e["safe_completed"] for e in episodes)
    return {
        "episodes": episodes,
        "count": n,
        "safe_completed": successes,
        "all_collisions": sum(e["collisions"] for e in episodes),
        "safe_completion_wilson95": wilson_interval(successes, n) if n else None,
        "statistical_note": "Wilson interval assumes independent trials; repeated routes need "
        "clustered analysis before generalization claims",
    }
