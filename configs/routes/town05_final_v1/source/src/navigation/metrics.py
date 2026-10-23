"""Honest headline metrics over the existing harness's raw episode records."""

import math

PROGRESS_ONLY_INFRACTIONS = ("min_speed_infractions",)


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


def summarize_records(records, execution_failure=False, last_route_only=False):
    """execution_failure marks every record failed, or only the route that was running
    when the attempt stopped or crashed (last_route_only, for multi-route attempts)."""
    episodes = []
    for index, record in enumerate(records):
        failed = execution_failure and (not last_route_only or index == len(records) - 1)
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
            record.get("status") in ("Completed", "Perfect") and completion >= 99.99 and not failed
        )
        # Any infraction prevents the stricter safe-and-compliant headline.
        safe = completed and not any(infractions.values())
        # Encounters impose waits (held reds, stops, stopped leads), which trip the
        # evaluator's speed-relative-to-traffic check. That is reported separately;
        # every collision, rule, route and blocking infraction still counts.
        rule_safe = completed and not any(
            v for k, v in infractions.items() if k not in PROGRESS_ONLY_INFRACTIONS
        )
        episodes.append(
            {
                "route_id": record.get("route_id"),
                "status": record.get("status"),
                "completion_percent": completion,
                "completed": completed,
                "safe_completed": safe,
                "rule_safe_completed": rule_safe,
                "min_speed_infractions": len(infractions.get("min_speed_infractions", [])),
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
        "rule_safe_completed": sum(e["rule_safe_completed"] for e in episodes),
        "all_collisions": sum(e["collisions"] for e in episodes),
        "safe_completion_wilson95": wilson_interval(successes, n) if n else None,
        "statistical_note": "Wilson interval assumes independent trials; repeated routes need "
        "clustered analysis before generalization claims",
    }
