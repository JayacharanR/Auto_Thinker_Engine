"""Training/audit-only world observations; never inputs to a camera policy."""

import numpy as np


def capture_world_state(world):
    snapshot = world.get_snapshot()
    actors = []
    for actor in sorted(world.get_actors(), key=lambda a: a.id):
        if not actor.type_id.startswith(("vehicle.", "walker.", "traffic.traffic_light")):
            continue
        state = snapshot.find(actor.id)
        if state is None:
            raise RuntimeError(f"Actor {actor.id} missing from synchronous snapshot")
        transform = state.get_transform()
        velocity = state.get_velocity()
        row = {
            "id": actor.id,
            "type": actor.type_id,
            "role": actor.attributes.get("role_name", ""),
            "pose": [
                transform.location.x,
                transform.location.y,
                transform.location.z,
                transform.rotation.pitch,
                transform.rotation.yaw,
                transform.rotation.roll,
            ],
            "velocity": [velocity.x, velocity.y, velocity.z],
        }
        if actor.type_id == "traffic.traffic_light":
            row["signal"] = str(actor.get_state())
            row["phase_elapsed"] = actor.get_elapsed_time()
        actors.append(row)
    return {"frame": snapshot.frame, "actors": actors}


def compare_world_prefixes(first, second):
    """Pair seeded actors by creation order, requiring matching types/roles.

    This checks observable state agreement only, not hidden simulator state or
    divergence after branching. Absolute actor IDs may change between servers.
    """
    if not first or len(first) != len(second):
        raise ValueError("Need equally long, nonempty world traces")
    maxima = dict(position_m=0.0, rotation_deg=0.0, velocity_mps=0.0, signal_phase_s=0.0)
    topology_equal = True
    signals_equal = True
    max_dynamic_actors = 0
    for a, b in zip(first, second):
        aa, bb = a["actors"], b["actors"]
        max_dynamic_actors = max(
            max_dynamic_actors,
            sum(x["type"].startswith(("vehicle.", "walker.")) for x in aa),
        )
        if [(x["type"], x["role"]) for x in aa] != [(x["type"], x["role"]) for x in bb]:
            topology_equal = False
            continue
        for x, y in zip(aa, bb):
            p, q = np.asarray(x["pose"]), np.asarray(y["pose"])
            maxima["position_m"] = max(maxima["position_m"], float(np.linalg.norm(p[:3] - q[:3])))
            maxima["rotation_deg"] = max(
                maxima["rotation_deg"], float(np.abs((p[3:] - q[3:] + 180) % 360 - 180).max())
            )
            maxima["velocity_mps"] = max(
                maxima["velocity_mps"],
                float(np.linalg.norm(np.asarray(x["velocity"]) - np.asarray(y["velocity"]))),
            )
            signals_equal &= x.get("signal") == y.get("signal")
            maxima["signal_phase_s"] = max(
                maxima["signal_phase_s"], abs(x.get("phase_elapsed", 0) - y.get("phase_elapsed", 0))
            )
    tolerances = dict(position_m=0.1, rotation_deg=0.5, velocity_mps=0.1, signal_phase_s=0.051)
    return {
        "frames_compared": len(first),
        "actor_type_role_sequence_equal": topology_equal,
        "signals_equal": bool(signals_equal),
        "max_dynamic_actors_including_ego": max_dynamic_actors,
        "maximum_differences": maxima,
        "tolerances": tolerances,
        "observable_prefix_within_tolerance": bool(
            topology_equal and signals_equal and all(maxima[k] <= tolerances[k] for k in maxima)
        ),
        "limitation": "Seeded actor creation order defines correspondence; observable state "
        "agreement does not prove hidden-state or post-branch determinism.",
    }
