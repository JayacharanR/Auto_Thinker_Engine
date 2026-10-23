#!/usr/bin/env python3
"""Read-only20Hz geography and actual-braking audit of completed TRAIN collection."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def audit(out, destination):
    import hashlib

    import carla

    data = json.loads((out / "data_audit.json").read_text())
    guard = json.loads((out / "guard.json").read_text())
    request = json.loads((out / "request.json").read_text())
    if not data["audit_passed"] or any(
        digest(out / "source" / p) != h for p, h in request["source_hashes"].items()
    ):
        raise ValueError("Completed original integrity/source audit required")
    maps = {}
    episodes = []
    for e in data["continuous_approach_episodes"]:
        path = Path(e["path"])
        meta = json.loads((path / "episode.json").read_text())
        town = meta["town"]
        if meta["strict_training_guard_sha256"] != digest(out / "guard.json"):
            raise ValueError("Episode training guard changed")
        if town not in maps:
            map_root = Path.home() / "software/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive"
            source = map_root / f"{town.removesuffix('_Opt')}.xodr"
            maps[town] = carla.Map(town, source.read_text())
            if (
                hashlib.sha256(maps[town].to_opendrive().encode()).hexdigest()
                != guard["geometry_eligibility"]["map_sha256"][town]
            ):
                raise ValueError("Map changed since collection guard")
        trace = path.parent / "trace.jsonl"
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        moving = None
        braking = None
        actual = []
        for index, r in enumerate(rows):
            if index and (
                r["frame"] != rows[index - 1]["frame"] + 1
                or abs(r["timestamp"] - rows[index - 1]["timestamp"] - 0.05) > 1e-5
            ):
                raise ValueError("Collection trace is not contiguous20Hz")
            wp = maps[town].get_waypoint(carla.Location(*r["ego"][:3]))
            if wp.is_junction:
                actual.append(wp.junction_id)
                if wp.junction_id not in guard["allowed_junctions"][town]:
                    raise ValueError("An actual20Hz position touches reserved geography")
            if moving is None and r["speed_mps"] >= 1:
                moving = index
            if (
                moving is not None
                and braking is None
                and index > moving
                and r["control"][2] > 0.5
                and r["speed_mps"] > 0.2
            ):
                braking = index
        episodes.append(
            dict(
                route_id=e["route_id"],
                town=town,
                trace_sha256=digest(trace),
                trace_path=str(trace),
                control_ticks=len(rows),
                actual_junction_ids=sorted(set(actual)),
                approach_imitation_eligible=e["approach_imitation_eligible"],
                first_actual_motion_s=rows[moving]["timestamp"] if moving is not None else None,
                first_moving_actual_brake_s=rows[braking]["timestamp"]
                if braking is not None
                else None,
                semantic_zero_speed_intent_onset_index=e["coverage"].get("onset_index"),
            )
        )
    result = dict(
        audit_passed=True,
        all_actual_20hz_positions_train=True,
        total_control_ticks=sum(e["control_ticks"] for e in episodes),
        episodes=episodes,
        source_sha256=digest(Path(__file__)),
        collection_audit_sha256=digest(out / "data_audit.json"),
        guard_sha256=digest(out / "guard.json"),
        scope="TRAIN diagnostic only, not camera behavior",
    )
    atomic_json(destination, result)
    print(
        json.dumps(
            dict(
                episodes=len(episodes),
                control_ticks=result["total_control_ticks"],
                audit_passed=True,
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    activate()
    audit(args.out.resolve(), args.report)


if __name__ == "__main__":
    main()
