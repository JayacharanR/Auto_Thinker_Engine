#!/usr/bin/env python3
"""Assess a completed M1 collection without changing raw data or eligibility."""

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def assess(root):
    root = Path(root).resolve()
    audit = json.loads((root / "data_audit.json").read_text())
    request = json.loads((root / "request.json").read_text())
    if not audit["audit_passed"] or audit["status"] != "complete":
        raise ValueError("Collection has not passed its full completion/integrity audit")
    changed = [
        p
        for p, h in request["source_hashes"].items()
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h
    ]
    vendor = activate()
    import carla

    split = json.loads((Path(request["catalog"]).parent / "locations.json").read_text())
    maps = {}
    for town, expected in request["geometry_eligibility"]["map_sha256"].items():
        path = (
            Path(vendor["carla_root"])
            / "CarlaUE4/Content/Carla/Maps/OpenDrive"
            / f"{town.removesuffix('_Opt')}.xodr"
        )
        maps[town] = carla.Map(town, path.read_text())
        if hashlib.sha256(maps[town].to_opendrive().encode()).hexdigest() != expected:
            raise ValueError("Assessment map changed")
    totals, groups = Counter(), defaultdict(Counter)
    failures, violations, signals = [], [], Counter()
    speeds, locations, frames_checked = [], set(), 0
    for episode in audit["episodes"]:
        data = Path(episode["path"])
        meta = json.loads((data / "episode.json").read_text())
        outcome = episode["outcome"]
        record = outcome["evaluator_record"] if outcome else None
        profile_path = data.parent / "collection_profile.json"
        profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
        row = Counter(
            episodes=1,
            samples=episode["samples"],
            imitation_samples=episode["samples"]
            if outcome and outcome["imitation_eligible"]
            else 0,
            imitation_episodes=int(bool(outcome and outcome["imitation_eligible"])),
            rule_safe=int(bool(record and record["rule_safe_completed"])),
            arrivals=int(bool(record and record["completed"])),
            censored=int(not outcome or outcome["episode_censored"]),
            collisions=record["collisions"] if record else 0,
            red=record["red_violations"] if record else 0,
            stop=record["stop_violations"] if record else 0,
            noise_ticks=profile.get("noise_ticks", 0),
            safety_stop_ticks=profile.get("safety_stop_ticks", 0),
            feedback_mismatches=profile.get("control_differences", 0),
            missing_profile=int(not profile_path.exists()),
        )
        totals.update(row)
        for key in ("family", "town", "mode"):
            groups[f"{key}:{meta[key]}"].update(row)
        if not outcome or not outcome["imitation_eligible"]:
            failures.append(
                {
                    "data": str(data),
                    "route_id": meta["route_id"],
                    "family": meta["family"],
                    "outcome": outcome,
                    "profile": profile,
                }
            )
        allowed = set(split["assignments"][meta["town"]]["train"])
        if meta["route_id"] not in request["geometry_eligibility"]["kept_route_ids"]:
            raise ValueError("Collection used a geometrically excluded route")
        for shard in sorted(data.glob("shard_*.zip")):
            with zipfile.ZipFile(shard) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                for sample in manifest["samples"]:
                    labels = json.loads(archive.read(f"{sample['prefix']}/targets.json"))
                    facts = labels["facts"]
                    waypoint = maps[meta["town"]].get_waypoint(
                        carla.Location(*facts["ego_pose_world"][:3])
                    )
                    if waypoint.is_junction:
                        locations.add((meta["town"], waypoint.junction_id))
                        if waypoint.junction_id not in allowed:
                            violations.append(
                                {
                                    "data": str(data),
                                    "frame": sample["frame"],
                                    "junction_id": waypoint.junction_id,
                                }
                            )
                    frames_checked += 1
                    if outcome and outcome["imitation_eligible"]:
                        speeds.append(facts["teacher_target_speed_mps"])
                        signals[str(facts["next_light_state"])] += 1
    if frames_checked != audit["totals"]["samples"]:
        raise ValueError("Assessment sample count differs from data audit")
    progress = json.loads((root / "progress.json").read_text())
    return {
        "collection": str(root),
        "integrity_audit_sha256": hashlib.sha256(
            (root / "data_audit.json").read_bytes()
        ).hexdigest(),
        "request_sha256": hashlib.sha256((root / "request.json").read_bytes()).hexdigest(),
        "source_files_changed_since_collection": changed,
        "totals": dict(totals),
        "groups": dict(groups),
        "ineligible_episodes": failures,
        "spatial_audit": {
            "sampled_ego_positions_checked": frames_checked,
            "observed_train_junctions": sorted(locations),
            "partition_violations": violations,
        },
        "labels": {
            "imitation_light_states": dict(signals),
            "stopped_target_fraction": float(np.mean(np.asarray(speeds) < 0.1)),
            "target_speed_percentiles_mps": np.percentile(speeds, [0, 25, 50, 75, 100]).tolist(),
        },
        "wall_seconds": progress["elapsed_seconds_this_invocation"],
        "shards": audit["totals"],
        "m1_initial_data_gate_passed": not violations,
        "limits": "Collection includes perturbations, one infrastructure failure and a final "
        "final cap; these are not held-out benchmark success rates. Three observed collisions "
        "are counted; the failed evaluator has no final outcome. Student rollouts and reactive "
        "counterfactual branches are still absent. Town05 and heldout_dev remain unused.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    result = assess(args.out)
    atomic_json(args.report, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("totals", "labels", "wall_seconds", "m1_initial_data_gate_passed")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
