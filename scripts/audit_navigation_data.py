#!/usr/bin/env python3
"""Verify every committed RGB/label sample and conservatively report pilot coverage."""

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.contracts import ControlCommand  # noqa: E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def audit(root):
    root = Path(root)
    episodes = []
    totals = Counter()
    for metadata_path in sorted(root.rglob("data/episode.json")):
        data = metadata_path.parent
        metadata = json.loads(metadata_path.read_text())
        outcome_path = data / "outcome.json"
        outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else None
        prior = None
        count = differences = 0
        files = sorted(data.glob("shard_*.zip"))
        for shard in files:
            with zipfile.ZipFile(shard) as archive:
                for record in shard_records(shard):
                    packet = load_policy_sample(shard, record)
                    labels = json.loads(archive.read(f"{record['prefix']}/targets.json"))
                    if packet.frame != record["frame"] or labels["frame"] != packet.frame:
                        raise ValueError("Policy/label/manifest frame mismatch")
                    if prior and (
                        packet.frame != prior[0] + 2
                        or abs(packet.timestamp - prior[1] - 0.1) > 1e-5
                    ):
                        raise ValueError("Lost or repeated 10Hz samples")
                    prior = packet.frame, packet.timestamp
                    for control in (labels["expert_control"], labels["executed_control"]):
                        ControlCommand(*control).validate()
                    facts = labels["facts"]
                    if (
                        not facts["action_execution_observed"]
                        or facts["actuator_feedback_frame"] != packet.frame + 1
                    ):
                        raise ValueError("Action execution lacks next-frame feedback")
                    if facts["world"]["frame"] != packet.frame:
                        raise ValueError("World labels were captured on a different physics frame")
                    changed = not np.allclose(
                        facts["requested_control"], labels["executed_control"], atol=1e-5, rtol=0
                    )
                    if changed and not labels["intervention"]:
                        raise ValueError("Unreported actuator difference")
                    differences += int(changed)
                    count += 1
        profile_path = data.parent / "collection_profile.json"
        if profile_path.exists() and json.loads(profile_path.read_text())["samples"] != count:
            raise ValueError("Collection profile and committed samples disagree")
        size = sum(p.stat().st_size for p in files)
        episodes.append(
            {
                "path": str(data),
                "route_id": metadata["route_id"],
                "family": metadata["family"],
                "town": metadata["town"],
                "samples": count,
                "shards": len(files),
                "bytes": size,
                "control_differences": differences,
                "partial_archives": len(list(data.glob("*.part"))),
                "outcome": outcome,
            }
        )
        totals.update(samples=count, shards=len(files), bytes=size)
    sample_count = totals["samples"]
    if not episodes or not sample_count:
        raise ValueError("No committed training samples found")
    return {
        "output": str(root),
        "audit_passed": True,
        "episodes": episodes,
        "totals": dict(totals),
        "coverage": {
            "towns": dict(Counter(e["town"] for e in episodes)),
            "families": dict(Counter(e["family"] for e in episodes)),
        },
        "bytes_per_timestamp": totals["bytes"] / sample_count,
        "estimated_rgb_shards_gb": {
            str(n): totals["bytes"] * n / sample_count / 1e9 for n in (20000, 100000)
        },
        "scope": "Every committed RGB frame decoded and member hashed; partial archives ignored. "
        "Eligible imitation episodes require durable safe outcomes; no policy learning claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    result = audit(args.out)
    if args.report:
        atomic_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
