#!/usr/bin/env python3
"""Independently match native next-tick controls to controlled-rule proposals."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve every execution audit")
    args.out.mkdir(parents=True)
    for source in [Path(__file__), ROOT / "src/navigation/recorder_actions.py"]:
        shutil.copyfile(source, args.out / source.name)
    paths = sorted(
        {
            p
            for root in args.root
            for p in root.glob("*/episodes/*/controlled_rule/requested_executed.jsonl")
        }
    )
    results, errors = [], []
    for path in paths:
        episode = path.parents[1]
        output = episode.parents[1]
        try:
            if not (output / "summary.json").exists():
                raise ValueError("Controlled rule attempt is not terminal")
            world = [
                json.loads(line)
                for line in (episode / "world_trace.jsonl").read_text().splitlines()
            ]
            controls = [json.loads(line) for line in path.read_text().splitlines()]
            hero = next(a["id"] for a in world[0]["actors"] if a["role"] == "hero")
            recordings = list((output / "recordings").glob("*.log"))
            if len(recordings) != 1:
                raise ValueError("One physical recording per controlled case required")
            native = vehicle_frames(recordings[0], hero)
            aligned, receipt = align_world_frames(native, world)
            checked, unacknowledged, differences = 0, [], []
            for row in controls:
                actual = aligned.get(row["frame"] + 1)
                if actual is None:
                    unacknowledged.append(row["frame"])
                    continue
                gap = float(np.max(np.abs(np.array(actual["control"]) - row["executed_control"])))
                if gap > 1e-5:
                    differences.append(
                        dict(
                            frame=row["frame"],
                            maximum_difference=gap,
                            scripted_setup=row["scripted_setup"],
                        )
                    )
                if (
                    not row["scripted_setup"]
                    and row["executed_control"] != row["requested_control"]
                ):
                    differences.append(
                        dict(frame=row["frame"], unexpected_post_handoff_override=True)
                    )
                checked += 1
            results.append(
                dict(
                    output=str(output),
                    control_log_sha256=digest(path),
                    recorder_sha256=digest(recordings[0]),
                    alignment=receipt,
                    checked_acknowledgments=checked,
                    missing_final_ack_frames=unacknowledged,
                    differences=differences,
                    same_frame_observations=len(world),
                    every_observed_next_tick_matches=not differences,
                    all_commands_have_native_ack=not unacknowledged,
                    limitation=(
                        'Final commands without a subsequent recorder tick remain '
                        'unacknowledged, never guessed.'
                    ),
                )
            )
        except Exception as error:
            errors.append(dict(output=str(output), error=str(error)))
    result = dict(
        records=results,
        errors=errors,
        candidates=len(paths),
        verified_recorders=len(results),
        differences=sum(len(r["differences"]) for r in results),
        missing_ack_frames=sum(len(r["missing_final_ack_frames"]) for r in results),
        stage="M3 physical actuation audit; no student-rule behavior claim",
        source_hashes={p.name: digest(p) for p in args.out.glob("*.py")},
    )
    atomic_json(args.out / "assessment.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("records", "errors")}), flush=True
    )


if __name__ == "__main__":
    main()
