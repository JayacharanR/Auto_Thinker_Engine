#!/usr/bin/env python3
"""Audit physical rule prototypes without promoting expert behavior to student evidence."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def assess(output):
    output = Path(output)
    manifest = read(output / "manifest.json")
    source_hashes = read(output / "source_hashes.json")
    source_valid = all(
        digest(output / "source" / path) == expected for path, expected in source_hashes.items()
    )
    episodes = list((output / "episodes").iterdir())
    if len(episodes) != 1:
        raise ValueError("Exactly one physical component episode required")
    episode = episodes[0]
    setup = episode / "controlled_rule"
    trace = lines(setup / "setup_trace.jsonl")
    controls = lines(setup / "requested_executed.jsonl")
    native_trace = lines(episode / "trace.jsonl")
    profile = read(episode / "profile.json")
    binding = read(setup / "binding.json")
    acknowledgment = read(setup / "dynamic_setup_acknowledgment.json")
    frames = [row["frame"] for row in trace]
    aligned = (
        bool(trace)
        and frames == [row["frame"] for row in controls]
        and frames == [row["frame"] for row in native_trace]
        and all(b == a + 1 for a, b in zip(frames, frames[1:]))
        and profile["frames"] == len(trace)
        and profile["all_camera_frames_aligned"]
    )
    scored = [row for row in trace if not row["scripted_setup"]]
    proposals_unchanged = bool(scored) and all(
        row["executed_control"] == row["requested_control"] and row["source"] == "evaluated_agent"
        for row in controls
        if not row["scripted_setup"]
    )
    no_reactivation = bool(scored) and all(
        not row["scripted_setup"] for row in trace[trace.index(scored[0]) :]
    )
    expected_ids = acknowledgment["expected_ids"]
    observed_ids = sorted(row["id"] for row in acknowledgment["observed"])
    recordings = sorted((output / "recordings").glob("*.log"))
    if len(recordings) != 1:
        raise ValueError("Exactly one unfiltered native recording required")
    contacts = recorder_contacts(recordings[0])
    rgb = read(setup / "raw_rgb_manifest.json")
    snapshots = []
    for sample in rgb["samples"]:
        if sample["frame"] not in frames:
            raise ValueError("Raw setup RGB frame absent from aligned trace")
        snapshots.extend(dict(path=path, sha256=digest(path)) for path in sample["images"].values())
    video = episode / "three_cameras.mp4"
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,nb_frames",
                "-of",
                "json",
                str(video),
            ],
            text=True,
        )
    )["streams"][0]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    unseen = bool(scored) and all(
        bool(boxes) and all(box["in_image"] is False for box in boxes)
        for row in scored
        for boxes in row["housing_frusta"].values()
    )
    displacement = (
        float(np.linalg.norm(np.array(scored[-1]["ego"][:2]) - scored[0]["ego"][:2]))
        if scored
        else None
    )
    opportunities = [
        dict(path=str(path), sha256=digest(path), data=read(path))
        for path in sorted((output / "rule_opportunities").glob("*.json"))
    ]
    return dict(
        output=str(output),
        manifest_sha256=digest(output / "manifest.json"),
        policy_kind=manifest["policy_kind"],
        source_hashes_verified=source_valid,
        synchronized_frames=bool(aligned),
        declared_actor_setup=expected_ids == observed_ids,
        evaluated_controls_unchanged=proposals_unchanged,
        prefix_never_reactivated=no_reactivation,
        binding=binding,
        frames=len(trace),
        scored_frames=len(scored),
        actual_handoff=None if not scored else scored[0],
        actual_scored_seconds=None
        if not scored
        else scored[-1]["timestamp"] - scored[0]["timestamp"],
        scored_displacement_m=displacement,
        maximum_scored_speed_mps=max((row["speed_mps"] for row in scored), default=None),
        all_applicable_housings_outside_every_view=unseen,
        native_summary=read(output / "summary.json"),
        native_opportunities=opportunities,
        recorder=contacts,
        raw_rgb_snapshots=snapshots,
        video=dict(info, full_decode_passed=True),
        setup_certified=False,
        student_rule_success_claim=False,
        note=(
            'Individual prototype audit only; paired rendering/condition '
            'validity must precede certification.'
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve every prior audit")
    source = args.out.parent / (args.out.stem + "_source.py")
    shutil.copyfile(__file__, source)
    request = read(args.root / "request.json")
    records, errors = [], []
    for row in request["cases"]:
        output = args.root / row["name"]
        if not (output / "summary.json").exists():
            errors.append(dict(case=row["name"], error="No terminal summary"))
            continue
        try:
            records.append(dict(case=row["name"], **assess(output)))
        except Exception as error:
            errors.append(dict(case=row["name"], error=str(error)))
    atomic_json(
        args.out,
        dict(
            status="complete" if not errors else "complete_with_failures",
            source_sha256=digest(source),
            request_sha256=digest(args.root / "request.json"),
            records=records,
            errors=errors,
            setup_certified=False,
            no_student_rule_outcomes=True,
        ),
    )
    print(json.dumps(dict(prototypes=len(records), errors=errors)), flush=True)


if __name__ == "__main__":
    main()
