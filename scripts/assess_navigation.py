#!/usr/bin/env python3
"""Audit an M0 attempt and optionally compare two same-seed control/pose prefixes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.metrics import execution_status, summarize_records  # noqa: E402
from src.navigation.replay import compare_world_prefixes  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def read_trace(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    raw = json.loads((output / "leaderboard.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    expected = summarize_records(
        raw["_checkpoint"]["records"],
        execution_failure=bool(
            summary.get("harness_crashed")
            or summary.get("stop_reason")
            or summary.get("execution_status") == "failed"
        ),
    )
    for key in ("count", "safe_completed", "all_collisions", "episodes"):
        if expected[key] != summary[key]:
            raise ValueError(f"Summary mismatch for {key}")
    source_hashes = output / "source_hashes.json"
    if source_hashes.exists():
        for name, digest in json.loads(source_hashes.read_text()).items():
            if hashlib.sha256((output / "source" / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source snapshot changed: {name}")
    profiles = []
    for path in sorted((output / "episodes").glob("*/trace.jsonl")):
        trace = read_trace(path)
        if not trace:
            continue
        frames = np.array([row["frame"] for row in trace])
        timestamps = np.array([row["timestamp"] for row in trace])
        if not (np.diff(frames) == 1).all():
            raise ValueError("Trace contains repeated or skipped frames")
        if not np.allclose(np.diff(timestamps), 1 / 20, atol=1e-5):
            raise ValueError("Trace timestamp spacing differs from20Hz")
        controls = np.array([row["control"] for row in trace])
        if not np.isfinite(controls).all():
            raise ValueError("Non-finite recorded controls")
        profile = json.loads(path.with_name("profile.json").read_text())
        if profile["frames"] != len(trace):
            raise ValueError("Profile and trace frame counts differ")
        with av.open(str(path.with_name("three_cameras.mp4"))) as container:
            stream = container.streams.video[0]
            if stream.codec_context.name != "h264" or stream.codec_context.format.name != "yuv420p":
                raise ValueError("Unexpected showcase video codec")
            count = sum(1 for _ in container.decode(stream))
        if count != (len(trace) + 1) // 2:
            raise ValueError("Video/trace frame counts differ")
        profile.update(video_frames=count, episode=str(path.parent))
        profiles.append(profile)
    return {
        "output": str(output),
        "audit_passed": True,
        "summary": summary,
        "profiles": profiles,
        "source_snapshot_present": source_hashes.exists(),
        "seed": manifest["seed"],
        "route_sha256": manifest["route"]["route_sha256"],
        "limitation": "This is development infrastructure evidence, not a trained-camera "
        "result or a calibrated encounter reliability estimate.",
    }


def compare_prefixes(first, second, limit=200):
    manifests = [json.loads((Path(p) / "manifest.json").read_text()) for p in (first, second)]
    for key in ("seed", "route", "configuration"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Cannot compare different {key}")
    traces = [
        read_trace(next((Path(p) / "episodes").glob("*/trace.jsonl"))) for p in (first, second)
    ]
    n = min(limit, *(len(t) for t in traces))
    if n < limit:
        raise ValueError(f"Need {limit} frames in both runs; only {n} available")
    poses = [np.array([row["ego"] for row in t[:n]]) for t in traces]
    controls = [np.array([row["control"] for row in t[:n]]) for t in traces]
    position_error = np.linalg.norm(poses[0][:, :3] - poses[1][:, :3], axis=1)
    yaw_error = np.abs((poses[0][:, 3] - poses[1][:, 3] + 180) % 360 - 180)
    control_error = np.abs(controls[0] - controls[1])
    result = {
        "frames_compared": n,
        "max_position_difference_m": float(position_error.max()),
        "max_yaw_difference_deg": float(yaw_error.max()),
        "max_control_difference": float(control_error.max()),
        "prefix_within_tolerance": bool(
            position_error.max() <= 0.1 and yaw_error.max() <= 0.5 and control_error.max() <= 0.01
        ),
        "tolerances": {"position_m": 0.1, "yaw_deg": 0.5, "control": 0.01},
        "limitation": "Same ego prefix is necessary but not sufficient for reactive "
        "counterfactual replay; surrounding actor states need auditing too.",
    }
    world_paths = [
        next((Path(p) / "episodes").glob("*/world_trace.jsonl"), None) for p in (first, second)
    ]
    if all(world_paths):
        worlds = [read_trace(p) for p in world_paths]
        if any(len(w) < n for w in worlds):
            raise ValueError("World traces shorter than requested prefix")
        for world, trace in zip(worlds, traces):
            if [r["frame"] for r in world[:n]] != [r["frame"] for r in trace[:n]]:
                raise ValueError("World/ego trace frame mismatch")
        result["world_state"] = compare_world_prefixes(worlds[0][:n], worlds[1][:n])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare")
    parser.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="Recompute derived counts; preserve initial summary and raw records",
    )
    args = parser.parse_args()
    if args.rebuild_summary:
        path = Path(args.out) / "summary.json"
        backup = path.with_name("summary.before_rebuild.json")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        summary = json.loads(path.read_text())
        raw = json.loads(path.with_name("leaderboard.json").read_text())
        manifest = json.loads(path.with_name("manifest.json").read_text())
        records = raw["_checkpoint"]["records"]
        status = execution_status(
            records,
            summary.get("harness_crashed"),
            summary.get("stop_reason"),
            manifest["max_steps"],
        )
        summary.update(
            summarize_records(
                records,
                execution_failure=bool(
                    summary.get("harness_crashed")
                    or summary.get("stop_reason")
                    or status == "failed"
                ),
            )
        )
        summary["parser_version"] = 3
        summary["execution_status"] = status
        atomic_json(path, summary)
        progress_path = path.with_name("progress.json")
        progress = json.loads(progress_path.read_text())
        progress["summary"] = summary
        progress["status"] = status
        atomic_json(progress_path, progress)
    result = audit(args.out)
    if args.compare:
        result["repeatability"] = compare_prefixes(args.out, args.compare)
    atomic_json(Path(args.out) / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
