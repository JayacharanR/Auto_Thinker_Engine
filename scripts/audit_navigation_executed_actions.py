#!/usr/bin/env python3
"""Verify preserved20Hz recorder controls against every10Hz shard acknowledgment."""

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import shard_records  # noqa:E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve every factual-action audit")
    args.out.mkdir(parents=True)
    sources = [Path(__file__), ROOT / "src/navigation/recorder_actions.py"]
    for source in sources:
        shutil.copyfile(source, args.out / source.name)
    manifest = ROOT / "outputs/navigation_spatial_data_20261019/manifest.json"
    data = json.loads(manifest.read_text())
    results, errors = [], []
    started = time.monotonic()
    for i, e in enumerate(data["episodes"]):
        raw = Path(e["data"])
        root = raw.parents[2]
        inside = list((root / "recordings").glob("*.log"))
        recording = inside[0] if len(inside) == 1 else root.with_suffix(".log")
        try:
            world = [
                json.loads(line)
                for line in (raw.parent / "world_trace.jsonl").read_text().splitlines()
            ]
            hero = next(a["id"] for a in world[0]["actors"] if a["role"] == "hero")
            native = vehicle_frames(recording, hero)
            aligned, receipt = align_world_frames(native, world)
            acknowledgments, differences = 0, []
            for shard in sorted(raw.glob("shard_*.zip")):
                with zipfile.ZipFile(shard) as archive:
                    for record in shard_records(shard):
                        target = json.loads(archive.read(record["prefix"] + "/targets.json"))
                        frame = target["facts"]["actuator_feedback_frame"]
                        actual = aligned.get(frame)
                        if actual is None:
                            raise ValueError("Recorded acknowledged frame unavailable")
                        error = float(
                            np.max(np.abs(np.array(actual["control"]) - target["executed_control"]))
                        )
                        if error > 1e-5:
                            differences.append(
                                dict(input_frame=record["frame"], ack_frame=frame, error=error)
                            )
                        acknowledgments += 1
            results.append(
                dict(
                    id=e["id"],
                    role=e["role"],
                    domain=e["domain"],
                    recorder_path=str(recording),
                    recorder_sha256=digest(recording),
                    world_trace_sha256=digest(raw.parent / "world_trace.jsonl"),
                    alignment=receipt,
                    acknowledged_samples=acknowledgments,
                    control_differences=differences,
                )
            )
        except Exception as error:
            errors.append(dict(id=e["id"], error=str(error)))
        atomic_json(
            args.out / "progress.json",
            dict(
                status="running",
                stage="M5 factual control audit",
                frames=i + 1,
                target_frames=len(data["episodes"]),
                updated_at=time.time(),
            ),
        )
    result = dict(
        status="complete" if not errors else "complete_with_failures",
        episodes=results,
        errors=errors,
        data_manifest_sha256=digest(manifest),
        source_hashes={p.name: digest(args.out / p.name) for p in sources},
        elapsed_s=time.monotonic() - started,
    )
    atomic_json(args.out / "assessment.json", result)
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            stage="M5 factual control audit",
            frames=len(data["episodes"]),
            target_frames=len(data["episodes"]),
            updated_at=time.time(),
        ),
    )
    print(
        json.dumps(
            dict(
                episodes=len(results),
                errors=errors,
                control_differences=sum(len(r["control_differences"]) for r in results),
                seconds=result["elapsed_s"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
