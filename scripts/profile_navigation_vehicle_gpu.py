#!/usr/bin/env python3
"""Serialize a full fixed165-frame GPU perception profile after factual encoding."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("Preserve prior GPU profile attempts")
    args.out.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "scripts/assess_navigation_cv_perception.py",
        ROOT / "src/navigation/road_user_fpn.py",
        ROOT / "src/navigation/vehicle_perception.py",
        ROOT / "src/navigation/camera_ground.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            source_hashes=hashes,
            after=str(args.after.resolve()),
            fixed_frames=165,
            original_diagnostic="outputs/navigation_cv_fpn_perception_retry_20261019",
            float_precision="FP32, no autocast",
            warmups=3,
            scope=(
                "Detector+ground projection+fusion; excludes loading/renderer/controller. "
                "No planner qualification."
            ),
            full_fixed_frame_comparison=True,
        ),
    )
    status = "running"

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M5 GPU road-user profile",
                frames=0,
                target_frames=165,
                phase=phase,
                active_output=str(args.out / "evaluation"),
                updated_at=time.time(),
            ),
        )

    try:
        progress("Waiting for factual encoder cache; no concurrent simulator/trainer")
        while True:
            if (args.out / "STOP").exists():
                raise InterruptedError("Road-user GPU profile STOP")
            p = args.after / "progress.json"
            if p.exists():
                prior = json.loads(p.read_text())["status"]
                if prior != "running":
                    if prior != "complete":
                        raise RuntimeError(f"Factual cache ended {prior}; inspect before profiling")
                    break
            time.sleep(3)
        if any(digest(ROOT / p) != h for p, h in hashes.items()):
            raise ValueError("Queued GPU profile source changed")
        with RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 25)
            with (args.out / "console.log").open("x") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "scripts/assess_navigation_cv_perception.py",
                        "--fpn",
                        "--device",
                        "cuda",
                        "--root",
                        "outputs/navigation_rule_setup_pilot_ack_20261019",
                        "--out",
                        str(args.out / "evaluation"),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                while child.poll() is None:
                    if (args.out / "STOP").exists():
                        (args.out / "evaluation").mkdir(parents=True, exist_ok=True)
                        (args.out / "evaluation/STOP").touch()
                    progress("Profiling fixed RGB observations; no model updates")
                    time.sleep(2)
            if child.returncode:
                raise RuntimeError("GPU perception profile failed; retain attempt")
            result = json.loads((args.out / "evaluation/assessment.json").read_text())
            atomic_json(
                args.out / "assessment.json", {k: v for k, v in result.items() if k != "rows"}
            )
            status = "complete"
    except InterruptedError:
        status = "stopped"
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error)))
        raise
    finally:
        progress("finished")
    print(json.dumps(dict(status=status)), flush=True)


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation_vehicle_gpu_profile.lock"):
        main()
