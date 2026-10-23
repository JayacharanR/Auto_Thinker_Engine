#!/usr/bin/env python3
"""Run only the unstarted camera arm; preserve failed expert/native attempt once."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.apply_navigation_compact_generation import live_dependencies  # noqa:E402
from scripts.run_navigation_train_integration_pair import actual_junction_audit  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.original, args.out = args.original.resolve(), args.out.resolve()
    original = read(args.original / "request.json")
    if (
        read(args.original / "progress.json")["status"] != "failed"
        or not (args.original / "expert/error.json").exists()
        or (args.original / "student").exists()
        or live_dependencies([args.original])
    ):
        raise ValueError(
            "Only a terminal failed expert with a genuinely unstarted camera arm may continue"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(
        args.out / "request.json",
        dict(
            original=str(args.original),
            original_request_sha256=digest(args.original / "request.json"),
            original_failed_expert_error_sha256=digest(args.original / "expert/error.json"),
            source_sha256=digest(__file__),
            planned_pair_denominator=2,
            retry_failed_expert=False,
            only_original_unstarted_student=True,
            original_seed_route_checkpoint_budget_preserved=True,
            no_selection_or_training=True,
        ),
    )
    shutil.copyfile(__file__, args.out / "continuation_source.py")
    active = args.out / "student"
    status = "running"

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="TRAIN expert/camera long integration",
                encounters_done=1 if status == "running" else 2,
                encounters_total=2,
                active_output=str(active),
                phase=phase,
                updated_at=time.time(),
            ),
        )

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            if any(
                digest(ROOT / p) != h for p, h in original["future_inference_source_sha256"].items()
            ):
                raise ValueError("Original camera evaluation source changed")
            if (
                digest(original["route"]) != original["route_sha256"]
                or digest(original["checkpoint"]) != original["checkpoint_sha256"]
            ):
                raise ValueError("Original registered route or selected camera checkpoint changed")
            progress("Original expert crash retained; only unstarted camera arm, same900s cap")
            command = [
                sys.executable,
                "-u",
                "scripts/run_navigation.py",
                "student",
                "--route",
                original["route"],
                "--seed",
                str(original["seed"]),
                "--checkpoint",
                original["checkpoint"],
                "--max-wall",
                str(original["maximum_wall_seconds_per_arm"]),
                "--out",
                str(active),
            ]
            with (args.out / "student.log").open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if (args.out / "STOP").exists() and active.exists():
                        (active / "STOP").touch()
                    progress("Camera-only TRAINdrive; original expert failure remains in pair")
                    time.sleep(2)
            outcomes = []
            for arm, output in (("expert", args.original / "expert"), ("student", active)):
                contacts, errors = [], []
                for recording in sorted((output / "recordings").glob("*.log")):
                    try:
                        contacts.append(recorder_contacts(recording))
                    except Exception as error:
                        errors.append(dict(path=str(recording), error=str(error)))
                outcomes.append(
                    dict(
                        arm=arm,
                        output=str(output),
                        summary=read(output / "summary.json")
                        if (output / "summary.json").exists()
                        else None,
                        recorders=contacts,
                        recorder_errors=errors,
                        unfiltered_contact_starts=sum(r["ego_contact_starts"] for r in contacts),
                        contact_lower_bound=bool(errors)
                        or not contacts
                        or any(not r["complete"] for r in contacts),
                        geography=actual_junction_audit(
                            output, read(Path(original["route"]).with_suffix(".json"))
                        ),
                        infrastructure_failed=(output / "error.json").exists(),
                    )
                )
            atomic_json(
                args.out / "assessment.json",
                dict(
                    outcomes=outcomes,
                    planned_pair_denominator=2,
                    failed_expert_not_retried=True,
                    unstarted_camera_only=True,
                    driving_qualification=False,
                    matched_valid_expert_comparison_available=False,
                    no_promotion=True,
                ),
            )
            if child.returncode:
                raise RuntimeError("Camera infrastructure failure retained; no native retry")
            status = "stopped" if (args.out / "STOP").exists() else "complete"
            progress("Original pair now terminal; expert failure retained, no model promotion")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
