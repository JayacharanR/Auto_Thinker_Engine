#!/usr/bin/env python3
"""Drive only protected selected residuals on the unchanged six-case TRAIN pilot."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402

OLD = ROOT / "outputs/navigation_m2_stop_go_pilot_20261016/protocol.json"


def selected_candidates():
    result = []
    for arm in ("rgb", "null"):
        root = ROOT / f"outputs/navigation_m2_signal_residual_{arm}_20261018"
        review = json.loads((root / "review.json").read_text())
        if json.loads((root / "progress.json").read_text())["status"] != "complete":
            raise ValueError("Both predeclared fits must complete before driving")
        if review["selected_step"] == 0:
            continue
        checkpoint = root / "best_policy.pt"
        if digest(checkpoint) != review["policy_sha256"]:
            raise ValueError("Selected policy identity changed")
        result.append(
            dict(
                name=f"{arm}_residual_selected",
                checkpoint=str(checkpoint),
                sha256=digest(checkpoint),
                selected_step=review["selected_step"],
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/navigation_m2_signal_residual_pilot_20261018"
    )
    args = parser.parse_args()
    candidates = selected_candidates()
    if args.out.exists():
        raise FileExistsError("Fresh pilot only; preserve all prior attempts")
    args.out.mkdir(parents=True)
    old = json.loads(OLD.read_text())
    protocol = dict(
        old,
        created_at=time.strftime("%Y-%m-%d %H:%M"),
        candidates=[c["name"] for c in candidates],
        original_protocol_sha256=digest(OLD),
        note=(
            "Same six routes/seed/criteria; historical component development locations, "
            "no untouched-test claim. No new imitation data."
        ),
    )
    atomic_json(args.out / "protocol.json", protocol)
    atomic_json(
        args.out / "request.json", dict(candidates=candidates, source_sha256=digest(__file__))
    )
    shutil.copyfile(__file__, args.out / "runner_source.py")
    ledger_path = ROOT / "reports/navigation_execution_20261018.json"
    status = "complete"
    with RunLock(ROOT / "outputs/.navigation.lock"):
        for candidate in candidates:
            if (args.out / "STOP").exists():
                status = "stopped"
                break
            out = args.out / candidate["name"]
            ledger = json.loads(ledger_path.read_text())
            ledger["active_output"] = str(out.relative_to(ROOT))
            atomic_json(ledger_path, ledger)
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/run_navigation.py"),
                "student",
                "--route",
                protocol["route"],
                "--routes-subset",
                ",".join(str(e["route_id"]) for e in protocol["encounters"]),
                "--seed",
                str(protocol["seed"]),
                "--seed-per-route",
                "--checkpoint",
                candidate["checkpoint"],
                "--max-wall",
                str(protocol["max_wall_per_candidate"]),
                "--out",
                str(out),
            ]
            with (args.out / f"{candidate['name']}_console.log").open("a") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if (args.out / "STOP").exists():
                        out.mkdir(exist_ok=True)
                        (out / "STOP").touch()
                    time.sleep(1)
            if child.returncode:
                status = "incomplete"
                break
    atomic_json(
        args.out / "progress.json",
        dict(
            status=status,
            stage="M2 residual TRAIN pilot",
            candidates=len(candidates),
            note="Only protected non-parent selections driven; not a promotion score.",
            updated_at=time.time(),
        ),
    )
    if status == "complete" and candidates:
        from scripts.assess_navigation_student_pilot import assess

        try:
            atomic_json(args.out / "assessment.json", assess(args.out))
        except Exception as error:
            atomic_json(
                args.out / "assessment_error.json",
                dict(error=str(error), all_raw_attempts_preserved=True),
            )
            raise
    print(json.dumps(dict(status=status, candidates=[c["name"] for c in candidates])), flush=True)


if __name__ == "__main__":
    main()
