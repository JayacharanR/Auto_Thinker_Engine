#!/usr/bin/env python3
"""Evaluate frozen spatial candidates on the unchanged six-case engineering pilot."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_navigation_student_pilot import assess  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402

ORIGINAL = ROOT / "outputs/navigation_m2_stop_go_pilot_20261016/protocol.json"


def candidate(fit):
    fit = Path(fit).resolve()
    if json.loads((fit / "progress.json").read_text())["status"] != "complete":
        raise ValueError("Complete the registered fit before freezing its driving candidate")
    review = json.loads((fit / "review.json").read_text())
    checkpoint = fit / "best_policy.pt"
    if digest(checkpoint) != review["policy_sha256"]:
        raise ValueError("Selected model identity changed")
    return dict(
        name=fit.name,
        checkpoint=str(checkpoint),
        sha256=digest(checkpoint),
        selected_step=review["selected_step"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out = args.out.resolve()
    candidates = [candidate(fit) for fit in args.fit]
    if args.out.exists():
        raise FileExistsError("Keep every prior pilot attempt; use a fresh output")
    args.out.mkdir(parents=True)
    protocol = json.loads(ORIGINAL.read_text())
    protocol.update(
        created_at=time.strftime("%Y-%m-%d %H:%M"),
        original_protocol_sha256=digest(ORIGINAL),
        candidates=[c["name"] for c in candidates],
        scope=(
            "Same six-case engineering diagnostic; internal development, "
            "no heldout_dev/Town05 inference or new training data"
        ),
    )
    atomic_json(args.out / "protocol.json", protocol)
    atomic_json(
        args.out / "request.json", dict(candidates=candidates, runner_sha256=digest(__file__))
    )
    shutil.copyfile(__file__, args.out / "runner_source.py")
    outcomes = []
    with RunLock(ROOT / "outputs/.navigation.lock"):
        for c in candidates:
            if (args.out / "STOP").exists():
                break
            output = args.out / c["name"]
            ledger_path = ROOT / "reports/navigation_execution_20261018.json"
            ledger = json.loads(ledger_path.read_text())
            ledger["active_output"] = str(output.relative_to(ROOT))
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
                c["checkpoint"],
                "--max-wall",
                str(protocol["max_wall_per_candidate"]),
                "--out",
                str(output),
            ]
            with (args.out / f"{c['name']}.log").open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if (args.out / "STOP").exists():
                        output.mkdir(exist_ok=True)
                        (output / "STOP").touch()
                    time.sleep(1)
            outcomes.append(dict(candidate=c["name"], exit_code=child.returncode))
            atomic_json(args.out / "outcomes.json", outcomes)
            if child.returncode:
                break
    status = (
        "complete"
        if len(outcomes) == len(candidates) and all(o["exit_code"] == 0 for o in outcomes)
        else "incomplete"
    )
    if (args.out / "STOP").exists():
        status = "stopped"
    atomic_json(
        args.out / "progress.json",
        dict(status=status, stage="M2 spatial pilot", updated_at=time.time()),
    )
    if status == "complete":
        try:
            atomic_json(args.out / "assessment.json", assess(args.out, decode_videos=True))
        except Exception as error:
            atomic_json(
                args.out / "assessment_error.json",
                dict(error=str(error), raw_trials_preserved=True),
            )
            raise
    print(json.dumps(dict(status=status, candidates=[c["name"] for c in candidates])), flush=True)


if __name__ == "__main__":
    main()
