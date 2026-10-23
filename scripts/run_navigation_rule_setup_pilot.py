#!/usr/bin/env python3
"""Seven predeclared native M3 setup probes; no student trial or outcome filtering."""

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
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402

CASES = (
    "signal0_empty_light_a0",
    "signal0_empty_light_a1",
    "signal0_green_stopped_lead_a1",
    "signal0_out_of_view_memory_a0",
    "signal0_yellow_distance_a0",
    "stop0_stop_and_release_a0",
    "signal0_green_clearing_cross_a1",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "outputs/navigation_rule_cases_v3_20261019"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--after-study", type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.cases = args.cases.resolve()
    if args.out.exists():
        raise FileExistsError("Preserve every setup-pilot attempt")
    specs = []
    for name in CASES:
        path = args.cases / name / "case.json"
        spec = json.loads(path.read_text())
        if digest(path) != path.with_suffix(".sha256").read_text().strip():
            raise ValueError("Case changed")
        specs.append(
            dict(
                name=name,
                path=str(path),
                case_sha256=digest(path),
                route_sha256=spec["route_sha256"],
            )
        )
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, args.out / "runner_source.py")
    atomic_json(
        args.out / "request.json",
        dict(
            cases=specs,
            seed=61001,
            after_study=str(args.after_study) if args.after_study else None,
            source_sha256=digest(__file__),
            max_wall_per_case=300,
            purpose=(
                'Native setup/reference prototype; no student/model selection; '
                'every failure retained'
            ),
        ),
    )
    records = []
    status = "failed"
    active = None

    def progress(phase="native setup reference"):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M3 native setup pilot",
                encounters_done=len(records),
                encounters_total=len(specs),
                phase=phase,
                active_output=str(active) if active else None,
                updated_at=time.time(),
            ),
        )

    try:
        status = "running"
        progress("Waiting for exact completed M4 matrix")
        if args.after_study:
            while True:
                if (args.out / "STOP").exists():
                    raise InterruptedError("Setup pilot STOP while waiting")
                try:
                    with RunLock(ROOT / "outputs/.navigation_visual_study.lock"):
                        p = json.loads((args.after_study / "progress.json").read_text())
                        if p["status"] != "complete":
                            raise RuntimeError(f"Study ended {p['status']}; do not proceed blindly")
                        break
                except RuntimeError as error:
                    if "Another navigation run holds" not in str(error):
                        raise
                progress("Waiting for active study to release its lock; no duplicate trainer")
                time.sleep(5)
        for row in specs:
            if (args.out / "STOP").exists():
                raise InterruptedError("Setup pilot STOP")
            active = args.out / row["name"]
            progress()
            ledger_path = ROOT / "reports/navigation_execution_20261018.json"
            ledger = json.loads(ledger_path.read_text())
            ledger["active_output"] = str(args.out.relative_to(ROOT))
            ledger["milestones"]["M3"]["status"] = "native_setup_pilot_running"
            atomic_json(ledger_path, ledger)
            command = [
                sys.executable,
                "-u",
                "scripts/run_navigation.py",
                "expert",
                "--route",
                str(Path(row["path"]).parent / "route.xml"),
                "--rule-case",
                row["path"],
                "--seed",
                "61001",
                "--max-wall",
                "300",
                "--out",
                str(active),
            ]
            with RunLock(ROOT / "outputs/.navigation.lock"):
                with (args.out / f"{row['name']}.log").open("x") as log:
                    child = subprocess.Popen(
                        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                    )
                    while child.poll() is None:
                        if (args.out / "STOP").exists():
                            active.mkdir(parents=True, exist_ok=True)
                            (active / "STOP").touch()
                        progress()
                        time.sleep(2)
            records.append(dict(case=row["name"], exit_code=child.returncode, output=str(active)))
            atomic_json(args.out / "outcomes.json", records)
            if child.returncode:
                # Retain remaining planned tests as unstarted; inspect infrastructure
                # before continuing, not a favorable-reference filter.
                raise RuntimeError(
                    f"Native setup execution failed:{row['name']}; inspect retained logs"
                )
        status = "complete"
    except InterruptedError:
        status = "stopped"
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), active_output=str(active)))
        raise
    finally:
        progress("finished")
    print(json.dumps(dict(status=status, probed=len(records), planned=len(specs))), flush=True)


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation_rule_setup_pilot.lock"):
        main()
