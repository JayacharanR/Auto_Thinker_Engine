#!/usr/bin/env python3
"""Verify repaired memory geometry after GPU work, reusing exact native cases."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_navigation_rule_prototypes import assess  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.rule_suite import SETUP_RUNTIME_SOURCES  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("Preserve all memory-reference attempts")
    cases_path = ROOT / "outputs/navigation_memory_cases_20261019/cases.json"
    cases = json.loads(cases_path.read_text())["cases"]
    previous = json.loads(
        (ROOT / "outputs/navigation_rule_references_20261019/records.json").read_text()
    )
    reuse = {row["case_sha256"]: row["audit"]["output"] for row in previous}
    sources = [
        *SETUP_RUNTIME_SOURCES,
        "scripts/run_navigation_memory_references.py",
        "scripts/assess_navigation_rule_prototypes.py",
        "scripts/audit_navigation_rule_execution.py",
    ]
    hashes = {p: digest(ROOT / p) for p in sources}
    args.out.mkdir(parents=True)
    for path in sources:
        target = args.out / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    atomic_json(
        args.out / "request.json",
        dict(
            cases=cases,
            source_hashes=hashes,
            cases_manifest_sha256=digest(cases_path),
            after=str(args.after.resolve()),
            no_student_trials=True,
            canonical_benchmark_changed=False,
        ),
    )
    active, records, status = None, [], "running"

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M3 complete physical reference",
                encounters_done=len(records),
                encounters_total=4,
                active_output=str(active) if active else None,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    try:
        while True:
            progress("Waiting for GPU profile; no concurrent simulator/trainer")
            if (args.out / "STOP").exists():
                raise InterruptedError("Memory-reference STOP")
            path = args.after / "progress.json"
            if path.exists():
                prior = json.loads(path.read_text())["status"]
                if prior != "running":
                    if prior != "complete":
                        raise RuntimeError(f"Prior stage ended {prior}; inspect first")
                    break
            time.sleep(3)
        for row in cases:
            if (args.out / "STOP").exists():
                raise InterruptedError("Memory-reference STOP")
            if any(digest(ROOT / p) != h for p, h in hashes.items()):
                raise ValueError("Memory-reference registered source changed")
            case = Path(row["path"])
            if digest(case) != row["sha256"]:
                raise ValueError("Prospective memory geometry changed")
            retained = row["sha256"] in reuse
            active = Path(reuse[row["sha256"]]) if retained else args.out / case.parent.name
            if not retained:
                progress("Native expert setup; no student rule claim")
                with RunLock(ROOT / "outputs/.navigation.lock"):
                    with (args.out / (case.parent.name + ".log")).open("x") as log:
                        child = subprocess.Popen(
                            [
                                sys.executable,
                                "-u",
                                "scripts/run_navigation.py",
                                "expert",
                                "--route",
                                str(case.parent / "route.xml"),
                                "--rule-case",
                                str(case),
                                "--seed",
                                "61001",
                                "--max-wall",
                                "300",
                                "--out",
                                str(active),
                            ],
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        while child.poll() is None:
                            if (args.out / "STOP").exists():
                                active.mkdir(parents=True, exist_ok=True)
                                (active / "STOP").touch()
                            progress("Native expert setup; no student rule claim")
                            time.sleep(2)
                    if child.returncode:
                        raise RuntimeError(f"Native memory setup failed:{case.parent.name}")
            audit = assess(active)
            if audit["binding"]["case_sha256"] != row["sha256"]:
                raise ValueError("Native reference is not the declared physical case")
            records.append(
                dict(case=case.parent.name, case_sha256=row["sha256"], reused=retained, audit=audit)
            )
            atomic_json(args.out / "records.json", records)
        active = args.out / "actuation_audit"
        progress("Independent next-tick native execution audit")
        subprocess.run(
            [
                sys.executable,
                "scripts/audit_navigation_rule_execution.py",
                "--root",
                str(args.out),
                "--out",
                str(active),
            ],
            cwd=ROOT,
            check=True,
        )
        checked = json.loads((active / "assessment.json").read_text())
        if (
            checked["verified_recorders"] != 2
            or checked["errors"]
            or checked["differences"]
            or checked["missing_ack_frames"]
        ):
            raise ValueError("Two new physical recordings need exact native control evidence")
        status = "complete"
    except InterruptedError as error:
        status = "stopped"
        atomic_json(args.out / "error.json", dict(error=str(error)))
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error)))
        raise
    finally:
        progress("finished; physical validity still requires paired certification")


if __name__ == "__main__":
    main()
