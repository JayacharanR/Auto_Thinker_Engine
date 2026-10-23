#!/usr/bin/env python3
"""Run one frozen camera candidate on all72 certified rule interventions."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_navigation import checkpoint_metadata  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.rule_suite import FAMILIES, SEEDS, setup_certificate_valid  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402
from src.navigation.storage_budget import BudgetWatch, require_budget  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def planned_trials(suite):
    manifest = read(suite / "cases.json")
    if (
        digest(suite / "cases.json") != (suite / "cases.sha256").read_text().strip()
        or manifest.get("setup_certified") is not True
        or manifest["seeds"] != list(SEEDS)
        or manifest["student_trials"] != 72
        or len(manifest["cases"]) != 24
    ):
        raise ValueError("A complete frozen certified24-case/72-trial suite is required")
    trials = []
    identities = set()
    for row in manifest["cases"]:
        case = Path(row["path"]).resolve()
        certificate = case.parent / "setup_certificate.json"
        identity = row["family"], row["location"], row["arm"]
        if identity in identities or row["family"] not in FAMILIES or row["arm"] not in (0, 1):
            raise ValueError("Duplicate or unregistered physical rule arm")
        identities.add(identity)
        if (
            not case.is_relative_to(suite.resolve())
            or digest(case) != row["sha256"]
            or digest(certificate) != row["certificate_sha256"]
            or not setup_certificate_valid(read(certificate), case_path=case)
        ):
            raise ValueError("Case/certificate/source binding failed before any student trial")
        for seed in SEEDS:
            trials.append(dict(
                name=f"{case.parent.name}_s{seed}",
                family=row["family"], location=row["location"], arm=row["arm"], seed=seed,
                case=str(case), case_sha256=row["sha256"],
                certificate_sha256=row["certificate_sha256"],
            ))
    for family in FAMILIES:
        locations = {location for f, location, _ in identities if f == family}
        if len(locations) != 2 or any(
            (family, location, arm) not in identities for location in locations for arm in (0, 1)
        ):
            raise ValueError("Both arms at two locations required for every family")
    return trials


def live_worker(output):
    for process in psutil.process_iter(["cmdline", "status"]):
        try:
            command = process.info["cmdline"] or []
            if (
                process.info["status"] != psutil.STATUS_ZOMBIE
                and "--out" in command
                and str(output) == command[command.index("--out") + 1]
                and any(arg.endswith("scripts/run_navigation.py") for arg in command)
            ):
                return process.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True,
                        help="Prospective candidate selection artifact; identity retained")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=Path, default=ROOT /
                        "outputs/navigation_compact_execution_20261020/storage_guard")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out, args.suite = args.out.resolve(), args.suite.resolve()
    trials = planned_trials(args.suite)
    inference = [
        *sorted((ROOT / "src/navigation").glob("*.py")),
        ROOT / "scripts/run_navigation.py", ROOT / "patches/navigation_garage.patch",
        ROOT / "configs/navigation_vendor.json",
    ]
    source = inference + [
        Path(__file__), ROOT / "scripts/assess_navigation_rule_student.py",
        ROOT / "scripts/assess_navigation_rule_prototypes.py",
        ROOT / "scripts/audit_navigation_rule_execution.py", ROOT / "configs/navigation.yaml",
    ]
    request = dict(
        schema="navigation-camera-rule-trials-v1", trials=trials,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
        architecture=checkpoint_metadata(args.checkpoint)["architecture"],
        suite=str(args.suite), suite_sha256=digest(args.suite / "cases.json"),
        selection=str(args.selection.resolve()), selection_sha256=digest(args.selection),
        rule_overrides=False, emergency_overrides=False,
        one_attempt_per_case_seed=True, missing_stays_failed_in72=True,
        maximum_wall_per_trial_seconds=300,
        inference_source_sha256={str(p.relative_to(ROOT)): digest(p) for p in inference},
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in source},
    )
    if args.out.exists():
        if not args.resume or read(args.out / "request.json") != request:
            raise ValueError("Only identical-source/candidate/suite resume is allowed")
        if (args.out / "STOP").exists():
            raise ValueError("Retained STOP requires deliberate inspection before resume")
    else:
        if args.resume:
            raise FileNotFoundError("No registered rule run to resume")
        args.out.mkdir(parents=True)
        atomic_json(args.out / "request.json", request)
        for path in source:
            target = args.out / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    outcomes_path = args.out / "outcomes.json"
    outcomes = read(outcomes_path) if outcomes_path.exists() else []
    previous = {row["name"]: row for row in outcomes}
    if len(previous) != len(outcomes):
        raise ValueError("Duplicate prior trials; do not overwrite scientific outcomes")
    active, status = None, "running"

    def progress(phase):
        atomic_json(args.out / "progress.json", dict(
            status=status, stage="M3 override-disabled camera rules",
            encounters_done=sum(r["status"] != "started" for r in outcomes),
            encounters_total=72, active_output=str(active) if active else None,
            phase=phase, updated_at=time.time(),
        ))

    try:
        with BudgetWatch(args.out, args.budget), RunLock(ROOT / "outputs/.navigation.lock"):
            for trial in trials:
                active = args.out / trial["name"]
                if live_worker(active):
                    raise RuntimeError("Owned native worker still alive; never duplicate its trial")
                if trial["name"] in previous:
                    row = previous[trial["name"]]
                    if row["status"] == "started":
                        shutil.copyfile(
                            outcomes_path, args.out / f"interrupted_{time.time_ns()}.json"
                        )
                        row.update(status="interrupted_unknown", exit_code=None,
                                   retained_without_rerun=True)
                        atomic_json(outcomes_path, outcomes)
                    continue
                if active.exists():
                    raise FileExistsError("An unregistered attempt directory must be inspected")
                if (args.out / "STOP").exists():
                    raise InterruptedError("Student rule queue STOP")
                require_budget(args.budget)
                if any(digest(ROOT / p) != h for p, h in request["source_hashes"].items()):
                    raise ValueError("Source-frozen camera/control/scoring code changed")
                if digest(args.checkpoint) != request["checkpoint_sha256"]:
                    raise ValueError("Frozen camera candidate changed")
                row = dict(name=trial["name"], status="started", started_at=time.time())
                outcomes.append(row)
                atomic_json(outcomes_path, outcomes)
                progress("Certified physical intervention; post-handoff overrides disabled")
                command = [sys.executable, "-u", "scripts/run_navigation.py", "student",
                           "--route", str(Path(trial["case"]).parent / "route.xml"),
                           "--rule-case", trial["case"], "--seed", str(trial["seed"]),
                           "--checkpoint", str(args.checkpoint), "--max-wall", "300",
                           "--out", str(active)]
                with (args.out / (trial["name"] + ".log")).open("x") as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                             stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        if (args.out / "STOP").exists():
                            active.mkdir(exist_ok=True)
                            (active / "STOP").touch()
                        progress("Native camera trial; all contacts and proposals retained")
                        time.sleep(2)
                row.update(status="terminal", exit_code=child.returncode, ended_at=time.time())
                atomic_json(outcomes_path, outcomes)
                if child.returncode:
                    raise RuntimeError(
                        "Native infrastructure failed; preserve before further trials"
                    )
            status = "complete"
    except InterruptedError as error:
        status = "stopped"
        atomic_json(args.out / "error.json", dict(error=str(error)))
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error)))
        raise
    finally:
        progress("Execution finished; independent evidence/behavior assessment required")
    audit = args.out / f"actuation_audit_{time.time_ns()}"
    subprocess.run([sys.executable, "scripts/audit_navigation_rule_execution.py", "--root",
                    str(args.out), "--out", str(audit)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/assess_navigation_rule_student.py", "--root",
                    str(args.out), "--actuation", str(audit / "assessment.json"), "--out",
                    str(args.out / f"assessment_{time.time_ns()}")], cwd=ROOT, check=True)
    print(json.dumps(dict(status=status, attempted_trials=len(outcomes), planned_trials=72)),
          flush=True)


if __name__ == "__main__":
    main()
