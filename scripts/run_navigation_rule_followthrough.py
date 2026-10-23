#!/usr/bin/env python3
"""After approved activation, certify cases and test the prospectively frozen M2primary."""

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
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-activation", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    primary = read(args.selection)["m2_primary"]
    if primary["name"] != "cnn_scratch_facts" or digest(primary["checkpoint"]) != primary["sha256"]:
        raise ValueError("Use the exact M2primary selected before any corrected pilot")
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *[ROOT / f"scripts/{name}.py" for name in (
        "audit_navigation_branches", "certify_navigation_rule_pairs",
        "run_navigation_rule_student", "assess_navigation_rule_student",
        "audit_navigation_rule_execution",
    )]]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for path in sources:
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(args.out / "request.json", dict(
        after_activation=str(args.after_activation.resolve()),
        selection=str(args.selection.resolve()), selection_sha256=digest(args.selection),
        candidate=primary, source_hashes=hashes,
        physical_preflight="outputs/navigation_rule_full_preflight_20261020",
        no_checkpoint_reselection_from_native_outcomes=True,
        planned_student_trials=72, overrides=False, no_final_town05_driving=True,
    ))
    status, active, done = "running", None, 0

    def progress(phase):
        atomic_json(args.out / "progress.json", dict(
            status=status, stage="M3 frozen-primary followthrough", phase=phase,
            encounters_done=done, encounters_total=72,
            active_output=str(active) if active else None, updated_at=time.time(),
        ))

    def child(command, output, phase):
        nonlocal active, done
        if (args.out / "STOP").exists():
            raise InterruptedError("Rule followthrough STOP")
        active = output
        progress(phase)
        with (args.out / (output.name + ".log")).open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if (args.out / "STOP").exists() and output.is_dir():
                    (output / "STOP").touch()
                p = output / "progress.json"
                if p.exists() and output.name == "student":
                    done = read(p).get("encounters_done", 0)
                progress(phase)
                time.sleep(2)
        if process.returncode:
            raise RuntimeError(f"Stage {output.name} failed; preserve evidence and inspect")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation_rule_followthrough.lock"):
            while True:
                if (args.out / "STOP").exists():
                    raise InterruptedError("Rule followthrough STOP while queued")
                previous = args.after_activation / "progress.json"
                if previous.exists():
                    state = read(previous)["status"]
                    if state == "complete":
                        break
                    if state != "running":
                        raise RuntimeError(f"Storage activation ended {state}; no student launched")
                progress("Waiting for pilots/clock repair and approved runtime generation")
                time.sleep(3)
            if any(digest(ROOT / p) != h for p, h in hashes.items()):
                raise ValueError("Queued certificate/student runner source changed")
            if digest(args.selection) != read(args.out / "request.json")["selection_sha256"]:
                raise ValueError("Prospective candidate selection changed")
            ledger_path = ROOT / "reports/navigation_execution_20261018.json"
            ledger = read(ledger_path)
            ledger["active_output"] = str(args.out.relative_to(ROOT))
            atomic_json(ledger_path, ledger)
            child([sys.executable, "scripts/audit_navigation_branches.py", "--root",
                   str(args.branch_root), "--out", str(args.out / "branch_audit")],
                  args.out / "branch_audit", "Independently audit closed TRAINbranch evidence")
            suite = args.out / "certified_suite"
            child([sys.executable, "scripts/certify_navigation_rule_pairs.py",
                   "--references", "outputs/navigation_rule_references_20261019",
                   "--references", "outputs/navigation_memory_references_v2_20261020",
                   "--actuation",
                   "outputs/navigation_rule_actuation_all24_20261020/assessment.json",
                   "--actuation",
                   "outputs/navigation_memory_references_v2_20261020/actuation_audit/assessment.json",
                   "--out", str(suite)], suite, "Certify all24physical rule arms")
            child([sys.executable, "-u", "scripts/run_navigation_rule_student.py",
                   "--suite", str(suite), "--checkpoint", primary["checkpoint"],
                   "--selection", str(args.selection), "--out", str(args.out / "student")],
                  args.out / "student", "All72override-disabled trials of frozen M2primary")
            measured = read(args.out / "student/progress.json")
            if measured["status"] != "complete" or measured["encounters_done"] != 72:
                raise RuntimeError("Student queue stopped/incomplete; never report72executed")
            done, status = 72, "complete"
    except InterruptedError as error:
        status = "stopped"
        atomic_json(args.out / "error.json", dict(error=str(error)))
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error)))
        raise
    finally:
        progress("Execution finished; behavior failures remain failures, no automatic promotion")
    print(json.dumps(dict(status=status, planned_trials=72)), flush=True)


if __name__ == "__main__":
    main()
