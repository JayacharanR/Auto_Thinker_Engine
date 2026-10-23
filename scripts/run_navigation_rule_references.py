#!/usr/bin/env python3
"""Complete all physical rule-reference arms, retaining and reusing exact prototypes."""

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
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--after-pilots", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-unstarted", action="store_true")
    args = parser.parse_args()
    for name in ("cases", "reuse", "after_pilots", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out.exists() and not args.resume_unstarted:
        raise FileExistsError("Preserve prior physical-reference attempts")
    cases = json.loads((args.cases / "cases.json").read_text())["cases"]
    reuse = {row["case"]: row for row in json.loads((args.reuse / "outcomes.json").read_text())}
    sources = [
        "scripts/run_navigation.py",
        "src/navigation/rule_case_runtime.py",
        "src/navigation/rule_case_contract.py",
        "src/navigation/rule_prefix.py",
        "src/navigation/rule_suite.py",
        "src/navigation/harness.py",
        "src/navigation/pdm_agent.py",
        "src/navigation/sensors.py",
        "src/navigation/controller.py",
        "src/navigation/video.py",
        "src/navigation/native_opportunities.py",
        "scripts/assess_navigation_rule_prototypes.py",
        "scripts/run_navigation_rule_references.py",
    ]
    source_hashes = {path: digest(ROOT / path) for path in sources}
    if args.resume_unstarted:
        prior = json.loads((args.out / "request.json").read_text())
        progress_before = json.loads((args.out / "progress.json").read_text())
        if progress_before["encounters_done"] or progress_before["active_output"]:
            raise ValueError("This resume path is only for a queue interrupted before any arm")
        for path, expected in prior["source_hashes"].items():
            if (
                path != "scripts/run_navigation_rule_references.py"
                and digest(ROOT / path) != expected
            ):
                raise ValueError("Physical reference inference/audit source changed")
        atomic_json(args.out / "interrupted_wait.json", progress_before)
        atomic_json(
            args.out / "resume_request.json",
            dict(
                original_request_sha256=digest(args.out / "request.json"),
                source_hashes=source_hashes,
                only_queued_before_first_arm=True,
            ),
        )
        shutil.copyfile(__file__, args.out / "resume_source.py")
    args.out.mkdir(parents=True, exist_ok=args.resume_unstarted)
    for path in sources:
        target = args.out / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    atomic_json(
        args.out / ("active_resume_request.json" if args.resume_unstarted else "request.json"),
        dict(
            cases=cases,
            seed=61001,
            reused_root=str(args.reuse),
            after_pilots=str(args.after_pilots),
            source_hashes=source_hashes,
            no_student_rule_trials=True,
            canonical_benchmark_changed=False,
            selection="Every24 declared geometry arm; reference outcome never filters setup",
        ),
    )
    records, active, status = [], None, "running"

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M3 complete physical reference",
                encounters_done=len(records),
                encounters_total=len(cases),
                phase=phase,
                active_output=str(active) if active else None,
                updated_at=time.time(),
            ),
        )

    try:
        progress("Waiting for four registered encoder pilots; no duplicate simulator")
        while True:
            if (args.out / "STOP").exists():
                raise InterruptedError("Reference queue STOP")
            path = args.after_pilots / "progress.json"
            if path.exists():
                terminal = json.loads(path.read_text())["status"]
                if terminal == "running":
                    time.sleep(3)
                    continue
                if terminal != "complete":
                    raise RuntimeError(
                        f"Encoder pilot queue ended {terminal}; inspect before reference start"
                    )
                break
            time.sleep(3)
        for row in cases:
            if (args.out / "STOP").exists():
                raise InterruptedError("Reference queue STOP")
            if any(digest(ROOT / p) != h for p, h in source_hashes.items()):
                raise ValueError("Registered reference source changed while queued")
            case = Path(row["path"])
            if digest(case) != row["sha256"]:
                raise ValueError("Registered case changed")
            name = case.parent.name
            if name in reuse:
                previous = reuse[name]
                active = Path(previous["output"])
                recorded = json.loads((active / "manifest.json").read_text())["rule_test"]
                if previous["exit_code"] or recorded["case_sha256"] != row["sha256"]:
                    raise ValueError("Reused prototype identity/terminal state changed")
                retained = True
            else:
                active = args.out / name
                retained = False
                progress("Native expert reference; no student evaluation")
                ledger_path = ROOT / "reports/navigation_execution_20261018.json"
                ledger = json.loads(ledger_path.read_text())
                ledger["active_output"] = str(args.out.relative_to(ROOT))
                ledger["milestones"]["M3"]["status"] = "native_reference_running"
                atomic_json(ledger_path, ledger)
                command = [
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
                ]
                with RunLock(ROOT / "outputs/.navigation.lock"):
                    with (args.out / (name + ".log")).open("x") as log:
                        child = subprocess.Popen(
                            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                        )
                        while child.poll() is None:
                            if (args.out / "STOP").exists():
                                active.mkdir(parents=True, exist_ok=True)
                                (active / "STOP").touch()
                            progress("Native expert reference; no student evaluation")
                            time.sleep(2)
                if child.returncode:
                    raise RuntimeError(
                        f"Reference infrastructure failed:{name},exit{child.returncode}"
                    )
            audited = assess(active)
            records.append(
                dict(case=name, case_sha256=row["sha256"], reused=retained, audit=audited)
            )
            atomic_json(args.out / "records.json", records)
            progress("Audited physical prototype; not yet paired setup certification")
        status = "complete"
    except InterruptedError:
        status = "stopped"
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), active_output=str(active)))
        raise
    finally:
        progress("finished")
    print(json.dumps(dict(status=status, arms=len(records), planned=len(cases))), flush=True)


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation_rule_reference.lock"):
        main()
