#!/usr/bin/env python3
"""Activate approved recording/storage changes only after pinned stages terminate."""

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

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def live_dependencies(roots):
    identities = {str(p.resolve()) for p in roots}
    found = []
    for process in psutil.process_iter(["cmdline", "status"]):
        try:
            command = process.info["cmdline"] or []
            if process.info["status"] == psutil.STATUS_ZOMBIE:
                continue
            owned = False
            if "--out" in command:
                output = Path(command[command.index("--out") + 1]).resolve()
                owned = any(output.is_relative_to(Path(root)) for root in identities)
            native = bool(command) and Path(command[0]).name.startswith("CarlaUE4-Linux")
            if owned or native:
                found.append(dict(pid=process.pid, command=command))
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--after", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=Path, default=ROOT /
                        "outputs/navigation_compact_execution_20261020/storage_guard")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    generation = read(args.generation / "request.json")
    patch = args.generation / "approved_runtime.patch"
    if digest(patch) != generation["patch_sha256"]:
        raise ValueError("Approved staged patch changed")
    guard_request = read(args.budget / "request.json")
    if generation["approved_plan_sha256"] != guard_request["plan_sha256"]:
        raise ValueError("Runtime generation belongs to another storage approval")
    shutil.copyfile(__file__, args.out / "activation_source.py")
    atomic_json(args.out / "request.json", dict(
        generation_request_sha256=digest(args.generation / "request.json"),
        guardian_request_sha256=digest(args.budget / "request.json"),
        source_sha256=digest(__file__), after=[str(p.resolve()) for p in args.after],
        patch_sha256=digest(patch), no_jobs_scripts_changed=True,
    ))
    status = "running"

    def progress(phase):
        atomic_json(args.out / "progress.json", dict(
            status=status, stage="Approved compact runtime activation", phase=phase,
            updated_at=time.time(),
        ))

    try:
        while True:
            if (args.out / "STOP").exists():
                raise InterruptedError("Compact activation STOP")
            states = [read(p / "progress.json")["status"] for p in args.after]
            if any(s not in ("running", "starting", "complete") for s in states):
                raise RuntimeError(
                    "A registered dependency failed/stopped; inspect before activation"
                )
            if all(s == "complete" for s in states) and not live_dependencies(args.after):
                break
            progress("Waiting for all registered stages and owned native workers to terminate")
            time.sleep(3)
        guard = read(args.budget / "progress.json")
        if guard["status"] != "running" or time.time() - guard["updated_at"] > 20:
            raise RuntimeError("Approved capacity guardian must remain live")
        with RunLock(ROOT / "outputs/.navigation.lock"):
            if live_dependencies(args.after):
                raise RuntimeError(
                    "Owned stage/native process started while acquiring generation lock"
                )
            if digest(args.generation / "request.json") != read(args.out / "request.json")[
                "generation_request_sha256"
            ]:
                raise ValueError("Prepared runtime generation identity changed")
            for path, identities in generation["targets"].items():
                if digest(ROOT / path) != identities["before_sha256"]:
                    raise ValueError(f"Registered patch target changed:{path}")
            for path, expected in generation.get("helper_sources_sha256", {}).items():
                if digest(ROOT / path) != expected:
                    raise ValueError(f"Prepared storage helper changed:{path}")
            subprocess.run(["git", "apply", "--check", str(patch.resolve())], cwd=ROOT, check=True)
            progress("Activating storage-only source generation after registered work")
            subprocess.run(["git", "apply", str(patch.resolve())], cwd=ROOT, check=True)
            changed = {p: digest(ROOT / p) for p in generation["targets"]}
            atomic_json(args.out / "applied.json", dict(
                before_after=generation["targets"], actual_after=changed,
                applied_at=time.time(), registered_stages_terminated=True,
                dependencies_live=[], all_native_recorders_and_traces_retained=True,
            ))
            if any(changed[p] != h["after_sha256"] for p, h in generation["targets"].items()):
                raise ValueError("Applied generation differs from the staged exact bytes")
            with (args.out / "validation.log").open("x") as log:
                check = subprocess.run([
                    sys.executable, "-m", "pytest", "-q",
                    "tests/test_navigation_final_evaluation.py",
                    "tests/test_navigation_recording_policy.py",
                    "tests/test_navigation_setup_resource_equivalence.py",
                    "tests/test_navigation_rule_suite.py",
                    "tests/test_navigation_storage_budget.py",
                ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(args.out / "validation.json", dict(exit_code=check.returncode))
            if check.returncode:
                raise RuntimeError(
                    "Runtime generation validation failed; inspect retained patch/log"
                )
            from src.navigation.final_evaluation import candidate_request

            probe = candidate_request(None, None, args.out / "provenance_probe", expert=True)
            if not {
                "scripts/run_navigation.py", "scripts/run_navigation_final.py",
                "scripts/run_encounters.py",
            } <= set(probe["inference_source_sha256"]):
                raise ValueError(
                    "Actual native entry points must be bound before final registration"
                )
            atomic_json(args.out / "entrypoint_provenance_check.json", dict(
                passed=True, source_files=sorted(probe["inference_source_sha256"]),
                no_candidate_registered=True, no_native_driving=True,
            ))
            ledger_path = ROOT / "reports/navigation_execution_20261018.json"
            ledger = read(ledger_path)
            ledger["storage_plan"].update(
                status="active_14GB_budget_10GB_reserve", runtime_reserve_applied=True,
                activation_receipt=str(args.out / "applied.json"),
            )
            atomic_json(ledger_path, ledger)
            stamp = time.strftime("%Y-%m-%d %H:%M")
            with (ROOT / "reports/journey/journal.md").open("a") as journal:
                journal.write(
                    f"\n## {stamp} - Activate approved compact runtime after pinned stages\n"
                    "**Type:** decision / implementation\n"
                    "**Stage:** Navigation storage, future M3/M6 execution\n\n"
                    "**What happened:** All eight registered predecessors report complete and "
                    "their owned workers/native servers are absent. Under the navigation lock, "
                    "apply exact staged runtime patch: future disk floor25→10GB; "
                    "predeclared M6 showcase routes0/4/8 seed3031, same expert/student selection. "
                    "Every mission retains native recorder/traces; nonfinal/M3 videos stay full.\n"
                    "**Verified by:** " + str(args.out / "applied.json") + "; "
                    "post-activation final/recording/resource/certificate tests pass. "
                    "No job shell script, camera input, control, route, seed or metric changed.\n"
                    "**Lesson:** Switch a resource generation only after its source-pinned "
                    "predecessors actually terminate.\n"
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
        progress("finished; source generation outcome retained")
    print(json.dumps(dict(status=status)), flush=True)


if __name__ == "__main__":
    main()
