#!/usr/bin/env python3
"""Execute only the unstarted candidates in the source-frozen corrected pilot."""

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
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def unstarted_candidates(request, results):
    """Reject ambiguity rather than accidentally retry an already settled case."""
    declared = request["candidates"]
    names = [r["candidate"]["name"] for r in results]
    if names != [c["name"] for c in declared[:len(results)]]:
        raise ValueError("Prior results must be an exact prefix of the frozen selection")
    for result, candidate in zip(results, declared):
        covered = result["settled_routes"] == 6 or (
            result.get("planned_wall_budget_verified") is True
            and result["settled_routes"] + result.get("unknown_failure_routes", 0) == 6
        )
        if result["candidate"] != candidate or not covered:
            raise ValueError("Inspect incomplete/changed native attempts before continuing")
    return declared[len(results):]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior", type=Path,
                        help="Inspected terminal continuation; never rerun its partial cases")
    args = parser.parse_args()
    args.original, args.out = args.original.resolve(), args.out.resolve()
    request = read(args.original / "request.json")
    prior = args.prior.resolve() if args.prior else args.original
    settled = read(prior / "results.json")
    for result in settled:
        if result["settled_routes"] != 6:
            native = prior / result["candidate"]["name"] / result["candidate"]["name"]
            summary = read(native / "summary.json")
            manifest = read(native / "manifest.json")
            if summary.get("stop_reason") != "wall_budget" or manifest["max_wall"] != 900:
                raise ValueError("Partial prior attempt is not the unchanged planned900s cap")
            result.update(planned_wall_budget_verified=True,
                          unknown_failure_routes=6 - result["settled_routes"],
                          original_native_summary_sha256=digest(native / "summary.json"))
    remaining = unstarted_candidates(request, settled)
    if not remaining or read(prior / "progress.json")["status"] != "failed":
        raise ValueError("This continuation requires an inspected, terminal original queue")
    if any((root / c["name"]).exists() for c in remaining
           for root in (args.original, prior)):
        raise ValueError("A supposedly unstarted candidate has prior artifacts")
    runtime_hashes = request["inference_source_sha256"]

    def verify_runtime():
        changed = [p for p, h in runtime_hashes.items() if digest(ROOT / p) != h]
        if changed:
            raise ValueError(f"Frozen inference/runtime changed: {changed}")

    verify_runtime()
    args.out.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__), ROOT / "scripts/assess_navigation_student_pilot.py"):
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(args.out / "request.json", dict(
        original_request_sha256=digest(args.original / "request.json"),
        original_results_sha256=digest(prior / "results.json"),
        original_progress_sha256=digest(prior / "progress.json"),
        prior=str(prior),
        original=str(args.original), candidates=remaining,
        settled_native_cases_retained=sum(r["settled_routes"] for r in settled),
        reason="Retain first-response IPC failure and planned900s-censored attempts. "
               "Missing cases stay unknown failures; only entirely unstarted candidates execute.",
        inference_source_sha256=runtime_hashes,
        assessor_sha256=digest(ROOT / "scripts/assess_navigation_student_pilot.py"),
        continuation_sha256=digest(__file__),
        m2_primary=request["m2_primary"], m4_primary=request["m4_primary"],
        candidate_case_once=True, overrides=False,
    ))
    status, active, results = "running", None, list(settled)

    def progress(phase):
        atomic_json(args.out / "progress.json", dict(
            status=status, stage="M2 corrected camera pilots, exact unstarted continuation",
            encounters_done=len(results) * 6, encounters_total=30,
            native_cases_observed=sum(r["settled_routes"] for r in results),
            unknown_cases=sum(6 - r["settled_routes"] for r in results),
            active_output=str(active) if active else None, phase=phase, updated_at=time.time(),
        ))

    try:
        with (
            RunLock(ROOT / "outputs/.navigation_corrected_pilot_continuation.lock"),
            BudgetWatch(args.out),
        ):
            for candidate in remaining:
                if (args.out / "STOP").exists():
                    raise InterruptedError("Continuation STOP")
                verify_runtime()
                if digest(candidate["checkpoint"]) != candidate["sha256"]:
                    raise ValueError("Frozen candidate changed")
                active = args.out / candidate["name"]
                progress("Same frozen six cases/seed/runtime; no settled case is retried")
                with (args.out / (candidate["name"] + ".log")).open("x") as log:
                    child = subprocess.Popen([
                        sys.executable, "-u", "scripts/run_navigation_spatial_pilot.py",
                        "--fit", str(Path(candidate["checkpoint"]).parent),
                        "--out", str(active),
                    ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        if (args.out / "STOP").exists():
                            active.mkdir(parents=True, exist_ok=True)
                            (active / "STOP").touch()
                        progress("Unassisted native driving; budget watch active")
                        time.sleep(2)
                native = active / candidate["name"]
                leaderboard = native / "leaderboard.json"
                rows = read(leaderboard)["_checkpoint"]["records"] if leaderboard.exists() else []
                contacts = [
                    recorder_contacts(p) for p in sorted((native / "recordings").glob("*.log"))
                ]
                assessment = active / "assessment.json"
                native_summary = read(native / "summary.json") if leaderboard.exists() else {}
                budget_end = native_summary.get("stop_reason") == "wall_budget"
                results.append(dict(
                    candidate=candidate, exit_code=child.returncode,
                    settled_routes=len(rows), planned_routes=6,
                    missing_routes_retained_as_unknown_failures=6 - len(rows),
                    planned_wall_budget_verified=budget_end,
                    unknown_failure_routes=6 - len(rows),
                    native_records=rows, recorders=contacts,
                    unfiltered_contact_starts=sum(r["ego_contact_starts"] for r in contacts),
                    all_recorders_complete=bool(contacts) and all(r["complete"] for r in contacts),
                    contact_count_is_lower_bound=any(not r["complete"] for r in contacts),
                    independent_assessment=str(assessment) if assessment.exists() else None,
                    primary_m2=candidate["sha256"] == request["m2_primary"]["sha256"],
                    primary_m4=candidate["sha256"] == request["m4_primary"]["policy_sha256"],
                ))
                atomic_json(args.out / "results.json", results)
                if child.returncode or (len(rows) != 6 and not budget_end):
                    raise RuntimeError("Incomplete native/audit execution; preserve and inspect")
            status = "complete"
    except InterruptedError as error:
        status = "stopped"
        atomic_json(args.out / "error.json", dict(error=str(error)))
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error)))
        raise
    finally:
        progress("Finished; unsafe outcomes retained without promotion")
    print(json.dumps(dict(status=status, candidates=len(results))), flush=True)


if __name__ == "__main__":
    main()
