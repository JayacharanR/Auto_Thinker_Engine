#!/usr/bin/env python3
"""Recover validation of an exact already-applied patch; never apply it again."""

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
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    activation = read(args.activation / "request.json")
    applied = read(args.activation / "applied.json")
    generation = read(args.generation / "request.json")
    dependencies = [Path(p) for p in activation["after"]]
    tests = ["test_navigation_final_evaluation.py", "test_navigation_recording_policy.py",
             "test_navigation_setup_resource_equivalence.py", "test_navigation_rule_suite.py",
             "test_navigation_storage_budget.py"]
    shutil.copyfile(__file__, args.out / "recovery_source.py")
    atomic_json(args.out / "request.json", dict(
        activation_request_sha256=digest(args.activation / "request.json"),
        applied_receipt_sha256=digest(args.activation / "applied.json"),
        generation_request_sha256=digest(args.generation / "request.json"),
        original_failed_validation_sha256=digest(args.activation / "validation.log"),
        source_sha256=digest(__file__), no_reapplication=True,
        tests_sha256={name: digest(ROOT / "tests" / name) for name in tests},
        reason="Test raced STOP-first publication with later metadata receipt; "
               "test now joins monitor before checking receipt. Production helper unchanged.",
    ))
    status = "running"

    def progress(phase):
        atomic_json(args.out / "progress.json", dict(
            status=status, stage="Approved compact runtime activation", phase=phase,
            updated_at=time.time(),
        ))

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            progress("Verify exact applied bytes and remaining validation; no patch reapplication")
            if live_dependencies(dependencies) or any(
                read(p / "progress.json")["status"] != "complete" for p in dependencies
            ):
                raise RuntimeError("Pinned predecessor/native workers must be terminal")
            if digest(args.generation / "request.json") != activation["generation_request_sha256"]:
                raise ValueError("Applied generation request changed")
            if any(digest(ROOT / p) != h["after_sha256"]
                   or applied["actual_after"][p] != h["after_sha256"]
                   for p, h in generation["targets"].items()):
                raise ValueError("Actual runtime does not match exact approved applied bytes")
            if any(digest(ROOT / p) != h
                   for p, h in generation["helper_sources_sha256"].items()):
                raise ValueError("Approved production storage helper changed")
            with (args.out / "validation.log").open("x") as log:
                subprocess.run([sys.executable, "-m", "pytest", "-q",
                                *[str(ROOT / "tests" / t) for t in tests]],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            from src.navigation.final_evaluation import candidate_request

            probe = candidate_request(None, None, args.out / "unused_probe", expert=True)
            if not {"scripts/run_navigation.py", "scripts/run_navigation_final.py",
                    "scripts/run_encounters.py"} <= set(probe["inference_source_sha256"]):
                raise ValueError("Final native entrypoint provenance missing")
            atomic_json(args.out / "assessment.json", dict(
                exact_approved_after_bytes_verified=True, production_helpers_unchanged=True,
                no_reapplication=True, all_required_tests_passed=True,
                final_entrypoints_bound=True, no_candidate_registered=True,
                original_applied_receipt=str(args.activation / "applied.json"),
            ))
            ledger_path = ROOT / "reports/navigation_execution_20261018.json"
            ledger = read(ledger_path)
            ledger["storage_plan"].update(
                status="active_14GB_budget_10GB_reserve", runtime_reserve_applied=True,
                current_runtime_reserve_bytes=10000000000,
                activation_receipt=str(args.activation / "applied.json"),
                validation_recovery=str(args.out / "assessment.json"),
            )
            atomic_json(ledger_path, ledger)
            status = "complete"
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error)))
        raise
    finally:
        progress("Validation complete; original failure and source generation retained")
    print(json.dumps(dict(status=status, patch_reapplied=False)), flush=True)


if __name__ == "__main__":
    main()
