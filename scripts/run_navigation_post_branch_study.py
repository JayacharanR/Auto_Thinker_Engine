#!/usr/bin/env python3
"""Audit prospective branches, finish matched M4pilots and a no-color diagnostic."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_navigation_spatial_pilot import candidate  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    study = ROOT / "outputs/navigation_repaired_study_v4_20261020/fits"
    fits = [study / f"{name}_f100_seed42" for name in ("cnn", "resnet", "dino", "carla")]
    fits.append(study / "comma_f100_seed42_no_color")
    declared = [candidate(fit) for fit in fits]
    for fit, encoder in zip(fits, ("cnn_scratch", "imagenet_resnet18", "dinov2_small",
                                   "jepa_carla", "jepa_comma2k19")):
        registered = read(fit / "request.json")
        if (registered["encoder"] != encoder or registered["seed"] != 42
                or registered["fraction"] != 1.0
                or registered["color_enabled"] != (fit != fits[-1])):
            raise ValueError("Use only the original registered matched full-label selections")
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "scripts/audit_navigation_fixed_branches.py",
               ROOT / "scripts/audit_navigation_branches.py",
               ROOT / "scripts/run_navigation_spatial_pilot.py",
               ROOT / "scripts/assess_navigation_student_pilot.py",
               ROOT / "reports/navigation_no_color_native_protocol_20261020.md",
               ROOT / "reports/navigation_m4_matched_native_protocol_20261020.md"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    runtime = [*sorted((ROOT / "src/navigation").glob("*.py")),
               ROOT / "scripts/run_navigation.py", ROOT / "configs/navigation.yaml",
               ROOT / "patches/navigation_garage.patch", ROOT / "configs/navigation_vendor.json"]
    runtime_hashes = {str(p.relative_to(ROOT)): digest(p) for p in runtime}
    for path in sources + runtime:
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(args.out / "request.json", dict(
        after=str(args.after.resolve()), candidates=declared, source_hashes=hashes,
        inference_source_sha256=runtime_hashes,
        reuse_completed_comma_primary=(
            "outputs/navigation_corrected_pilot_continuation_v2_20261020/comma_f100_seed42"
        ),
        original_primary_request_sha256=digest(
            ROOT / "outputs/navigation_corrected_camera_pilots_20261020/request.json"),
        replaces_original_primary=False, posthoc_engineering_ablation=True,
        student_M3_outcomes_used_for_selection=False, cases_per_candidate=6,
        maximum_native_wall_seconds_per_candidate=900, new_candidates=5,
        case_once=True, no_training_updates=True, keep_unknown_failures=True,
        reason="Four remaining matched M4heads are required by the original selection protocol; "
               "the earlier cold-start/fact heads do not substitute for them. "
               "Then a registered component control already trained on corrected data. "
               "Its seed42 offline score.34311 versus matched color-on.35141; "
               "test whether the handcrafted color branch hurts closed-loop behavior. "
               "No statistical/safety or primary replacement claim from one-seed control.",
    ))
    status, active, done, observed, unknown = "running", None, 0, 0, 0

    def progress(phase):
        atomic_json(args.out / "progress.json", dict(
            status=status, stage="M4 remaining matched pilots and no-color diagnostic", phase=phase,
            encounters_done=done, encounters_total=30,
            native_cases_observed=observed, unknown_cases=unknown,
            active_output=str(active) if active else None, updated_at=time.time(),
        ))

    def run(command, output):
        nonlocal active, done
        active = output
        with (args.out / (output.name + ".log")).open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                if (args.out / "STOP").exists() and output.exists():
                    (output / "STOP").touch()
                progress("Serial evidence audit/native pilot; all outcomes retained")
                time.sleep(2)
        if child.returncode:
            raise RuntimeError(f"{output.name} infrastructure/audit failed; preserve and inspect")

    try:
        with BudgetWatch(args.out):
            while True:
                if (args.out / "STOP").exists():
                    raise InterruptedError("Post-branch study STOP")
                path = args.after / "progress.json"
                if path.exists() and read(path)["status"] != "running":
                    if read(path)["status"] != "complete":
                        raise RuntimeError("Native branch pilot failed; inspect before continuing")
                    break
                progress("Waiting for M3, protected paired fit and prospective action collection")
                time.sleep(3)
            if any(digest(ROOT / p) != h for p, h in hashes.items()):
                raise ValueError("Queued audit/pilot source changed")
            if any(digest(ROOT / p) != h for p, h in runtime_hashes.items()):
                raise ValueError("Frozen inference/physics changed before matched pilots")
            if any(digest(c["checkpoint"]) != c["sha256"] for c in declared):
                raise ValueError("Frozen matched/no-color candidate changed")
            audit = args.out / "fixed_branch_audit"
            run([sys.executable, "scripts/audit_navigation_fixed_branches.py", "--root",
                 str(args.after), "--out", str(audit)], audit)
            measured = read(audit / "assessment.json")
            if measured["errors"]:
                raise RuntimeError("Prospective action evidence failed; inspect retained errors")
            ledger_path = ROOT / "reports/navigation_execution_20261018.json"
            ledger = read(ledger_path)
            ledger["active_output"] = str(args.out.relative_to(ROOT))
            atomic_json(ledger_path, ledger)
            results = []
            for fit, c in zip(fits, declared):
                if (args.out / "STOP").exists():
                    raise InterruptedError("Matched-pilot queue STOP")
                if digest(c["checkpoint"]) != c["sha256"]:
                    raise ValueError("Frozen matched candidate changed")
                if any(digest(ROOT / p) != h for p, h in runtime_hashes.items()):
                    raise ValueError("Source-frozen matched comparison changed between candidates")
                output = args.out / c["name"]
                run([sys.executable, "-u", "scripts/run_navigation_spatial_pilot.py", "--fit",
                     str(fit), "--out", str(output)], output)
                native = output / c["name"]
                contacts = [recorder_contacts(p)
                            for p in sorted((native / "recordings").glob("*.log"))]
                rows = read(native / "leaderboard.json")["_checkpoint"]["records"]
                results.append(dict(
                    candidate=c, native_records=rows, planned_cases=6,
                    native_cases_observed=len(rows), missing_unknown_failures=6 - len(rows),
                    recorders=contacts,
                    unfiltered_contact_starts=sum(r["ego_contact_starts"] for r in contacts),
                    lower_bound=any(not r["complete"] for r in contacts),
                    native_summary=read(native / "summary.json"),
                    independent_assessment=str(output / "assessment.json"),
                    no_automatic_promotion=True, original_primary_unchanged=True,
                    component_diagnostic_only=c["name"].endswith("no_color"),
                ))
                atomic_json(args.out / "assessment.json", dict(
                    results=results, planned_new_cases=30, original_comma_primary_reused=True,
                    no_new_primary_selected=True, no_automatic_promotion=True,
                ))
                done += 6
                observed += len(rows)
                unknown += 6 - len(rows)
            status = "complete"
    except InterruptedError as error:
        status = "stopped"
        atomic_json(args.out / "error.json", dict(error=str(error)))
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error)))
        raise
    finally:
        progress("Finished; all native failures/unknowns retained, no automatic promotion")


if __name__ == "__main__":
    main()
