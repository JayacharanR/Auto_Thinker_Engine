#!/usr/bin/env python3
"""Freeze corrected-study winners before unchanged six-case camera diagnostics."""

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


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("Retain every corrected camera pilot attempt")
    args.out.mkdir(parents=True)
    atomic_json(
        args.out / "queued_request.json",
        dict(
            after=str(args.after.resolve()),
            study=str(args.study.resolve()),
            facts=str(args.facts.resolve()),
            runner_sha256=digest(__file__),
            primary="Lowest4candidate full-label selector, fixedseed42",
            m4_primary="Lowestfive-encoder mean100%score over3seeds, fixedseed42",
            planned_candidates=5,
            cases_per_candidate=6,
            overrides=False,
            keep_every_outcome=True,
            not_heldout_or_final_evaluation=True,
        ),
    )
    shutil.copyfile(__file__, args.out / "queue_source.py")
    active, status, results = None, "running", []

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M2 corrected camera pilots",
                encounters_done=sum(r["settled_routes"] for r in results),
                encounters_total=30,
                active_output=str(active) if active else None,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    try:
        for dependency in (args.after, args.study, args.facts):
            while True:
                progress("Waiting for registered training/reference stages; no duplicate simulator")
                if (args.out / "STOP").exists():
                    raise InterruptedError("Camera-pilot queue STOP")
                path = dependency / "progress.json"
                if path.exists():
                    state = read(path)["status"]
                    if state != "running":
                        if state != "complete":
                            raise RuntimeError(f"Dependency {dependency} ended {state}")
                        break
                time.sleep(3)
        for path in (
            "scripts/run_navigation.py",
            "scripts/run_navigation_spatial_pilot.py",
            "scripts/summarize_navigation_repaired_study.py",
        ):
            target = args.out / "source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, target)
        subprocess.run(
            [
                sys.executable,
                "scripts/summarize_navigation_repaired_study.py",
                "--study",
                str(args.study),
                "--out",
                str(args.out / "m4_analysis.json"),
            ],
            cwd=ROOT,
            check=True,
        )
        m4 = read(args.out / "m4_analysis.json")["primary_candidate"]
        fits = [
            args.facts / f"{encoder}_{arm}"
            for encoder in ("imagenet_resnet18", "cnn_scratch")
            for arm in ("plain", "facts")
        ]
        m2_primary = min(fits, key=lambda f: (read(f / "review.json")["best_score"], f.name))
        declared = [candidate(f) for f in fits] + [candidate(Path(m4["fit"]))]
        runtime_sources = [
            *sorted((ROOT / "src/navigation").glob("*.py")),
            ROOT / "scripts/run_navigation.py",
            ROOT / "scripts/run_navigation_spatial_pilot.py",
            ROOT / "configs/navigation.yaml",
            ROOT / "patches/navigation_garage.patch",
        ]
        runtime_hashes = {str(p.relative_to(ROOT)): digest(p) for p in runtime_sources}
        for source in runtime_sources:
            target = args.out / "inference_source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        atomic_json(
            args.out / "request.json",
            dict(
                candidates=declared,
                m2_primary=candidate(m2_primary),
                m4_primary=m4,
                selection_frozen_before_any_driving=True,
                fact_study_request_sha256=digest(args.facts / "request.json"),
                corrected_study_request_sha256=digest(args.study / "request.json"),
                original_pilot_protocol_sha256=digest(
                    ROOT / "outputs/navigation_m2_stop_go_pilot_20261016/protocol.json"
                ),
                candidate_case_once=True,
                inference_source_sha256=runtime_hashes,
                overrides=False,
            ),
        )
        for fit, c in zip(fits + [Path(m4["fit"])], declared):
            if (args.out / "STOP").exists():
                raise InterruptedError("Camera-pilot queue STOP")
            if digest(c["checkpoint"]) != c["sha256"]:
                raise ValueError("Frozen corrected candidate changed")
            if any(digest(ROOT / p) != h for p, h in runtime_hashes.items()):
                raise ValueError("Frozen inference/audit source changed between candidate pilots")
            active = args.out / c["name"]
            progress("Unchanged6cases; candidate frozen before driving")
            ledger_path = ROOT / "reports/navigation_execution_20261018.json"
            ledger = read(ledger_path)
            ledger["active_output"] = str(args.out.relative_to(ROOT))
            atomic_json(ledger_path, ledger)
            with (args.out / (c["name"] + ".log")).open("x") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "scripts/run_navigation_spatial_pilot.py",
                        "--fit",
                        str(fit),
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
                    progress("Unassisted camera pilot; full native contacts retained")
                    time.sleep(2)
            native = active / c["name"]
            leaderboard = native / "leaderboard.json"
            rows = read(leaderboard)["_checkpoint"]["records"] if leaderboard.exists() else []
            contacts = [recorder_contacts(p) for p in sorted((native / "recordings").glob("*.log"))]
            assessment = active / "assessment.json"
            results.append(
                dict(
                    candidate=c,
                    exit_code=child.returncode,
                    settled_routes=len(rows),
                    planned_routes=6,
                    missing_routes_retained_as_unknown_failures=6 - len(rows),
                    native_records=rows,
                    recorders=contacts,
                    unfiltered_contact_starts=sum(r["ego_contact_starts"] for r in contacts),
                    all_recorders_complete=bool(contacts) and all(r["complete"] for r in contacts),
                    contact_count_is_lower_bound=any(not r["complete"] for r in contacts),
                    independent_assessment=str(assessment) if assessment.exists() else None,
                    primary_m2=c["name"] == m2_primary.name,
                    primary_m4=c["sha256"] == m4["policy_sha256"],
                )
            )
            atomic_json(args.out / "results.json", results)
            if child.returncode:
                raise RuntimeError(
                    "Pilot infrastructure/audit failed; raw contacts preserved, "
                    "inspect before continuing"
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
        progress("finished")
    print(
        json.dumps(
            dict(
                status=status,
                candidates=len(results),
                unfiltered_contact_starts=sum(r["unfiltered_contact_starts"] for r in results),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
