#!/usr/bin/env python3
"""Continue exactly unstarted dense-label sources without retrying failed geometry."""

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
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def unstarted(original):
    plan = read(original / "plan.json")
    rows = read(original / "assessment.json")["results"]
    expected = plan["episodes"]
    if (
        read(original / "progress.json")["status"] != "failed"
        or not rows
        or any(row["item"] != entry for row, entry in zip(rows, expected))
    ):
        raise ValueError("Only the exact terminal original attempted prefix can continue")
    if len(rows) >= len(expected):
        raise ValueError("No original source remains unstarted")
    for item in expected[len(rows) :]:
        path = original / "episodes" / f"{item['town']}_{item['route_id']}_{item['role']}"
        if path.exists():
            raise ValueError("Do not rerun a started or failed source")
    return plan, rows, expected[len(rows) :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.original = args.original.resolve()
    args.out = args.out.resolve()
    plan, results, remaining = unstarted(args.original)
    request = read(args.original / "request.json")
    if live_dependencies([args.original]):
        raise RuntimeError("Original owner/native worker still live")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "episodes").mkdir()
    # Preparation preceded execution: this conservative deadline includes those
    # earlier test/wait seconds; never gives the corpus a new30minute allowance.
    deadline = (args.original / "request.json").stat().st_mtime + plan["maximum_wall_seconds"]
    atomic_json(
        args.out / "request.json",
        dict(
            original=str(args.original),
            original_request_sha256=digest(args.original / "request.json"),
            plan_sha256=digest(args.original / "plan.json"),
            original_results_sha256=digest(args.original / "assessment.json"),
            remaining_route_ids=[e["route_id"] for e in remaining],
            failed_sources_never_retried=True,
            original_planned_denominator=len(plan["episodes"]),
            original_conservative_deadline=deadline,
            remaining_seconds_at_registration=max(0, deadline - time.time()),
            original_artifact_cap_shared_bytes=plan["maximum_artifact_bytes"],
            source_sha256=digest(__file__),
        ),
    )
    shutil.copyfile(__file__, args.out / "continuation_source.py")
    status, active = "running", None

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="TRAIN dense-label collection",
                encounters_done=len(results),
                encounters_total=len(plan["episodes"]),
                active_output=str(active) if active else None,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def stopped():
        return (
            (args.out / "STOP").exists()
            or time.time() > deadline
            or sum(
                p.stat().st_size
                for root in (args.original, args.out)
                for p in root.rglob("*")
                if p.is_file()
            )
            > 3e9
        )

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            for item in remaining:
                if stopped():
                    raise InterruptedError("Original corpus STOP/deadline/shared3GBcap")
                if any(digest(ROOT / p) != h for p, h in request["source_hashes"].items()):
                    raise ValueError(
                        "Original source generation changed; do not resume this recipe"
                    )
                active = args.out / "episodes" / f"{item['town']}_{item['route_id']}_{item['role']}"
                command = [
                    sys.executable,
                    "-u",
                    "scripts/run_navigation.py",
                    "expert",
                    "--route",
                    str(Path(plan["catalog"]) / f"{item['town']}.xml"),
                    "--routes-subset",
                    str(item["route_id"]),
                    "--collection-catalog",
                    plan["catalog"],
                    "--collection-guard",
                    str(args.original / "plan.json"),
                    "--training-dense-labels",
                    "--collection-max-samples",
                    str(item["rgb_timestamp_cap"]),
                    "--seed",
                    str(item["seed"]),
                    "--max-wall",
                    str(item["maximum_wall_seconds"]),
                    "--out",
                    str(active),
                ]
                with (args.out / f"{item['route_id']}.log").open("x") as log:
                    child = subprocess.Popen(
                        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                    )
                    while child.poll() is None:
                        if stopped() and active.exists():
                            (active / "STOP").touch()
                        progress(
                            "Original failed geometry preserved; collecting unstarted sources only"
                        )
                        time.sleep(2)
                profiles = [
                    dict(path=str(p), sha256=digest(p), **read(p))
                    for p in active.glob("episodes/*/label_profile.json")
                ]
                contacts = []
                errors = []
                for path in sorted((active / "recordings").glob("*.log")):
                    try:
                        contacts.append(recorder_contacts(path))
                    except Exception as error:
                        errors.append(dict(path=str(path), error=str(error)))
                row = dict(
                    item=item,
                    output=str(active),
                    exit_code=child.returncode,
                    label_profiles=profiles,
                    label_timestamps=sum(p["label_timestamps"] for p in profiles),
                    contacts=contacts,
                    contact_errors=errors,
                    unfiltered_contact_starts=sum(c["ego_contact_starts"] for c in contacts),
                    contact_count_lower_bound=bool(errors)
                    or not contacts
                    or any(not c["complete"] for c in contacts),
                    native_summary=read(active / "summary.json")
                    if (active / "summary.json").exists()
                    else None,
                )
                results.append(row)
                atomic_json(
                    args.out / "assessment.json",
                    dict(
                        results=results,
                        planned_episodes=len(plan["episodes"]),
                        original_failed_sources_retained=True,
                        dataset_not_yet_independently_audited=True,
                        training_started=False,
                        no_driving_claim=True,
                    ),
                )
                if child.returncode or row["label_timestamps"] < 2:
                    raise RuntimeError(
                        "New collection failure retained; no retry of any attempted source"
                    )
            status = "complete"
            progress("Original13source attempts settled; failed source remains failed")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(
            args.out / "error.json",
            dict(error=str(error), at=time.time(), remaining_unstarted_unknown=True),
        )
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
