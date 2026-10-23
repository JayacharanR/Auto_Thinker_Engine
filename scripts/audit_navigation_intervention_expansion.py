#!/usr/bin/env python3
"""Independently audit all63frozen cases after their exact collection owner exits."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_interventions import audit_case, lines  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.replay import compare_world_prefixes  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def owner_live(pid, root):
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        argv = process.cmdline()
        if (
            process.uids().real != os.getuid()
            or not any(
                a.endswith("scripts/collect_navigation_intervention_expansion.py") for a in argv
            )
            or not any(not a.startswith("-") and Path(a).resolve() == root for a in argv)
        ):
            raise ValueError("Saved collection PIDnow belongs to an unrelated process")
        if abs(process.create_time() - (root / "request.json").stat().st_mtime) > 30:
            raise ValueError("Collection PIDcreation time mismatch")
        return True
    except psutil.NoSuchProcess:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    request = read(args.root / "request.json")
    paths = [
        Path(__file__),
        ROOT / "scripts/audit_navigation_interventions.py",
        ROOT / "scripts/audit_navigation_dense_labels.py",
        ROOT / "scripts/audit_navigation_fixed_branches.py",
        ROOT / "src/navigation/recorder_actions.py",
        ROOT / "src/navigation/recorder_collisions.py",
        ROOT / "src/navigation/replay.py",
        ROOT / "src/navigation/dataset.py",
    ]
    for p in paths:
        dst = args.out / "source" / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    atomic_json(
        args.out / "request.json",
        dict(
            collection=str(args.root),
            collection_request_sha256=digest(args.root / "request.json"),
            exact_owner_pid=args.owner_pid,
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in paths},
            maximum_wait_seconds=6000,
            maximum_audit_seconds=900,
            read_only_no_model_or_native_run=True,
        ),
    )
    status, done, started = "running", 0, time.monotonic()

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Expansion independent action audit",
                frames=done,
                target_frames=63,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 900:
            raise InterruptedError("Audit STOPor900sverification cap")

    try:
        with BudgetWatch(args.out):
            progress("Waiting for exact verified collector PIDto exit; no duplicate native owner")
            while owner_live(args.owner_pid, args.root):
                if (args.out / "STOP").exists() or time.monotonic() - started > 6000:
                    raise InterruptedError("Owned audit STOPorwait cap")
                progress("Verified collector remains live; audit waits for closed native files")
                time.sleep(3)
            started = time.monotonic()
            if read(args.root / "progress.json")["status"] not in ("complete", "failed", "stopped"):
                raise ValueError(
                    "Collector handle ended with stale running metadata; inspect before auditing"
                )
            with RunLock(ROOT / "outputs/.navigation.lock"):
                if any(
                    digest(args.root / "source" / p) != sha
                    for p, sha in request["source_hashes"].items()
                ):
                    raise ValueError("Collector source archive changed")
                for name in (
                    "candidate_controls.py",
                    "trajectory_candidates.py",
                    "frozen_ego.py",
                    "controller.py",
                ):
                    key = "src/navigation/" + name
                    if digest(ROOT / key) != request["source_hashes"][key]:
                        raise ValueError("Proposal reproduction equations changed")
                geometry = read(args.root / "source" / request["geometry_file"])
                original_plan = Path(request["original_data_plan"])
                if digest(original_plan) != request["original_data_plan_sha256"]:
                    raise ValueError("Original source/calibration plan changed")
                data_plan = read(original_plan)
                from src.navigation.label_collection import validate_label_guard

                for context in geometry["contexts"]:
                    validate_label_guard(
                        original_plan,
                        data_plan["catalog"],
                        context["town"],
                        str(context["route_id"]),
                    )
                contexts = {c["context_id"]: c for c in geometry["contexts"]}
                records, errors, raw_recordings, comparisons, prefixes = [], [], [], [], {}
                for case in request["cases"]:
                    check()
                    root = args.root / "cases" / case["case_id"]
                    episode = root / "episode"
                    native = episode / "native.log"
                    raw = None
                    decode_error = None
                    if native.exists():
                        try:
                            raw = recorder_contacts(native)
                        except Exception as error:
                            decode_error = str(error)
                    raw_recordings.append(
                        dict(
                            case=case,
                            physical_output=str(episode),
                            native_exists=native.exists(),
                            complete=bool(raw and raw["complete"]),
                            observed_contact_starts=raw["ego_contact_starts"] if raw else None,
                            native_sha256=digest(native) if native.exists() else None,
                            decode_error=decode_error,
                            process_result=read(root / "worker_result.json")
                            if (root / "worker_result.json").exists()
                            else None,
                        )
                    )
                    try:
                        canonical = {
                            k: case[k] for k in ("case_id", "context_id", "branch", "repeat")
                        }
                        result = audit_case(
                            episode,
                            contexts[case["context_id"]],
                            canonical,
                            args.root / "source" / request["ego_descriptor"],
                            Path(request["original_data_plan"]),
                        )
                        state = lines(episode / "world_trace.jsonl")
                        key = case["context_id"]
                        if key in prefixes:
                            comparisons.append(
                                dict(
                                    case_id=case["case_id"],
                                    **compare_world_prefixes(prefixes[key], state[:80]),
                                )
                            )
                        else:
                            prefixes[key] = state[:80]
                        records.append(
                            result | dict(physical_output=str(episode), declared_case=case)
                        )
                    except Exception as error:
                        errors.append(
                            dict(case=case, physical_output=str(episode), error=str(error))
                        )
                    done += 1
                    progress(
                        "Closed pixels/native controls/plans/first events/both actor geographies"
                    )
                result = dict(
                    planned=63,
                    independently_verified=len(records),
                    rejected_or_unknown=len(errors),
                    records=records,
                    errors=errors,
                    all_attempt_recordings=raw_recordings,
                    comparisons=comparisons,
                    all_observable_comparisons_pass=bool(comparisons)
                    and all(c["observable_prefix_within_tolerance"] for c in comparisons),
                    all_observable_comparisons_exact=all(
                        all(v == 0 for v in c["maximum_differences"].values()) for c in comparisons
                    ),
                    observed_unfiltered_contact_starts=sum(
                        r["observed_contact_starts"] or 0 for r in raw_recordings
                    ),
                    unknown_native_recordings=sum(not r["complete"] for r in raw_recordings),
                    verified_by_role={
                        role: sum(r["source_role"] == role for r in records)
                        for role in ("train", "calibration")
                    },
                    native_shutdown_errors_not_hidden=True,
                    no_fit_or_driving_promotion=True,
                    elapsed_audit_seconds=time.monotonic() - started,
                )
                atomic_json(args.out / "assessment.json", result)
                status = "failed" if errors else "complete"
                progress(
                    "All63attempts audited; retain every rejection and observable-replay result"
                )
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
