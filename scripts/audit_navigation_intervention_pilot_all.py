#!/usr/bin/env python3
"""Audit all ten original attempts across retained failed owner generations."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    roots = [
        ROOT / "outputs" / name
        for name in (
            "navigation_intervention_pilot_20261020",
            "navigation_intervention_continuation_20261020",
            "navigation_intervention_finish_20261020",
            "navigation_intervention_last2_20261020",
        )
    ]
    original = roots[0]
    request = read(original / "request.json")
    if any(
        read(p / "progress.json")["status"] != "failed"
        or not (p / "external_client_abort.json").exists()
        for p in roots
    ):
        raise ValueError(
            "Every source owner must remain failed, with its retained client-abort receipt"
        )
    for p in roots:
        r = read(p / "request.json")
        if any(digest(p / "source" / name) != sha for name, sha in r["source_hashes"].items()):
            raise ValueError("A generating source archive changed")
    for name in (
        "candidate_controls.py",
        "trajectory_candidates.py",
        "frozen_ego.py",
        "controller.py",
    ):
        key = "src/navigation/" + name
        if digest(ROOT / key) != request["source_hashes"][key]:
            raise ValueError("Restore original proposal equations for independent reproduction")
    paths = [
        Path(__file__),
        ROOT / "scripts/audit_navigation_interventions.py",
        ROOT / "scripts/audit_navigation_dense_labels.py",
        ROOT / "scripts/audit_navigation_fixed_branches.py",
        ROOT / "src/navigation/recorder_actions.py",
        ROOT / "src/navigation/recorder_collisions.py",
    ]
    for source in paths:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            original_request_sha256=digest(original / "request.json"),
            source_owner_roots=[str(p) for p in roots],
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in paths},
            scope="Read-only closed-file verification, no additional native/model execution",
            maximum_wall_seconds=300,
            maximum_artifact_bytes=20_000_000,
        ),
    )
    started = time.monotonic()
    status = "running"
    done = 0

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Intervention independent audit",
                frames=done,
                target_frames=10,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 300:
            raise InterruptedError("Owned audit STOPor300sCPUverification cap")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            geometry = read(original / "source" / request["geometry_file"])
            contexts = {c["context_id"]: c for c in geometry["contexts"]}
            records, errors, raw_recordings, comparisons, prefixes, repeats = [], [], [], [], {}, {}
            for index, case in enumerate(request["cases"]):
                check()
                owner = roots[0 if index < 2 else 1 if index < 6 else 2 if index < 8 else 3]
                episode = owner / "episodes" / case["case_id"]
                if sum((r / "episodes" / case["case_id"]).exists() for r in roots) != 1:
                    raise ValueError(
                        "Missing or duplicated attempted case; never select a favorable retry"
                    )
                native = episode / "native.log"
                raw = recorder_contacts(native) if native.exists() else None
                raw_recordings.append(
                    dict(
                        case=case,
                        output=str(episode),
                        closed_native=bool(raw and raw["complete"]),
                        observed_contact_starts=raw["ego_contact_starts"] if raw else None,
                        native_recording_sha256=raw["sha256"] if raw else None,
                    )
                )
                try:
                    result = audit_case(
                        episode,
                        contexts[case["context_id"]],
                        case,
                        original / "source" / request["ego_descriptor"],
                        original / "source" / request["original_data_plan"],
                    )
                    state = lines(episode / "world_trace.jsonl")
                    key = case["context_id"]
                    if key in prefixes:
                        comparisons.append(
                            dict(
                                case_id=case["case_id"],
                                kind="same_context_prefix",
                                **compare_world_prefixes(prefixes[key], state[:80]),
                            )
                        )
                    else:
                        prefixes[key] = state[:80]
                    key = case["context_id"], case["branch"]
                    if key in repeats:
                        comparisons.append(
                            dict(
                                case_id=case["case_id"],
                                kind="same_option_repeat",
                                **compare_world_prefixes(repeats[key], state),
                            )
                        )
                    else:
                        repeats[key] = state
                    records.append(result | dict(physical_output=str(episode)))
                except Exception as error:
                    errors.append(dict(case=case, physical_output=str(episode), error=str(error)))
                done += 1
                progress(
                    "Verify pixels/native actions/all15plans/actual ego+other geography"
                )
            result = dict(
                original_planned=10,
                verified=len(records),
                rejected_or_unknown=len(errors),
                records=records,
                errors=errors,
                all_attempt_recordings=raw_recordings,
                comparisons=comparisons,
                all_observable_comparisons_pass=bool(comparisons)
                and all(c["observable_prefix_within_tolerance"] for c in comparisons),
                observed_unfiltered_contact_starts=sum(
                    r["observed_contact_starts"] or 0 for r in raw_recordings
                ),
                unknown_native_recordings=sum(not r["closed_native"] for r in raw_recordings),
                all_original_owner_failures_retained=True,
                no_new_model_or_native_execution=True,
                no_safety_or_fitting_promotion=True,
                elapsed_seconds=time.monotonic() - started,
            )
            atomic_json(args.out / "assessment.json", result)
            status = "failed" if errors else "complete"
            progress(
                "Original10audit retains all rejections; verified source list is separate evidence"
            )
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
