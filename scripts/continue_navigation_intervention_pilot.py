#!/usr/bin/env python3
"""Continue exactly eight unstarted pilot cases after the scoped timeout repair."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_interventions import audit, audit_case, lines  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.frozen_ego import FrozenExpertEgo  # noqa:E402
from src.navigation.intervention_collection import collect_case  # noqa:E402
from src.navigation.replay import compare_world_prefixes  # noqa:E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def require_only_timeout_changes(original):
    request = read(original / "request.json")
    allowed = {"src/navigation/native_cleanup.py", "src/navigation/intervention_collection.py"}
    changed = {path for path, sha in request["source_hashes"].items() if digest(ROOT / path) != sha}
    if changed != allowed:
        raise ValueError(f"Only the scoped timeout repair may differ: {changed}")
    collector = (ROOT / "src/navigation/intervention_collection.py").read_text()
    if (
        collector.replace("    client.set_timeout(20.0)\n", "", 1)
        != (original / "source/src/navigation/intervention_collection.py").read_text()
    ):
        raise ValueError("Collector physics, controls or data recipe changed")
    cleanup = (ROOT / "src/navigation/native_cleanup.py").read_text()
    block = """        # CarlaServer establishes20sfor normal work. A short teardown bound is
        # scoped to cleanup and must not leak into the next map reload.
        attempt("restore_normal_client_timeout", lambda: client.set_timeout(20.0))
        receipt["normal_client_timeout_restored"] = not any(
            e["operation"] == "restore_normal_client_timeout" for e in errors
        )
"""
    if (
        cleanup.replace(block, "", 1)
        != (original / "source/src/navigation/native_cleanup.py").read_text()
    ):
        raise ValueError("Cleanup changed beyond restoring the established normal timeout")
    return changed


def combined_audit(original, continuation, out, check):
    out.mkdir(parents=True, exist_ok=False)
    request = read(original / "request.json")
    geometry = read(original / "source" / request["geometry_file"])
    contexts = {c["context_id"]: c for c in geometry["contexts"]}
    records, errors, comparisons, prefixes, repeats = [], [], [], {}, {}
    for index, case in enumerate(request["cases"]):
        check()
        owner = original if index < 2 else continuation
        episode = owner / "episodes" / case["case_id"]
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
        atomic_json(
            out / "progress.json",
            dict(
                status="running",
                stage="Intervention independent audit",
                frames=len(records) + len(errors),
                target_frames=10,
                updated_at=time.time(),
            ),
        )
    result = dict(
        original_planned=10,
        verified=len(records),
        errors=errors,
        records=records,
        comparisons=comparisons,
        all_registered_observable_comparisons_pass=bool(comparisons)
        and all(r["observable_prefix_within_tolerance"] for r in comparisons),
        unfiltered_contact_starts=sum(r["unfiltered_contact_starts"] for r in records),
        original_failed_brake_case_retained=True,
        no_model_fit_or_driving_promotion=True,
    )
    atomic_json(out / "assessment.json", result)
    atomic_json(
        out / "progress.json",
        dict(
            status="failed" if errors else "complete",
            stage="Intervention independent audit",
            frames=10,
            target_frames=10,
            updated_at=time.time(),
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    original = args.original.resolve()
    if (
        read(original / "progress.json")["status"] != "failed"
        or not (original / "external_client_abort.json").exists()
        or not read(original / "servers/server_000/normal_cleanup_retry.json")[
            "normal_shutdown_finished"
        ]
        or (original / "storage_stop.json").exists()
    ):
        raise ValueError(
            "Verified original client abort/normal cleanup only; no storage-stop bypass"
        )
    request = read(original / "request.json")
    initial = read(original / "records.json")
    if len(initial) != 2 or initial[0]["status"] != "captured" or initial[1]["status"] != "failed":
        raise ValueError("Original one captured/one failed prefix must remain unchanged")
    remaining = request["cases"][2:]
    if len(remaining) != 8 or any(
        (original / "episodes" / c["case_id"]).exists() for c in remaining
    ):
        raise ValueError("Only all eight entirely unstarted original cases may continue")
    changed = require_only_timeout_changes(original)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "episodes").mkdir()
    deadline = (original / "request.json").stat().st_mtime + 900
    sources = [ROOT / p for p in request["source_hashes"]] + [Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    continuation = request | dict(
        cases=remaining,
        source_hashes=hashes,
        original=str(original),
        original_request_sha256=digest(original / "request.json"),
        only_changed_original_sources=sorted(changed),
        original_absolute_deadline=deadline,
        no_new_wall_or_storage_allowance=True,
        first_original_brake_failure_not_retried=True,
    )
    atomic_json(args.out / "request.json", continuation)
    contexts = {c["context_id"]: c for c in read(ROOT / request["geometry_file"])["contexts"]}
    ego_path = ROOT / request["ego_descriptor"]
    status, active, current_samples = "running", None, 0
    records = []

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Diverse causal-action TRAINpilot",
                frames=80 + sum(r.get("samples", 0) for r in records) + current_samples,
                target_frames=800,
                encounters_done=2 + len(records),
                encounters_total=10,
                phase=phase,
                original_failed_cases=1,
                active_output=str(active) if active else None,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists() or time.time() > deadline:
            raise InterruptedError("Owned STOP or original900sdeadline")
        size = sum(
            p.stat().st_size
            for root in (original, args.out)
            for p in root.rglob("*")
            if p.is_file()
        )
        if size > 600_000_000:
            raise InterruptedError("Original shared600MBartifact cap")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check()
            check_resources(args.out, 10)
            vendor = activate()
            model = FrozenExpertEgo(ego_path)
            config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
            pending, server_index = 0, 0
            progress("Only8original unstarted cases; failedcase retained, original deadline/cap")
            while pending < len(remaining):
                check()
                town = contexts[remaining[pending]["context_id"]]["town"]
                server_out = args.out / "servers" / f"server_{server_index:03d}"
                server_out.mkdir(parents=True)
                (server_out / "STOP").symlink_to("../../STOP")
                server = CarlaServer(vendor["carla_root"], server_out, town=town)
                with server as client:
                    while (
                        pending < len(remaining)
                        and contexts[remaining[pending]["context_id"]]["town"] == town
                    ):
                        check()
                        case = remaining[pending]
                        active = args.out / "episodes" / case["case_id"]
                        current_samples = 0
                        progress("Continue only original unstarted case; no retry")

                        def on_tick(samples, tick):
                            nonlocal current_samples
                            current_samples = samples
                            if tick % 5 == 0:
                                progress("Native actual actions; original prospective controls")

                        try:
                            result = collect_case(
                                server,
                                client,
                                case,
                                contexts[case["context_id"]],
                                active,
                                config,
                                model,
                                ego_path,
                                check,
                                on_tick,
                            )
                        except BaseException as error:
                            records.append(
                                dict(
                                    case=case,
                                    status="failed",
                                    samples=current_samples,
                                    error=str(error),
                                )
                            )
                            atomic_json(args.out / "records.json", records)
                            current_samples = 0
                            raise
                        records.append(result | dict(status="captured"))
                        pending += 1
                        current_samples = 0
                        atomic_json(args.out / "records.json", records)
                server_index += 1
            active = None
            if any(digest(ROOT / p) != sha for p, sha in hashes.items()):
                raise ValueError("New source generation changed during continuation")
            progress("Independently audit8new captures then all10original attempted cases")
            result = audit(args.out, args.out / "continuation_independent_audit")
            combined = combined_audit(
                original, args.out, args.out / "original10_independent_audit", check
            )
            atomic_json(
                args.out / "assessment.json",
                dict(
                    original_planned=10,
                    original_failed_brake_retained=True,
                    continued8native_verified=result["independently_verified"],
                    combined_verified=combined["verified"],
                    combined_errors=combined["errors"],
                    replay_comparisons_pass=combined["all_registered_observable_comparisons_pass"],
                    all_unfiltered_contacts=combined["unfiltered_contact_starts"],
                    original10audit_still_failed=bool(combined["errors"]),
                    no_fit_or_driving_promotion=True,
                    no_automatic_extension=True,
                ),
            )
            status = "complete"
            progress(
                "Original10settled; failedcase retained; inspect independent audit before fitting"
            )
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(
            args.out / "error.json",
            dict(error=str(error), at=time.time(), active_output=str(active)),
        )
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
