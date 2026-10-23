#!/usr/bin/env python3
"""Run only the original four unstarted cases, each with a fresh owned simulator."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_interventions import audit_case, lines  # noqa:E402
from scripts.continue_navigation_intervention_pilot import require_only_timeout_changes  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.frozen_ego import FrozenExpertEgo  # noqa:E402
from src.navigation.intervention_collection import collect_case  # noqa:E402
from src.navigation.replay import compare_world_prefixes  # noqa:E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    original = ROOT / "outputs/navigation_intervention_pilot_20261020"
    previous = ROOT / "outputs/navigation_intervention_continuation_20261020"
    request = read(original / "request.json")
    if (
        read(previous / "progress.json")["status"] != "failed"
        or not read(previous / "servers/server_001/normal_cleanup_retry.json")[
            "normal_shutdown_finished"
        ]
    ):
        raise ValueError("Original failed continuation and its normal STOPcleanup required")
    require_only_timeout_changes(original)
    remaining = request["cases"][6:]
    if len(remaining) != 4 or any(
        (root / "episodes" / case["case_id"]).exists()
        for root in (original, previous)
        for case in remaining
    ):
        raise ValueError("Only original final4entirely unstarted cases")
    if (previous / "storage_stop.json").exists() or (original / "storage_stop.json").exists():
        raise ValueError("No storage guardian STOPbypass")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "episodes").mkdir()
    deadline = (original / "request.json").stat().st_mtime + 900
    sources = [ROOT / p for p in request["source_hashes"]] + [
        Path(__file__),
        ROOT / "scripts/continue_navigation_intervention_pilot.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        request
        | dict(
            cases=remaining,
            original=str(original),
            previous=str(previous),
            source_hashes=hashes,
            original_absolute_deadline=deadline,
            original_storage_cap_shared=600_000_000,
            isolated_native_server_per_case=True,
            failed2cases_not_retried=True,
            original_case_or_control_equations_changed=False,
        ),
    )
    contexts = {c["context_id"]: c for c in read(ROOT / request["geometry_file"])["contexts"]}
    status, active, current_samples = "running", None, 0
    records = []

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Diverse causal-action TRAINpilot",
                frames=320 + sum(r.get("samples", 0) for r in records) + current_samples,
                target_frames=800,
                encounters_done=6 + len(records),
                encounters_total=10,
                original_failed_cases=2,
                phase=phase,
                active_output=str(active) if active else None,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists() or time.time() > deadline:
            raise InterruptedError("Owned STOPororiginal900sdeadline")
        size = sum(
            p.stat().st_size
            for root in (original, previous, args.out)
            for p in root.rglob("*")
            if p.is_file()
        )
        if size > 600_000_000:
            raise InterruptedError("Original shared600MBcap")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check()
            check_resources(args.out, 10)
            vendor = activate()
            ego_path = ROOT / request["ego_descriptor"]
            model = FrozenExpertEgo(ego_path)
            config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
            progress("Only original4unstarted; fresh native server each, original deadline/cap")
            for index, case in enumerate(remaining):
                check()
                active = args.out / "episodes" / case["case_id"]
                current_samples = 0
                server_out = args.out / "servers" / f"server_{index:03d}"
                server_out.mkdir(parents=True)
                (server_out / "STOP").symlink_to("../../STOP")
                server = CarlaServer(
                    vendor["carla_root"], server_out, town=contexts[case["context_id"]]["town"]
                )
                with server as client:

                    def on_tick(samples, tick):
                        nonlocal current_samples
                        current_samples = samples
                        if tick % 5 == 0:
                            progress("Original prospective plan, own actual controls; no retries")

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
                    current_samples = 0
                    atomic_json(args.out / "records.json", records)
            active = None
            if any(digest(ROOT / p) != sha for p, sha in hashes.items()):
                raise ValueError("Pinned source changed during final unstarted continuation")
            audit_out = args.out / "original10_independent_audit"
            audit_out.mkdir()
            audited, errors, prefixes, repeats, comparisons = [], [], {}, {}, []
            for index, case in enumerate(request["cases"]):
                check()
                owner = original if index < 2 else previous if index < 6 else args.out
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
                    audited.append(result | dict(physical_output=str(episode)))
                except Exception as error:
                    errors.append(dict(case=case, physical_output=str(episode), error=str(error)))
                atomic_json(
                    audit_out / "progress.json",
                    dict(
                        status="running",
                        stage="Intervention independent audit",
                        frames=len(audited) + len(errors),
                        target_frames=10,
                        updated_at=time.time(),
                    ),
                )
            assessment = dict(
                original_planned=10,
                records=audited,
                errors=errors,
                independently_verified=len(audited),
                comparisons=comparisons,
                replay_observables_pass=bool(comparisons)
                and all(c["observable_prefix_within_tolerance"] for c in comparisons),
                unfiltered_contact_starts=sum(r["unfiltered_contact_starts"] for r in audited),
                original2failedcases_retained=True,
                no_driving_qualification_or_model_fitting=True,
            )
            atomic_json(audit_out / "assessment.json", assessment)
            atomic_json(
                audit_out / "progress.json",
                dict(
                    status="failed" if errors else "complete",
                    stage="Intervention independent audit",
                    frames=10,
                    target_frames=10,
                    updated_at=time.time(),
                ),
            )
            atomic_json(args.out / "assessment.json", assessment)
            status = "complete"
            progress(
                "Original10settled;2failures retained; independent audit determines eligible data"
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
