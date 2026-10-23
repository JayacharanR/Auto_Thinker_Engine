#!/usr/bin/env python3
"""Ten original TRAIN-only action attempts; no retry, fitting or driver deployment."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_interventions import audit  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.frozen_ego import FrozenExpertEgo  # noqa:E402
from src.navigation.intervention_collection import collect_case  # noqa:E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def pilot_cases(geometry):
    selected = []
    for mechanism in ("lead", "crossing", "rear"):
        choices = [
            c for c in geometry["contexts"] if c["role"] == "train" and c["mechanism"] == mechanism
        ]
        if not choices:
            raise ValueError("All three registered TRAINmechanisms required")
        context = min(choices, key=lambda c: c["route_id"])
        if context["route_id"] == 6:
            raise ValueError("Old observed diagnostic cannot become TRAIN")
        for branch in ("go", "brake", "wait_go"):
            selected.append(
                dict(
                    case_id=context["context_id"] + "_" + branch + "_repeat0",
                    context_id=context["context_id"],
                    branch=branch,
                    repeat=0,
                )
            )
    first = selected[0]
    selected.append(first | dict(case_id=first["context_id"] + "_go_repeat1", repeat=1))
    if [c["context_id"] for c in selected[::3]][:3] != ["r35_lead", "r134_crossing", "r35_rear"]:
        raise ValueError("Frozen pilot context selection changed")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "episodes").mkdir()
    geometry_path = ROOT / "outputs/navigation_intervention_geometry_v2_20261020/plan.json"
    geometry = read(geometry_path)
    if geometry["schema"] != "navigation-action-intervention-geometry-v2":
        raise ValueError("Only before-outcome revised geometry")
    cases = pilot_cases(geometry)
    contexts = {c["context_id"]: c for c in geometry["contexts"]}
    ego_path = ROOT / "outputs/navigation_models/frozen_expert_ego_20261020.json"
    data_plan_path = Path(geometry["original_data_plan"])
    if digest(data_plan_path) != geometry["original_data_plan_sha256"]:
        raise ValueError("Original source/calibration split changed")
    data_plan = read(data_plan_path)
    sources = [
        Path(__file__),
        geometry_path,
        data_plan_path,
        ego_path,
        ROOT / "configs/navigation.yaml",
        ROOT / "configs/navigation_vendor.json",
        ROOT / "reports/navigation_intervention_pilot_protocol_20261020.md",
        Path(data_plan["actual_training_guard"]),
        *sorted((ROOT / "src/navigation").glob("*.py")),
        *[
            ROOT / "scripts" / name
            for name in (
                "collect_navigation_branches.py",
                "audit_navigation_interventions.py",
                "audit_navigation_dense_labels.py",
                "audit_navigation_fixed_branches.py",
                "navigation_label_rig.py",
            )
        ],
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            cases=cases,
            source_hashes=hashes,
            geometry_file=str(geometry_path.relative_to(ROOT)),
            ego_descriptor=str(ego_path.relative_to(ROOT)),
            original_data_plan=str(data_plan_path.relative_to(ROOT)),
            source_role="TRAINonly",
            maximum_wall_seconds=900,
            maximum_artifact_bytes=600_000_000,
            freeze_commands_before_outcome=True,
            imitation_eligible=False,
            no_automatic_training=True,
            no_failed_retry=True,
            physics_hz=20,
            RGB_hz=10,
            prefix_ticks=80,
            branch_ticks=80,
            planned_attempts=10,
        ),
    )
    status, started, active, current_samples = "running", time.monotonic(), None, 0
    records = []

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Diverse causal-action TRAINpilot",
                frames=sum(r.get("samples", 0) for r in records) + current_samples,
                target_frames=800,
                encounters_done=len(records),
                encounters_total=10,
                phase=phase,
                active_output=str(active) if active else None,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 900:
            raise InterruptedError("Owned pilot STOP or900swhole-stage cap")
        if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > 600_000_000:
            raise InterruptedError("Pilot600MBartifact cap")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 10)
            vendor = activate()
            model = FrozenExpertEgo(ego_path)
            config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
            progress("All ten TRAINcases frozen before native outcomes; no fitting")
            # One fresh owned engine per physical town block; source/whole-stage
            # limits do not restart. A failure stops this owner, never retries.
            server_index = 0
            pending = 0
            while pending < len(cases):
                check()
                case = cases[pending]
                town = contexts[case["context_id"]]["town"]
                server_out = args.out / "servers" / f"server_{server_index:03d}"
                server_out.mkdir(parents=True)
                (server_out / "STOP").symlink_to("../../STOP")
                server = CarlaServer(vendor["carla_root"], server_out, town=town)
                with server as client:
                    while (
                        pending < len(cases)
                        and contexts[cases[pending]["context_id"]]["town"] == town
                    ):
                        check()
                        case = cases[pending]
                        active = args.out / "episodes" / case["case_id"]
                        current_samples = 0
                        progress("Collect the original next case; no retry or replacement")

                        def on_tick(samples, tick):
                            nonlocal current_samples
                            current_samples = samples
                            if tick % 5 == 0:
                                progress("Actual20Hzcontrols/10HzRGB; plans fixed at decision")

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
                raise ValueError("Pinned runtime/collector source changed during pilot")
            progress("Closed native cases; independent pixels/actuators/geography/replay audit")
            result = audit(args.out, args.out / "independent_audit")
            if result["errors"] or not result["all_registered_observable_comparisons_pass"]:
                raise ValueError(
                    "Independent pilot audit/replay failed; no further fits or auto-extension"
                )
            atomic_json(
                args.out / "assessment.json",
                dict(
                    planned_attempts=10,
                    captured_attempts=len(records),
                    independent_audit_sha256=digest(args.out / "independent_audit/assessment.json"),
                    native_control_acknowledgments=sum(
                        r["native_controls_verified"] for r in result["records"]
                    ),
                    all_unfiltered_contact_starts=result["unfiltered_contact_starts"],
                    no_model_updates=True,
                    no_driving_or_M5qualification=True,
                    auto_extension_or_training=False,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status = "complete"
            progress(
                "Ten original attempts independently audited; evidence only, no model promotion"
            )
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(
            args.out / "error.json",
            dict(error=str(error), active_output=str(active), at=time.time()),
        )
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
