#!/usr/bin/env python3
"""Bounded M0 route generation and full-harness expert evaluation."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.collection import finalize_collection, verify_collection_catalog  # noqa: E402
from src.navigation.metrics import execution_status, summarize_records  # noqa: E402
from src.navigation.routes import (  # noqa: E402
    create_development_route,
    signal_inventory,
    verify_route,
)
from src.navigation.runtime import (  # noqa: E402
    CarlaServer,
    RunLock,
    atomic_json,
    check_resources,
    navigation_ports,
    resource_shortfall,
    slot_ports,
)
from src.navigation.vendor import activate  # noqa: E402


def checkpoint_metadata(path):
    """Identify an own exported policy without relying on a stale generic tag."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    from src.navigation.policy_input_contract import policy_input_schema

    architecture = state.get("architecture")
    if architecture not in (
        "camera-student-v1",
        "camera-stop-go-v2",
        "camera-signal-residual-v3",
        "spatial-stop-go-v1",
        "spatial-visual-decision-v2",
        "spatial-visual-facts-v3",
    ):
        raise ValueError(f"Unsupported camera checkpoint architecture: {architecture}")
    metadata = {
        "architecture": architecture,
        "hidden": int(state["hidden"]),
        "selected_training_step": state.get("step"),
        "policy_input_schema": policy_input_schema(state),
    }
    if architecture in (
        "spatial-stop-go-v1",
        "spatial-visual-decision-v2",
        "spatial-visual-facts-v3",
    ):
        metadata.update(
            encoder_kind=state["encoder_kind"],
            encoder_dim=state["encoder_dim"],
            color_enabled=state["color_enabled"],
            neural_ego_speed_enabled=architecture == "spatial-stop-go-v1",
            fact_supervision=bool(state.get("fact_supervision", False)),
            burn_in=state.get("burn_in", 8),
        )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "make-route", "expert", "student", "dagger"])
    parser.add_argument("--config", default="configs/navigation.yaml")
    parser.add_argument(
        "--carla-root", default=os.environ.get("CARLA_ROOT", "~/software/CARLA_0.9.15")
    )
    parser.add_argument("--route", default="configs/routes/navigation_m0_town03_opt_v2.xml")
    parser.add_argument("--town", default="Town03_Opt", help="Development route generation map")
    parser.add_argument("--min-length", type=float, default=1000)
    parser.add_argument("--max-length", type=float, default=1500)
    parser.add_argument("--out", default="outputs/navigation_m0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="0: whole route; >0: profile only")
    parser.add_argument("--max-wall", type=float, default=1800)
    parser.add_argument("--routes-subset", default="", help="Evaluator route-id subset, e.g. 3-12")
    parser.add_argument(
        "--seed-per-route", action="store_true", help="Traffic seed = --seed + XML route id"
    )
    parser.add_argument("--slot", type=int, default=0, help="Parallel server slot (ports/lock)")
    parser.add_argument(
        "--collection-catalog",
        default="",
        help="Collect lossless training shards from a frozen train catalog",
    )
    parser.add_argument("--collection-mode", choices=["expert", "noisy"], default="expert")
    parser.add_argument("--collection-max-samples", type=int, default=0)
    parser.add_argument(
        "--collection-guard", default="", help="Strict TRAIN component-location guard"
    )
    parser.add_argument("--training-empty-background", action="store_true")
    parser.add_argument("--training-signal-wait", type=float)
    parser.add_argument("--training-light-gap", type=float)
    parser.add_argument(
        "--training-dense-labels",
        action="store_true",
        help="Separate instance/depth TRAINlabels; guarded expert collection only",
    )
    parser.add_argument("--checkpoint", default="", help="Camera student policy checkpoint")
    parser.add_argument("--rule-case", default="", help="Separate frozen M3 component-test case")
    parser.add_argument(
        "--final-registration", default="", help="Immutable qualified Town05 candidate registration"
    )
    parser.add_argument(
        "--dagger-guard", default="", help="Frozen training/internal-validation guard"
    )
    parser.add_argument(
        "--rule-overrides", action="store_true", help="Enable learned-perception rule layer"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    # These flags alter criteria at import time as well as expert behavior.
    for name in ("SAVE_PATH", "DATAGEN", "DEBUG_CHALLENGE", "HISTOGRAM", "TP_STATS"):
        os.environ.pop(name, None)
    vendor = activate(args.carla_root)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if args.command == "doctor":
        from autopilot import AutoPilot

        from src.navigation.harness import NavigationEvaluator

        print(
            json.dumps(
                {
                    "vendor": vendor,
                    "signals": signal_inventory(args.carla_root),
                    "imports": [AutoPilot.__name__, NavigationEvaluator.__name__],
                },
                indent=2,
            )
        )
        return
    if args.command == "make-route":
        print(
            json.dumps(
                create_development_route(
                    args.carla_root,
                    args.route,
                    town=args.town,
                    minimum=args.min_length,
                    maximum=args.max_length,
                ),
                indent=2,
            )
        )
        return
    route = Path(args.route).resolve()
    manifest = verify_route(route)
    collection = (
        verify_collection_catalog(args.collection_catalog, route)
        if args.collection_catalog
        else None
    )
    student = None
    policy_kind = "privileged_pdm_lite_modified"
    if args.command in ("student", "dagger"):
        if args.command == "student" and collection:
            raise ValueError(
                "Student rollouts require a separate reviewed DAgger collection adapter"
            )
        checkpoint = Path(args.checkpoint).resolve()
        if not args.checkpoint or not checkpoint.is_file():
            raise ValueError("Camera evaluation requires --checkpoint")
        student = {
            **checkpoint_metadata(checkpoint),
            "checkpoint": str(Path(args.out).resolve() / "policy.pt"),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "rule_overrides": args.rule_overrides,
            "ego_inputs": "Ground-truth ego odometry/localization for declared navigation; "
            "no other actor state or ground-truth rule/hazard input",
        }
        from src.navigation.policy_input_contract import validate_config_schema

        validate_config_schema(config, student["policy_input_schema"])
        policy_kind = f"{student['architecture']}_map_ego"
    if args.command == "dagger":
        from src.navigation.dagger import validate_guard

        if not collection or not args.dagger_guard or args.rule_overrides:
            raise ValueError("DAgger needs a frozen train catalog/guard and learned overrides OFF")
        if not 0 < args.max_wall <= 1800:
            raise ValueError("DAgger pilots are bounded to at most30 minutes per invocation")
        collection["dagger_guard"] = validate_guard(
            args.dagger_guard, args.collection_catalog, collection["town"], args.routes_subset
        )
        policy_kind = f"dagger_training_{student['architecture']}_map_ego"
    if collection:
        if args.collection_max_samples < 0:
            raise ValueError("Collection sample limit cannot be negative")
        collection.update(
            mode="dagger" if args.command == "dagger" else args.collection_mode,
            max_samples=args.collection_max_samples,
        )
    if args.collection_guard:
        if args.training_dense_labels:
            from src.navigation.label_collection import (
                validate_label_guard as validate_collection_guard,
            )
        else:
            from src.navigation.approach_data import (
                validate_approach_guard as validate_collection_guard,
            )

        if not collection or args.command != "expert" or args.collection_mode != "expert":
            raise ValueError("Approach guard requires clean expert TRAIN collection")
        collection["learning_guard"] = validate_collection_guard(
            args.collection_guard, args.collection_catalog, collection["town"], args.routes_subset
        )
    if args.training_empty_background:
        if not args.collection_guard or not collection or args.command != "expert":
            raise ValueError("Empty background requires guarded expert TRAIN collection")
        collection["background"] = "empty"
    if args.training_signal_wait is not None:
        from src.navigation.collection import training_signal_wait

        if not collection:
            raise ValueError("Training signal wait requires collection")
        collection["signal_stationary_seconds"] = args.training_signal_wait
        training_signal_wait(dict(collection=collection, student=student))
    if args.training_light_gap is not None:
        from src.navigation.collection import training_light_gap

        if not collection or args.command != "expert":
            raise ValueError("Light-gap trial is expert TRAIN collection only")
        collection["training_light_gap_m"] = args.training_light_gap
        training_light_gap(dict(collection=collection, student=student))
        policy_kind = "privileged_pdm_lite_training_gap"
    if args.training_dense_labels:
        from src.navigation.label_collection import require_label_scope

        require_label_scope(
            args.command,
            collection,
            student,
            args.final_registration,
            args.rule_case,
            args.training_empty_background,
            args.training_light_gap,
        )
        collection["dense_labels"] = True
        policy_kind = "privileged_pdm_lite_dense_label_training"
    rule_test = None
    if args.rule_case:
        from src.navigation.rule_case_contract import verify_rule_case

        if args.command not in ("expert", "student") or args.seed_per_route:
            raise ValueError(
                "Controlled component tests use explicit expert/student and paired seeds"
            )
        rule_test = verify_rule_case(
            args.rule_case,
            route,
            args.routes_subset,
            args.seed,
            collection,
            args.final_registration,
            args.rule_overrides,
            args.max_steps,
        )
        if args.command == "student":
            certificate = Path(args.rule_case).resolve().parent / "setup_certificate.json"
            if not certificate.exists():
                raise ValueError("Certify this physical component setup before any student trial")
            from src.navigation.rule_suite import setup_certificate_valid

            certified = json.loads(certificate.read_text())
            if certified.get("case_sha256") != rule_test[
                "case_sha256"
            ] or not setup_certificate_valid(certified, args.rule_case):
                raise ValueError("Physical rule-test setup certificate is missing or invalid")
        policy_kind = f"CONTROLLED_COMPONENT_{policy_kind}"
    final = None
    if args.final_registration:
        from src.navigation.final_evaluation import validate_final_request

        if manifest["town"].removesuffix("_Opt") != "Town05" or args.command not in (
            "expert",
            "student",
        ):
            raise ValueError(
                "Final registration is only for the frozen Town05 expert/student suite"
            )
        final = validate_final_request(
            args.final_registration,
            route,
            args.out,
            args.seed,
            args.routes_subset,
            student,
            collection,
            args.seed_per_route,
            args.max_steps,
        )
    elif manifest["town"].removesuffix("_Opt") not in config["data"]["train_towns"]:
        raise ValueError("M0 only evaluates development towns")
    output = Path(args.out).resolve()
    if args.worker:
        run_worker(args, config, output, route)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory; attempts must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    runtime = config["runtime"]
    port, _ = slot_ports(runtime, args.slot)
    # Slot 0 keeps the original lock, so single-server tools stay mutually exclusive with it.
    lock = ".gpu.lock" if args.slot == 0 else f".slot{args.slot}.lock"
    with RunLock(ROOT / "outputs/navigation" / lock):
        free = check_resources(
            output, runtime["minimum_free_gb"], navigation_ports(runtime) - {port}
        )
        shortfall = resource_shortfall(runtime)
        if shortfall:
            raise RuntimeError(f"Refusing to start another server: {shortfall}")
        if student:
            shutil.copyfile(checkpoint, output / "policy.pt")
        request = {
            "config": str(config_path),
            "out": str(output),
            "collection": collection,
            "student": student,
            "rule_test": rule_test,
        }
        atomic_json(output / "agent.json", request)
        atomic_json(
            output / "manifest.json",
            {
                "vendor": vendor,
                "route": manifest,
                "seed": args.seed,
                "configuration": config,
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "free_gb_at_start": free,
                "max_steps": args.max_steps,
                "max_wall": args.max_wall,
                "slot": args.slot,
                "ports": slot_ports(runtime, args.slot),
                "routes_subset": args.routes_subset,
                "seed_per_route": args.seed_per_route,
                "policy_kind": policy_kind,
                "final_benchmark": bool(final),
                "final_registration": final,
                "training_collection": bool(collection),
                "privileged_assistance": args.command == "dagger",
                "collection": collection,
                "student": student,
                "rule_test": rule_test,
            },
        )
        sources = [
            *sorted((ROOT / "src/navigation").glob("*.py")),
            Path(__file__).resolve(),
            ROOT / "patches/navigation_garage.patch",
            ROOT / "configs/navigation_vendor.json",
        ]
        if args.training_dense_labels:
            sources.append(ROOT / "scripts/navigation_label_rig.py")
        hashes = {}
        for source in sources:
            relative = source.relative_to(ROOT)
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        atomic_json(output / "source_hashes.json", hashes)
        atomic_json(
            output / "progress.json",
            {"status": "starting", "target_frames": args.max_steps, "updated_at": time.time()},
        )
        try:
            with CarlaServer(args.carla_root, output, port, town=manifest["town"]):
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    args.command,
                    "--worker",
                    "--config",
                    str(config_path),
                    "--carla-root",
                    args.carla_root,
                    "--route",
                    str(route),
                    "--out",
                    str(output),
                    "--seed",
                    str(args.seed),
                    "--max-steps",
                    str(args.max_steps),
                    "--max-wall",
                    str(args.max_wall),
                    "--routes-subset",
                    args.routes_subset,
                    "--slot",
                    str(args.slot),
                    *(["--seed-per-route"] if args.seed_per_route else []),
                    *(
                        ["--final-registration", str(Path(args.final_registration).resolve())]
                        if final
                        else []
                    ),
                    *(["--checkpoint", str(output / "policy.pt")] if student else []),
                    *(["--rule-case", str(Path(args.rule_case).resolve())] if rule_test else []),
                    *(
                        ["--dagger-guard", str(Path(args.dagger_guard).resolve())]
                        if args.command == "dagger"
                        else []
                    ),
                    *(["--rule-overrides"] if student and args.rule_overrides else []),
                    *(["--collection-catalog", collection["catalog"]] if collection else []),
                    *(["--training-empty-background"] if args.training_empty_background else []),
                    *(["--training-dense-labels"] if args.training_dense_labels else []),
                    *(
                        ["--training-signal-wait", str(args.training_signal_wait)]
                        if args.training_signal_wait is not None
                        else []
                    ),
                    *(
                        ["--collection-guard", str(Path(args.collection_guard).resolve())]
                        if args.collection_guard
                        else []
                    ),
                    *(
                        ["--training-light-gap", str(args.training_light_gap)]
                        if args.training_light_gap is not None
                        else []
                    ),
                    *(
                        [
                            "--collection-mode",
                            args.collection_mode,
                            "--collection-max-samples",
                            str(args.collection_max_samples),
                        ]
                        if collection
                        else []
                    ),
                ]
                # A native abort in CARLA's client remains inside this child. The parent
                # owns the server and still executes cleanup and preserves the failure.
                worker = subprocess.Popen(command, cwd=ROOT)
                atomic_json(output / "worker.json", {"pid": worker.pid})
                deadline = time.monotonic() + args.max_wall + 180
                stop_started = None
                while worker.poll() is None:
                    if time.monotonic() >= deadline:
                        (output / "STOP").touch()
                    if (output / "STOP").exists():
                        stop_started = stop_started or time.monotonic()
                        if time.monotonic() - stop_started > 60:
                            worker.terminate()  # owned evaluator, after STOP grace period
                            worker.wait(timeout=30)
                            break
                    time.sleep(1)
                code = worker.returncode
                if code:
                    raise RuntimeError(f"Evaluator child exited {code}; preserved console/attempt")
        except BaseException as exc:
            atomic_json(output / "error.json", {"type": type(exc).__name__, "error": str(exc)})
            progress = json.loads((output / "progress.json").read_text())
            progress.update(status="failed", error=str(exc), updated_at=time.time())
            atomic_json(output / "progress.json", progress)
            raise
        finally:
            if collection:
                finalize_collection(output)


def run_worker(args, config, output, route):
    from leaderboard.utils.statistics_manager_local import StatisticsManager

    from src.navigation.harness import NavigationEvaluator

    runtime = config["runtime"]
    port, traffic_manager_port = slot_ports(runtime, args.slot)
    settings = SimpleNamespace(
        host=runtime["host"],
        port=port,
        traffic_manager_port=traffic_manager_port,
        traffic_manager_seed=args.seed,
        debug=0,
        record=str(output),
        timeout=30.0,
        routes=str(route),
        routes_subset=args.routes_subset,
        seed_per_route=args.seed_per_route,
        repetitions=1,
        agent=str(
            ROOT
            / "src/navigation"
            / (
                "dagger_agent.py"
                if args.command == "dagger"
                else "camera_agent.py"
                if args.command == "student"
                else "training_gap_agent.py"
                if args.training_light_gap is not None
                else "labeled_collection_agent.py"
                if args.training_dense_labels
                else ("collection_agent.py" if args.collection_catalog else "pdm_agent.py")
            )
        ),
        agent_config=str(output / "agent.json"),
        track="MAP",
        resume=False,
        checkpoint=str(output / "leaderboard.json"),
        debug_checkpoint=str(output / "live_results.txt"),
        navigation_out=str(output),
        navigation_town=verify_route(route)["town"],
        max_steps=args.max_steps,
        max_wall=args.max_wall,
    )
    stats = StatisticsManager(settings.checkpoint, settings.debug_checkpoint)
    evaluator = NavigationEvaluator(settings, stats)
    crashed = evaluator.run(settings) and not evaluator.navigation_halted
    result = json.loads((output / "leaderboard.json").read_text())
    records = result.get("_checkpoint", {}).get("records", [])
    reason = evaluator.manager.navigation_stop_reason
    status = execution_status(records, crashed, reason, args.max_steps)
    summary = summarize_records(
        records,
        execution_failure=bool(crashed or reason or status == "failed"),
        # Earlier routes in a multi-route attempt finished before the stop or crash.
        last_route_only=bool(args.routes_subset),
    )
    summary.update(
        harness_crashed=bool(crashed),
        profile_only=args.max_steps > 0,
        stop_reason=reason,
        policy_kind=json.loads((output / "manifest.json").read_text())["policy_kind"],
        execution_status=status,
        training_mixture_not_camera_evaluation=bool(args.collection_catalog),
    )
    atomic_json(output / "summary.json", summary)
    if (output / "STOP").exists():
        status = "stopped"
    progress = json.loads((output / "progress.json").read_text())
    if records:
        progress["route_progress_percent"] = records[-1].get("scores", {}).get("score_route", 0)
    progress.update(status=status, summary=summary, updated_at=time.time())
    atomic_json(output / "progress.json", progress)
    print(json.dumps(summary, indent=2))
    del evaluator
    if status == "failed":
        raise RuntimeError("Harness failed; see leaderboard.json and summary.json")


if __name__ == "__main__":
    main()
