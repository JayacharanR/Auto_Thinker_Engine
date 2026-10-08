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
)
from src.navigation.vendor import activate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "make-route", "expert"])
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
    if manifest["town"].removesuffix("_Opt") not in config["data"]["train_towns"]:
        raise ValueError("M0 only evaluates development towns")
    output = Path(args.out).resolve()
    if args.worker:
        run_worker(args, config, output, route)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory; attempts must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    runtime = config["runtime"]
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        free = check_resources(output, runtime["minimum_free_gb"])
        request = {"config": str(config_path), "out": str(output)}
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
                "policy_kind": "privileged_pdm_lite_modified",
                "final_benchmark": False,
            },
        )
        sources = [
            *sorted((ROOT / "src/navigation").glob("*.py")),
            Path(__file__).resolve(),
            ROOT / "patches/navigation_garage.patch",
            ROOT / "configs/navigation_vendor.json",
        ]
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
            with CarlaServer(args.carla_root, output, runtime["port"], town=manifest["town"]):
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "expert",
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


def run_worker(args, config, output, route):
    from leaderboard.utils.statistics_manager_local import StatisticsManager

    from src.navigation.harness import NavigationEvaluator

    runtime = config["runtime"]
    settings = SimpleNamespace(
        host=runtime["host"],
        port=runtime["port"],
        traffic_manager_port=runtime["traffic_manager_port"],
        traffic_manager_seed=args.seed,
        debug=0,
        record=str(output),
        timeout=30.0,
        routes=str(route),
        routes_subset="",
        repetitions=1,
        agent=str(ROOT / "src/navigation/pdm_agent.py"),
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
    crashed = evaluator.run(settings)
    result = json.loads((output / "leaderboard.json").read_text())
    records = result.get("_checkpoint", {}).get("records", [])
    reason = evaluator.manager.navigation_stop_reason
    status = execution_status(records, crashed, reason, args.max_steps)
    summary = summarize_records(
        records, execution_failure=bool(crashed or reason or status == "failed")
    )
    summary.update(
        harness_crashed=bool(crashed),
        profile_only=args.max_steps > 0,
        stop_reason=reason,
        policy_kind="privileged_pdm_lite_modified",
        execution_status=status,
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
