#!/usr/bin/env python3
"""Keep native callbacks out of the supervisor; two new excluded instrument cycles."""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.native_process_owner import (  # noqa:E402
    require_process_identity,
    run_native_worker,
)
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def worker(args):
    import psutil
    import yaml

    from scripts.audit_navigation_dense_labels import actual_geography
    from scripts.collect_navigation_branches import selected_entry
    from scripts.probe_navigation_native_session import exercise
    from src.navigation.vendor import activate

    identity = read(args.engine_identity)
    if identity["parent_pid"] != os.getppid():
        raise ValueError("Worker requires its direct owning parent")
    started = time.monotonic()

    def live_process():
        process = psutil.Process(identity["pid"])
        require_process_identity(
            identity,
            process.cmdline(),
            process.uids().real,
            os.getpgid(process.pid),
            process.create_time(),
        )
        return process

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 120:
            raise InterruptedError("Owned instrument worker STOPor120s cap")
        process = live_process()
        if process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("Owned engine exited")

    # Wait for this exact owned PID's listening socket, never attach to another
    # process merely because a loopback RPCport is responding.
    while True:
        check()
        process = live_process()
        if any(
            c.status == psutil.CONN_LISTEN and c.laddr.port == identity["port"]
            for c in process.net_connections(kind="tcp")
        ):
            break
        time.sleep(0.2)
    activate()
    import carla

    native = carla.Client("127.0.0.1", identity["port"])
    native.set_timeout(2.0)
    while True:
        check()
        try:
            ready = native.get_world().get_map().name
            break
        except RuntimeError:
            time.sleep(0.2)
    native.set_timeout(20.0)
    atomic_json(args.out / "startup_map.json", dict(actual=ready, requested=identity["town"]))

    def poll_owned():
        try:
            p = live_process()
            return None if p.status() != psutil.STATUS_ZOMBIE else -1
        except psutil.NoSuchProcess:
            return -1

    view = SimpleNamespace(process=SimpleNamespace(poll=poll_owned))
    entry, _ = selected_entry()
    config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
    result = exercise(view, native, args.out, entry, config, check)
    plan = read(ROOT / "outputs/navigation_dense_label_collection_20261020/plan.json")
    item = next(r for r in plan["episodes"] if r["route_id"] == 6)
    result["actual_geography"] = actual_geography(read(args.out / "world_trace.json"), item, plan)
    native = None
    gc.collect()
    atomic_json(
        args.out / "child_complete.json",
        result | dict(native_handles_not_returned=True, instrument_only_no_fitting=True),
    )


def main(args):
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        ROOT / "reports/navigation_process_isolation_protocol_20261020.md",
        ROOT / "scripts/probe_navigation_native_session.py",
        ROOT / "configs/navigation.yaml",
        ROOT / "src/navigation/native_process_owner.py",
        ROOT / "src/navigation/native_cleanup.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for p in paths:
        dst = args.out / "source" / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    atomic_json(
        args.out / "request.json",
        dict(
            source_hashes=hashes,
            cycles=2,
            maximum_wall_seconds=300,
            maximum_artifact_bytes=20_000_000,
            no_driving_or_model_or_training_data=True,
            previous_instrument_not_retried=True,
            outer_never_imports_carla=True,
        ),
    )
    status, records, started = "running", [], time.monotonic()

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Native process isolation instrument",
                frames=len(records),
                target_frames=2,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 300:
            raise InterruptedError("Outer instrument STOPor300s cap")
        if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > 20_000_000:
            raise InterruptedError("Outer instrument20MBcap")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 10)
            from src.navigation.vendor import activate

            vendor = activate()
            for index in range(2):
                check()
                progress("Stationary child; no native bindings in supervising process")
                result = run_native_worker(
                    vendor["carla_root"],
                    args.out / f"cycle_{index}",
                    "Town01_Opt",
                    [
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(Path(__file__).resolve()),
                        "--worker",
                    ],
                    check,
                    lambda state: progress("Native child alive; owned STOPcontrols teardown"),
                )
                records.append(result)
                atomic_json(args.out / "records.json", records)
            if any(digest(ROOT / p) != sha for p, sha in hashes.items()):
                raise ValueError("Pinned instrument source changed")
            passed = all(
                r["child_returncode"] == 0
                and r["native_cleanup"]["normal_shutdown_finished"]
                and not r["parent_native_bindings_imported"]
                for r in records
            )
            native_abnormal = sum(r["native_returncode"] not in (0, -15) for r in records)
            atomic_json(
                args.out / "assessment.json",
                dict(
                    records=records,
                    process_containment_verified=passed,
                    native_abnormal_exit_count=native_abnormal,
                    engine_stability_qualified=False,
                    no_driving_or_world_model_qualification=True,
                    all_instrument_data_excluded_from_fitting=True,
                ),
            )
            status = "complete"
            progress(
                "Both child outcomes retained; containment is not engine or driving qualification"
            )
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--engine-identity", type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args)
        except BaseException as error:
            atomic_json(args.out / "worker_error.json", dict(error=str(error), at=time.time()))
            raise
    else:
        main(args)
