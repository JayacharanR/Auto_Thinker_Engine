#!/usr/bin/env python3
"""Enforce the approved local capacity budget using only owned queue STOP files.

The budget uses net filesystem free-space consumption from one fixed baseline;
unrelated writes also count conservatively. No retained artifact is removed.
Existing source-registered jobs retain their own stricter resource checks.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.runtime import RunLock, atomic_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned_root(path):
    path = Path(path).resolve()
    outputs = (ROOT / "outputs").resolve()
    if not path.is_relative_to(outputs) or len(path.relative_to(outputs).parts) < 1:
        raise ValueError("STOP targets must be explicitly registered project output roots")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.out = owned_root(args.out)
    plan = read(args.plan)
    if plan.get("user_approved") is not True or not plan.get("approval_text"):
        raise ValueError("The compact local plan needs the user's explicit reserve decision")
    if args.out.exists():
        if not args.resume:
            raise FileExistsError("Retain the existing fixed budget baseline; resume explicitly")
        request = read(args.out / "request.json")
        if request["source_sha256"] != digest(__file__):
            raise ValueError("Resume only the same guardian source")
        if request["plan_sha256"] != digest(args.plan):
            raise ValueError("Approved budget plan changed; preserve the original baseline")
    else:
        if args.resume:
            raise FileNotFoundError("No registered storage baseline to resume")
        args.out.mkdir(parents=True)
        shutil.copyfile(__file__, args.out / "guard_source.py")
        baseline = shutil.disk_usage(ROOT).free
        request = dict(
            schema="navigation-storage-budget-v1",
            plan=str(args.plan.resolve()),
            plan_sha256=digest(args.plan),
            source_sha256=digest(__file__),
            baseline_free_bytes=baseline,
            additional_cap_bytes=plan["requested_additional_artifact_cap_bytes"],
            minimum_free_bytes=plan["minimum_free_bytes_proposed"],
            effective_floor_bytes=max(
                plan["minimum_free_bytes_proposed"],
                baseline - plan["requested_additional_artifact_cap_bytes"],
            ),
            check_headroom_bytes=64 * 1024 * 1024,
            budget_is_net_filesystem_consumption=True,
            unrelated_writes_count_conservatively=True,
            existing_job_guards_unchanged=True,
            registered_at=time.time(),
        )
        atomic_json(args.out / "request.json", request)
        atomic_json(args.out / "watched_roots.json", dict(roots=[]))
    with RunLock(ROOT / "outputs/.navigation_storage_guard.lock"):
        while True:
            free = shutil.disk_usage(ROOT).free
            used = max(0, request["baseline_free_bytes"] - free)
            shortfall = free < request["effective_floor_bytes"] + request["check_headroom_bytes"]
            stopped = []
            if shortfall:
                for item in read(args.out / "watched_roots.json")["roots"]:
                    target = owned_root(item)
                    state = target / "progress.json"
                    if state.exists() and read(state).get("status") in ("running", "starting"):
                        (target / "STOP").touch()
                        stopped.append(str(target))
                atomic_json(
                    args.out / "shortfall.json",
                    dict(
                        free_bytes=free,
                        net_consumed_bytes=used,
                        stopped_owned_roots=stopped,
                        stop_mechanism="Only touch <registered root>/STOP; never signal processes",
                        occurred_at=time.time(),
                    ),
                )
            status = "budget_reached" if shortfall else "running"
            if (args.out / "STOP").exists():
                status = "stopped"
            atomic_json(
                args.out / "progress.json",
                dict(
                    status=status,
                    stage="Approved local storage budget",
                    free_bytes=free,
                    net_consumed_bytes=used,
                    cap_bytes=request["additional_cap_bytes"],
                    minimum_free_bytes=request["minimum_free_bytes"],
                    effective_floor_bytes=request["effective_floor_bytes"],
                    phase="Existing job limits still apply; scientific artifacts retained",
                    updated_at=time.time(),
                ),
            )
            if args.once or status != "running":
                print(json.dumps(dict(status=status, free_bytes=free, net_consumed_bytes=used)))
                return
            time.sleep(5)


if __name__ == "__main__":
    main()
