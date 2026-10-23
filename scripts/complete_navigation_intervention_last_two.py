#!/usr/bin/env python3
"""Only the final two original untouched cases, within the original native cap."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.frozen_ego import FrozenExpertEgo  # noqa:E402
from src.navigation.intervention_collection import collect_case  # noqa:E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    old = ROOT / "outputs/navigation_intervention_pilot_20261020"
    roots = [
        old,
        ROOT / "outputs/navigation_intervention_continuation_20261020",
        ROOT / "outputs/navigation_intervention_finish_20261020",
    ]
    request = read(old / "request.json")
    cases = request["cases"][8:]
    if (
        len(cases) != 2
        or any((p / "episodes" / c["case_id"]).exists() for p in roots for c in cases)
        or not read(roots[-1] / "servers/server_001/normal_cleanup_retry.json")[
            "normal_shutdown_finished"
        ]
    ):
        raise ValueError("Only2original untouched cases after verified STOPcleanup")
    allowed = {"src/navigation/native_cleanup.py", "src/navigation/intervention_collection.py"}
    if {p for p, sha in request["source_hashes"].items() if digest(ROOT / p) != sha} != allowed:
        raise ValueError("Only native cleanup lifecycle/timeout changes")
    if (ROOT / "src/navigation/intervention_collection.py").read_text().replace(
        "    client.set_timeout(20.0)\n", "", 1
    ) != (old / "source/src/navigation/intervention_collection.py").read_text():
        raise ValueError("No control or physics-equation changes")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "episodes").mkdir()
    paths = [ROOT / p for p in request["source_hashes"]] + [Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for p in paths:
        dst = args.out / "source" / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    deadline = (old / "request.json").stat().st_mtime + 900
    atomic_json(
        args.out / "request.json",
        request
        | dict(
            cases=cases,
            source_hashes=hashes,
            original_absolute_deadline=deadline,
            source_roots=[str(p) for p in roots],
            only2original_unstarted=True,
            original2failedcases_retained=True,
            native_lifecycle_change="Stop/destroy sensors before async; restore normal20sRPC",
        ),
    )
    status, active, current = "running", None, 0
    records = []

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Diverse causal-action TRAINpilot",
                frames=480 + sum(r["samples"] for r in records) + current,
                target_frames=800,
                encounters_done=8 + len(records),
                encounters_total=10,
                original_failed_cases=2,
                active_output=str(active) if active else None,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists() or time.time() > deadline:
            raise InterruptedError("Owned STOP/original900sdeadline")
        if (
            sum(
                p.stat().st_size
                for root in roots + [args.out]
                for p in root.rglob("*")
                if p.is_file()
            )
            > 600_000_000
        ):
            raise InterruptedError("Original shared600MBcap")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check()
            check_resources(args.out, 10)
            vendor = activate()
            geometry = read(ROOT / request["geometry_file"])
            contexts = {c["context_id"]: c for c in geometry["contexts"]}
            ego_path = ROOT / request["ego_descriptor"]
            model = FrozenExpertEgo(ego_path)
            config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
            for index, case in enumerate(cases):
                check()
                active = args.out / "episodes" / case["case_id"]
                current = 0
                server_out = args.out / "servers" / f"server_{index:03d}"
                server_out.mkdir(parents=True)
                (server_out / "STOP").symlink_to("../../STOP")
                progress("Only original2untouched cases; unchanged native allowance")
                server = CarlaServer(
                    vendor["carla_root"], server_out, town=contexts[case["context_id"]]["town"]
                )
                with server as client:

                    def on_tick(samples, tick):
                        nonlocal current
                        current = samples
                        if tick % 5 == 0:
                            progress("Native capture; full failed-attempt denominator retained")

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
                    records.append(result)
                    current = 0
                    atomic_json(args.out / "records.json", records)
            active = None
            status = "complete"
            progress(
                "All10settled;2unknownfailures retained; independent audit follows"
            )
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
