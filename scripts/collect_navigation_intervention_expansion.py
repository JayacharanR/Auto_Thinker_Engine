#!/usr/bin/env python3
"""Only63original unstarted option cases, with native bindings isolated per child."""

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.native_process_owner import run_native_worker  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def remaining_cases(geometry, pilot_request):
    excluded = {c["context_id"] for c in pilot_request["cases"]}
    if excluded != {"r35_lead", "r134_crossing", "r35_rear"}:
        raise ValueError("Retain all three original pilot contexts, including failed options")
    cases = []
    for context in geometry["contexts"]:
        if context["context_id"] in excluded:
            continue
        if context["route_id"] == 6:
            raise ValueError("Old observed diagnostic cannot become TRAIN")
        for branch in ("go", "brake", "wait_go"):
            cases.append(
                dict(
                    case_id=context["context_id"] + "_" + branch + "_repeat0",
                    context_id=context["context_id"],
                    branch=branch,
                    repeat=0,
                    role=context["role"],
                    town=context["town"],
                )
            )
    if len(cases) != 63 or sum(c["role"] == "train" for c in cases) != 45:
        raise ValueError("Exact frozen45TRAIN/18calibration remaining cases")
    return cases


def worker(args):
    import yaml

    from src.navigation.frozen_ego import FrozenExpertEgo
    from src.navigation.intervention_collection import collect_case
    from src.navigation.native_process_client import connect_owned_engine

    task = read(args.task_file)
    request = read(task["parent_request"])
    if digest(task["parent_request"]) != task["parent_request_sha256"]:
        raise ValueError("Parent recipe changed")
    geometry = read(ROOT / request["geometry_file"])
    context = next(c for c in geometry["contexts"] if c["context_id"] == task["case"]["context_id"])
    if task["case"] not in request["cases"]:
        raise ValueError("Undeclared worker case")
    for name, sha in request["source_hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Pinned runtime/config changed before child")
    model_path = ROOT / request["ego_descriptor"]
    config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
    server, client, check = connect_owned_engine(args.engine_identity, args.out)
    case = {k: task["case"][k] for k in ("case_id", "context_id", "branch", "repeat")}
    episode = args.out / "episode"

    def on_tick(samples, tick):
        atomic_json(
            args.out / "progress.json",
            dict(
                status="running",
                stage="Native factual action worker",
                frames=samples,
                target_frames=80,
                tick=tick,
                phase="Actual acknowledged actions, no model inference",
                updated_at=time.time(),
            ),
        )

    result = collect_case(
        server,
        client,
        case,
        context,
        episode,
        config,
        FrozenExpertEgo(model_path),
        model_path,
        check,
        on_tick,
    )
    client = None
    gc.collect()
    atomic_json(
        args.out / "child_complete.json",
        dict(
            case=task["case"],
            samples=result["samples"],
            source_role=context["role"],
            case_evidence=str(episode),
            imitation_eligible=False,
            all_native_handles_scoped_to_child=True,
        ),
    )
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            stage="Native factual action worker",
            frames=80,
            target_frames=80,
            phase="Native capture closed; independent audit pending",
            updated_at=time.time(),
        ),
    )


def main(args):
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    geometry_path = ROOT / "outputs/navigation_intervention_geometry_v2_20261020/plan.json"
    geometry = read(geometry_path)
    original = ROOT / "outputs/navigation_intervention_pilot_20261020"
    pilot = read(original / "request.json")
    cases = remaining_cases(geometry, pilot)
    preflight = read(ROOT / "outputs/navigation_process_isolation_probe_20261020/assessment.json")
    if not preflight["process_containment_verified"]:
        raise ValueError("Native child containment must verify before broader collection")
    # A contained engine failure is still reported, never counted as normal.
    (args.out / "tasks").mkdir()
    (args.out / "cases").mkdir()
    ego_path = ROOT / "outputs/navigation_models/frozen_expert_ego_20261020.json"
    paths = [
        Path(__file__),
        geometry_path,
        ego_path,
        ROOT / "configs/navigation.yaml",
        ROOT / "configs/navigation_vendor.json",
        ROOT / "reports/navigation_intervention_expansion_protocol_20261020.md",
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
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for p in paths:
        dst = args.out / "source" / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    atomic_json(
        args.out / "request.json",
        dict(
            cases=cases,
            source_hashes=hashes,
            geometry_file=str(geometry_path.relative_to(ROOT)),
            geometry_sha256=digest(geometry_path),
            ego_descriptor=str(ego_path.relative_to(ROOT)),
            original_pilot=str(original),
            original_data_plan=geometry["original_data_plan"],
            original_data_plan_sha256=geometry["original_data_plan_sha256"],
            maximum_native_wall_seconds=5400,
            maximum_artifact_bytes=4_000_000_000,
            planned_cases=63,
            train_cases=45,
            calibration_cases=18,
            no_failed_case_retry=True,
            containment_preflight_sha256=digest(
                ROOT / "outputs/navigation_process_isolation_probe_20261020/assessment.json"
            ),
            known_native_abnormal_shutdowns_disclosed=True,
            automatic_training=False,
            cleanup_may_finish_after_cutoff=True,
            no_benchmark_or_reward_change=True,
        ),
    )
    status, started, active = "running", time.monotonic(), None
    records = []
    last_space_check = 0.0

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Frozen multi-location action data",
                frames=len(records),
                target_frames=63,
                encounters_done=len(records),
                encounters_total=63,
                active_output=str(active) if active else None,
                captured=sum(
                    r["capture_receipt_present"] and r["child_returncode"] == 0 for r in records
                ),
                abnormal_native_exits=sum(r["native_returncode"] not in (0, -15) for r in records),
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def check():
        nonlocal last_space_check
        if (args.out / "STOP").exists() or time.monotonic() - started > 5400:
            raise InterruptedError("Owned stage STOPor5400sstartup/capture cap")
        if time.monotonic() - last_space_check > 2:
            last_space_check = time.monotonic()
            if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > 4_000_000_000:
                raise InterruptedError("Stage4GBartifact cap")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 10)
            from src.navigation.vendor import activate

            vendor = activate()
            progress("Exactly63frozen unstarted options; failures retained, no fitting")
            for index, case in enumerate(cases):
                check()
                if any(digest(ROOT / p) != sha for p, sha in hashes.items()):
                    raise ValueError("Pinned source changed before original next case")
                active = args.out / "cases" / case["case_id"]
                task = args.out / "tasks" / f"{index:03d}.json"
                atomic_json(
                    task,
                    dict(
                        case=case,
                        parent_request=str(args.out / "request.json"),
                        parent_request_sha256=digest(args.out / "request.json"),
                    ),
                )
                result = run_native_worker(
                    vendor["carla_root"],
                    active,
                    case["town"],
                    [
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--task-file",
                        str(task),
                    ],
                    check,
                    lambda state: progress(
                        "Frozen native child; acknowledged actions, no policy/model fit"
                    ),
                )
                receipt = active / "episode/outcome.json"
                result.update(
                    case=case,
                    physical_output=str(active),
                    capture_receipt_present=receipt.exists(),
                    raw_samples=read(receipt)["samples"] if receipt.exists() else None,
                    original_case_not_retried=True,
                )
                records.append(result)
                atomic_json(args.out / "records.json", records)
                failure_path = active / "worker_error.json"
                if failure_path.exists() and read(failure_path).get("type") in {
                    "AttributeError",
                    "TypeError",
                    "KeyError",
                    "ImportError",
                    "NameError",
                    "UnboundLocalError",
                    "FileNotFoundError",
                }:
                    raise RuntimeError(
                        "Child programming/configuration error; inspect before any next case"
                    )
                progress("Original next option settled; child/native errors remain in denominator")
            active = None
            atomic_json(
                args.out / "assessment.json",
                dict(
                    planned=63,
                    settled=len(records),
                    captured=sum(
                        r["capture_receipt_present"] and r["child_returncode"] == 0 for r in records
                    ),
                    child_failures=sum(r["child_returncode"] != 0 for r in records),
                    abnormal_native_exits=sum(
                        r["native_returncode"] not in (0, -15) for r in records
                    ),
                    whole_source_independent_audit_pending=True,
                    no_automatic_fit_or_driving_promotion=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status = "complete"
            progress("All63original options settled; independent data audit required, no promotion")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(
            args.out / "error.json",
            dict(error=str(error), at=time.time(), active_output=str(active)),
        )
        progress(str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--engine-identity", type=Path)
    parser.add_argument("--task-file", type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args)
        except BaseException as error:
            atomic_json(
                args.out / "worker_error.json",
                dict(error=str(error), type=type(error).__name__, at=time.time()),
            )
            raise
    else:
        main(args)
