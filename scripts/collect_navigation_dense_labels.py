#!/usr/bin/env python3
"""Freeze a five-town TRAIN label corpus, then collect bounded canonical episodes."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.label_collection import validate_label_guard  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def prepare(out):
    catalog = ROOT / "configs/encounters/location_split_v1/train"
    guard_path = (
        ROOT / "outputs/navigation_spatial_data_repaired_20261019/actual_training_guard.json"
    )
    guard = read(guard_path)
    cases = sorted(
        (ROOT / "outputs/navigation_rule_followthrough_v3_20261020/certified_suite").glob(
            "*/case.json"
        )
    )
    if len(cases) != 24:
        raise ValueError("Preserve all24current rule-location reservations")
    reserved = {}
    for path in cases:
        case = read(path)
        reserved.setdefault(case["town"], set()).update(case["crossed_junction_ids"])
    entries = []
    priorities = [
        "following_stopped_vehicle",
        "signal_stop",
        "signal_release_memory",
        "stop_sign",
        "crossing_yield",
        "route_junction",
    ]
    for town in sorted(guard["allowed_junctions"]):
        pool = [
            e
            for e in read(catalog / "catalog.json")["encounters"]
            if e["town"] == town
            and e["route_id"] in guard["geometry_eligibility"]["kept_route_ids"]
            and set(e["crossed_junction_ids"])
            <= set(guard["allowed_junctions"][town]) - reserved.get(town, set())
        ]
        pool.sort(key=lambda e: (priorities.index(e["family"]), e["route_id"]))
        chosen = []
        junctions = set()
        families = set()
        for entry in pool:
            if entry["junction_id"] not in junctions and entry["family"] not in families:
                chosen.append(entry)
                junctions.add(entry["junction_id"])
                families.add(entry["family"])
        for entry in pool:
            if entry["junction_id"] not in junctions:
                chosen.append(entry)
                junctions.add(entry["junction_id"])
        needed = 3 if town in ("Town03_Opt", "Town04_Opt") else 2
        if len(chosen) < needed:
            raise ValueError("Insufficient strict TRAINlocations; never relax geography")
        selected = chosen[:needed]
        if town == "Town04_Opt":
            signs = [e for e in chosen if e["family"] == "stop_sign"]
            if not signs:
                raise ValueError("A genuine strict TRAINstop-sign encounter is required")
            if signs[0] not in selected:
                selected.append(signs[0])
        for index, entry in enumerate(selected):
            role = "calibration" if town in ("Town03_Opt", "Town04_Opt") and index == 2 else "train"
            entries.append(
                dict(
                    route_id=entry["route_id"],
                    town=town,
                    role=role,
                    entry=entry,
                    seed=62100 + entry["route_id"],
                    rgb_timestamp_cap=200,
                    maximum_wall_seconds=180,
                )
            )
    if len(entries) != 13 or len({e["town"] for e in entries}) != 5:
        raise ValueError("Expected13predeclared episodes across all5TRAINtowns")
    plan = dict(
        schema="navigation-dense-label-collection-v1",
        catalog=str(catalog),
        catalog_sha256=digest(catalog / "catalog.json"),
        actual_training_guard=str(guard_path),
        actual_training_guard_sha256=digest(guard_path),
        rule_case_sha256={str(p): digest(p) for p in cases},
        reserved_rule_junctions={t: sorted(ids) for t, ids in reserved.items()},
        episodes=entries,
        label_hz=2,
        rgb_hz=10,
        maximum_wall_seconds=1800,
        maximum_artifact_bytes=3_000_000_000,
        calibration_note="New dense-label heads/first-event model calibration "
        "is location-separated "
        "before collection. Existing frozen encoders/old imitation models have "
        "TRAINvideo/label exposure here; not unseen-encoder or untouched benchmark.",
        no_training_or_model_selection=True,
        no_reruns_after_crash=True,
        canonical_background_and_scenarios=True,
        charge_all_future_supervised_timestamps=True,
        benchmark_and_policy_inputs_unchanged=True,
        selection="Fixed family priority then routeID; unique locations first. Third Town03/04 "
        "location reserved for calibration, add first actual Town04stop encounter.",
    )
    atomic_json(out / "plan.json", plan)
    for item in entries:
        validate_label_guard(out / "plan.json", catalog, item["town"], str(item["route_id"]))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-prepared", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.prepare_only == args.execute_prepared:
        raise ValueError("Choose prepare-only or execute-prepared")
    if args.prepare_only:
        args.out.mkdir(parents=True, exist_ok=False)
        plan = prepare(args.out)
        sources = [
            Path(__file__),
            ROOT / "scripts/run_navigation.py",
            ROOT / "scripts/navigation_label_rig.py",
            ROOT / "reports/navigation_dense_label_collection_protocol_20261020.md",
            ROOT / "configs/navigation.yaml",
            *sorted((ROOT / "src/navigation").glob("*.py")),
        ]
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
        for source in sources:
            target = args.out / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        atomic_json(
            args.out / "request.json",
            dict(
                plan_sha256=digest(args.out / "plan.json"),
                source_hashes=hashes,
                protocol_sha256=digest(
                    ROOT / "reports/navigation_dense_label_collection_protocol_20261020.md"
                ),
            ),
        )
        atomic_json(
            args.out / "progress.json",
            dict(
                status="prepared",
                stage="TRAIN dense-label collection",
                encounters_done=0,
                encounters_total=len(plan["episodes"]),
                updated_at=time.time(),
            ),
        )
        print(json.dumps([(e["town"], e["route_id"], e["role"]) for e in plan["episodes"]]))
        return
    plan = read(args.out / "plan.json")
    request = read(args.out / "request.json")
    if read(args.out / "progress.json")["status"] != "prepared" or (args.out / "episodes").exists():
        raise ValueError("Execute exactly one prepared generation; never restart from stale state")
    if digest(args.out / "plan.json") != request["plan_sha256"] or any(
        digest(ROOT / p) != h for p, h in request["source_hashes"].items()
    ):
        raise ValueError("Prepared collection sources or immutable plan changed")
    (args.out / "episodes").mkdir()
    status, done, active = "running", 0, None
    started = time.monotonic()
    results = []

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="TRAIN dense-label collection",
                encounters_done=done,
                encounters_total=len(plan["episodes"]),
                active_output=str(active) if active else None,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            for item in plan["episodes"]:
                if (args.out / "STOP").exists():
                    raise InterruptedError("Dense-label collection STOP")
                if time.monotonic() - started > plan["maximum_wall_seconds"]:
                    raise InterruptedError("Dense-label1800s cap; unstarted sources remain unknown")
                used = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
                if used > plan["maximum_artifact_bytes"] - 250_000_000:
                    raise InterruptedError("Dense-label3GBartifact budget; preserve every source")
                if any(digest(ROOT / p) != h for p, h in request["source_hashes"].items()):
                    raise ValueError("Source-frozen label collection changed between episodes")
                active = args.out / "episodes" / f"{item['town']}_{item['route_id']}_{item['role']}"
                progress(
                    "Canonical privileged TRAINcollection; RGBpolicy/label shards remain separate"
                )
                command = [
                    sys.executable,
                    "-u",
                    "scripts/run_navigation.py",
                    "expert",
                    "--route",
                    str(Path(plan["catalog"]) / f"{item['town']}.xml"),
                    "--routes-subset",
                    str(item["route_id"]),
                    "--collection-catalog",
                    plan["catalog"],
                    "--collection-guard",
                    str(args.out / "plan.json"),
                    "--training-dense-labels",
                    "--collection-max-samples",
                    str(item["rgb_timestamp_cap"]),
                    "--seed",
                    str(item["seed"]),
                    "--max-wall",
                    str(item["maximum_wall_seconds"]),
                    "--out",
                    str(active),
                ]
                with (args.out / f"{item['route_id']}.log").open("x") as log:
                    child = subprocess.Popen(
                        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                    )
                    while child.poll() is None:
                        if (
                            (args.out / "STOP").exists()
                            or time.monotonic() - started > 1800
                            or sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
                            > 3e9
                        ):
                            if active.exists():
                                (active / "STOP").touch()
                        progress(
                            "Dense labels2Hz / original RGB10Hz; no labels supplied to controller"
                        )
                        time.sleep(2)
                labels = [
                    dict(path=str(p), sha256=digest(p), **read(p))
                    for p in active.glob("episodes/*/label_profile.json")
                ]
                contacts = []
                errors = []
                for p in sorted((active / "recordings").glob("*.log")):
                    try:
                        contacts.append(recorder_contacts(p))
                    except Exception as error:
                        errors.append(dict(path=str(p), error=str(error)))
                result = dict(
                    item=item,
                    output=str(active),
                    exit_code=child.returncode,
                    label_profiles=labels,
                    native_summary=read(active / "summary.json")
                    if (active / "summary.json").exists()
                    else None,
                    contacts=contacts,
                    contact_errors=errors,
                    label_timestamps=sum(p["label_timestamps"] for p in labels),
                    unfiltered_contact_starts=sum(r["ego_contact_starts"] for r in contacts),
                    contact_count_lower_bound=bool(errors)
                    or not contacts
                    or any(not r["complete"] for r in contacts),
                )
                results.append(result)
                done += 1
                atomic_json(
                    args.out / "assessment.json",
                    dict(
                        results=results,
                        planned_episodes=len(plan["episodes"]),
                        dataset_not_yet_independently_audited=True,
                        training_started=False,
                        no_driving_claim=True,
                    ),
                )
                if child.returncode or not labels or result["label_timestamps"] < 2:
                    raise RuntimeError(
                        "Collection infrastructure/label failure preserved; inspect, no case retry"
                    )
            status = "complete"
            progress("All13original label attempts terminal; independent data audit still required")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(
            args.out / "error.json",
            dict(error=str(error), at=time.time(), remaining_unstarted_unknown=True),
        )
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
