#!/usr/bin/env python3
"""Bounded expert/camera TRAIN integration after the approved feedback validation."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.apply_navigation_compact_generation import live_dependencies  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402

AFTER = ROOT / "outputs/navigation_ego_feedback_activation_20261020"
ROUTE = ROOT / "outputs/navigation_train_integration_geometry_20261020/route.xml"
FIT = ROOT / "outputs/navigation_repaired_study_v4_20261020/fits/resnet_f100_seed42"


def read(path):
    return json.loads(Path(path).read_text())


def actual_junction_audit(output, route_manifest):
    """Post-drive geography audit only; never an intervention or policy input."""
    from src.navigation.vendor import activate

    vendor = activate()
    import carla

    town = route_manifest["town"]
    path = Path(vendor["carla_root"]) / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
    if digest(path) != route_manifest["xodr_sha256"]:
        raise ValueError("Strict TRAIN geometry map changed")
    world_map = carla.Map(town, path.read_text())
    forbidden, frames = [], 0
    for trace in sorted((output / "episodes").glob("*/world_trace.jsonl")):
        for row in map(json.loads, trace.read_text().splitlines()):
            ego = next(a for a in row["actors"] if a["role"] == "hero")
            waypoint = world_map.get_waypoint(carla.Location(*ego["pose"][:3]))
            if (
                waypoint.is_junction
                and waypoint.junction_id not in route_manifest["allowed_junctions"]
            ):
                forbidden.append(dict(frame=row["frame"], junction=waypoint.junction_id))
            frames += 1
    return dict(
        observed_frames=frames,
        unexpected_reserved_junction_frames=forbidden,
        actual_junction_ids_within_train=bool(frames) and not forbidden,
        centerline_clearance_only_prevalidated=True,
        all_native_outcomes_retained_even_if_geography_invalid=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = read(ROUTE.with_suffix(".json"))
    checkpoint = FIT / "best_policy.pt"
    review = read(FIT / "review.json")
    if (
        digest(ROUTE) != manifest["route_sha256"]
        or digest(checkpoint) != review["policy_sha256"]
        or read(FIT / "progress.json")["status"] != "complete"
        or not manifest["native_route_interpolation_verified"]
        or not 1000 <= manifest["length_m"] <= 1500
    ):
        raise ValueError("Freeze the existing selected checkpoint and strictly TRAIN route")
    # Pin the exact future bytes already tested/approved, rather than mutating
    # or weakening the current running comparison's source identity.
    expected = dict(
        read(ROOT / "outputs/navigation_matched_native_completion_20261020/request.json")[
            "inference_source_sha256"
        ]
    )
    generation = read(ROOT / "outputs/navigation_feedback_generation_20261020/generation.json")
    for path, identities in generation["targets"].items():
        if path.startswith("src/navigation/") or path in expected:
            expected[path] = identities["after_sha256"]
    atomic_json(
        args.out / "request.json",
        dict(
            after=str(AFTER),
            route=str(ROUTE),
            route_sha256=digest(ROUTE),
            route_manifest_sha256=digest(ROUTE.with_suffix(".json")),
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint),
            selected_step=review["selected_step"],
            future_inference_source_sha256=expected,
            runner_sha256=digest(__file__),
            seed=62071,
            maximum_wall_seconds_per_arm=900,
            rule_overrides=False,
            role="Familiar TRAINengineering, not heldout/final benchmark or safety qualification",
            candidate_selection="Post-hoc engineering reference "
            "from the first two completed matched "
            "TRAINpilots: ResNet6/6arrivals/0contacts/2rulefailures. Frozen now; "
            "do not reselect after DINO/CARLA/no-color outcomes.",
            original_m4_comma_primary_replaced=False,
            route_selection_uses_agent_outcomes=False,
            no_training=True,
            raw_contacts_and_failures_all_retained=True,
            no_automatic_promotion=True,
        ),
    )
    shutil.copyfile(__file__, args.out / "runner_source.py")
    status, active, done = "running", None, 0

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="TRAIN expert/camera long integration",
                phase=phase,
                encounters_done=done,
                encounters_total=2,
                active_output=str(active) if active else None,
                updated_at=time.time(),
            ),
        )

    try:
        with BudgetWatch(args.out):
            while True:
                if (args.out / "STOP").exists():
                    raise InterruptedError("TRAINintegration STOP while queued")
                path = AFTER / "progress.json"
                if path.exists() and read(path)["status"] not in ("starting", "running"):
                    if read(path)["status"] != "complete":
                        raise RuntimeError(
                            "Approved feedback validation failed/stopped; inspect first"
                        )
                    if not live_dependencies([AFTER]):
                        break
                progress("Waiting for matched pilots and approved feedback native validation")
                time.sleep(3)
            with RunLock(ROOT / "outputs/.navigation.lock"):
                results = []
                for arm in ("expert", "student"):
                    if (args.out / "STOP").exists():
                        raise InterruptedError("TRAINintegration STOP before next arm")
                    if any(digest(ROOT / p) != h for p, h in expected.items()):
                        raise ValueError("Registered future inference source differs")
                    if (
                        digest(ROUTE) != manifest["route_sha256"]
                        or digest(checkpoint) != review["policy_sha256"]
                    ):
                        raise ValueError("Registered TRAIN route/camera parent changed")
                    active = args.out / arm
                    progress(f"{arm} follows the frozen1.376km TRAINroute; all failures retained")
                    command = [
                        sys.executable,
                        "-u",
                        "scripts/run_navigation.py",
                        arm,
                        "--route",
                        str(ROUTE),
                        "--seed",
                        "62071",
                        "--max-wall",
                        "900",
                        "--out",
                        str(active),
                    ]
                    if arm == "student":
                        command.extend(["--checkpoint", str(checkpoint)])
                    with (args.out / f"{arm}.log").open("x") as log:
                        child = subprocess.Popen(
                            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                        )
                        while child.poll() is None:
                            if (args.out / "STOP").exists() and active.exists():
                                (active / "STOP").touch()
                            progress(f"{arm} long TRAINdrive; no rule/emergency overrides")
                            time.sleep(2)
                    summary = (
                        read(active / "summary.json")
                        if (active / "summary.json").exists()
                        else None
                    )
                    contacts = [
                        recorder_contacts(p) for p in sorted((active / "recordings").glob("*.log"))
                    ]
                    result = dict(
                        arm=arm,
                        exit_code=child.returncode,
                        native_summary=summary,
                        recorders=contacts,
                        unfiltered_contact_starts=sum(r["ego_contact_starts"] for r in contacts),
                        lower_bound=not contacts or any(not r["complete"] for r in contacts),
                        geography=actual_junction_audit(active, manifest),
                    )
                    results.append(result)
                    atomic_json(
                        args.out / "assessment.json",
                        dict(
                            results=results,
                            driving_qualification=False,
                            comparison_with_old_town03_invalid=True,
                            interpretation="One familiar TRAINroute with one junction. "
                            "No urban-rule mastery, untouched-evaluation claim, "
                            "M4primary reselection or M6qualification.",
                        ),
                    )
                    done += 1
                    if child.returncode:
                        raise RuntimeError("Native infrastructure failure retained; no retries")
                    if (args.out / "STOP").exists():
                        raise InterruptedError("TRAINintegration stopped; retain observed outcomes")
                status = "complete"
                progress("Both TRAINarms terminal; all outcomes retained without model promotion")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
