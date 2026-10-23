#!/usr/bin/env python3
"""Guarded, outcome-blind full-approach TRAIN collection with coverage audit."""

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_data import audit  # noqa: E402
from scripts.collect_navigation import collection_error, committed_samples  # noqa: E402
from src.navigation.approach_data import (  # noqa: E402
    approach_coverage,
    reserve_components,
    select_approaches,
)
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.dataset import shard_records  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.splits import collection_geometry_eligibility  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def prepare(out, pilot, empty_background=False, varied_waits=False, light_gap=None):
    if light_gap is not None and (not empty_background or not 6 <= light_gap <= 16):
        raise ValueError("Light gap requires guarded empty TRAIN scenes and6–16m")
    catalog = ROOT / "configs/encounters/location_split_v1/train"
    base_path = ROOT / "reports/navigation_m2_dagger_guard_20261016.json"
    component = ROOT / "outputs/navigation_m2_signal_corpus_v2_20261016/plan.json"
    base = read(base_path)
    allowed, reserved = reserve_components(base, read(component)["cases"])
    additional = {
        t: sorted(set(base["reserved_internal_junctions"][t]) | set(reserved.get(t, [])))
        for t in allowed
    }
    geometry = collection_geometry_eligibility(catalog, additional_reserved=additional)
    guard = dict(
        kind="continuous-approach-train-v1",
        base_guard=str(base_path),
        base_guard_sha256=digest(base_path),
        component_plan=str(component),
        component_plan_sha256=digest(component),
        allowed_junctions=allowed,
        reserved_component_junctions=reserved,
        geometry_eligibility=geometry,
    )
    entries = select_approaches(read(catalog / "catalog.json"), allowed, geometry["kept_route_ids"])
    if pilot:
        entries = [
            next(e for e in entries if e["town"] == town) for town in ("Town01_Opt", "Town03_Opt")
        ]
        if empty_background:
            entries = [e for e in entries if e["town"] == "Town03_Opt"]
    schedule = [
        dict(entry=e, seed=202610180 + index * 100 + repetition)
        for index, e in enumerate(entries)
        for repetition in range(1 if pilot else 3)
    ]
    if varied_waits:
        if not empty_background:
            raise ValueError("Variable waits require the no-lead TRAIN scene")
        for index, case in enumerate(schedule):
            case["stationary_seconds"] = 7 if pilot else (3, 7, 11)[index % 3]
    plan = dict(
        kind="Continuous signal-approach TRAIN collection",
        pilot=pilot,
        background="empty" if empty_background else "canonical",
        varied_waits=varied_waits,
        training_light_gap_m=light_gap,
        catalog=str(catalog),
        catalog_sha256=digest(catalog / "catalog.json"),
        guard_sha256=None,
        schedule=schedule,
        no_eligible_signal_towns=sorted(
            set(read(catalog / "catalog.json")["towns"]) - {e["town"] for e in entries}
        ),
        maximum_wall_seconds=900 if pilot else 5400,
        per_episode_maximum_wall_seconds=300,
        scope="TRAIN data only; no student, new state-label release or driving-benchmark change",
    )
    if out.exists():
        raise FileExistsError("Fresh approach output required; old attempts must be preserved")
    check_resources(out, 25)
    atomic_json(out / "guard.json", guard)
    plan["guard_sha256"] = digest(out / "guard.json")
    atomic_json(out / "plan.json", plan)
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/run_navigation.py",
        ROOT / "scripts/audit_navigation_data.py",
        ROOT / "scripts/collect_navigation.py",
        ROOT / "reports/navigation_m2_approach_protocol_20261018.md",
        ROOT / "reports/navigation_m2_empty_approach_protocol_20261018.md",
        ROOT / "reports/navigation_m2_varied_wait_protocol_20261018.md",
        ROOT / "configs/navigation.yaml",
        *sorted((ROOT / "src/navigation").glob("*.py")),
    ]
    if light_gap is not None:
        sources.append(ROOT / "reports/navigation_m2_light_gap_protocol_20261019.md")
    for path in sources:
        destination = out / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    atomic_json(
        out / "request.json",
        dict(
            plan_sha256=digest(out / "plan.json"),
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        ),
    )
    return plan


def coverage_audit(out):
    result = audit(out)
    episodes = []
    for episode in result["episodes"]:
        data, rows = Path(episode["path"]), []
        for shard in sorted(data.glob("shard_*.zip")):
            with zipfile.ZipFile(shard) as archive:
                for record in shard_records(shard):
                    prefix = record["prefix"]
                    policy, label = (
                        json.loads(archive.read(f"{prefix}/{name}.json"))
                        for name in ("policy", "targets")
                    )
                    f = label["facts"]
                    if read(out / "plan.json")["background"] == "empty" and any(
                        a["type"].startswith("vehicle.") and a["role"] != "hero"
                        for a in f["world"]["actors"]
                    ):
                        raise ValueError("No-lead curriculum contains an unexpected vehicle")
                    rows.append(
                        dict(
                            frame=policy["frame"],
                            timestamp=policy["timestamp"],
                            ego_speed=policy["ego_motion"][0],
                            teacher_speed=f["teacher_target_speed_mps"],
                            light_id=f["applicable_light_id"],
                            light_state=f["next_light_state"],
                            light_distance=f["next_light_distance_m"],
                            light_hazard=f["teacher_hazards"]["light_hazard"],
                        )
                    )
        coverage = approach_coverage(rows)
        clean = bool(episode["outcome"] and episode["outcome"]["imitation_eligible"])
        meta = read(data / "episode.json")
        if meta.get("strict_training_guard_sha256") != digest(out / "guard.json"):
            raise ValueError("Recorded strict training guard differs from frozen plan")
        episodes.append(
            dict(
                route_id=episode["route_id"],
                town=episode["town"],
                path=str(data),
                clean_expert_eligible=clean,
                coverage=coverage,
                approach_imitation_eligible=clean and coverage["continuous_signal_sequence"],
            )
        )
    result.update(
        continuous_approach_episodes=episodes,
        eligible_approach_episodes=sum(e["approach_imitation_eligible"] for e in episodes),
        all_episode_failures_retained=True,
        visibility_certified=False,
        state_labels_released=False,
        driving_policy_changed=False,
        plan_sha256=digest(out / "plan.json"),
        guard_sha256=digest(out / "guard.json"),
    )
    atomic_json(out / "data_audit.json", result)
    return result


def run(out, prepare_only=False):
    if not out.exists():
        raise ValueError("Run --prepare first; inspect tests/plan before simulator launch")
    plan, request = read(out / "plan.json"), read(out / "request.json")
    if (
        digest(out / "plan.json") != request["plan_sha256"]
        or digest(out / "guard.json") != plan["guard_sha256"]
        or any(
            digest(ROOT / p) != h or digest(out / "source" / p) != h
            for p, h in request["source_hashes"].items()
        )
    ):
        raise ValueError("Frozen approach plan or source changed")
    if prepare_only:
        print(json.dumps(plan, indent=2))
        return
    if (out / "progress.json").exists():
        raise ValueError("Attempt already launched; use a new plan, never restart blindly")
    started, outcomes, errors = time.monotonic(), [], 0
    total = len(plan["schedule"])

    def progress(done, current=None, status="running"):
        atomic_json(
            out / "progress.json",
            dict(
                status=status,
                stage="M2 approach collection",
                encounters_done=done,
                encounters_total=total,
                current_chunk=current,
                frames=committed_samples(out),
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    for index, case in enumerate(plan["schedule"]):
        if (out / "STOP").exists() or time.monotonic() - started >= plan["maximum_wall_seconds"]:
            break
        entry, child_out = case["entry"], out / f"episode_{index:05d}"
        remaining = plan["maximum_wall_seconds"] - (time.monotonic() - started)
        if remaining < 90:
            break  # Do not knowingly start a new episode with only a few seconds left.
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/run_navigation.py"),
            "expert",
            "--route",
            str(Path(plan["catalog"]) / f"{entry['town']}.xml"),
            "--collection-catalog",
            plan["catalog"],
            "--collection-guard",
            str(out / "guard.json"),
            "--routes-subset",
            str(entry["route_id"]),
            "--seed",
            str(case["seed"]),
            "--max-wall",
            str(min(remaining, plan["per_episode_maximum_wall_seconds"])),
            "--out",
            str(child_out),
        ]
        if plan["background"] == "empty":
            command.append("--training-empty-background")
        if "stationary_seconds" in case:
            command += ["--training-signal-wait", str(case["stationary_seconds"])]
        if plan.get("training_light_gap_m") is not None:
            command += ["--training-light-gap", str(plan["training_light_gap_m"])]
        progress(index, str(child_out))
        with (out / f"episode_{index:05d}.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                if (out / "STOP").exists() or time.monotonic() - started >= plan[
                    "maximum_wall_seconds"
                ]:
                    if child_out.exists():
                        (child_out / "STOP").touch()
                progress(index, str(child_out))
                time.sleep(2)
        error = collection_error(child_out, child.returncode)
        outcomes.append(
            dict(
                route_id=entry["route_id"],
                town=entry["town"],
                seed=case["seed"],
                exit_code=child.returncode,
                infrastructure_error=error,
                out=str(child_out),
            )
        )
        atomic_json(out / "outcomes.json", outcomes)
        errors = errors + 1 if error else 0
        if error == "partition_violation" or errors >= 3:
            break
    status = "complete" if len(outcomes) == total and errors < 3 else "stopped"
    if any(e["infrastructure_error"] == "partition_violation" for e in outcomes) or errors >= 3:
        status = "failed"
    result = coverage_audit(out) if committed_samples(out) else None
    progress(len(outcomes), status=status)
    print(
        json.dumps(
            dict(
                status=status,
                episodes=len(outcomes),
                eligible_approaches=result["eligible_approach_episodes"] if result else 0,
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--empty-background", action="store_true")
    parser.add_argument("--varied-waits", action="store_true")
    parser.add_argument("--light-gap", type=float)
    args = parser.parse_args()
    out = args.out.resolve()
    with RunLock(ROOT / "outputs/.approach_collection.lock"):
        if args.prepare:
            prepare(out, args.pilot, args.empty_background, args.varied_waits, args.light_gap)
        run(out, args.prepare)


if __name__ == "__main__":
    main()
