#!/usr/bin/env python3
"""Audit DAgger labels/exposure without treating assisted arrivals as policy scores."""

import argparse
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_encounters import light_identity  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.dataset import shard_records  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def assess(root, decode_videos=False):
    root = Path(root)
    audit, request, guard = (
        read(root / name) for name in ("data_audit.json", "request.json", "guard.json")
    )
    if audit["status"] != "complete" or not audit["audit_passed"]:
        raise ValueError("Completed pixel/frame audit required")
    if digest(root / "guard.json") != request["guard_sha256"]:
        raise ValueError("Frozen correction guard changed")
    if any(digest(root / "source" / p) != h for p, h in request["source_hashes"].items()):
        raise ValueError("Collector source archive changed")
    vendor = activate()
    import carla

    xodr = Path(vendor["carla_root"]) / "CarlaUE4/Content/Carla/Maps/OpenDrive/Town03_Opt.xodr"
    world_map = carla.Map("Town03_Opt", xodr.read_text())
    allowed = set(guard["allowed_junctions"]["Town03_Opt"])
    child = root / "Town03_Opt"
    manifest = read(child / "manifest.json")
    if (
        not manifest["privileged_assistance"]
        or manifest["student"]["rule_overrides"]
        or digest(child / "policy.pt") != request["checkpoint_sha256"]
    ):
        raise ValueError("Wrong correction behavior/policy")
    sources = read(child / "source_hashes.json")
    if any(digest(child / "source" / p) != h for p, h in sources.items()):
        raise ValueError("Driving source archive changed")
    summary = read(child / "summary.json")
    catalog_dir = Path(read(child / "agent.json")["collection"]["catalog"])
    catalog = read(catalog_dir / "catalog.json")
    inventory_path = ROOT / catalog["inventory"] / "Town03_Opt.json"
    if digest(inventory_path) != catalog["inventory_sha256"]["Town03_Opt"]:
        raise ValueError("Signal inventory changed")
    inventory, runtime = read(inventory_path), read(child / "runtime_sign_inventory.json")
    entries = {e["route_id"]: e for e in request["episodes"]}
    counts, sources10, rows = Counter(), Counter(), []
    spatial_violations = []
    eligible_paths = []
    for entry in audit["episodes"]:
        data = Path(entry["path"])
        episode = data.parent
        meta, outcome, profile = (
            read(data / "episode.json"),
            read(data / "outcome.json"),
            read(episode / "collection_profile.json"),
        )
        rid = meta["route_id"]
        if (
            meta["mode"] != "dagger"
            or meta["internal_manifest_sha256"] != guard["internal_manifest_sha256"]
            or outcome["imitation_eligible"]
            or outcome["counterfactual_safe_label"]
        ):
            raise ValueError("Incorrect correction/imitation semantics")
        if meta["policy_checkpoint_sha256"] != request["checkpoint_sha256"]:
            raise ValueError("Episode source policy mismatch")
        if profile["control_differences"]:
            raise ValueError("Unexplained actuator difference")
        control_trace = {
            r["frame"]: r
            for r in map(json.loads, (episode / "dagger_trace.jsonl").read_text().splitlines())
        }
        setup_status = "no_setup_declared"
        if entries[rid]["scenario"]:
            started = [
                s
                for s in read(child / "scenario_setup" / f"{rid}.json").get("started", [])
                if s["type"] == entries[rid]["scenario"]
            ]
            setup_status = "setup_started" if len(started) == 1 else "setup_failed"
            if (
                setup_status == "setup_started"
                and entries[rid]["scenario"] == "NavigationSignalHold"
            ):
                setup_status = light_identity(
                    entries[rid], started[0]["ego_light_id"], runtime, inventory
                )["status"]
        physics_frames = 0
        for line in (episode / "trace.jsonl").read_text().splitlines():
            row = json.loads(line)
            wp = world_map.get_waypoint(carla.Location(*row["ego"][:3]))
            physics_frames += 1
            if wp.is_junction and wp.junction_id not in allowed:
                spatial_violations.append(
                    dict(route_id=rid, frame=row["frame"], junction_id=wp.junction_id)
                )
        eligible = bool(outcome.get("correction_labels_eligible"))
        if eligible:
            eligible_paths.append(str(data))
        local = Counter()
        for shard in sorted(data.glob("shard_*.zip")):
            with zipfile.ZipFile(shard) as archive:
                for record in shard_records(shard):
                    target = json.loads(archive.read(f"{record['prefix']}/targets.json"))
                    policy = json.loads(archive.read(f"{record['prefix']}/policy.json"))
                    facts = target["facts"]
                    trace = control_trace[target["frame"]]
                    for first, second in (
                        (target["expert_control"], trace["teacher_control"]),
                        (facts["student_requested_control"], trace["student_requested_control"]),
                        (facts["requested_control"], trace["selected_control"]),
                        (target["executed_control"], trace["selected_control"]),
                    ):
                        if not np.allclose(first, second, rtol=0, atol=1e-5):
                            raise ValueError(
                                "Teacher/student/selected/executed controls were conflated"
                            )
                    source = facts["selected_control_source"]
                    expected_control = (
                        target["expert_control"]
                        if source != "student"
                        else facts["student_requested_control"]
                    )
                    if not np.allclose(
                        expected_control, facts["requested_control"], rtol=0, atol=1e-5
                    ):
                        raise ValueError("Unreported behavior-policy selection")
                    if not facts["expert_label_valid"] or facts["counterfactual_safe_label"]:
                        raise ValueError("Invalid expert/counterfactual label")
                    light_id = facts["applicable_light_id"]
                    if light_id is not None:
                        light = next(a for a in facts["world"]["actors"] if a["id"] == light_id)
                        if light["signal"] != facts["next_light_state"]:
                            raise ValueError("Applicable light and same-frame world state disagree")
                    if eligible:
                        sources10[source] += 1
                        speed = facts["teacher_target_speed_mps"]
                        go = speed > 1 and target["expert_control"][2] < 0.5
                        stationary_go = go and policy["ego_motion"][0] < 0.2
                        local.update(
                            samples=1,
                            teacher_stop=int(speed < 0.1),
                            teacher_go=int(go),
                            stationary_go=int(stationary_go),
                            stationary_go_after_5s=int(stationary_go and policy["timestamp"] > 5),
                        )
        video = episode / "three_cameras.mp4"
        if decode_videos:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-xerror",
                    "-threads",
                    "2",
                    "-i",
                    str(video),
                    "-f",
                    "null",
                    "-",
                ],
                check=True,
                capture_output=True,
                timeout=180,
            )
        counts.update(local)
        counts.update(
            recovery_events=profile["dagger_control_ticks"].get("recovery_events", 0),
            privileged_ticks=profile["privileged_intervention_ticks"],
            physics_frames=physics_frames,
        )
        rows.append(
            dict(
                route_id=rid,
                correction_eligible=eligible,
                counts=dict(local),
                setup_status=setup_status,
                profile=profile,
                data=str(data),
                full_video_decode_passed=decode_videos,
            )
        )
    exposure = bool(
        counts["teacher_stop"]
        and counts["teacher_go"]
        and counts["recovery_events"]
        and counts["stationary_go_after_5s"] >= 30
        and sum(r["counts"].get("stationary_go_after_5s", 0) > 0 for r in rows) >= 2
    )
    setup_valid = len(rows) == len(entries) and all(
        r["setup_status"] in ("setup_started", "no_setup_declared") for r in rows
    )
    unknown_failures = [
        e["status"]
        for e in summary["episodes"]
        if "agent crashed" in e["status"].lower() or "couldn't be set up" in e["status"].lower()
    ]
    passed = (
        exposure
        and setup_valid
        and not spatial_violations
        and not unknown_failures
        and not summary["harness_crashed"]
        and summary["stop_reason"] is None
    )
    return dict(
        audit_passed=True,
        correction_data_gate_passed=passed,
        exposure_gate_passed=exposure,
        all_recorded_collisions=summary["all_collisions"],
        recorded_red_violations=sum(e["red_violations"] for e in summary["episodes"]),
        recorded_stop_violations=sum(e["stop_violations"] for e in summary["episodes"]),
        counts=dict(counts),
        correction_sources_10hz=dict(sources10),
        episodes=rows,
        spatial_violations=spatial_violations,
        unknown_failures=unknown_failures,
        eligible_correction_paths=eligible_paths,
        request_sha256=digest(root / "request.json"),
        data_audit_sha256=digest(root / "data_audit.json"),
        internal_manifest_sha256=guard["internal_manifest_sha256"],
        scope="Privileged assisted TRAIN data, not camera evaluation. Instantaneous expert "
        "corrections only; censored future/unsafe/executed-student actions are not labels.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decode-videos", action="store_true")
    args = parser.parse_args()
    result = assess(args.out, args.decode_videos)
    atomic_json(args.report, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("episodes", "eligible_correction_paths")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
