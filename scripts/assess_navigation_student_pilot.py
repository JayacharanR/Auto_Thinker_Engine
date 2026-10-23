#!/usr/bin/env python3
"""Audit the frozen TRAIN-only driver pilot, preserving every raw-attempt collision."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_encounters import light_identity  # noqa: E402
from src.navigation.metrics import summarize_records, wilson_interval  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_evidence_issue(episode):
    """A crashed case stays in the denominator even without finalized artifacts."""
    required = ("profile.json", "trace.jsonl", "world_trace.jsonl", "three_cameras.mp4")
    missing = [name for name in required if not (episode / name).is_file()]
    empty = [
        name for name in required if name not in missing and (episode / name).stat().st_size == 0
    ]
    return dict(missing=missing, empty=empty) if missing or empty else None


def spatial_audit(output):
    """Offline centre-position check against the unchanged junction assignment."""
    from src.navigation.vendor import activate

    vendor = activate()
    import carla

    xodr = Path(vendor["carla_root"]) / "CarlaUE4/Content/Carla/Maps/OpenDrive/Town03_Opt.xodr"
    world_map = carla.Map("Town03_Opt", xodr.read_text())
    assignments = ROOT / "configs/encounters/location_split_v1/locations.json"
    allowed = set(read(assignments)["assignments"]["Town03_Opt"]["train"])
    counts, violations = {}, []
    for trace in sorted(Path(output).glob("*/episodes/*/trace.jsonl")):
        candidate = trace.parents[2].name
        counts.setdefault(candidate, 0)
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            waypoint = world_map.get_waypoint(carla.Location(*row["ego"][:3]))
            counts[candidate] += 1
            if waypoint.is_junction and waypoint.junction_id not in allowed:
                violations.append(
                    dict(trace=str(trace), frame=row["frame"], junction_id=waypoint.junction_id)
                )
    return dict(
        timestamps=counts,
        violations=violations,
        assignments_sha256=sha(assignments),
        map_sha256=sha(xodr),
        scope="Stored ego centre mapped to nearest driving waypoint; "
        "not proof of visual/geographic independence.",
    )


def assess(output, previous=(), decode_videos=False):
    output = Path(output)
    protocol = read(output / "protocol.json")
    request = read(output / "request.json")
    catalog_path = ROOT / "configs/encounters/location_split_v1/train/catalog.json"
    if sha(catalog_path) != protocol["catalog_sha256"]:
        raise ValueError("Frozen TRAIN catalog changed")
    if sha(ROOT / protocol["route"]) != protocol["route_sha256"]:
        raise ValueError("Frozen pilot route changed")
    catalog = read(catalog_path)
    expected = {int(e["route_id"]): e for e in protocol["encounters"]}
    reference = ROOT / catalog["inventory"] / "Town03_Opt.json"
    if sha(reference) != catalog["inventory_sha256"]["Town03_Opt"]:
        raise ValueError("Signal inventory changed")
    inventory = read(reference)
    candidates = {}
    for candidate in request["candidates"]:
        name = candidate["name"]
        run = output / name
        manifest = read(run / "manifest.json")
        if (
            manifest["seed"] != protocol["seed"]
            or not manifest["seed_per_route"]
            or {int(r) for r in manifest["routes_subset"].split(",")} != set(expected)
        ):
            raise ValueError(f"Pilot route/seed mismatch for {name}")
        if candidate["checkpoint"]:
            if (
                manifest["student"]["rule_overrides"]
                or sha(run / "policy.pt") != candidate["sha256"]
            ):
                raise ValueError(f"Policy/override mismatch for {name}")
        sources = read(run / "source_hashes.json")
        if any(sha(run / "source" / p) != h for p, h in sources.items()):
            raise ValueError(f"Archived source changed for {name}")
        summary = read(run / "summary.json")
        records = read(run / "leaderboard.json")["_checkpoint"]["records"]
        measured = summarize_records(
            records, execution_failure=bool(summary.get("stop_reason")), last_route_only=True
        )
        route_ids = [int(r["route_id"].split("_")[1]) for r in records]
        if len(route_ids) != len(set(route_ids)) or set(route_ids) - set(expected):
            raise ValueError(f"Duplicated/unexpected pilot outcomes for {name}")
        missing_native_routes = sorted(set(expected) - set(route_ids))
        measured.update(
            native_record_count=len(records), count=len(expected),
            unobserved_outcomes=[dict(route_id=rid, status="unknown_failure_not_executed")
                                 for rid in missing_native_routes],
            safe_completion_wilson95=wilson_interval(measured["safe_completed"], len(expected)),
        )
        setups = {}
        runtime = read(run / "runtime_sign_inventory.json")
        for rid, entry in expected.items():
            if rid in missing_native_routes:
                setups[rid] = dict(status="not_executed_unknown_failure")
                continue
            if not entry["scenario"]:
                setups[rid] = {"status": "no_setup_declared"}
                continue
            evidence = read(run / "scenario_setup" / f"{rid}.json")
            started = [s for s in evidence.get("started", []) if s["type"] == entry["scenario"]]
            if len(started) != 1:
                setups[rid] = {"status": "setup_failed", "evidence": evidence}
            elif entry["scenario"] == "NavigationSignalHold":
                setups[rid] = light_identity(entry, started[0]["ego_light_id"], runtime, inventory)
            else:
                setups[rid] = {"status": "setup_started"}
        profiles, incomplete_episodes, episode_routes = [], [], []
        for episode in sorted((run / "episodes").iterdir()):
            rid = int(read(episode / "route_validation.json")["route_id"])
            episode_routes.append(rid)
            if rid not in expected:
                raise ValueError(f"Unexpected episode route: {episode}")
            issue = episode_evidence_issue(episode)
            if issue:
                incomplete_episodes.append(dict(
                    episode=str(episode), route_id=rid, **issue,
                    counted_as_evidence_failure=True,
                    native_record=next(
                        r for r in records if int(r["route_id"].split("_")[1]) == rid
                    ),
                ))
                continue
            profile = read(episode / "profile.json")
            if not profile["all_camera_frames_aligned"] or profile.get("intervention_ticks", 0):
                raise ValueError(f"Camera alignment/intervention failure: {episode}")
            trace = [
                json.loads(line) for line in (episode / "trace.jsonl").read_text().splitlines()
            ]
            if len(trace) != profile["frames"] or any(
                b["frame"] != a["frame"] + 1 for a, b in zip(trace, trace[1:])
            ):
                raise ValueError(f"Missing/repeated trace frames: {episode}")
            if candidate["checkpoint"] and any(
                row["rule_overrides_enabled"] or row["interventions"] for row in trace
            ):
                raise ValueError(f"Unexpected rule intervention in trace: {episode}")
            trace_summary = {
                "frames": len(trace),
                "brake_ticks": sum(row["control"][2] > 0.5 for row in trace),
                "zero_speed_ticks": sum(row["speed_mps"] < 0.1 for row in trace),
                "expert_hazard_ticks": {
                    k: sum(row.get("expert_hazards", {}).get(k, False) for row in trace)
                    for k in ("light", "stop", "vehicle")
                }
                if not candidate["checkpoint"]
                else None,
                "last_state": {
                    k: v
                    for k, v in trace[-1].items()
                    if k in ("speed_mps", "target_speed_mps", "brake_probability", "control")
                },
            }
            if expected[rid]["scenario"] == "NavigationSignalHold":
                evidence = read(run / "scenario_setup" / f"{rid}.json")
                light_id = next(
                    s["ego_light_id"]
                    for s in evidence["started"]
                    if s["type"] == "NavigationSignalHold"
                )
                states = []
                with (episode / "world_trace.jsonl").open() as stream:
                    for line in stream:
                        world = json.loads(line)
                        states.append(
                            next(a["signal"] for a in world["actors"] if a["id"] == light_id)
                        )
                setups[rid]["observed_signal_frames"] = dict(Counter(states))
                if "Red" not in states:
                    setups[rid]["status"] = "declared_red_not_observed"
            video = episode / "three_cameras.mp4"
            info = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,pix_fmt,width,height,nb_frames",
                        "-of",
                        "json",
                        str(video),
                    ]
                )
            )["streams"][0]
            if (info["codec_name"], info["pix_fmt"], info["width"], info["height"]) != (
                "h264",
                "yuv420p",
                1536,
                288,
            ):
                raise ValueError(f"Unexpected video format: {video}")
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
                    timeout=180,
                    capture_output=True,
                )
            profiles.append(
                dict(
                    episode=str(episode),
                    profile=profile,
                    trace_summary=trace_summary,
                    video=info,
                    full_video_decode_passed=decode_videos,
                )
            )
        if len(set(episode_routes)) != len(episode_routes):
            raise ValueError(f"Duplicate episode routes for {name}")
        missing_episode_routes = sorted(set(expected) - set(episode_routes))
        recorders = list((run / "recordings").glob("*.log"))
        recorder_valid = len(recorders) == len(records) and all(
            b"CARLA_RECORDER" in p.open("rb").read(30) for p in recorders
        )
        console_clean = b"\0" not in (output / f"{name}.log").read_bytes()
        if not recorder_valid or not console_clean:
            raise ValueError(f"Recorder/console corruption for {name}")
        candidates[name] = dict(
            **measured,
            execution_status=summary["execution_status"],
            setup_audit=setups,
            profiles=profiles,
            incomplete_episodes=incomplete_episodes,
            missing_episode_routes=missing_episode_routes,
            missing_native_routes=missing_native_routes,
            stop_reason=summary.get("stop_reason"),
            complete_episode_evidence=not incomplete_episodes
            and not missing_episode_routes and not missing_native_routes,
            source_archive_verified=True,
            checkpoint_sha256=candidate["sha256"],
            distinct_recorders_verified=True,
        )
    # Every recorded collision is counted, including a stopped infrastructure attempt.
    raw_attempts = []
    for attempt in [*map(Path, previous), output]:
        for path in sorted(attempt.glob("*/leaderboard.json")):
            records = read(path).get("_checkpoint", {}).get("records", [])
            s = summarize_records(records)
            raw_attempts.append(
                dict(attempt=str(path.parent), count=s["count"], all_collisions=s["all_collisions"])
            )
    spatial = spatial_audit(output)
    return dict(
        scope="Six fixed TRAIN-location engineering diagnostics; not a held-out benchmark, "
        "generalization estimate or training dataset. Every raw-attempt collision reported.",
        protocol=protocol,
        candidates=candidates,
        raw_attempts=raw_attempts,
        all_attempt_collisions=sum(a["all_collisions"] for a in raw_attempts),
        spatial_audit=spatial,
        audit_passed=not spatial["violations"]
        and all(
            c["execution_status"] == "evaluation_complete"
            and c["complete_episode_evidence"]
            and all(
                s["status"] in ("setup_started", "no_setup_declared")
                for s in c["setup_audit"].values()
            )
            for c in candidates.values()
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--previous", action="append", type=Path, default=[])
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--decode-videos", action="store_true")
    args = parser.parse_args()
    result = assess(args.out, args.previous, args.decode_videos)
    atomic_json(args.report, result)
    print(
        json.dumps(
            {
                "audit_passed": result["audit_passed"],
                "all_attempt_collisions": result["all_attempt_collisions"],
                "candidates": {
                    n: {
                        k: c[k]
                        for k in (
                            "count",
                            "safe_completed",
                            "rule_safe_completed",
                            "all_collisions",
                        )
                    }
                    for n, c in result["candidates"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
