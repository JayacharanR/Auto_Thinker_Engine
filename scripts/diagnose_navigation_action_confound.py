#!/usr/bin/env python3
"""Read-only audit of collision context and reaction controls in the old factual corpus."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = ROOT / "outputs/navigation_dynamics_sources_repaired_20261019/manifest.json"
    windows = ROOT / "outputs/navigation_dynamics_windows_20261019"
    source = json.loads(sources.read_text())
    request = json.loads((windows / "request.json").read_text())
    manifest = json.loads((windows / "manifest.json").read_text())
    if request["source_manifest_sha256"] != digest(sources):
        raise ValueError("Use only the original repaired-source factual windows")
    if digest(windows / "windows.npz") != manifest["window_arrays_sha256"]:
        raise ValueError("Original factual arrays changed")
    values = dict(np.load(windows / "windows.npz", allow_pickle=False))
    results = []
    for episode_id, entry in enumerate(source["accepted"]):
        if not entry["contacts"]["ego_contact_starts"]:
            continue
        data = Path(entry["data"])
        path = data.parent / "world_trace.jsonl"
        if digest(path) != entry["world_trace_sha256"]:
            raise ValueError("Original world trace changed")
        world = {r["frame"]: r for r in map(json.loads, path.read_text().splitlines())}
        first = min(entry["contacts"]["starts"], key=lambda r: r["frame"])
        first_world = min(world) + first["frame"] - 1 - entry["alignment"]["offset"]
        present = world[first_world]["actors"]
        ego = next(a for a in present if a["role"] == "hero")
        other_id = first["actor2"] if first["actor1"] == ego["id"] else first["actor1"]
        other = next((a for a in present if a["id"] == other_id), None)
        relative = None
        if other is not None:
            yaw = np.deg2rad(ego["pose"][4])
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            relative = ((np.array(other["pose"][:2]) - ego["pose"][:2]) @ rotation).tolist()
        meta = manifest["episodes"][episode_id]
        if entry["samples"] != meta["samples"] or entry["last_rgb_frame"] - entry[
            "first_rgb_frame"
        ] != 2 * (meta["samples"] - 1):
            raise ValueError("Original10Hz episode mapping differs")
        positive = np.flatnonzero(
            (values["episode_id"] == episode_id)
            & (((values["outcomes"][..., 0] == 1) & (values["cost_valid"][..., 0] > 0)).any(1))
        )
        horizons, input_after_first, post_event_actions, post_event_brake = 0, 0, 0, 0
        pre_event_brake = 0
        for i in positive:
            frame = int(entry["first_rgb_frame"] + 2 * (values["input_index"][i] - meta["start"]))
            for horizon in np.flatnonzero(
                (values["outcomes"][i, :, 0] == 1) & (values["cost_valid"][i, :, 0] > 0)
            ):
                horizons += 1
                if frame >= first_world:
                    input_after_first += 1
                    continue
                tick, end = first_world - frame, 10 * (int(horizon) + 1)
                pre_event_brake += int((values["controls"][i, : min(tick, end), 2] > 0).any())
                if tick < end:
                    post_event_actions += 1
                    post_event_brake += int((values["controls"][i, tick:end, 2] > 0).any())
        results.append(
            dict(
                episode_id=episode_id,
                role=entry["dynamics_role"],
                mode=entry["mode"],
                source_data=str(data),
                total_contact_starts=entry["contacts"]["ego_contact_starts"],
                declared_privileged_interventions=len(entry["interventions"]),
                first_contact_world_frame=first_world,
                first_contact_native_frame=first["frame"],
                ego_speed_mps=float(np.linalg.norm(ego["velocity"])),
                other_actor_type=other["type"] if other else None,
                other_actor_xy_in_ego=relative,
                collision_positive_windows=len(positive),
                collision_positive_horizons=horizons,
                positive_horizons_with_input_at_or_after_first_contact=input_after_first,
                positive_horizons_containing_post_first_contact_actions=post_event_actions,
                those_containing_positive_brake=post_event_brake,
                pre_first_contact_horizons_containing_positive_brake=pre_event_brake,
                world_trace_sha256=digest(path),
            )
        )
    totals = {
        key: sum(row[key] for row in results)
        for key in (
            "collision_positive_windows",
            "collision_positive_horizons",
            "positive_horizons_with_input_at_or_after_first_contact",
            "positive_horizons_containing_post_first_contact_actions",
            "those_containing_positive_brake",
            "pre_first_contact_horizons_containing_positive_brake",
        )
    }
    if totals["collision_positive_horizons"] != manifest["collision_positive_horizons"]:
        raise ValueError("Audit did not cover every original collision-positive horizon")
    shutil.copyfile(__file__, args.out / "audit_source.py")
    result = dict(
        episodes=results,
        totals=totals,
        source_manifest_sha256=digest(sources),
        window_arrays_sha256=digest(windows / "windows.npz"),
        reporting_source_sha256=digest(__file__),
        no_model_or_target_changes=True,
        shortcut_causal_explanation_proven=False,
        limitation="Observed action/outcome association only; later braking can reflect expert "
        "reaction or traffic context. No causal attribution or filtering of headline contacts.",
    )
    atomic_json(args.out / "assessment.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
