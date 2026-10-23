#!/usr/bin/env python3
"""Bind68verified recordings, preserving five unknown attempts in the73denominator."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    old = ROOT / "outputs/navigation_interventions_verified_20261020/manifest.json"
    audit = ROOT / "outputs/navigation_intervention_expansion_audit_20261020/assessment.json"
    replay = ROOT / "outputs/navigation_intervention_prefix_identity_20261021/assessment.json"
    parent = ROOT / "outputs/navigation_intervention_expansion_20261020"
    previous, data, comparisons = read(old), read(audit), read(replay)
    if (
        previous["accepted"] != 8
        or data["independently_verified"] != 60
        or data["rejected_or_unknown"] != 3
        or comparisons["original_audit_sha256"] != digest(audit)
        or comparisons["comparisons_count"] != 39
        or not comparisons["all_pass"]
        or not comparisons["all_exact"]
        or read(audit.parent / "progress.json")["status"] != "failed"
    ):
        raise ValueError(
            "Require unchanged original8/new60verified and39exact geometry comparisons"
        )
    with BudgetWatch(args.out):
        episodes = list(previous["episodes"])
        for row in data["records"]:
            source = Path(row["physical_output"])
            context = read(source / "case.json")["context"]
            metadata = read(source / "data/episode.json")
            if (
                row["source_role"] != context["role"]
                or metadata["partition"] != context["role"]
                or metadata["imitation_eligible"] is not False
                or context["route_id"] == 6
                or not all(
                    row[k]
                    for k in (
                        "all_pixel_hashes_verified",
                        "all15prospective_plans_reproduced",
                        "two_actual_prior_controls_verified",
                        "all_next_tick_controls_equal",
                    )
                )
            ):
                raise ValueError("Original roles, source exclusions and native proof required")
            paths = [
                source / name
                for name in (
                    "case.json",
                    "frozen_plan.json",
                    "actions.jsonl",
                    "world_trace.jsonl",
                    "native_boxes.json",
                    "outcome.json",
                    "data/episode.json",
                    "native.log",
                )
            ]
            event = row["prefix_support"].get("first_contact_tick")
            episodes.append(
                dict(
                    case=row["case"],
                    source_role=row["source_role"],
                    physical_output=str(source),
                    town=context["town"],
                    route_id=context["route_id"],
                    context_id=context["context_id"],
                    mechanism=context["mechanism"],
                    RGB_timestamps=80,
                    labeled_physics_ticks=160,
                    source_hashes={str(p): digest(p) for p in paths},
                    RGB_shards_sha256=row["source_shards_sha256"],
                    original_plan=read(source / "frozen_plan.json"),
                    prefix_support=row["prefix_support"],
                    first_event_index0=None if event is None else event - 1,
                    unfiltered_contact_starts=row["unfiltered_contact_starts"],
                    actual_geography=row["actual_geography"],
                    clean_imitation_eligible=False,
                    capture_receipt_before_original_deadline=(source / "outcome.json")
                    .stat()
                    .st_mtime
                    <= (parent / "request.json").stat().st_mtime + 5400,
                )
            )
        if len({e["case"]["case_id"] for e in episodes}) != 68:
            raise ValueError("Exactly68unique episodes; no option retry")
        for episode in episodes:
            for path, sha in episode["source_hashes"].items():
                if digest(path) != sha:
                    raise ValueError("Original native/plan/pose source changed")
            for path, sha in episode["RGB_shards_sha256"].items():
                if digest(path) != sha:
                    raise ValueError("Original RGBshard changed")
            if not episode["capture_receipt_before_original_deadline"]:
                raise ValueError("Capture outside original allowance")
        roles = {
            role: dict(
                episodes=sum(e["source_role"] == role for e in episodes),
                contacts=sum(
                    e["unfiltered_contact_starts"] for e in episodes if e["source_role"] == role
                ),
                independent_locations=len(
                    {(e["town"], e["route_id"]) for e in episodes if e["source_role"] == role}
                ),
            )
            for role in ("train", "calibration")
        }
        manifest = dict(
            schema="navigation-verified-action-interventions-v2",
            episodes=episodes,
            original_planned=73,
            accepted=68,
            rejected_or_unknown=previous["rejected_or_unknown"] + data["errors"],
            roles=roles,
            RGB_timestamps=5440,
            labeled_physics_ticks=10880,
            audit_sha256={str(p): digest(p) for p in (old, audit, replay)},
            all_original_failed_artifacts_preserved=True,
            static_correspondence_correction_reported=True,
            scripted_not_reactive=True,
            positive_red_examples=0,
            observed_unfiltered_contact_starts=sum(
                e["unfiltered_contact_starts"] for e in episodes
            ),
            unknown_native_recordings=5,
            model_fitting_not_performed=True,
            no_camera_driving_or_M5_qualification=True,
        )
        atomic_json(args.out / "manifest.json", manifest)
        shutil.copyfile(__file__, args.out / "curation_source.py")
        atomic_json(
            args.out / "assessment.json",
            dict(
                manifest_sha256=digest(args.out / "manifest.json"),
                accepted=68,
                rejected_or_unknown=5,
                roles=roles,
                all_original_sources_hash_verified=True,
            ),
        )
        atomic_json(
            args.out / "progress.json",
            dict(
                status="complete",
                stage="Verified factual action sources",
                frames=73,
                target_frames=73,
                accepted_sources=68,
                unknown_sources=5,
                phase="50TRAIN/18calibration; five unknowns retained; no fitting or promotion",
                updated_at=time.time(),
            ),
        )
        print(json.dumps(roles), flush=True)


if __name__ == "__main__":
    main()
