#!/usr/bin/env python3
"""Publish verified factual TRAINsources without turning the failed ten-case audit green."""

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
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    audit = ROOT / "outputs/navigation_intervention_audit_20261020"
    result = read(audit / "assessment.json")
    expected = {"r35_lead_brake_repeat0", "r134_crossing_wait_go_repeat0"}
    if (
        read(audit / "progress.json")["status"] != "failed"
        or result["verified"] != 8
        or result["original_planned"] != 10
        or not result["all_observable_comparisons_pass"]
        or {e["case"]["case_id"] for e in result["errors"]} != expected
        or any(
            "world_trace.jsonl" not in e["error"] or "No such file" not in e["error"]
            for e in result["errors"]
        )
    ):
        raise ValueError("Only original8verified/2knownzero-observation failures can be curated")
    with BudgetWatch(args.out):
        episodes = []
        original = ROOT / "outputs/navigation_intervention_pilot_20261020"
        deadline = (original / "request.json").stat().st_mtime + 900
        for row in result["records"]:
            source = Path(row["physical_output"])
            if row["source_role"] != "train" or row["case"]["context_id"].startswith("r6_"):
                raise ValueError("Only original TRAINroles; exclude old observed route6diagnostic")
            for path, sha in row["source_shards_sha256"].items():
                if digest(path) != sha:
                    raise ValueError("Verified original RGBshard changed")
            if digest(source / "native.log") != row["native_recording_sha256"]:
                raise ValueError("Verified native recording changed")
            if (
                not row["all15prospective_plans_reproduced"]
                or not row["all_next_tick_controls_equal"]
            ):
                raise ValueError("Causal plan/native actuator proof required")
            physics = read(source / "case.json")["context"]
            metadata = read(source / "data/episode.json")
            if metadata["imitation_eligible"] is not False or metadata["partition"] != "train":
                raise ValueError("Factual actions must never become clean expert imitation")
            support = row["prefix_support"]
            event_tick = support.get("first_contact_tick")
            validity = support.get("matching_commands_before_safety")
            if support["new_first_event_hazard_eligible"] and (
                validity is None
                or not 0 < validity <= 80
                or event_tick is not None
                and validity < event_tick
            ):
                raise ValueError("Observed prefix does not support declared first-event target")
            paths = [
                source / p
                for p in (
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
            episodes.append(
                dict(
                    case=row["case"],
                    source_role="train",
                    physical_output=str(source),
                    town=physics["town"],
                    route_id=physics["route_id"],
                    context_id=physics["context_id"],
                    mechanism=physics["mechanism"],
                    RGB_timestamps=80,
                    labeled_physics_ticks=160,
                    source_hashes={str(p): digest(p) for p in paths},
                    RGB_shards_sha256=row["source_shards_sha256"],
                    original_plan=read(source / "frozen_plan.json"),
                    prefix_support=support,
                    first_event_index0=None if event_tick is None else event_tick - 1,
                    unfiltered_contact_starts=row["unfiltered_contact_starts"],
                    actual_geography=row["actual_geography"],
                    clean_imitation_eligible=False,
                    capture_receipt_before_original_deadline=(source / "outcome.json")
                    .stat()
                    .st_mtime
                    <= deadline,
                )
            )
        if not all(e["capture_receipt_before_original_deadline"] for e in episodes):
            raise ValueError("Do not silently extend original native collection allowance")
        manifest = dict(
            schema="navigation-verified-action-interventions-v1",
            original_planned=10,
            accepted=8,
            rejected_or_unknown=result["errors"],
            episodes=episodes,
            calibration_episodes=0,
            RGB_timestamps=640,
            labeled_physics_ticks=1280,
            independent_contexts=3,
            independent_locations=2,
            repeated_go_is_not_independent=True,
            source_audit=str(audit / "assessment.json"),
            source_audit_sha256=digest(audit / "assessment.json"),
            original_audit_remains_failed=True,
            all_observable_comparisons_exact=all(
                all(v == 0 for v in c["maximum_differences"].values())
                for c in result["comparisons"]
            ),
            no_model_fit_or_driving_promotion=True,
            scripted_not_reactive=True,
            no_positive_rear_or_red_outcomes=True,
        )
        atomic_json(args.out / "manifest.json", manifest)
        shutil.copyfile(__file__, args.out / "curation_source.py")
        atomic_json(
            args.out / "assessment.json",
            dict(
                manifest_sha256=digest(args.out / "manifest.json"),
                all_included_sources_independently_verified=True,
                original_denominator=10,
                accepted=8,
                rejected=2,
                original_audit_remains_failed=True,
                no_model_promotion=True,
            ),
        )
        atomic_json(
            args.out / "progress.json",
            dict(
                status="complete",
                stage="Verified factual action sources",
                frames=10,
                target_frames=10,
                accepted_sources=8,
                unknown_sources=2,
                phase="8verified/2unknown; no dynamics fitting or promotion",
                updated_at=time.time(),
            ),
        )
        print(
            json.dumps(
                dict(accepted=8, unknown=2, manifest_sha256=digest(args.out / "manifest.json"))
            )
        )


if __name__ == "__main__":
    main()
