"""Predeclared showcase-video selection; every native scientific record remains."""

import json
from pathlib import Path

from src.navigation.dagger import digest

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "reports/navigation_local_storage_plan_20261020.json"


def frozen_showcase_policy(plan_path=PLAN):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("user_approved") is not True
        or plan["requested_additional_artifact_cap_bytes"] != 14_000_000_000
        or plan["minimum_free_bytes_proposed"] != 10_000_000_000
        or plan["m6_estimate"]["full_preview_routes"] != [0, 4, 8]
        or plan["m6_estimate"]["full_preview_seed"] != 3031
        or plan["m6_estimate"]["full_preview_both_arms"] is not True
        or plan["m6_estimate"]["all_native_recorders_and_traces_retained"] is not True
    ):
        raise ValueError("Use the exact user-approved recording-only compact plan")
    return dict(
        schema="navigation-showcase-selection-v1",
        plan=str(plan_path), plan_sha256=digest(plan_path),
        full_preview_route_ids=[0, 4, 8], full_preview_seeds=[3031],
        native_recordings_all_missions=True, complete_traces_all_missions=True,
        sensor_and_policy_cadence_unchanged=True,
        expert_and_student_same_selection=True,
        existing_preview_codec_and_fps_unchanged=True,
        selection_independent_of_driving_outcomes=True,
    )


def preview_enabled(manifest, current_route):
    """Only the registered final suite uses selection; all M3 videos stay full."""
    if manifest.get("final_benchmark") is not True:
        return True
    registration = manifest.get("final_registration") or {}
    policy = registration.get("recording_policy")
    if not policy or policy != frozen_showcase_policy(policy["plan"]):
        raise ValueError("Final preview selection must match the frozen approved registration")
    if manifest.get("rule_test") or manifest.get("training_collection"):
        raise ValueError("Controlled rule/training episodes cannot use selective final recording")
    return (
        int(current_route["route_id"]) in policy["full_preview_route_ids"]
        and current_route["seed"] in policy["full_preview_seeds"]
    )
