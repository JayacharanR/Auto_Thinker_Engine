#!/usr/bin/env python3
"""Verify frozen prospective plans against context, native actuators and first contacts."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_branches import audit as factual_audit  # noqa:E402
from src.navigation.candidate_controls import control_candidates  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa:E402
from src.navigation.frozen_ego import FrozenExpertEgo  # noqa:E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def matching_prefix(planned, applied, interventions, first_contact_tick):
    if planned.shape != (80, 3) or applied.shape != planned.shape or len(interventions) != 80:
        raise ValueError("Exactly80 prospective and applied commands required")
    if not np.isfinite(planned).all() or not np.isfinite(applied).all():
        raise ValueError("Finite controls required")
    first = next((i for i, value in enumerate(interventions) if value is not None), 80)
    if first_contact_tick is not None and not 0 <= first_contact_tick <= 80:
        raise ValueError("Contact before branch or beyond its observed horizon")
    if first < 80 and (first_contact_tick is None or first < first_contact_tick):
        raise ValueError("A privileged intervention preceded the actual first contact")
    if not np.allclose(planned[:first], applied[:first], atol=1e-5, rtol=0):
        raise ValueError("Proposed and applied commands differ before the declared safety stop")
    return dict(matching_commands_before_safety=first,
                tail_censored=first < 80, first_contact_tick=first_contact_tick,
                future_latent_progress_after_intervention_unknown=first < 80,
                absorbing_first_collision_supported=first_contact_tick is not None)


def audit(root, out):
    request = read(root / "request.json")
    if request.get("freeze_commands_before_outcome") is not True:
        raise ValueError("Do not relabel a feedback-control collection as prospective plans")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "audit_source.py")
    factual_audit(root, out / "factual")
    base = read(out / "factual/assessment.json")
    if base["errors"]:
        raise ValueError("Closed native/pixel/actuation audit failed; preserve its errors")
    for path in ("src/navigation/candidate_controls.py", "src/navigation/trajectory_candidates.py",
                 "src/navigation/frozen_ego.py", "src/navigation/controller.py"):
        if digest(ROOT / path) != request["source_hashes"][path]:
            raise ValueError("Restore the registered proposal equations before reproduction")
    descriptor = root / "source/outputs/navigation_models/frozen_expert_ego_20261020.json"
    if digest(descriptor) != request["frozen_ego_sha256"]:
        raise ValueError("Archived ego parameters changed")
    model = FrozenExpertEgo(descriptor)
    records, errors = [], []
    for row in base["records"]:
        episode = root / row["name"]
        try:
            plan = read(episode / "frozen_plan.json")
            actions = lines(episode / "actions.jsonl")
            world = lines(episode / "world_trace.jsonl")
            hero = next(a["id"] for a in world[0]["actors"] if a["role"] == "hero")
            native, alignment = align_world_frames(
                vehicle_frames(episode / "native.log", hero), world
            )
            frame = actions[80]["frame"]
            if (plan["branch_frame"] != frame or plan["branch_tick"] != 80
                    or plan["frozen_before_first_branch_command"] is not True):
                raise ValueError("Prospective plan does not bind the decision frame")
            history = np.array([native[frame - 1]["control"], native[frame]["control"]])
            if not np.allclose(history, plan["actual_previous_controls"], atol=1e-5, rtol=0):
                raise ValueError("Proposal history is not two previously applied native commands")
            packet = None
            for shard in sorted((episode / "data").glob("shard_*.zip")):
                for record in shard_records(shard):
                    if record["frame"] == frame:
                        packet = load_policy_sample(shard, record)
            if (packet is None
                    or not np.array_equal(packet.navigation_waypoints, plan["navigation"])
                    or abs(float(packet.ego_motion[0]) - plan["speed_mps"]) > 1e-6):
                raise ValueError("Proposal context differs from the current policy packet")
            slate = control_candidates(plan["navigation"], np.zeros(8), np.full(8, 5.),
                                       plan["speed_mps"], plan["actual_previous_controls"], model)
            if len(plan["slate"]) != 15:
                raise ValueError("Complete shared15candidate slate required")
            for calculated, stored in zip(slate, plan["slate"]):
                if ((calculated.desired.path_index, calculated.desired.speed_index)
                        != (stored["path"], stored["speed"])
                        or not np.array_equal(
                            calculated.ordered_controls, stored["ordered_controls"])
                        or not np.array_equal(
                            calculated.forecast_states, stored["forecast_states"])):
                    raise ValueError("Frozen plan does not reproduce from declared causal inputs")
            name = row["name"].split("_", 1)[1]
            expected_speed = {"go": 2, "brake": 0, "wait_go": 4}[name]
            if plan["selected_path"] != 1 or plan["selected_speed"] != expected_speed:
                raise ValueError("Outcome-dependent candidate substitution")
            selected = next(p for p in slate if p.desired.path_index == 1
                            and p.desired.speed_index == expected_speed)
            logged = np.array([r["frozen_plan_control"] for r in actions[80:]])
            if not np.array_equal(logged, selected.ordered_controls):
                raise ValueError("Per-tick original proposals were rewritten after the decision")
            applied = np.array([native[r["frame"] + 1]["control"] for r in actions[80:]])
            contacts = recorder_contacts(episode / "native.log")
            first_native = min((r["frame"] for r in contacts["starts"]), default=None)
            first_tick = first_native - native[frame]["frame"] if first_native is not None else None
            evidence = matching_prefix(selected.ordered_controls, applied,
                                       [r["safety_intervention"] for r in actions[80:]], first_tick)
            records.append(dict(name=row["name"], plan_sha256=digest(episode / "frozen_plan.json"),
                                history_native_verified=True, declared_context_verified=True,
                                all15plans_reproduced=True, prefix=evidence,
                                unfiltered_contact_starts=contacts["ego_contact_starts"],
                                alignment=alignment))
        except Exception as error:
            errors.append(dict(name=row["name"], error=str(error)))
    result = dict(
        records=records, errors=errors, collection_request_sha256=digest(root / "request.json"),
        factual_audit_sha256=digest(out / "factual/assessment.json"),
        executable_prefixes_verified=len(records) == 6 and not errors,
        replay_observables_verified=base["all_registered_observable_comparisons_pass"],
        unfiltered_all_attempt_contact_starts=base["unfiltered_contact_starts"],
        calibration_qualified=False, planner_qualified=False, no_model_updates=True,
        limitation="Stationary TRAINlead only. No reactive crossing traffic, red positives, "
                   "generalization, complete15way observed ranking, or model-calibration claim.",
    )
    atomic_json(out / "assessment.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
