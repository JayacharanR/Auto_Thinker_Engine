#!/usr/bin/env python3
"""Closed-file audit of generic prospective action data, with both actors' geography."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_dense_labels import actual_geography  # noqa:E402
from scripts.audit_navigation_fixed_branches import matching_prefix  # noqa:E402
from src.navigation.candidate_controls import control_candidates  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa:E402
from src.navigation.frozen_ego import FrozenExpertEgo  # noqa:E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.replay import compare_world_prefixes  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def audit_case(episode, context, case, ego_path, data_plan_path):
    evidence = read(episode / "case.json")
    if evidence != dict(context=context, case=case):
        raise ValueError("Case or frozen geometry substituted after outcome")
    state, actions = lines(episode / "world_trace.jsonl"), lines(episode / "actions.jsonl")
    if (
        len(state) != 160
        or len(actions) != 160
        or [r["frame"] for r in state] != [r["frame"] for r in actions]
        or any(b["frame"] != a["frame"] + 1 for a, b in zip(state, state[1:]))
    ):
        raise ValueError("Exactly160 contiguous physical action/world frames required")
    hero = next(a["id"] for a in state[0]["actors"] if a["role"] == "hero")
    native, alignment = align_world_frames(vehicle_frames(episode / "native.log", hero), state)
    contacts = recorder_contacts(episode / "native.log")
    if not contacts["complete"] or contacts["town"] != context["town"]:
        raise ValueError("Native physical identity/recorder incomplete")
    for action in actions:
        if not np.allclose(
            native[action["frame"] + 1]["control"], action["requested_control"], atol=1e-5, rtol=0
        ):
            raise ValueError("Actual next-tick ego command differs")
    frames, branch_packet, shards = [], None, {}
    for shard in sorted((episode / "data").glob("shard_*.zip")):
        shards[str(shard)] = digest(shard)
        with zipfile.ZipFile(shard) as archive:
            for row in shard_records(shard):
                packet = load_policy_sample(shard, row)
                target = json.loads(archive.read(row["prefix"] + "/targets.json"))
                ack = target["facts"]["actuator_feedback_frame"]
                if (
                    ack != packet.frame + 1
                    or not np.allclose(
                        native[ack]["control"], target["executed_control"], atol=1e-5, rtol=0
                    )
                    or target["facts"]["imitation_eligible"] is not False
                ):
                    raise ValueError(
                        "RGB source action acknowledgment or non-imitation eligibility failed"
                    )
                frames.append(packet.frame)
                if packet.frame == actions[80]["frame"]:
                    branch_packet = packet
    if frames != [r["frame"] for r in state[::2]] or len(frames) != 80:
        raise ValueError("All80original RGB frames required")
    metadata = read(episode / "data/episode.json")
    if metadata["partition"] != context["role"] or metadata["imitation_eligible"] is not False:
        raise ValueError("Source role or non-imitation eligibility changed")
    plan = read(episode / "frozen_plan.json")
    frame = actions[80]["frame"]
    history = np.array([native[frame - 1]["control"], native[frame]["control"]])
    if (
        plan["branch_frame"] != frame
        or plan["branch_tick"] != 80
        or not plan["frozen_before_first_branch_command"]
        or digest(ego_path) != plan["frozen_ego_sha256"]
        or not np.allclose(history, plan["actual_previous_controls"], atol=1e-5, rtol=0)
        or plan["actual_previous_observation_frames"] != [frame - 1, frame]
        or branch_packet is None
        or not np.array_equal(branch_packet.navigation_waypoints, plan["navigation"])
        or abs(float(branch_packet.ego_motion[0]) - plan["speed_mps"]) > 1e-6
    ):
        raise ValueError("Prospective context/actual own history does not reproduce")
    slate = control_candidates(
        plan["navigation"],
        np.zeros(8),
        np.full(8, 5.0),
        plan["speed_mps"],
        plan["actual_previous_controls"],
        FrozenExpertEgo(ego_path),
    )
    if len(plan["slate"]) != 15:
        raise ValueError("Complete15candidate slate required")
    for calculated, stored in zip(slate, plan["slate"]):
        if (
            (calculated.desired.path_index, calculated.desired.speed_index)
            != (stored["path"], stored["speed"])
            or not np.array_equal(calculated.ordered_controls, stored["ordered_controls"])
            or not np.array_equal(calculated.forecast_states, stored["forecast_states"])
        ):
            raise ValueError("Stored15plans changed from prospective inputs")
    speed_index = {"go": 2, "brake": 0, "wait_go": 4}[case["branch"]]
    if plan["selected_path"] != 1 or plan["selected_speed"] != speed_index:
        raise ValueError("Wrong frozen candidate selected")
    selected = next(
        p for p in slate if p.desired.path_index == 1 and p.desired.speed_index == speed_index
    )
    if not np.array_equal(
        selected.ordered_controls, np.array([r["frozen_plan_control"] for r in actions[80:]])
    ):
        raise ValueError("Prospective commands were revised")
    first_native = min((r["frame"] for r in contacts["starts"]), default=None)
    contact_tick = first_native - native[frame]["frame"] if first_native is not None else None
    if contact_tick is not None and contact_tick <= 0:
        support = dict(
            first_episode_contact_precedes_branch=True,
            new_first_event_hazard_eligible=False,
            first_contact_tick=contact_tick,
        )
    else:
        support = matching_prefix(
            selected.ordered_controls,
            np.array([native[r["frame"] + 1]["control"] for r in actions[80:]]),
            [r["safety_intervention"] for r in actions[80:]],
            contact_tick,
        )
        support["new_first_event_hazard_eligible"] = True
    data_plan = read(data_plan_path)
    item = next(r for r in data_plan["episodes"] if r["route_id"] == context["route_id"])
    own_geo = actual_geography(state, item, data_plan)
    other_state = [
        dict(
            actors=[
                a | dict(role="hero")
                for a in row["actors"]
                if a["role"] == "intervention_" + context["mechanism"]
            ]
        )
        for row in state
    ]
    other_geo = actual_geography(other_state, item, data_plan)
    return dict(
        case=case,
        source_role=context["role"],
        rgb_samples=80,
        native_controls_verified=160,
        all_pixel_hashes_verified=True,
        all15prospective_plans_reproduced=True,
        two_actual_prior_controls_verified=True,
        all_next_tick_controls_equal=True,
        source_shards_sha256=shards,
        native_recording_sha256=contacts["sha256"],
        unfiltered_contact_starts=contacts["ego_contact_starts"],
        contact_times_s=[r["time_s"] for r in contacts["starts"]],
        prefix_support=support,
        actual_geography=dict(ego=own_geo, other=other_geo),
        alignment=alignment,
        no_driving_qualification=True,
    )


def audit(root, out):
    out.mkdir(parents=True, exist_ok=False)
    request = read(root / "request.json")
    if any(digest(root / "source" / path) != sha for path, sha in request["source_hashes"].items()):
        raise ValueError("Original source archive changed")
    geometry = read(root / "source" / request["geometry_file"])
    contexts = {c["context_id"]: c for c in geometry["contexts"]}
    records, errors, prefixes, repeats, comparisons = [], [], {}, {}, []
    ego_path = root / "source" / request["ego_descriptor"]
    data_plan_path = root / "source" / request["original_data_plan"]
    for case in request["cases"]:
        episode = root / "episodes" / case["case_id"]
        try:
            result = audit_case(
                episode, contexts[case["context_id"]], case, ego_path, data_plan_path
            )
            state = lines(episode / "world_trace.jsonl")
            prefix_key = case["context_id"]
            if prefix_key in prefixes:
                comparisons.append(
                    dict(
                        case_id=case["case_id"],
                        kind="same_context_prefix",
                        **compare_world_prefixes(prefixes[prefix_key], state[:80]),
                    )
                )
            else:
                prefixes[prefix_key] = state[:80]
            repeat_key = case["context_id"], case["branch"]
            if repeat_key in repeats:
                comparisons.append(
                    dict(
                        case_id=case["case_id"],
                        kind="same_option_repeat",
                        **compare_world_prefixes(repeats[repeat_key], state),
                    )
                )
            else:
                repeats[repeat_key] = state
            records.append(result)
        except Exception as error:
            errors.append(dict(case=case, error=str(error)))
        atomic_json(
            out / "progress.json",
            dict(
                status="running",
                stage="Intervention independent audit",
                frames=len(records) + len(errors),
                target_frames=len(request["cases"]),
                updated_at=__import__("time").time(),
            ),
        )
    result = dict(
        records=records,
        errors=errors,
        planned=len(request["cases"]),
        independently_verified=len(records),
        observable_comparisons=comparisons,
        all_registered_observable_comparisons_pass=bool(comparisons)
        and all(r["observable_prefix_within_tolerance"] for r in comparisons),
        unfiltered_contact_starts=sum(r["unfiltered_contact_starts"] for r in records),
        no_fitting_or_driving_promotion=True,
    )
    atomic_json(out / "assessment.json", result)
    atomic_json(
        out / "progress.json",
        dict(
            status="complete" if not errors else "failed",
            stage="Intervention independent audit",
            frames=len(request["cases"]),
            target_frames=len(request["cases"]),
            updated_at=__import__("time").time(),
        ),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.out.resolve())
