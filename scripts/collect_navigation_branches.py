#!/usr/bin/env python3
"""Small factual TRAIN action branches with same-frame RGB and executed commands."""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.candidate_controls import control_candidates  # noqa:E402
from src.navigation.collection import (  # noqa:E402
    CollectionRecorder,
    RouteProjector,
    training_position,
)
from src.navigation.contracts import SensorPacket  # noqa:E402
from src.navigation.controller import track  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import EpisodeWriter  # noqa:E402
from src.navigation.frozen_ego import FrozenExpertEgo  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.replay import capture_world_state, compare_world_prefixes  # noqa:E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.sensors import CameraRig  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def selected_entry():
    catalog = json.loads(
        (ROOT / "configs/encounters/location_split_v1/train/catalog.json").read_text()
    )
    guard = json.loads(
        (ROOT / "outputs/navigation_m2_varied_waits_20261018/guard.json").read_text()
    )
    prototypes = json.loads(
        (ROOT / "outputs/navigation_rule_cases_v3_20261019/request.json").read_text()
    )["chosen"]
    excluded = {(e["town"], j) for e in prototypes.values() for j in e["crossed_junction_ids"]}
    choices = [
        e
        for e in catalog["encounters"]
        if e["family"] == "following_stopped_vehicle"
        and e["route_id"] in guard["geometry_eligibility"]["kept_route_ids"]
        and set(e["crossed_junction_ids"]) <= set(guard["allowed_junctions"][e["town"]])
        and not {(e["town"], j) for j in e["crossed_junction_ids"]} & excluded
    ]
    entry = min(choices, key=lambda e: e["route_id"])
    if entry["route_id"] != 6 or entry["town"] != "Town01_Opt":
        raise ValueError("Prospectively registered branch-pilot geometry changed")
    return entry, guard


def branch_target(name, seconds):
    if name not in ("go", "brake", "wait_go") or not 0 <= seconds < 4.0 + 1e-6:
        raise ValueError("Unknown fixed branch or time outside its4s horizon")
    return 0.0 if name == "brake" or (name == "wait_go" and seconds < 2) else 5.0


def frozen_branch_plan(navigation, speed, actual_previous_controls, name, ego_model):
    if name not in ("go", "brake", "wait_go"):
        raise ValueError("Only the registered three center-path instrumentation options")
    slate = control_candidates(navigation, np.zeros(8), np.full(8, 5.0), speed,
                               actual_previous_controls, ego_model)
    selected = next(p for p in slate if p.desired.path_index == 1
                    and p.desired.speed_index == {"go": 2, "brake": 0, "wait_go": 4}[name])
    for plan in slate:
        plan.ordered_controls.setflags(write=False)
    return slate, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--after-references", type=Path)
    parser.add_argument("--resume-unstarted", action="store_true")
    parser.add_argument("--reset-signal-clocks", action="store_true")
    parser.add_argument("--freeze-commands", action="store_true",
                        help="New TRAINpilot: freeze shared candidate commands before branching")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.freeze_commands and (not args.reset_signal_clocks or args.resume_unstarted):
        raise ValueError("Frozen commands require repaired clocks and a fresh prospective pilot")
    if args.out.exists() and not args.resume_unstarted:
        raise FileExistsError("Preserve every branch-pilot attempt")
    args.out.mkdir(parents=True, exist_ok=args.resume_unstarted)
    entry, guard = selected_entry()
    if args.reset_signal_clocks:
        actual = (
            ROOT / "outputs/navigation_spatial_data_repaired_20261019/actual_training_guard.json"
        )
        guard = json.loads(actual.read_text())
        if not set(entry["crossed_junction_ids"]) <= set(guard["allowed_junctions"][entry["town"]]):
            raise ValueError("Registered branch route violates corrected geography guard")
    config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
    sources = [
        Path(__file__),
        ROOT / "src/navigation/collection.py",
        ROOT / "src/navigation/controller.py",
        ROOT / "src/navigation/dataset.py",
        ROOT / "src/navigation/sensors.py",
        ROOT / "src/navigation/runtime.py",
        ROOT / "src/navigation/replay.py",
        ROOT / "src/navigation/recorder_collisions.py",
        ROOT / "reports/navigation_branch_pilot_protocol_20261019.md",
    ]
    if args.reset_signal_clocks:
        sources.append(ROOT / "reports/navigation_branch_clock_repair_protocol_20261020.md")
    ego_path = ROOT / "outputs/navigation_models/frozen_expert_ego_20261020.json"
    if args.freeze_commands:
        sources.append(ego_path)
        sources.extend([ROOT / p for p in (
            "src/navigation/candidate_controls.py", "src/navigation/trajectory_candidates.py",
            "src/navigation/frozen_ego.py", "src/navigation/storage_budget.py",
            "reports/navigation_fixed_branch_protocol_20261020.md",
            "configs/navigation.yaml", "configs/navigation_vendor.json",
        )])
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    if args.resume_unstarted:
        prior = json.loads((args.out / "request.json").read_text())
        before = json.loads((args.out / "progress.json").read_text())
        if before["frames"] or before["active_output"]:
            raise ValueError("Only an unstarted interrupted queue may use this resume")
        for path, expected in prior["source_hashes"].items():
            if path != "scripts/collect_navigation_branches.py" and digest(ROOT / path) != expected:
                raise ValueError("Branch collection dependencies changed")
        atomic_json(args.out / "interrupted_wait.json", before)
    for source in sources:
        folder = "resume_source" if args.resume_unstarted else "source"
        target = args.out / folder / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / ("resume_request.json" if args.resume_unstarted else "request.json"),
        dict(
            entry=entry,
            seed=62001,
            prefix_seconds=4.0,
            branch_seconds=4.0,
            prefix_requested_speed_mps=3.0,
            targets=["go", "brake", "wait_go"],
            repeats=2,
            reset_signal_clocks=args.reset_signal_clocks,
            freeze_commands_before_outcome=args.freeze_commands,
            frozen_ego_sha256=digest(ego_path) if args.freeze_commands else None,
            minimum_free_gb=10 if args.freeze_commands else 25,
            actual_validation_guard_sha256=digest(actual) if args.reset_signal_clocks else None,
            guard_sha256=digest(ROOT / "outputs/navigation_m2_varied_waits_20261018/guard.json"),
            source_hashes=hashes,
            training_only=True,
            imitation_eligible=False,
            canonical_benchmark_changed=False,
            maximum_wall_seconds=1800,
            after_references=str(args.after_references) if args.after_references else None,
        ),
    )
    records, prefixes, repeats, samples, status = [], {}, {}, 0, "running"
    started = time.monotonic()
    active = None
    current_samples = 0

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M5 factual branch pilot",
                frames=samples + current_samples,
                target_frames=480,
                encounters_done=len(records),
                encounters_total=6,
                phase=phase,
                active_output=str(active) if active else None,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists():
            raise InterruptedError("Branch pilot STOP")
        if time.monotonic() - started > 1800:
            raise InterruptedError("Branch pilot wall cap")
        minimum_free = 10e9 if args.freeze_commands else 25e9
        if shutil.disk_usage(args.out).free < minimum_free + 2e6:
            raise InterruptedError("Branch pilot disk reserve")

    budget_watch = None
    try:
        if args.freeze_commands:
            pending_watch = BudgetWatch(args.out)
            pending_watch.__enter__()
            budget_watch = pending_watch
        if args.after_references:
            progress("Waiting for exact physical reference queue; no duplicate server")
            while True:
                if (args.out / "STOP").exists():
                    raise InterruptedError("Branch pilot STOP while queued")
                path = args.after_references / "progress.json"
                if path.exists():
                    terminal = json.loads(path.read_text())["status"]
                    if terminal != "running":
                        if terminal != "complete":
                            raise RuntimeError(f"Prior reference queue ended {terminal}; inspect")
                        break
                time.sleep(3)
        # Waiting consumes no simulation budget; maximum wall begins at owned start.
        started = time.monotonic()
        if any(digest(ROOT / p) != h for p, h in hashes.items()):
            raise ValueError("Registered branch source changed while queued")
        with RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 10 if args.freeze_commands else 25)
            ego_model = FrozenExpertEgo(ego_path) if args.freeze_commands else None
            vendor = activate()
            import carla
            from agents.navigation.global_route_planner import GlobalRoutePlanner

            with CarlaServer(
                vendor["carla_root"], args.out, port=2100, town=entry["town"]
            ) as client:
                for repeat in range(2):
                    for name in ("go", "brake", "wait_go"):
                        check()
                        active = args.out / f"repeat{repeat}_{name}"
                        active.mkdir()
                        # CARLA can boot on its defaultTown03 despite the map
                        # command-line argument. Never reload that wrong world.
                        world = client.load_world(entry["town"], reset_settings=True)
                        actual_town = world.get_map().name.rsplit("/", 1)[-1]
                        if actual_town != entry["town"]:
                            raise RuntimeError("Branch simulator loaded the wrong map")
                        atomic_json(
                            active / "world_identity.json",
                            dict(
                                actual_town=actual_town,
                                requested_town=entry["town"],
                                opendrive_sha256=__import__("hashlib")
                                .sha256(world.get_map().to_opendrive().encode())
                                .hexdigest(),
                            ),
                        )
                        settings = world.get_settings()
                        settings.synchronous_mode = True
                        settings.fixed_delta_seconds = 0.05
                        settings.substepping = True
                        settings.max_substep_delta_time = 0.01
                        settings.max_substeps = 5
                        world.apply_settings(settings)
                        world.set_weather(carla.WeatherParameters.ClearNoon)
                        lights = sorted(
                            world.get_actors().filter("traffic.traffic_light*"), key=lambda a: a.id
                        )
                        if args.reset_signal_clocks:
                            reset_groups = set()
                            for light in lights:
                                group = tuple(
                                    sorted(a.id for a in light.get_group_traffic_lights())
                                )
                                if group not in reset_groups:
                                    light.reset_group()
                                    reset_groups.add(group)
                        for light in lights:
                            light.set_state(carla.TrafficLightState.Green)
                            light.freeze(True)
                        if args.reset_signal_clocks:
                            clock_frame = world.tick()
                            clock_rows = [
                                dict(
                                    id=a.id,
                                    state=str(a.get_state()),
                                    elapsed=a.get_elapsed_time(),
                                    frozen=a.is_frozen(),
                                )
                                for a in lights
                            ]
                            atomic_json(
                                active / "phase_clock_reset.json",
                                dict(
                                    acknowledgment_frame=clock_frame,
                                    groups_reset=len(reset_groups),
                                    lights=clock_rows,
                                    maximum_allowed_elapsed_s=0.051,
                                ),
                            )
                            if any(
                                abs(row["elapsed"]) > 0.051
                                or not row["frozen"]
                                or row["state"] != "Green"
                                for row in clock_rows
                            ):
                                asynchronous = world.get_settings()
                                asynchronous.synchronous_mode = False
                                asynchronous.fixed_delta_seconds = None
                                world.apply_settings(asynchronous)
                                raise RuntimeError(
                                    "Signal clock reset did not meet registered tolerance"
                                )
                        planner = GlobalRoutePlanner(world.get_map(), 1.0)
                        route = planner.trace_route(
                            carla.Location(*entry["start_xyz"]), carla.Location(*entry["end_xyz"])
                        )
                        xyz = np.array(
                            [[w.transform.location.x, w.transform.location.y] for w, _ in route]
                        )
                        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
                        lead_index = int(np.searchsorted(arc, 15.0))
                        actors, rig, writer, log, world_log = [], None, None, None, None
                        collision_events = []
                        intervention = None
                        recording = active / "native.log"
                        try:
                            for index, model, role in (
                                (0, "vehicle.lincoln.mkz_2020", "hero"),
                                (lead_index, "vehicle.lincoln.mkz_2017", "branch_stationary_lead"),
                            ):
                                transform = route[index][0].transform
                                transform.location.z += 0.5
                                bp = world.get_blueprint_library().find(model)
                                bp.set_attribute("role_name", role)
                                actor = world.try_spawn_actor(bp, transform)
                                if actor is None:
                                    raise RuntimeError(f"Branch {role} spawn failed")
                                actor.apply_control(carla.VehicleControl(brake=1.0))
                                actors.append(actor)
                            ego, lead = actors
                            for _ in range(20):
                                world.tick()
                            sensor = world.spawn_actor(
                                world.get_blueprint_library().find("sensor.other.collision"),
                                carla.Transform(),
                                attach_to=ego,
                            )
                            actors.append(sensor)
                            sensor.listen(
                                lambda event: collision_events.append(
                                    dict(frame=event.frame, other_actor=event.other_actor.id)
                                )
                            )
                            rig = CameraRig(world, ego, config)
                            # Native Leaderboard also warms newly registered
                            # cameras for10 braked ticks before measurement.
                            for _ in range(10):
                                check()
                                ego.apply_control(carla.VehicleControl(brake=1.0))
                                lead.apply_control(carla.VehicleControl(brake=1.0))
                                world.tick()
                            projector = RouteProjector(xyz)
                            writer = EpisodeWriter(
                                active / "data",
                                dict(
                                    town=entry["town"],
                                    partition="train",
                                    role="prediction_only",
                                    route_id=entry["route_id"],
                                    junction_ids=entry["crossed_junction_ids"],
                                    seed=62001,
                                    repeat=repeat,
                                    branch=name,
                                    imitation_eligible=False,
                                    purpose="Factual alternative actions; no driving claim",
                                    expert_control_field_semantics=(
                                        "Duplicate executed control; no expert labels"
                                    ),
                                ),
                                minimum_free_gb=10 if args.freeze_commands else 25,
                            )
                            recorder = CollectionRecorder(writer)
                            log = (active / "actions.jsonl").open("x")
                            world_log = (active / "world_trace.jsonl").open("x")
                            client.start_recorder(str(recording), True)
                            first, prefix_states, all_states = None, [], []
                            actual_history, selected_plan = [], None
                            for tick in range(161):
                                check()
                                frame = world.tick()
                                previous = ego.get_control()
                                actual_history.append([previous.steer, previous.throttle,
                                                       previous.brake])
                                recorder.acknowledge(
                                    frame, (previous.steer, previous.throttle, previous.brake)
                                )
                                current_samples = writer.samples
                                images = rig.barrier.collect(frame)
                                snapshot = world.get_snapshot()
                                first = (
                                    snapshot.timestamp.elapsed_seconds if first is None else first
                                )
                                elapsed = snapshot.timestamp.elapsed_seconds - first
                                if tick == 160:
                                    break
                                pose = training_position(
                                    world.get_map(),
                                    ego.get_transform(),
                                    set(guard["allowed_junctions"][entry["town"]]),
                                    frame,
                                    active / "partition_violation.json",
                                )
                                speed = ego.get_velocity().length()
                                yaw = math.radians(pose.rotation.yaw)
                                navigation = projector.project(
                                    np.array([pose.location.x, pose.location.y]), yaw
                                )
                                rgb = {view: value[2] for view, value in images.items()}
                                packet = SensorPacket(
                                    frame,
                                    elapsed,
                                    rgb,
                                    rig.calibration,
                                    navigation,
                                    np.array(
                                        [speed, yaw, math.radians(ego.get_angular_velocity().z)],
                                        dtype=np.float32,
                                    ),
                                )
                                # Integer ticks fix handoff independent of floating clock error.
                                target = (
                                    3.0 if tick < 80 else branch_target(name, (tick - 80) * 0.05)
                                )
                                command = track(navigation, np.zeros(8), speed, target, 0.0)
                                proposed = (command.steer, command.throttle, command.brake)
                                if args.freeze_commands and tick == 80:
                                    slate, selected_plan = frozen_branch_plan(
                                        navigation, speed, np.array(actual_history[-2:]), name,
                                        ego_model,
                                    )
                                    atomic_json(active / "frozen_plan.json", dict(
                                        branch_frame=frame, branch_tick=tick,
                                        navigation=navigation.tolist(), speed_mps=speed,
                                        actual_previous_controls=actual_history[-2:],
                                        selected_path=1,
                                        selected_speed={"go": 2, "brake": 0, "wait_go": 4}[name],
                                        frozen_before_first_branch_command=True,
                                        ego_model_sha256=digest(ego_path),
                                        slate=[dict(
                                            path=p.desired.path_index, speed=p.desired.speed_index,
                                            ordered_controls=p.ordered_controls.tolist(),
                                            desired_speeds=p.desired.speeds.tolist(),
                                            forecast_states=p.forecast_states.tolist(),
                                        ) for p in slate],
                                    ))
                                if selected_plan is not None:
                                    proposed = tuple(float(v) for v in
                                                     selected_plan.ordered_controls[tick - 80])
                                    target = float(selected_plan.desired.speeds[(tick - 80) // 10])
                                if collision_events:
                                    intervention = "privileged_post_contact_stop"
                                    chosen = (0.0, 0.0, 1.0)
                                else:
                                    chosen = proposed
                                state = capture_world_state(world)
                                if state["frame"] != packet.frame:
                                    raise RuntimeError("Branch camera/world frame mismatch")
                                all_states.append(state)
                                if tick < 80:
                                    prefix_states.append(state)
                                facts = dict(
                                    world=state,
                                    branch=name,
                                    tick=tick,
                                    branch_active=tick >= 80,
                                    branch_handoff_tick=80,
                                    requested_target_mps=target,
                                    safety_intervention=intervention,
                                    contacts_observed=list(collision_events),
                                    counterfactual_tail_censored=intervention is not None,
                                    imitation_eligible=False,
                                )
                                if args.freeze_commands:
                                    facts.update(
                                        frozen_plan_control=list(proposed),
                                        branch_commands_frozen=selected_plan is not None,
                                        proposed_control_before_safety=list(proposed),
                                    )
                                if tick % 2 == 0:
                                    recorder.capture(packet, chosen, chosen, facts, intervention)
                                log.write(
                                    json.dumps(
                                        dict(
                                            frame=frame,
                                            timestamp=elapsed,
                                            requested_control=list(chosen),
                                            branch_target_mps=target,
                                            prefix=tick < 80,
                                            safety_intervention=intervention,
                                            **(dict(frozen_plan_control=list(proposed))
                                               if args.freeze_commands else {}),
                                        )
                                    )
                                    + "\n"
                                )
                                world_log.write(json.dumps(state) + "\n")
                                ego.apply_control(
                                    carla.VehicleControl(
                                        steer=chosen[0], throttle=chosen[1], brake=chosen[2]
                                    )
                                )
                                lead.apply_control(carla.VehicleControl(brake=1.0))
                                progress("Collecting factual actions; no model training")
                            writer.close()
                            client.stop_recorder()
                            cleanup_frame = world.tick()
                            atomic_json(
                                active / "recorder_close_ack.json",
                                dict(
                                    cleanup_frame=cleanup_frame,
                                    measured_world_rows=160,
                                    measured_rgb_samples=80,
                                    extra_cleanup_tick_not_a_measured_observation=True,
                                ),
                            )
                            deadline = time.monotonic() + 5
                            last_identity = None
                            while True:
                                contacts = recorder_contacts(recording)
                                stat = recording.stat()
                                identity = stat.st_size, stat.st_mtime_ns
                                if (
                                    contacts["complete"]
                                    and contacts["frames"] >= 161
                                    and identity == last_identity
                                ):
                                    break
                                if time.monotonic() >= deadline:
                                    raise RuntimeError(
                                        "Native recorder stop/flush did not complete"
                                    )
                                last_identity = identity
                                time.sleep(0.05)
                            comparisons = []
                            if prefixes:
                                comparisons.append(
                                    compare_world_prefixes(
                                        next(iter(prefixes.values())), prefix_states
                                    )
                                )
                            prefixes[(repeat, name)] = prefix_states
                            if name in repeats:
                                comparisons.append(
                                    dict(
                                        same_option_repeat=True,
                                        **compare_world_prefixes(repeats[name], all_states),
                                    )
                                )
                            repeats[name] = all_states
                            result = dict(
                                repeat=repeat,
                                branch=name,
                                samples=writer.samples,
                                control_differences=recorder.control_differences,
                                recorder=contacts,
                                sensor_events=collision_events,
                                safety_intervention=intervention,
                                counterfactual_tail_censored=intervention is not None,
                                observable_comparisons=comparisons,
                                imitation_eligible=False,
                            )
                            atomic_json(active / "outcome.json", result)
                            records.append(result)
                            samples += writer.samples
                            current_samples = 0
                            atomic_json(args.out / "records.json", records)
                        except Exception as primary_error:
                            atomic_json(
                                active / "primary_failure.json",
                                dict(
                                    type=type(primary_error).__name__,
                                    error=str(primary_error),
                                    committed_samples=writer.samples if writer else 0,
                                ),
                            )
                            raise
                        finally:
                            client.stop_recorder()
                            if writer is not None:
                                writer.close()
                            if log:
                                log.close()
                            if world_log:
                                world_log.close()
                            if rig:
                                rig.close()
                            for actor in reversed(actors):
                                if actor.is_alive:
                                    if actor.type_id.startswith("sensor."):
                                        actor.stop()
                                    actor.destroy()
                            # Leave an advancing world for normal engine
                            # shutdown instead of a paused synchronous tick.
                            try:
                                asynchronous = world.get_settings()
                                asynchronous.synchronous_mode = False
                                asynchronous.fixed_delta_seconds = None
                                world.apply_settings(asynchronous)
                            except Exception as cleanup_error:
                                atomic_json(
                                    active / "world_settings_cleanup_error.json",
                                    dict(
                                        error=str(cleanup_error),
                                        primary_failure_preserved=True,
                                    ),
                                )
                status = "complete"
    except InterruptedError as error:
        status = "stopped"
        atomic_json(args.out / "error.json", dict(error=str(error)))
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), active_output=str(active)))
        raise
    finally:
        progress("finished")
        if budget_watch is not None:
            budget_watch.__exit__(None, None, None)
    print(json.dumps(dict(status=status, branches=len(records), samples=samples)), flush=True)


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation_branch_pilot.lock"):
        main()
