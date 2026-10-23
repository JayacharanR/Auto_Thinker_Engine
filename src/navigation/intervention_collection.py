"""Owned TRAIN-only factual-action recording; never part of camera inference."""

import math
import time

import numpy as np

from src.navigation.collection import CollectionRecorder, RouteProjector, training_position
from src.navigation.contracts import SensorPacket
from src.navigation.controller import track
from src.navigation.dagger import digest
from src.navigation.dataset import EpisodeWriter
from src.navigation.native_cleanup import cleanup_owned_collection
from src.navigation.recorder_collisions import recorder_contacts
from src.navigation.replay import capture_world_state
from src.navigation.runtime import atomic_json
from src.navigation.sensors import CameraRig, camera_specs


def native_transform(carla, row):
    return carla.Transform(
        carla.Location(*row[:3]), carla.Rotation(pitch=row[3], yaw=row[4], roll=row[5])
    )


def collect_case(
    server, client, case, context, output, config, ego_model, ego_descriptor_path, check, on_tick
):
    """One predeclared attempt, preserving partial evidence on every failure."""
    import carla

    from scripts.collect_navigation_branches import frozen_branch_plan

    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "case.json", dict(case=case, context=context))
    client.set_timeout(20.0)
    world = client.load_world(context["town"], reset_settings=True)
    if world.get_map().name.rsplit("/", 1)[-1] != context["town"]:
        raise ValueError("Intervention actual map identity mismatch")
    atomic_json(
        output / "world_identity.json",
        dict(
            actual_town=context["town"],
            requested_town=context["town"],
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
    lights = sorted(world.get_actors().filter("traffic.traffic_light*"), key=lambda a: a.id)
    reset_groups = set()
    for light in lights:
        group = tuple(sorted(a.id for a in light.get_group_traffic_lights()))
        if group not in reset_groups:
            light.reset_group()
            reset_groups.add(group)
    for light in lights:
        light.set_state(carla.TrafficLightState.Green)
        light.freeze(True)
    clock_frame = world.tick()
    clocks = [
        dict(id=a.id, elapsed=a.get_elapsed_time(), frozen=a.is_frozen(), state=str(a.get_state()))
        for a in lights
    ]
    atomic_json(output / "phase_clock_reset.json", dict(frame=clock_frame, lights=clocks))
    if any(abs(r["elapsed"]) > 0.051 or not r["frozen"] or r["state"] != "Green" for r in clocks):
        raise ValueError("Original observable signal clock tolerance failed")
    actors, rig, writer, action_log, world_log = [], None, None, None, None
    recording, contacts, states, actions, samples = False, [], [], [], 0
    try:
        for kind, blueprint, role in (
            ("ego_route", "vehicle.lincoln.mkz_2020", "hero"),
            ("other_route", "vehicle.lincoln.mkz_2017", "intervention_" + context["mechanism"]),
        ):
            bp = world.get_blueprint_library().find(blueprint)
            bp.set_attribute("role_name", role)
            pose = native_transform(carla, context[kind][0])
            pose.location.z += 0.5
            actor = world.try_spawn_actor(bp, pose)
            if actor is None:
                raise RuntimeError(f"Original intervention{role} spawn failed; no replacement")
            actor.apply_control(carla.VehicleControl(brake=1.0))
            actors.append(actor)
        ego, other = actors
        for _ in range(20):
            check()
            world.tick()
        atomic_json(
            output / "native_boxes.json",
            {
                actor.attributes["role_name"]: dict(
                    actor_id=actor.id,
                    actor_type=actor.type_id,
                    box_location=[
                        actor.bounding_box.location.x,
                        actor.bounding_box.location.y,
                        actor.bounding_box.location.z,
                    ],
                    box_extent=[
                        actor.bounding_box.extent.x,
                        actor.bounding_box.extent.y,
                        actor.bounding_box.extent.z,
                    ],
                    box_rotation=[
                        actor.bounding_box.rotation.pitch,
                        actor.bounding_box.rotation.yaw,
                        actor.bounding_box.rotation.roll,
                    ],
                )
                for actor in (ego, other)
            },
        )
        sensor = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.collision"),
            carla.Transform(),
            attach_to=ego,
        )
        actors.append(sensor)
        sensor.listen(lambda e: contacts.append(dict(frame=e.frame, other_actor=e.other_actor.id)))
        rig = CameraRig(world, ego, config)
        for _ in range(10):
            check()
            world.tick()
        atomic_json(output / "sensors.json", camera_specs(config))
        ego_projector = RouteProjector(np.asarray(context["ego_route"])[:, :2])
        other_projector = RouteProjector(np.asarray(context["other_route"])[:, :2])
        writer = EpisodeWriter(
            output / "data",
            dict(
                town=context["town"],
                partition=context["role"],
                route_id=context["route_id"],
                junction_ids=context["entry"]["crossed_junction_ids"],
                role="prediction_only",
                context_id=context["context_id"],
                mechanism=context["mechanism"],
                branch=case["branch"],
                repeat=case["repeat"],
                seed=context["seed"],
                imitation_eligible=False,
                expert_control_field_semantics="Duplicated actual action; no expert",
            ),
            minimum_free_gb=10,
        )
        recorder = CollectionRecorder(writer)
        action_log = (output / "actions.jsonl").open("x")
        world_log = (output / "world_trace.jsonl").open("x")
        client.start_recorder(str(output / "native.log"), True)
        recording = True
        if context["other_target_speed_mps"]:
            direction = other.get_transform().get_forward_vector()
            speed = context["other_target_speed_mps"]
            other.set_target_velocity(
                carla.Vector3D(direction.x * speed, direction.y * speed, direction.z * speed)
            )
        history, selected, first_time = [], None, None
        for tick in range(161):
            check()
            frame = world.tick()
            actual = ego.get_control()
            history.append([actual.steer, actual.throttle, actual.brake])
            recorder.acknowledge(frame, history[-1])
            images = rig.barrier.collect(frame)
            if tick == 160:
                break
            pose = training_position(
                world.get_map(),
                ego.get_transform(),
                context["allowed_junctions"],
                frame,
                output / "partition_violation.json",
            )
            training_position(
                world.get_map(),
                other.get_transform(),
                context["allowed_junctions"],
                frame,
                output / "other_partition_violation.json",
            )
            yaw = math.radians(pose.rotation.yaw)
            speed = ego.get_velocity().length()
            navigation = ego_projector.project(np.array([pose.location.x, pose.location.y]), yaw)
            physical_time = world.get_snapshot().timestamp.elapsed_seconds
            if first_time is None:
                first_time = physical_time
            elapsed = physical_time - first_time
            packet = SensorPacket(
                frame,
                elapsed,
                {v: r[2] for v, r in images.items()},
                rig.calibration,
                navigation,
                np.array([speed, yaw, math.radians(ego.get_angular_velocity().z)], np.float32),
            )
            prefix_command = track(navigation, np.zeros(8), speed, 3.0, 0.0)
            proposed = [prefix_command.steer, prefix_command.throttle, prefix_command.brake]
            if tick == 80:
                slate, selected = frozen_branch_plan(
                    navigation, speed, np.asarray(history[-2:]), case["branch"], ego_model
                )
                atomic_json(
                    output / "frozen_plan.json",
                    dict(
                        branch_frame=frame,
                        branch_tick=80,
                        navigation=navigation.tolist(),
                        speed_mps=speed,
                        actual_previous_controls=history[-2:],
                        actual_previous_observation_frames=[frame - 1, frame],
                        selected_path=1,
                        selected_speed={"go": 2, "brake": 0, "wait_go": 4}[case["branch"]],
                        frozen_before_first_branch_command=True,
                        frozen_ego_sha256=digest(ego_descriptor_path),
                        slate=[
                            dict(
                                path=p.desired.path_index,
                                speed=p.desired.speed_index,
                                ordered_controls=p.ordered_controls.tolist(),
                                desired_speeds=p.desired.speeds.tolist(),
                                forecast_states=p.forecast_states.tolist(),
                            )
                            for p in slate
                        ],
                    ),
                )
            if selected is not None:
                proposed = selected.ordered_controls[tick - 80].tolist()
            intervention = "privileged_post_contact_stop" if contacts else None
            chosen = [0.0, 0.0, 1.0] if intervention else proposed
            state = capture_world_state(world)
            if state["frame"] != frame:
                raise ValueError("Physical camera/actor frame mismatch")
            facts = dict(
                world=state,
                tick=tick,
                branch_active=tick >= 80,
                branch_handoff_tick=80,
                branch=case["branch"],
                context_id=context["context_id"],
                contacts_observed=list(contacts),
                safety_intervention=intervention,
                counterfactual_tail_censored=intervention is not None,
                imitation_eligible=False,
                frozen_plan_control=proposed,
                branch_commands_frozen=selected is not None,
                own_actual_prior_controls=history[-2:],
            )
            if tick % 2 == 0:
                recorder.capture(packet, chosen, chosen, facts, intervention)
            row = dict(
                frame=frame,
                timestamp=elapsed,
                requested_control=chosen,
                prefix=tick < 80,
                safety_intervention=intervention,
                frozen_plan_control=proposed,
            )
            action_log.write(__import__("json").dumps(row) + "\n")
            world_log.write(__import__("json").dumps(state) + "\n")
            actions.append(row)
            states.append(state)
            ego.apply_control(
                carla.VehicleControl(steer=chosen[0], throttle=chosen[1], brake=chosen[2])
            )
            other_pose = other.get_transform()
            if intervention or not context["other_target_speed_mps"]:
                other_control = carla.VehicleControl(brake=1.0)
            else:
                other_nav = other_projector.project(
                    np.array([other_pose.location.x, other_pose.location.y]),
                    math.radians(other_pose.rotation.yaw),
                )
                command = track(
                    other_nav,
                    np.zeros(8),
                    other.get_velocity().length(),
                    context["other_target_speed_mps"],
                    0.0,
                )
                other_control = carla.VehicleControl(
                    steer=command.steer, throttle=command.throttle, brake=command.brake
                )
            other.apply_control(other_control)
            on_tick(writer.samples, tick)
        writer.close()
        samples = writer.samples
        client.stop_recorder()
        recording = False
        cleanup_frame = world.tick()
        atomic_json(
            output / "recorder_close_ack.json",
            dict(
                cleanup_frame=cleanup_frame, measured_world_rows=160, measured_rgb_samples=samples
            ),
        )
        deadline, last_identity = time.monotonic() + 5, None
        while True:
            check()
            native = recorder_contacts(output / "native.log")
            stat = (output / "native.log").stat()
            identity = stat.st_size, stat.st_mtime_ns
            if native["complete"] and native["frames"] >= 161 and identity == last_identity:
                break
            if time.monotonic() > deadline:
                raise ValueError("Original native recorder close/flush incomplete")
            last_identity = identity
            time.sleep(0.05)
        outcome = dict(
            case=case,
            samples=samples,
            recorder=native,
            sensor_events=contacts,
            control_differences=recorder.control_differences,
            first_event_already_in_prefix=any(e["frame"] <= actions[80]["frame"] for e in contacts),
            no_model_or_expert_imitation=True,
        )
        atomic_json(output / "outcome.json", outcome)
        return outcome
    except BaseException as error:
        atomic_json(
            output / "primary_failure.json",
            dict(
                error=str(error),
                type=type(error).__name__,
                committed_samples=writer.samples if writer else 0,
                at=time.time(),
            ),
        )
        raise
    finally:
        if writer:
            writer.close()
        if action_log:
            action_log.close()
        if world_log:
            world_log.close()
        cleanup_owned_collection(server, client, world, rig, actors, output, recording)
