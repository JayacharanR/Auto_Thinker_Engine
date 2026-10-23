"""Privileged setup/measurement of separate controlled rule tests.

No simulator values enter camera inference. A bounded prefix may replace
requested controls before declared handover; every subsequent control is the
original agent proposal, without engineered rule or emergency assistance.
"""

import json
import math
from pathlib import Path

import numpy as np

from src.navigation.controller import track
from src.navigation.rule_prefix import PrefixSpec, RulePrefix
from src.navigation.runtime import atomic_json

ACTIVE = {"runtime": None}


class RuleCaseRuntime:
    def __init__(self, world, ego, spec, output, navigation_projector, calibrations):
        import carla

        self.world, self.ego, self.spec = world, ego, spec
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.treatment = spec["treatment"]
        self.projector = navigation_projector
        self.calibrations = calibrations
        self.housing_vertices = []
        self.prefix = RulePrefix(PrefixSpec(**self.treatment["prefix"]))
        self.started, self.last_timestamp, self.last_phase = None, None, None
        self.last_distance = float("inf")
        self.scripted_actors = []
        self.phase_changes = []
        self.closed = False
        self.last_rgb_timestamp = None
        self.rgb_records = []
        self.control_log = (self.output / "requested_executed.jsonl").open("x")
        self.trace = (self.output / "setup_trace.jsonl").open("x")
        self.allowed = set(spec["strict_allowed_junctions"])
        self.bound_light = None
        signals = list(world.get_actors().filter("traffic.traffic_light*"))
        # All groups use the same red state in both paired arms. This is a
        # controlled component world, explicitly separate from canonical traffic.
        for signal in signals:
            signal.set_state(carla.TrafficLightState.Red)
            signal.freeze(True)
        geometry = spec.get("signal_reference")
        if geometry:
            matched = [
                a
                for a in signals
                if a.get_location().distance(carla.Location(*geometry["xyz"])) < 0.01
            ]
            if len(matched) != 1:
                raise RuntimeError("Controlled setup has no unique physical signal")
            self.bound_light = matched[0]
            for box in self.bound_light.get_light_boxes():
                self.housing_vertices.append(
                    [
                        [vertex.x, vertex.y, vertex.z]
                        for vertex in box.get_world_vertices(carla.Transform())
                    ]
                )
            initial = self.treatment.get("prefix_signal", self.treatment["ego_signal"])
            self.set_phase(initial, -1.0)
        self.spawn_declared_actors()
        # Actor lists follow the last world snapshot. A spawn proxy can exist
        # before get_actors sees it. Use exactly one setup tick in EVERY arm.
        acknowledged_frame = world.tick()
        dynamic = [a for a in world.get_actors() if a.type_id.startswith(("vehicle.", "walker."))]
        atomic_json(
            self.output / "dynamic_setup_acknowledgment.json",
            dict(
                world_frame=acknowledged_frame,
                expected_ids=sorted({ego.id, *(row["actor"].id for row in self.scripted_actors)}),
                observed=[
                    dict(
                        id=actor.id, type=actor.type_id, role=actor.attributes.get("role_name", "")
                    )
                    for actor in dynamic
                ],
                common_acknowledgment_ticks=1,
                setup_before_policy_sensor_registration=True,
            ),
        )
        if {a.id for a in dynamic} != {ego.id, *(a["actor"].id for a in self.scripted_actors)}:
            raise RuntimeError("Unexpected dynamic background in controlled rule test")
        atomic_json(
            self.output / "binding.json",
            dict(
                case_sha256=spec["case_sha256"],
                ego_light_id=self.bound_light.id if self.bound_light else None,
                expected_signal=geometry,
                controlled_empty_background=True,
                scripted_actor_ids=[row["actor"].id for row in self.scripted_actors],
                rule_overrides=False,
                emergency_overrides=False,
                prefix=self.treatment["prefix"],
                autonomous_route_success_claim=False,
            ),
        )

    def spawn_declared_actors(self):
        import carla

        for kind in ("lead", "crossing"):
            if not self.treatment[kind]:
                continue
            geometry = self.spec["scripted_actor_paths"][kind]
            pose = geometry["spawn_pose"]
            blueprint = self.world.get_blueprint_library().find("vehicle.lincoln.mkz_2017")
            blueprint.set_attribute("role_name", f"rule_test_{kind}")
            actor = self.world.try_spawn_actor(
                blueprint,
                carla.Transform(
                    carla.Location(*pose[:3]),
                    carla.Rotation(pitch=pose[3], yaw=pose[4], roll=pose[5]),
                ),
            )
            if actor is None:
                raise RuntimeError(f"Scripted {kind} actor spawn failed")
            from src.navigation.collection import RouteProjector

            self.scripted_actors.append(
                dict(actor=actor, kind=kind, projector=RouteProjector(geometry["route_xy"]))
            )

    def set_phase(self, name, timestamp):
        import carla

        if name in ("None", self.last_phase):
            return
        if self.bound_light is None or name not in ("Red", "Yellow", "Green"):
            raise RuntimeError("Controlled phase has no valid bound traffic light")
        self.bound_light.set_state(getattr(carla.TrafficLightState, name))
        self.last_phase = name
        self.phase_changes.append(dict(requested_after_timestamp=timestamp, state=name))

    def background_tick(self):
        """Called after agent/measurement, before the next synchronous world tick."""
        import carla

        if self.closed or self.last_timestamp is None:
            return
        treatment = self.treatment
        after_handoff = (
            None
            if self.prefix.handoff_time is None
            else self.last_timestamp - self.prefix.handoff_time
        )
        elapsed = self.last_timestamp - self.started
        desired = treatment["ego_signal"]
        if after_handoff is None and self.prefix.spec.mode != "none":
            desired = treatment.get("prefix_signal", desired)
        if (
            desired == "Yellow"
            and after_handoff is not None
            and after_handoff >= treatment["yellow_seconds"]
        ):
            desired = "Red"
        if desired == "Red" and elapsed >= treatment.get("signal_hold_seconds", 1e9):
            desired = "Green"
        self.set_phase(desired, self.last_timestamp)
        for row in self.scripted_actors:
            actor = row["actor"]
            pose = actor.get_transform()
            yaw = math.radians(pose.rotation.yaw)
            xy = np.array([pose.location.x, pose.location.y])
            navigation = row["projector"].project(xy, yaw)
            speed = actor.get_velocity().length()
            target = (
                0.0
                if row["kind"] == "lead" and elapsed < treatment["lead_release_seconds"]
                else 4.0
            )
            if row["kind"] == "crossing" and after_handoff is None and self.last_distance > 12:
                target = 0.0
            # Scripted traffic remains reactive to ego; these privileged checks
            # control only the non-ego actor, never the evaluated driving proposal.
            ego_delta = self.ego.get_location() - pose.location
            local = np.array([ego_delta.x, ego_delta.y]) @ np.array(
                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
            )
            if 0 < local[0] < 6 and abs(local[1]) < 2:
                target = 0.0
            command = track(navigation, np.zeros(8), speed, target, 0.0)
            actor.apply_control(
                carla.VehicleControl(
                    steer=command.steer, throttle=command.throttle, brake=command.brake
                )
            )

    def selected_control(self, timestamp, requested, rgb=None):
        """Record current state; phase mutation happens only on next background tick."""
        pose = self.ego.get_transform()
        self.started = timestamp if self.started is None else self.started
        self.last_timestamp = timestamp
        waypoint = self.world.get_map().get_waypoint(pose.location)
        if waypoint.is_junction and waypoint.junction_id not in self.allowed:
            raise RuntimeError("Controlled component left strict TRAIN geography")
        yaw = math.radians(pose.rotation.yaw)
        xy = np.array([pose.location.x, pose.location.y])
        navigation = self.projector.project(xy, yaw)
        direction = np.array(self.spec["line_heading_xy"])
        distance = float((np.array(self.spec["line_xyz"][:2]) - xy) @ direction)
        self.last_distance = distance
        speed = self.ego.get_velocity().length()
        prefix = self.prefix.step(timestamp, navigation, speed, distance)
        selected = requested if prefix is None else [prefix.steer, prefix.throttle, prefix.brake]
        if rgb is not None and (
            self.last_rgb_timestamp is None or timestamp - self.last_rgb_timestamp >= 1.0 - 1e-5
        ):
            from PIL import Image

            frame = self.world.get_snapshot().frame
            folder = self.output / "raw_setup_rgb"
            folder.mkdir(exist_ok=True)
            images = {}
            for view, image in rgb.items():
                path = folder / f"{frame:08d}_{view}.png"
                if path.exists():
                    raise FileExistsError("Refuse to overwrite setup RGB evidence")
                Image.fromarray(image).save(path)
                images[view] = str(path)
            self.rgb_records.append(
                dict(
                    frame=frame,
                    timestamp=timestamp,
                    images=images,
                    observed_light=str(self.bound_light.get_state()) if self.bound_light else None,
                    prefix_state=self.prefix.state(timestamp),
                )
            )
            self.last_rgb_timestamp = timestamp
        self.control_log.write(
            json.dumps(
                dict(
                    frame=self.world.get_snapshot().frame,
                    timestamp=timestamp,
                    requested_control=list(requested),
                    executed_control=list(selected),
                    source="evaluated_agent" if prefix is None else "SCRIPTED SETUP PREFIX",
                    **self.prefix.state(timestamp),
                )
            )
            + "\n"
        )
        self.trace.write(
            json.dumps(
                dict(
                    frame=self.world.get_snapshot().frame,
                    timestamp=timestamp,
                    ego=[pose.location.x, pose.location.y, pose.location.z, pose.rotation.yaw],
                    speed_mps=speed,
                    distance_to_registered_line_m=distance,
                    observed_light=str(self.bound_light.get_state()) if self.bound_light else None,
                    housing_frusta=self.housing_frusta(pose),
                    **self.prefix.state(timestamp),
                )
            )
            + "\n"
        )
        if int(round(timestamp * 20)) % 20 == 0:
            self.control_log.flush()
            self.trace.flush()
        return selected

    def housing_frusta(self, pose):
        from src.navigation.rule_geometry import project_box

        full_pose = [
            pose.location.x,
            pose.location.y,
            pose.location.z,
            pose.rotation.pitch,
            pose.rotation.yaw,
            pose.rotation.roll,
        ]
        return {
            view: [
                project_box(vertices, full_pose, calibration) for vertices in self.housing_vertices
            ]
            for view, calibration in self.calibrations.items()
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.control_log.close()
        self.trace.close()
        atomic_json(self.output / "phase_events.json", self.phase_changes)
        atomic_json(
            self.output / "raw_rgb_manifest.json",
            dict(
                samples=self.rgb_records,
                purpose="Lossless1Hz physical setup/paired-render audit; never imitation data",
                local_training_labels_released=False,
            ),
        )
        for row in self.scripted_actors:
            if row["actor"].is_alive:
                row["actor"].destroy()
        atomic_json(
            self.output / "handoff.json",
            dict(
                handoff_time=self.prefix.handoff_time,
                last_timestamp=self.last_timestamp,
                prefix_mode=self.prefix.spec.mode,
                control_after_handoff="Original evaluated-agent controls only",
            ),
        )
