"""Privileged TRAIN-only RGB/semantic pixel registration probe, never inference."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.navigation.rule_geometry import camera_matrix
from src.navigation.runtime import atomic_json
from src.navigation.sensors import FrameBarrier, calibrations, camera_specs

LENS_NAMES = ("lens_k", "lens_kcube", "lens_x_size", "lens_y_size", "lens_circle_multiplier")


def lens_values(blueprint):
    return {
        key: blueprint.get_attribute(key).as_float()
        for key in LENS_NAMES
        if blueprint.has_attribute(key)
    }


def capture_label_pixels(world, config, corpus, case_id, out, stop=None):
    import carla

    from scripts.probe_navigation_signal_pixels import capture_static_frame
    from src.navigation.signal_learning import junction_roles

    corpus, out = Path(corpus), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((corpus / "plan.json").read_text())
    town = world.get_map().name.split("/")[-1]
    case = next(
        c
        for c in json.loads((corpus / town / "cases.json").read_text())
        if c["route_id"] == case_id
    )
    if junction_roles(plan["cases"])[town, case["junction_id"]] != "train":
        raise ValueError("Semantic diagnostic requires component-TRAIN pose")
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    def full_pose(actor):
        t = actor.get_transform()
        return [
            t.location.x,
            t.location.y,
            t.location.z,
            t.rotation.pitch,
            t.rotation.yaw,
            t.rotation.roll,
        ]

    rng = np.random.default_rng(case["background_seed"])
    for light in sorted(lights, key=lambda a: tuple(full_pose(a))):
        light.freeze(True)
        light.set_state(
            [
                carla.TrafficLightState.Red,
                carla.TrafficLightState.Green,
                carla.TrafficLightState.Yellow,
            ][int(rng.integers(3))]
        )
    target = [
        light
        for light in lights
        if np.linalg.norm(np.array(full_pose(light)[:3]) - case["inventory_light_xyz"]) < 0.002
    ]
    if len(target) != 1:
        raise ValueError("Source target light lacks unique physical identity")
    target[0].set_state(carla.TrafficLightState.Red)
    p = case["ego_pose"]
    transform = carla.Transform(
        carla.Location(*p[:3]), carla.Rotation(pitch=p[3], yaw=p[4], roll=p[5])
    )
    settings = world.get_settings()
    synchronous = world.get_settings()
    synchronous.synchronous_mode, synchronous.fixed_delta_seconds = True, 0.05
    hero, sensors = None, []
    names = [
        f"{kind}_{view}" for view in ("front", "left", "right") for kind in ("rgb", "semantic")
    ]
    barrier = FrameBarrier(names)
    optics = {}

    def callback(name, kind):
        def receive(image):
            bgra = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
            value = dict(
                pose=[
                    image.transform.location.x,
                    image.transform.location.y,
                    image.transform.location.z,
                    image.transform.rotation.pitch,
                    image.transform.rotation.yaw,
                    image.transform.rotation.roll,
                ],
                fov=image.fov,
            )
            if kind == "semantic":
                value["tags"] = bgra[:, :, 2].copy()
                image.convert(carla.ColorConverter.CityScapesPalette)
                bgra = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
            value["rgb"] = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
            barrier.put(name, image.frame, image.timestamp, value)

        return receive

    world.apply_settings(synchronous)
    try:
        bp = world.get_blueprint_library().find(case["ego_blueprint"])
        bp.set_attribute("role_name", "hero")
        hero = world.spawn_actor(
            bp, carla.Transform(carla.Location(p[0], p[1], p[2] + 3), transform.rotation)
        )
        hero.set_simulate_physics(False)
        hero.set_transform(transform)
        for spec in camera_specs(config):
            rgb_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            for kind, blueprint in (
                ("rgb", rgb_bp),
                (
                    "semantic",
                    world.get_blueprint_library().find("sensor.camera.semantic_segmentation"),
                ),
            ):
                for key, value in {
                    "image_size_x": spec["width"],
                    "image_size_y": spec["height"],
                    "fov": spec["fov"],
                    "sensor_tick": 0.0,
                }.items():
                    blueprint.set_attribute(key, str(value))
                if kind == "semantic":
                    for key in LENS_NAMES:
                        if blueprint.has_attribute(key) and rgb_bp.has_attribute(key):
                            blueprint.set_attribute(key, str(rgb_bp.get_attribute(key).as_float()))
                name = f"{kind}_{spec['id']}"
                optics[name] = lens_values(blueprint)
                mount = carla.Transform(
                    carla.Location(x=spec["x"], y=spec["y"], z=spec["z"]),
                    carla.Rotation(pitch=spec["pitch"], yaw=spec["yaw"], roll=spec["roll"]),
                )
                sensor = world.spawn_actor(blueprint, mount, attach_to=hero)
                sensors.append(sensor)
                if any(
                    abs(float(sensor.attributes[key]) - value) > 1e-6
                    for key, value in optics[name].items()
                ):
                    raise ValueError("Spawned sensor lens attributes differ from request")
                sensor.listen(callback(name, kind))
            if optics[f"rgb_{spec['id']}"] != optics[f"semantic_{spec['id']}"]:
                raise ValueError("RGB/semantic lens attributes do not match")
        frame, captured = capture_static_frame(
            world, SimpleNamespace(barrier=barrier), stop or out / "STOP", 20
        )
        enum = {
            name: int(getattr(carla.CityObjectLabel, name))
            for name in (
                "TrafficLight",
                "TrafficSigns",
                "Poles",
                "Buildings",
                "Vegetation",
                "Roads",
                "Sky",
            )
        }
        metadata = {}
        for name, (seen_frame, timestamp, value) in captured.items():
            if seen_frame != frame:
                raise ValueError("Frame mismatch")
            view = name.split("_", 1)[1]
            expected = camera_matrix(case["ego_pose"], calibrations(config)[view])
            actual = carla.Transform(
                carla.Location(*value["pose"][:3]),
                carla.Rotation(pitch=value["pose"][3], yaw=value["pose"][4], roll=value["pose"][5]),
            )
            error = float(np.max(np.abs(expected - np.array(actual.get_matrix()))))
            if error > 1e-3:
                raise ValueError("Camera image pose mismatch")
            Image.fromarray(value["rgb"]).save(out / f"probe_{name}.png")
            metadata[name] = dict(
                frame=seen_frame,
                timestamp=timestamp,
                pose=value["pose"],
                fov=value["fov"],
                matrix_error=error,
            )
            if "tags" in value:
                Image.fromarray(value["tags"]).save(out / f"probe_tags_{view}.png")
                metadata[name]["tag_counts"] = {
                    str(int(k)): int(v)
                    for k, v in zip(*np.unique(value["tags"], return_counts=True))
                }
        atomic_json(
            out / "pixel_probe.json",
            dict(
                case=case,
                frame=frame,
                sensors=metadata,
                optics=optics,
                semantic_enum=enum,
                weather=str(world.get_weather()),
                diagnostic_cadence="20Hz all-frame synchronous capture; no driving/policy",
                inputs="RGB unchanged mount/FOV/resolution; semantic sensor labels only",
            ),
        )
    finally:
        for sensor in sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        if hero is not None and hero.is_alive:
            hero.destroy()
        world.apply_settings(settings)
