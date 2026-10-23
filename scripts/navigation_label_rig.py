"""TRAIN-only synchronized RGB/instance/depth label rig; never a policy dependency."""

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from src.navigation.contracts import VIEWS
from src.navigation.sensors import FrameBarrier, calibrations, camera_specs


def decode_depth(bgra):
    if bgra.dtype != np.uint8 or bgra.ndim != 3 or bgra.shape[-1] != 4:
        raise ValueError("Raw CARLA BGRA depth bytes required")
    raw = bgra.astype(np.uint32)
    return (
        1000 * (raw[..., 2] + 256 * raw[..., 1] + 65536 * raw[..., 0]).astype(np.float64) / 16777215
    ).astype(np.float32)


def decode_instances(bgra):
    if bgra.dtype != np.uint8 or bgra.ndim != 3 or bgra.shape[-1] != 4:
        raise ValueError("Raw CARLA BGRA instance bytes required")
    # Keep this tuple lossless. Render-component IDs are NOT claimed as actor IDs.
    return bgra[..., 2].copy(), bgra[..., [1, 0]].copy()


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class LabelRig:
    def __init__(self, world, ego, config, labels, include_rgb=True):
        import carla

        self.sensors = []
        specs = camera_specs(config)
        self.names = [(spec["id"], "rgb") for spec in specs] if include_rgb else []
        if labels:
            self.names += [(spec["id"], kind) for spec in specs for kind in ("instance", "depth")]
        self.barrier = FrameBarrier([f"{view}_{kind}" for view, kind in self.names])
        self.calibration = calibrations(config)
        self.specification = []
        try:
            for view, kind in self.names:
                spec = next(row for row in specs if row["id"] == view)
                type_id = {
                    "rgb": "sensor.camera.rgb",
                    "depth": "sensor.camera.depth",
                    "instance": "sensor.camera.instance_segmentation",
                }[kind]
                bp = world.get_blueprint_library().find(type_id)
                for key, value in dict(
                    image_size_x=spec["width"],
                    image_size_y=spec["height"],
                    fov=spec["fov"],
                    sensor_tick=0.0,
                ).items():
                    bp.set_attribute(key, str(value))
                transform = carla.Transform(
                    carla.Location(x=spec["x"], y=spec["y"], z=spec["z"]),
                    carla.Rotation(pitch=spec["pitch"], yaw=spec["yaw"], roll=spec["roll"]),
                )
                sensor = world.spawn_actor(bp, transform, attach_to=ego)
                self.sensors.append(sensor)
                self.specification.append(
                    dict(view=view, kind=kind, type=type_id, relative_transform=spec)
                )
                sensor.listen(self.callback(f"{view}_{kind}"))
        except BaseException:
            self.close()
            raise

    def callback(self, name):
        def receive(image):
            raw = (
                np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4).copy()
            )
            transform = image.transform
            pose = [
                transform.location.x,
                transform.location.y,
                transform.location.z,
                transform.rotation.pitch,
                transform.rotation.yaw,
                transform.rotation.roll,
            ]
            self.barrier.put(name, image.frame, image.timestamp, (raw, pose))

        return receive

    def collect(self, frame):
        values = self.barrier.collect(frame)
        rgb = {
            view: np.ascontiguousarray(values[f"{view}_rgb"][2][0][..., :3][..., ::-1])
            for view in VIEWS
            if f"{view}_rgb" in values
        }
        labels = {
            f"{view}_{kind}": values[f"{view}_{kind}"][2][0]
            for view, kind in self.names
            if kind != "rgb"
        }
        poses = {name: value[2][1] for name, value in values.items()}
        return rgb, labels, next(iter(values.values()))[1], poses

    def close(self):
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()


class LabelShards:
    """Small exclusive atomic shards; raw pixels are verified before publication."""

    def __init__(self, root, labels, samples_per_shard=20):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.labels, self.limit = labels, samples_per_shard
        self.archive, self.part, self.count, self.index = None, None, 0, 0
        self.raw_bytes = 0
        self.shards = []

    def append(self, frame, timestamp, rgb, labels):
        expected = {f"{view}_{kind}" for view in VIEWS for kind in ("instance", "depth")}
        if set(labels) != (expected if self.labels else set()) or set(rgb) != set(VIEWS):
            raise ValueError("RGB inputs and privileged label fields must match the registered rig")
        arrays = {f"rgb_{view}": rgb[view] for view in VIEWS} | labels
        for name, value in arrays.items():
            channels = 3 if name.startswith("rgb_") else 4
            if value.dtype != np.uint8 or value.shape != (288, 512, channels):
                raise ValueError("Preserve native512×288 raw bytes")
        if self.archive is None:
            self.part = self.root / f"shard_{self.index:06d}.zip.part"
            self.archive = zipfile.ZipFile(
                self.part, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1
            )
        prefix = f"{self.count:08d}"
        hashes = {}
        for name, value in arrays.items():
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            self.archive.writestr(f"{prefix}/{name}.npy", stream.getvalue())
            hashes[name] = digest_array(value)
            self.raw_bytes += value.nbytes
        self.archive.writestr(
            f"{prefix}/metadata.json",
            json.dumps(
                dict(
                    frame=frame,
                    timestamp=timestamp,
                    pixel_sha256=hashes,
                    privileged_labels_not_policy_inputs=self.labels,
                )
            ),
        )
        self.count += 1
        if self.count % self.limit == 0:
            self.commit()

    def commit(self):
        if self.archive is None:
            return
        self.archive.close()
        self.archive = None
        with zipfile.ZipFile(self.part) as archive:
            for name in archive.namelist():
                if not name.endswith("metadata.json"):
                    continue
                metadata = json.loads(archive.read(name))
                prefix = name.rsplit("/", 1)[0]
                for field, expected in metadata["pixel_sha256"].items():
                    array = np.load(
                        io.BytesIO(archive.read(f"{prefix}/{field}.npy")), allow_pickle=False
                    )
                    if digest_array(array) != expected:
                        raise ValueError("Raw label/RGB shard changed on readback")
        with self.part.open("rb") as stream:
            os.fsync(stream.fileno())
        target = self.part.with_suffix("")
        os.replace(self.part, target)
        self.shards.append(
            dict(
                path=str(target),
                bytes=target.stat().st_size,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
        )
        self.part = None
        self.index += 1

    def close(self):
        self.commit()
