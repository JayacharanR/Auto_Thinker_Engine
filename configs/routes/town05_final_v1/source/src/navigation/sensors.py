"""Frame-exact RGB collection; no latest-frame or silent zero-image fallback."""

import queue
import time

import numpy as np

from src.navigation.contracts import VIEWS, CameraCalibration


class SensorTimeout(RuntimeError):
    pass


class FrameBarrier:
    def __init__(self, names=VIEWS, capacity=64):
        self.names = tuple(names)
        self.queues = {name: queue.Queue(maxsize=capacity) for name in names}
        self.pending = {}
        self.overflow = False

    def put(self, name, frame, timestamp, array):
        if name not in self.queues:
            raise KeyError(name)
        try:
            self.queues[name].put_nowait((frame, timestamp, array))
        except queue.Full:
            self.overflow = True

    def collect(self, frame, timeout=10.0):
        deadline = time.monotonic() + timeout
        output = {}
        for name in self.names:
            while True:
                if self.overflow:
                    raise SensorTimeout("Camera queue overflow; frames were lost")
                item = self.pending.pop(name, None)
                if item is None:
                    try:
                        item = self.queues[name].get(timeout=max(0, deadline - time.monotonic()))
                    except queue.Empty as exc:
                        raise SensorTimeout(f"Missing {name} frame {frame}") from exc
                if item[0] < frame:
                    continue
                if item[0] > frame:
                    self.pending[name] = item
                    raise SensorTimeout(f"Missing {name} frame {frame}; received {item[0]}")
                output[name] = item
                break
        times = [item[1] for item in output.values()]
        if max(times) - min(times) > 1e-6:
            raise SensorTimeout("Camera timestamps disagree for the same frame")
        return output


def camera_specs(config):
    c = config["camera"]
    return [
        {
            "type": "sensor.camera.rgb",
            "id": view,
            "x": c["x"],
            "y": c["y"],
            "z": c["z"],
            "pitch": c["pitch"],
            "roll": c["roll"],
            "yaw": c["views"][view],
            "width": c["width"],
            "height": c["height"],
            "fov": c["fov"],
        }
        for view in VIEWS
    ]


def calibrations(config):
    return {
        s["id"]: CameraCalibration(
            s["width"],
            s["height"],
            s["fov"],
            (s["x"], s["y"], s["z"]),
            (s["pitch"], s["yaw"], s["roll"]),
        )
        for s in camera_specs(config)
    }


class CameraRig:
    def __init__(self, world, ego, config):
        import carla

        self.barrier = FrameBarrier()
        self.sensors = []
        self.calibration = calibrations(config)
        try:
            for spec in camera_specs(config):
                bp = world.get_blueprint_library().find(spec["type"])
                for key, value in {
                    "image_size_x": spec["width"],
                    "image_size_y": spec["height"],
                    "fov": spec["fov"],
                    "sensor_tick": 1 / config["runtime"]["camera_hz"],
                }.items():
                    bp.set_attribute(key, str(value))
                transform = carla.Transform(
                    carla.Location(x=spec["x"], y=spec["y"], z=spec["z"]),
                    carla.Rotation(pitch=spec["pitch"], yaw=spec["yaw"], roll=spec["roll"]),
                )
                sensor = world.spawn_actor(bp, transform, attach_to=ego)
                self.sensors.append(sensor)
                sensor.listen(self._callback(spec["id"]))
        except BaseException:
            self.close()
            raise

    def _callback(self, name):
        def receive(image):
            bgra = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
            rgb = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
            self.barrier.put(name, image.frame, image.timestamp, rgb)

        return receive

    def close(self):
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
