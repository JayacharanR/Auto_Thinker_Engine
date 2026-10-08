"""Owned-process CARLA lifecycle and atomic, inspectable progress records."""

import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import psutil


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class RunLock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError(f"Another navigation run holds {self.path}") from None
        return self

    def __exit__(self, *args):
        self.file.close()


def check_resources(output, minimum_free_gb):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free / 1e9
    if free < minimum_free_gb:
        raise RuntimeError(f"Only {free:.1f} GB free; require {minimum_free_gb}")
    pattern = re.compile(
        r"^bash jobs/|^[^ ]*python[0-9.]* (-u )?[^ ]*scripts/"
        r"(train_|evaluate_|probe_|run_bev_|run_traffic_|run_collision_|run_overnight_)"
        r"|^[^ ]*CarlaUE4-Linux"
    )
    for proc in psutil.process_iter(["pid", "cmdline", "status"]):
        try:
            command = " ".join(proc.info["cmdline"] or [])
            if proc.info["status"] != psutil.STATUS_ZOMBIE and pattern.search(command):
                raise RuntimeError(f"Existing simulation/training process {proc.pid}: {command}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return free


class CarlaServer:
    """Never attach to or stop an unowned server; never escalate to SIGKILL."""

    def __init__(self, root, output, port=2100, startup_timeout=120, town="Town03"):
        self.root = Path(root).expanduser().resolve()
        self.output = Path(output).resolve()
        self.port = port
        self.startup_timeout = startup_timeout
        if not re.fullmatch(r"Town[0-9]+(?:HD)?(?:_Opt)?", town):
            raise ValueError("Invalid CARLA map name")
        self.town = town
        self.process = None
        self.log = None

    def __enter__(self):
        import carla

        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"RPC port {self.port} is already occupied")
        cache = self.root / "CarlaUE4/Content/Carla/Maps/TM"
        if cache.exists():
            raise RuntimeError(
                "Traffic Manager cache is enabled; apply existing configure_carla fix"
            )
        binary = self.root / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
        if not binary.exists():
            raise FileNotFoundError(binary)
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = (self.output / "carla.log").open("a")
        env = dict(os.environ, SDL_AUDIODRIVER="dummy")
        self.process = subprocess.Popen(
            [
                str(binary),
                "CarlaUE4",
                f"/Game/Carla/Maps/{self.town}",
                "-RenderOffScreen",
                "-nosound",
                "-quality-level=Low",
                f"-carla-rpc-port={self.port}",
            ],
            cwd=self.root,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        atomic_json(
            self.output / "server.json",
            {"pid": self.process.pid, "port": self.port, "binary": str(binary)},
        )
        deadline = time.monotonic() + self.startup_timeout
        try:
            while time.monotonic() < deadline:
                if (self.output / "STOP").exists():
                    raise InterruptedError("STOP requested during simulator startup")
                if self.process.poll() is not None:
                    raise RuntimeError(f"CARLA exited {self.process.returncode}; see carla.log")
                client = carla.Client("127.0.0.1", self.port)
                client.set_timeout(2)
                try:
                    world = client.get_world()
                    atomic_json(
                        self.output / "startup_map.json",
                        {"actual": world.get_map().name, "requested": self.town},
                    )
                    client.set_timeout(20)
                    return client
                except RuntimeError:
                    time.sleep(1)
            raise TimeoutError("CARLA startup timed out")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                atomic_json(
                    self.output / "cleanup_failed.json",
                    {"pid": self.process.pid, "reason": "SIGTERM timeout; no force kill"},
                )
                raise RuntimeError(
                    "Owned CARLA did not exit after SIGTERM; see cleanup_failed.json"
                )
        if self.log:
            self.log.close()

    def __exit__(self, *args):
        self.close()
