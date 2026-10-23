"""Simulator ownership without importing CARLA into the supervising Python process."""

import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

from src.navigation.runtime import atomic_json


def require_process_identity(saved, command, uid, group, created):
    """Never adopt another process, including a reused saved PID."""
    if (
        uid != os.getuid()
        or not command
        or command[0] != saved["binary"]
        or f"-carla-rpc-port={saved['port']}" not in command
        or group != saved["pid"]
        or abs(created - saved["create_time"]) > 0.01
    ):
        raise ValueError("Owned native process identity changed; no signal permitted")


class NativeEngineOwner:
    """One direct Popen ownership handle; STOP permits only normal SIGTERM cleanup.

    A separate child creates and releases every CARLA Client/world/actor. The
    supervisor waits for that child to exit before handling engine shutdown.
    """

    def __init__(self, root, output, town, port=2100):
        self.root = Path(root).expanduser().resolve()
        self.output = Path(output).resolve()
        if not re.fullmatch(r"Town[0-9]+(?:HD)?(?:_Opt)?", town):
            raise ValueError("Invalid town")
        self.town, self.port = town, int(port)
        self.process = None
        self.log = None
        self.identity = None

    def start(self):
        if self.process is not None:
            raise RuntimeError("Never start an owned engine twice")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError("Owned RPCport already occupied")
        if (self.root / "CarlaUE4/Content/Carla/Maps/TM").exists():
            raise RuntimeError("Existing TrafficManagercache workaround required")
        binary = self.root / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
        if not binary.is_file():
            raise FileNotFoundError(binary)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "server.json").exists():
            raise FileExistsError("Preserve prior owned engine")
        self.log = (self.output / "carla.log").open("ab")
        command = [
            str(binary),
            "CarlaUE4",
            f"/Game/Carla/Maps/{self.town}",
            "-RenderOffScreen",
            "-nosound",
            "-quality-level=Low",
            f"-carla-rpc-port={self.port}",
        ]
        self.process = subprocess.Popen(
            command,
            cwd=self.root,
            env=dict(os.environ, SDL_AUDIODRIVER="dummy"),
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.identity = dict(
            pid=self.process.pid,
            port=self.port,
            binary=str(binary),
            town=self.town,
            create_time=psutil.Process(self.process.pid).create_time(),
            parent_pid=os.getpid(),
            command=command,
            parent_never_imports_carla=True,
        )
        atomic_json(self.output / "server.json", self.identity)
        return self.identity

    def stop_after_request(self):
        if not (self.output / "STOP").exists():
            raise ValueError("Only owned case STOP permits native shutdown")
        if self.process is None or self.identity is None:
            raise ValueError("No direct owned engine handle")
        receipt = dict(
            identity=self.identity,
            signals=[],
            STOP_present=True,
            started_at=time.time(),
            force_kill_permitted=False,
        )
        for attempt in range(2):
            if self.process.poll() is not None:
                break
            actual = psutil.Process(self.process.pid)
            require_process_identity(
                self.identity,
                actual.cmdline(),
                actual.uids().real,
                os.getpgid(actual.pid),
                actual.create_time(),
            )
            os.killpg(self.process.pid, signal.SIGTERM)
            receipt["signals"].append(
                dict(signal="SIGTERM", normal_attempt=attempt + 1, at=time.time())
            )
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                atomic_json(
                    self.output / "cleanup_failed.json",
                    dict(
                        pid=self.process.pid,
                        reason="Normal shutdown timeout; one normal retry, no force kill",
                        normal_attempt=attempt + 1,
                        at=time.time(),
                    ),
                )
        finished = self.process.poll() is not None
        receipt.update(
            normal_shutdown_finished=finished,
            returncode=self.process.returncode,
            finished_at=time.time(),
        )
        atomic_json(self.output / "native_owner_cleanup.json", receipt)
        if self.log:
            self.log.close()
        if not finished:
            raise RuntimeError("Owned engine remains alive; stop queue, no force kill")
        return receipt


def run_native_worker(root, output, town, worker_command, check, phase, worker_wall_seconds=120):
    """One child owns all bindings; parent survives a native callback abort.

    worker_command receives --engine-identity and --out. It must check its owned
    output/STOP before each tick. The owner waits for child exit before ordinary
    teardown; an outer STOP is also forwarded if the child is still running.
    """
    if "carla" in sys.modules:
        raise RuntimeError("Native supervisor must never import CARLA bindings")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    engine = NativeEngineOwner(root, output / "engine", town)
    (output / "engine").mkdir()
    (output / "engine/STOP").symlink_to("../STOP")
    started = time.monotonic()
    child = None
    child_log = None
    stop_reason = None
    stop_at = None
    engine.start()
    try:
        child_log = (output / "worker.log").open("xb")
        command = list(worker_command) + [
            "--engine-identity",
            str(output / "engine/server.json"),
            "--out",
            str(output),
        ]
        child = subprocess.Popen(
            command,
            stdout=child_log,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parents[2],
            start_new_session=True,
        )
        atomic_json(
            output / "worker_identity.json",
            dict(
                pid=child.pid,
                command=command,
                started_at=time.time(),
                server_identity=engine.identity,
                parent_native_bindings_imported=False,
            ),
        )
        while child.poll() is None:
            try:
                check()
                if time.monotonic() - started > worker_wall_seconds:
                    raise InterruptedError("Owned native worker wall cap")
            except BaseException as error:
                if stop_reason is None:
                    stop_reason = str(error)
                    stop_at = time.monotonic()
                    (output / "STOP").touch()
                    atomic_json(
                        output / "STOP_reason.json", dict(reason=stop_reason, at=time.time())
                    )
            phase(dict(worker_pid=child.pid, worker_alive=True, stop_requested=stop_reason))
            if stop_at is not None and time.monotonic() - stop_at > 30:
                # The existing case STOP is the only permission for normal owned
                # engine teardown; no child SIGKILL or process substitution.
                engine.stop_after_request()
                try:
                    child.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        "Owned child remains alive after STOP; no force kill"
                    ) from None
            time.sleep(0.25)
        # All native client threads have exited with their child before the
        # engine is asked to stop. Cooldown occurs outside measured data.
        if stop_reason is None and engine.process.poll() is None:
            time.sleep(1)
        (output / "STOP").touch()
        atomic_json(
            output / "STOP_reason.json",
            dict(reason=stop_reason or "Native child exited; normal case teardown", at=time.time()),
        )
        cleanup = engine.stop_after_request()
        result = dict(
            child_returncode=child.returncode,
            native_returncode=engine.process.returncode,
            child_exit_before_normal_teardown=stop_reason is None,
            native_cleanup=cleanup,
            stop_reason=stop_reason,
            elapsed_seconds=time.monotonic() - started,
            parent_native_bindings_imported="carla" in sys.modules,
        )
        atomic_json(output / "worker_result.json", result)
        if stop_reason is not None:
            raise InterruptedError(stop_reason)
        return result
    finally:
        if engine.process is not None and engine.process.poll() is None:
            (output / "STOP").touch()
            engine.stop_after_request()
        if child_log:
            child_log.close()
