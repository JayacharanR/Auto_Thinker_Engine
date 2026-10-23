"""Child-only connection to the exact simulator process started by its parent."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import psutil

from src.navigation.native_process_owner import require_process_identity
from src.navigation.runtime import atomic_json


def connect_owned_engine(identity_path, output, maximum_seconds=120):
    identity = json.loads(Path(identity_path).read_text())
    output = Path(output)
    if identity["parent_pid"] != os.getppid():
        raise ValueError("Direct native owning parent required")
    started = time.monotonic()

    def live_process():
        process = psutil.Process(identity["pid"])
        require_process_identity(
            identity,
            process.cmdline(),
            process.uids().real,
            os.getpgid(process.pid),
            process.create_time(),
        )
        return process

    def check():
        if (output / "STOP").exists() or time.monotonic() - started > maximum_seconds:
            raise InterruptedError("Owned child STOPorwall cap")
        if live_process().status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("Owned engine exited")

    while True:
        check()
        if any(
            c.status == psutil.CONN_LISTEN and c.laddr.port == identity["port"]
            for c in live_process().net_connections(kind="tcp")
        ):
            break
        time.sleep(0.2)
    from src.navigation.vendor import activate

    activate()
    import carla

    client = carla.Client("127.0.0.1", identity["port"])
    client.set_timeout(2.0)
    while True:
        check()
        try:
            actual = client.get_world().get_map().name
            break
        except RuntimeError:
            time.sleep(0.2)
    client.set_timeout(20.0)
    atomic_json(output / "startup_map.json", dict(actual=actual, requested=identity["town"]))

    def poll():
        try:
            return None if live_process().status() != psutil.STATUS_ZOMBIE else -1
        except psutil.NoSuchProcess:
            return -1

    return SimpleNamespace(process=SimpleNamespace(poll=poll)), client, check
