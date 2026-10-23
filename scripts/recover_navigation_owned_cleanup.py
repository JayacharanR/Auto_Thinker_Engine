#!/usr/bin/env python3
"""Retry only normal cleanup of a verified owned CARLA after its STOP request."""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.runtime import RunLock, atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.out / "STOP").exists():
        raise ValueError("Only an existing queue STOP request permits owned cleanup")
    server_path = args.out / "server.json"
    server = json.loads(server_path.read_text())
    failed = json.loads((args.out / "cleanup_failed.json").read_text())
    progress = json.loads((args.out / "progress.json").read_text())
    if failed["pid"] != server["pid"] or progress["status"] not in ("failed", "stopped"):
        raise ValueError("Do not adopt an active or unrelated simulator")
    pid = server["pid"]
    if not psutil.pid_exists(pid):
        print("Owned simulator already exited")
        return
    process = psutil.Process(pid)
    command = process.cmdline()
    if (
        process.uids().real != os.getuid()
        or not command
        or command[0] != server["binary"]
        or f"-carla-rpc-port={server['port']}" not in command
        or os.getpgid(pid) != pid
        or abs(process.create_time() - server_path.stat().st_mtime) > 10
    ):
        raise ValueError("Saved process identity/group/creation time does not match")
    receipt = dict(
        pid=pid,
        command=command,
        create_time=process.create_time(),
        stop_file_present=True,
        prior_cleanup_timeout=failed,
        signal="SIGTERM only; never SIGKILL or force escalation",
        requested_at=time.time(),
    )
    path = args.out / "normal_cleanup_retry.json"
    if path.exists():
        raise FileExistsError("One deliberate normal cleanup retry; preserve its receipt")
    atomic_json(path, receipt)
    os.killpg(pid, signal.SIGTERM)
    _, alive = psutil.wait_procs([process], timeout=20)
    receipt.update(
        normal_shutdown_finished=not alive,
        stopped_processes=[p.pid for p in alive],
        finished_at=time.time(),
    )
    atomic_json(path, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation.lock"):
        main()
