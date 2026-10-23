#!/usr/bin/env python3
"""Validate a mounted data partition and register space for future recordings.

This never mounts, formats, repairs, or removes data from a drive. With --wait,
it waits for the ordinary desktop mount authorization to finish. A budget is
an accounting allowance, not a filesystem quota or preallocated disk image.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.runtime import atomic_json  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mounted_partition(device, uuid):
    """Check both the block identity and the actual mounted filesystem."""
    info = json.loads(
        subprocess.check_output(
            ["lsblk", "--json", "-o", "PATH,TYPE,FSTYPE,LABEL,UUID", str(device)],
            text=True,
        )
    )["blockdevices"]
    if len(info) != 1 or info[0]["uuid"] != uuid or info[0]["type"] != "part":
        raise ValueError("Selected partition identity changed; do not use another disk")
    result = subprocess.run(
        ["findmnt", "--json", "--source", str(device), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 1 and not result.stdout.strip():
        return None
    result.check_returncode()
    mounts = json.loads(result.stdout)["filesystems"]
    if len(mounts) != 1:
        raise ValueError("Expected one unambiguous mount for the selected partition")
    mount = mounts[0]
    if "ro" in mount["options"].split(","):
        raise ValueError("Selected partition is read-only; no repair or force mount attempted")
    mount["block_identity"] = info[0]
    return mount


def probe_writes(directory):
    """Exercise actual exclusive creation, durability, and atomic replacement."""
    payload = os.urandom(1024 * 1024)
    expected = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory(prefix=".storage_probe_", dir=directory) as temporary:
        base = Path(temporary)
        pending, committed = base / "pending.bin", base / "committed.bin"
        with pending.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            with pending.open("xb"):
                pass
        except FileExistsError:
            pass
        else:
            raise RuntimeError("Exclusive file creation did not reject an existing path")
        with committed.open("xb") as stream:
            stream.write(b"synthetic prior version")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, committed)
        descriptor = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if pending.exists() or digest(committed) != expected:
            raise RuntimeError("Atomic replacement or read-back digest failed")
    return dict(
        bytes=1024 * 1024,
        sha256=expected,
        exclusive_create=True,
        file_fsync=True,
        atomic_replace=True,
        directory_fsync=True,
        read_back_verified=True,
        only_synthetic_probe_files_removed=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=Path, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--subdir", default="Auto_Thinker_Engine/navigation_20261020")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--working-budget-gb", type=float, default=100)
    parser.add_argument("--minimum-working-gb", type=float, default=40)
    parser.add_argument("--reserve-gb", type=float, default=25)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    relative = Path(args.subdir)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        parser.error("subdir must be a relative directory without parent traversal")
    if not 0 < args.minimum_working_gb <= args.working_budget_gb or args.reserve_gb < 25:
        parser.error("Require >=40 GB working capacity and the approved >=25 GB reserve")
    if args.minimum_working_gb < 40:
        parser.error("This recording plan needs at least40 GB of additional working capacity")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / "provision_source.py"
    shutil.copyfile(__file__, source)
    atomic_json(
        args.out / "request.json",
        dict(
            device=str(args.device),
            uuid=args.uuid,
            subdir=str(relative),
            desired_working_budget_bytes=int(args.working_budget_gb * 1e9),
            minimum_working_bytes=int(args.minimum_working_gb * 1e9),
            reserve_bytes=int(args.reserve_gb * 1e9),
            budget_is_accounting_allowance_not_quota=True,
            user_authorization="Choose a directory and use larger storage allocation",
            source_sha256=digest(source),
            created_at=time.time(),
        ),
    )

    def progress(status, phase, **extra):
        atomic_json(
            args.out / "progress.json",
            dict(status=status, phase=phase, updated_at=time.time(), **extra),
        )

    try:
        while True:
            if (args.out / "STOP").exists():
                progress("stopped", "Storage preparation STOP; no mount or training affected")
                return
            mount = mounted_partition(args.device, args.uuid)
            if mount is not None:
                break
            progress("waiting_for_mount", "Approve Games mount through the desktop or terminal")
            if not args.wait:
                return
            time.sleep(10)
        target = Path(mount["target"]).resolve()
        if os.stat(target).st_dev == os.stat(ROOT).st_dev:
            raise ValueError("Mounted path shares the project filesystem; no new capacity")
        if shutil.disk_usage(ROOT).free < 25e9:
            raise RuntimeError("Project filesystem is below its unchanged25 GB reserve")
        usage = shutil.disk_usage(target)
        allowance = min(int(args.working_budget_gb * 1e9), usage.free - int(args.reserve_gb * 1e9))
        if allowance < args.minimum_working_gb * 1e9:
            raise RuntimeError("Mounted partition lacks40 GB working space plus25 GB reserve")
        directory = target
        for component in relative.parts:
            directory = directory / component
            if directory.is_symlink():
                raise ValueError("Refuse an existing symlink in the chosen storage directory")
            directory.mkdir(exist_ok=True)
        receipt = directory / "navigation_storage.json"
        if receipt.exists() or any(directory.iterdir()):
            raise FileExistsError("Chosen storage directory already contains data; preserve it")
        progress(
            "running", "Checking actual mounted filesystem writes", storage_root=str(directory)
        )
        checks = probe_writes(directory)
        identity = mounted_partition(args.device, args.uuid)
        if identity != mount or os.stat(directory).st_dev != os.stat(target).st_dev:
            raise RuntimeError("Mount identity changed during validation")
        record = dict(
            schema="navigation-recording-storage-v1",
            storage_root=str(directory),
            mount=mount,
            filesystem_total_bytes=usage.total,
            filesystem_free_bytes_after_probe=shutil.disk_usage(directory).free,
            working_budget_bytes=allowance,
            reserve_bytes=int(args.reserve_gb * 1e9),
            budget_is_accounting_allowance_not_quota=True,
            no_space_preallocated=True,
            no_retained_artifacts_moved_or_deleted=True,
            probe=checks,
            source_sha256=digest(source),
            request_sha256=digest(args.out / "request.json"),
            validated_at=time.time(),
        )
        atomic_json(receipt, record)
        atomic_json(args.out / "storage.json", record)
        progress("complete", "External recording storage validated", storage_root=str(directory))
        print(
            json.dumps(dict(storage_root=str(directory), working_budget_bytes=allowance)),
            flush=True,
        )
    except Exception as error:
        atomic_json(args.out / "error.json", dict(type=type(error).__name__, error=str(error)))
        progress("failed", "Storage not qualified; existing artifacts untouched")
        raise


if __name__ == "__main__":
    main()
