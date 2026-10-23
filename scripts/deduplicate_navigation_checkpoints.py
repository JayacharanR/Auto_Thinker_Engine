#!/usr/bin/env python3
"""Share verified duplicate checkpoint extents via COW; preserve every path and byte."""

import argparse
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

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve prior extent-sharing receipt")
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, args.out / "source.py")
    inventory = json.loads(args.inventory.read_text())
    atomic_json(
        args.out / "request.json",
        dict(
            inventory_sha256=digest(args.inventory),
            source_sha256=digest(__file__),
            semantics="Btrfs copy-on-write reflink, never hardlinks; no deleted logical artifacts",
        ),
    )
    free_before = shutil.disk_usage(ROOT).free
    # Exercise independent writes on tiny files in the actual destination FS.
    with tempfile.TemporaryDirectory(prefix="cow_probe_", dir=args.out) as directory:
        first, second = Path(directory) / "first", Path(directory) / "second"
        first.write_bytes(b"independent checkpoint bytes")
        subprocess.run(["cp", "--reflink=always", "--", str(first), str(second)], check=True)
        assert first.stat().st_ino != second.stat().st_ino
        with second.open("r+b") as stream:
            stream.write(b"Independent")
            stream.flush()
            os.fsync(stream.fileno())
        assert first.read_bytes() == b"independent checkpoint bytes"
    records = []
    for group in inventory["groups"]:
        source = ROOT / group["paths"][0]
        expected = group["sha256"]
        for relative in group["paths"][1:]:
            target = ROOT / relative
            if target.is_symlink() or source.is_symlink():
                raise ValueError("Only regular existing duplicate files are eligible")
            if digest(source) != expected or digest(target) != expected:
                raise ValueError("Inventory bytes changed; preserve the live checkpoint")
            previous = target.stat()
            temporary = target.with_name(f".{target.name}.cow_{os.getpid()}")
            with temporary.open("xb"):
                pass
            try:
                subprocess.run(
                    ["cp", "--reflink=always", "--preserve=all", "--", str(source), str(temporary)],
                    check=True,
                )
                shutil.copystat(target, temporary)
                if digest(temporary) != expected or digest(source) != expected:
                    raise ValueError("Cloned bytes changed; leave original target untouched")
                current = target.stat()
                if (current.st_ino, current.st_mtime_ns, current.st_size) != (
                    previous.st_ino,
                    previous.st_mtime_ns,
                    previous.st_size,
                ):
                    raise RuntimeError("Target changed during copy; refuse replacement")
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                parent = os.open(target.parent, os.O_DIRECTORY)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
                if digest(target) != expected or target.stat().st_ino == source.stat().st_ino:
                    raise RuntimeError("Post-replacement exact-byte/COW verification failed")
                records.append(
                    dict(
                        source=str(source),
                        target=str(target),
                        sha256=expected,
                        bytes=current.st_size,
                        exact_bytes_verified=True,
                        original_mtime_preserved=target.stat().st_mtime_ns == previous.st_mtime_ns,
                        different_inodes=True,
                        copy_on_write=True,
                    )
                )
                atomic_json(args.out / "records.json", records)
            finally:
                if temporary.exists():
                    temporary.unlink()
    result = dict(
        status="complete",
        records=records,
        free_bytes_before=free_before,
        free_bytes_after=shutil.disk_usage(ROOT).free,
        time=time.time(),
        tiny_clone_independent_write_verified=True,
        note=(
            'Actual filesystem free-space delta may lag transactions; logical '
            'bytes are not physical savings.'
        ),
    )
    atomic_json(args.out / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation_storage.lock"):
        main()
