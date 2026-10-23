#!/usr/bin/env python3
"""Kernel-verified COW deduplication; never modify checkpoint logical bytes."""

import argparse
import fcntl
import json
import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402

FIDEDUPERANGE = 0xC0189436  # Linux fs.h _IOWR(0x94,54,24-byte file_dedupe_range).


def same_extent(source, target, source_offset, target_offset, length):
    buffer = bytearray(
        struct.pack(
            "=QQHHIqQQiI", source_offset, length, 1, 0, 0, target.fileno(), target_offset, 0, 0, 0
        )
    )
    fcntl.ioctl(source.fileno(), FIDEDUPERANGE, buffer, True)
    _, _, done, status, _ = struct.unpack_from("=qQQiI", buffer, 24)
    return dict(bytes_deduped=done, status=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve every extent-sharing receipt")
    inventory = json.loads(args.inventory.read_text())
    if inventory["errors"] or not inventory["read_only"]:
        raise ValueError("A complete read-only inventory is required")
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, args.out / "source.py")
    atomic_json(
        args.out / "request.json",
        dict(
            inventory_sha256=digest(args.inventory),
            source_sha256=digest(__file__),
            local_kernel_header_sha256=digest("/usr/include/linux/fs.h"),
            semantics="FIDEDUPERANGE compares all bytes in-kernel; "
            "independent COW extents, no logical deletion",
        ),
    )
    # Test actual filesystem safety and mismatching-byte rejection before models.
    with tempfile.TemporaryDirectory(prefix="extent_probe_", dir=args.out) as directory:
        a, b, c = [Path(directory) / name for name in ("source", "equal", "different")]
        original = b"0123456789abcdef" * 1024
        a.write_bytes(original)
        b.write_bytes(original)
        c.write_bytes(original[:-1] + b"!")
        with a.open("rb") as source, b.open("r+b") as target, c.open("r+b") as different:
            equal_result = same_extent(source, target, 0, 0, len(original))
            different_result = same_extent(source, different, 0, 0, len(original))
        if equal_result != dict(bytes_deduped=len(original), status=0):
            raise ValueError("Filesystem identical-byte sharing probe failed")
        if different_result != dict(bytes_deduped=0, status=1):
            raise ValueError("Filesystem must reject unequal bytes without changing either file")
        with b.open("r+b") as target:
            target.write(b"independent")
        if a.read_bytes() != original or c.read_bytes() != original[:-1] + b"!":
            raise ValueError("Copy-on-write or unequal-byte preservation failed")
    before_free = shutil.disk_usage(ROOT).free
    for relative, expected in inventory["files"].items():
        path = ROOT / relative
        if (
            not str(path.resolve()).startswith(str(ROOT / "outputs") + "/")
            or path.is_symlink()
            or path.stat().st_mtime_ns != expected["mtime_ns"]
            or digest(path) != expected["sha256"]
        ):
            raise ValueError(f"Historical file changed; refuse extent sharing:{relative}")
    records = []
    for group in inventory["groups"]:
        source = group["ranges"][0]
        with (ROOT / source["path"]).open("rb") as first:
            for target in group["ranges"][1:]:
                if any(
                    r["offset"] % 4096 or r["bytes"] != group["bytes"] for r in (source, target)
                ):
                    raise ValueError("Only matching4KiB-aligned tensor byte ranges are allowed")
                done = 0
                with (ROOT / target["path"]).open("r+b") as second:
                    while done < group["bytes"]:
                        length = min(1 << 20, group["bytes"] - done)
                        result = same_extent(
                            first, second, source["offset"] + done, target["offset"] + done, length
                        )
                        if result["status"] != 0 or result["bytes_deduped"] != length:
                            raise RuntimeError(
                                f"Kernel range sharing refused:{result}; originals remain intact"
                            )
                        done += length
                records.append(dict(source=source, target=target, kernel_compared_equal_bytes=done))
        atomic_json(args.out / "records.json", records)
    verified = []
    for relative, expected in inventory["files"].items():
        path = ROOT / relative
        if digest(path) != expected["sha256"] or path.stat().st_mtime_ns != expected["mtime_ns"]:
            raise RuntimeError(f"Post-sharing whole-file preservation check failed:{relative}")
        verified.append(relative)
    fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        import ctypes

        if ctypes.CDLL(None, use_errno=True).syncfs(fd):
            raise OSError(ctypes.get_errno(), "syncfs failed")
    finally:
        os.close(fd)
    after_free = shutil.disk_usage(ROOT).free
    summary = dict(
        status="complete",
        files_verified=len(verified),
        range_pairs=len(records),
        kernel_shared_bytes=sum(r["kernel_compared_equal_bytes"] for r in records),
        free_bytes_before=before_free,
        free_bytes_after=after_free,
        actual_free_space_delta=after_free - before_free,
        all_whole_file_sha256_and_mtimes_preserved=True,
        independent_write_and_unequal_byte_rejection_test_passed=True,
        records=records,
        time=time.time(),
    )
    atomic_json(args.out / "summary.json", summary)
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            stage="Verified COW storage recovery",
            frames=len(records),
            target_frames=len(records),
            phase="All historical bytes and mtimes preserved",
            updated_at=time.time(),
        ),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation_storage.lock"):
        main()
