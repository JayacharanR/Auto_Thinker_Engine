#!/usr/bin/env python3
"""Measure physical free-space deltas for fresh copies, retaining all originals."""

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


def sync(path):
    subprocess.run(["sync", "-f", str(path)], check=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.out / "profile_source.py")
    episode = next((args.source / "episodes").iterdir())
    paths = [
        *(episode / "world_trace.jsonl" for _ in range(3)),
        episode / "trace.jsonl",
        next((args.source / "recordings").glob("*.log")),
        episode / "three_cameras.mp4",
    ]
    rows = []
    sync(args.out)
    initial_free = shutil.disk_usage(args.out).free
    with tempfile.TemporaryDirectory(prefix=".physical_probe_", dir=args.out) as folder:
        for index, source in enumerate(paths):
            sync(folder)
            before = shutil.disk_usage(folder).free
            # Conservative preflight uses logical bytes, even if compression helps.
            if before - source.stat().st_size < 25e9:
                raise RuntimeError("Synthetic copy could breach the unchanged25 GB reserve")
            dest = Path(folder) / f"{index}.copy"
            digest = hashlib.sha256()
            started = time.monotonic()
            with source.open("rb") as reader, dest.open("xb") as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    digest.update(chunk)
                    writer.write(chunk)  # Explicit writes, never copy_file_range/reflink.
                writer.flush()
                os.fsync(writer.fileno())
            if hashlib.sha256(dest.read_bytes()).hexdigest() != digest.hexdigest():
                raise ValueError("Synthetic copy read-back differs from source")
            sync(dest)
            after = shutil.disk_usage(folder).free
            rows.append(
                dict(
                    source=str(source),
                    source_sha256=digest.hexdigest(),
                    logical_bytes=source.stat().st_size,
                    free_before_bytes=before,
                    free_after_bytes=after,
                    observed_physical_delta_bytes=before - after,
                    exact_read_back=True,
                    elapsed_s=time.monotonic() - started,
                )
            )
    sync(args.out)
    atomic_json(
        args.out / "profile.json",
        dict(
            results=rows,
            initial_free_bytes=initial_free,
            final_free_bytes=shutil.disk_usage(args.out).free,
            source_files_modified=False,
            only_owned_synthetic_copies_removed=True,
            note=(
                "Actual filesystem free-space changes on explicit fresh writes. "
                "Concurrent training writes/metadata introduce measurement noise. "
                "Not a promise that other scenes compress by the same ratio."
            ),
        ),
    )
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
