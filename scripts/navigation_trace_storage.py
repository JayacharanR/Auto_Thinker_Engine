#!/usr/bin/env python3
"""Archive completed world traces; restore exact legacy paths around a command."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402
from src.navigation.trace_storage import (  # noqa: E402
    compress_trace,
    materialized_traces,
    restore_trace,
)


def no_world_writers():
    pattern = re.compile(
        r"^[^ ]*CarlaUE4|^[^ ]*python[0-9.]* (-u )?[^ ]*scripts/"
        r"(run_navigation\.py|run_encounters\.py|collect_navigation)"
    )
    for process in psutil.process_iter(["cmdline", "status"]):
        try:
            if process.info["status"] != psutil.STATUS_ZOMBIE and pattern.search(
                " ".join(process.info["cmdline"] or [])
            ):
                raise RuntimeError(f"World trace may be active:{process.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compress", "materialize", "restore"))
    parser.add_argument("--root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/navigation_trace_storage_20261019"
    )
    options = sys.argv[1:]
    split = options.index("--") if "--" in options else len(options)
    args = parser.parse_args(options[:split])
    command = options[split + 1 :] if split < len(options) else []
    args.root, args.out = args.root.resolve(), args.out.resolve()
    args.root.relative_to(ROOT / "outputs")
    args.out.relative_to(ROOT / "outputs")
    with RunLock(ROOT / "outputs/.navigation_trace_storage.lock"):
        if args.command == "materialize":
            if not command:
                parser.error("Provide a legacy command after --")
            receipts = sorted(args.root.rglob("world_trace.jsonl.archive.json"))
            with materialized_traces(receipts):
                result = subprocess.run(command, cwd=ROOT)
            raise SystemExit(result.returncode)
        if command:
            parser.error("Compression/restoration takes no trailing command")
        no_world_writers()
        restoring = args.command == "restore"
        paths = sorted(
            args.root.rglob("world_trace.jsonl.archive.json" if restoring else "world_trace.jsonl")
        )
        sources = [Path(__file__), ROOT / "src/navigation/trace_storage.py"]
        request = dict(
            root=str(args.root),
            paths=[str(p.resolve()) for p in paths],
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
            format="gzip level1, deterministic header, exact round-trip SHA256/bytes",
            command=args.command,
            physical_free_bytes_before=shutil.disk_usage(args.root).free,
            original_receipts=[json.loads(p.read_text()) for p in paths] if restoring else None,
        )
        if (args.out / "request.json").exists():
            raise FileExistsError("Keep prior storage run; use a fresh --out for another batch")
        for source in sources:
            archived = args.out / "source" / source.relative_to(ROOT)
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, archived)
        atomic_json(args.out / "request.json", request)
        results = []
        status = "failed"
        try:
            for path in paths:
                if (args.out / "STOP").exists():
                    status = "stopped"
                    break
                no_world_writers()
                result = restore_trace(path) if restoring else compress_trace(path)
                results.append(result)
                atomic_json(
                    args.out / "progress.json",
                    dict(
                        status="running",
                        stage=(
                            "Exact original-trace restoration"
                            if restoring
                            else "Lossless completed-trace archive"
                        ),
                        files=len(results),
                        total_files=len(paths),
                        original_bytes=sum(r["original_bytes"] for r in results),
                        archive_bytes=sum(r["archive_bytes"] for r in results),
                        updated_at=time.time(),
                    ),
                )
            else:
                status = "complete"
        finally:
            atomic_json(args.out / "receipts.json", results)
            atomic_json(
                args.out / "summary.json",
                dict(
                    status=status,
                    files=len(results),
                    expected_files=len(paths),
                    logical_reduction_bytes=(
                        0
                        if restoring
                        else sum(r["original_bytes"] - r["archive_bytes"] for r in results)
                    ),
                    all_roundtrips_verified=all(r["verified_roundtrip"] for r in results),
                    original_layout_restored=restoring and status == "complete",
                    physical_free_bytes_after=shutil.disk_usage(args.root).free,
                ),
            )
            progress = json.loads((args.out / "progress.json").read_text()) if results else {}
            atomic_json(
                args.out / "progress.json", dict(progress, status=status, updated_at=time.time())
            )
        print(json.dumps(json.loads((args.out / "summary.json").read_text())), flush=True)


if __name__ == "__main__":
    main()
