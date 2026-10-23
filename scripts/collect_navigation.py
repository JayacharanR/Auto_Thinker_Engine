#!/usr/bin/env python3
"""Resume bounded, outcome-blind training collection from atomic committed shards."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_data import audit  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, resource_shortfall  # noqa: E402
from src.navigation.splits import collection_geometry_eligibility  # noqa: E402


def committed_samples(root):
    count = 0
    for path in Path(root).glob("episode_*/episodes/*/data/shard_*.zip"):
        with zipfile.ZipFile(path) as archive:
            count += len(json.loads(archive.read("manifest.json"))["samples"])
    return count


def episode_schedule(catalog, seed):
    """Cycle family/town strata without looking at expert outcomes."""
    pools = defaultdict(list)
    for entry in catalog["encounters"]:
        pools[entry["family"], entry["town"]].append(entry)
    rng = np.random.default_rng(seed)
    order = sorted(pools)
    for key, rows in pools.items():
        pools[key] = [rows[i] for i in rng.permutation(len(rows))]
    epoch = 0
    while True:
        for key in order:
            yield pools[key][epoch % len(pools[key])]
        epoch += 1


def collection_error(output, returncode):
    output = Path(output)
    if (output / "collection_partition_violation.json").exists():
        return "partition_violation"
    if returncode:
        return "worker_exit"
    summary = output / "summary.json"
    if summary.exists():
        episodes = json.loads(summary.read_text())["episodes"]
        if any("agent crashed" in e["status"].lower() for e in episodes):
            return "agent_crashed"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--catalog", default="configs/encounters/location_split_v1/train", type=Path
    )
    parser.add_argument("--target-samples", type=int, default=20000)
    parser.add_argument("--max-wall", type=float, default=5400, help="Hard queue budget, seconds")
    parser.add_argument("--seed", type=int, default=202610150)
    args = parser.parse_args()
    if not 0 < args.target_samples <= 100000 or not 0 < args.max_wall <= 5400:
        raise ValueError(
            "Bounded collector permits up to 100k samples and 90 minutes per invocation"
        )
    args.out, args.catalog = args.out.resolve(), args.catalog.resolve()
    shortfall = resource_shortfall({"minimum_free_vram_gb": 6, "minimum_available_ram_gb": 12})
    if shortfall:
        raise RuntimeError(shortfall)
    args.out.mkdir(parents=True, exist_ok=True)
    with RunLock(args.out / ".collection.lock"):
        data = (args.catalog / "catalog.json").read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != (args.catalog / "catalog.sha256").read_text().split()[0]:
            raise ValueError("Catalog hash mismatch")
        catalog = json.loads(data)
        if catalog.get("role") != "train":
            raise ValueError("Only training-partition collection is allowed")
        eligibility = collection_geometry_eligibility(args.catalog)
        sources = [
            *sorted((ROOT / "src/navigation").glob("*.py")),
            ROOT / "scripts/run_navigation.py",
            Path(__file__).resolve(),
            ROOT / "scripts/audit_navigation_data.py",
            ROOT / "configs/navigation.yaml",
            ROOT / "configs/navigation_vendor.json",
            ROOT / "patches/navigation_garage.patch",
        ]
        source_hashes = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        }
        request = {
            "catalog": str(args.catalog),
            "catalog_sha256": digest,
            "target_samples": args.target_samples,
            "seed": args.seed,
            "mode_schedule": "two expert episodes then one bounded steering-noise episode",
            "source_hashes": source_hashes,
            "geometry_eligibility": eligibility,
        }
        path = args.out / "request.json"
        if path.exists() and json.loads(path.read_text()) != request:
            raise ValueError("Resume request or driving source changed; use a reviewed new run")
        atomic_json(path, request)
        atomic_json(args.out / "geometry_eligibility.json", eligibility)
        samples = committed_samples(args.out)
        started = time.monotonic()
        errors = 0
        fatal_error = None
        outcomes = []

        def progress(status, episode=None):
            atomic_json(
                args.out / "progress.json",
                {
                    "status": status,
                    "stage": "M1 collection",
                    "frames": samples,
                    "target_frames": args.target_samples,
                    "route_progress_percent": 100 * samples / args.target_samples,
                    "current_episode": episode,
                    "updated_at": time.time(),
                    "elapsed_seconds_this_invocation": time.monotonic() - started,
                },
            )

        progress("running")
        eligible_catalog = {
            **catalog,
            "encounters": [
                e for e in catalog["encounters"] if e["route_id"] in eligibility["kept_route_ids"]
            ],
        }
        for index, entry in enumerate(episode_schedule(eligible_catalog, args.seed)):
            output = args.out / f"episode_{index:05d}"
            # Existing attempts remain data, including failed/censored ones. Never overwrite.
            if output.exists():
                continue
            if samples >= args.target_samples or (args.out / "STOP").exists():
                break
            if time.monotonic() - started >= args.max_wall:
                break
            mode = "noisy" if index % 3 == 2 else "expert"
            remaining = args.target_samples - samples
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/run_navigation.py"),
                "expert",
                "--route",
                str(args.catalog / f"{entry['town']}.xml"),
                "--collection-catalog",
                str(args.catalog),
                "--collection-mode",
                mode,
                "--collection-max-samples",
                str(remaining),
                "--routes-subset",
                str(entry["route_id"]),
                "--seed",
                str(args.seed + index),
                "--max-wall",
                str(min(300, max(1, args.max_wall - (time.monotonic() - started)))),
                "--out",
                str(output),
            ]
            progress("running", output.name)
            with (args.out / f"episode_{index:05d}_console.log").open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if (args.out / "STOP").exists() or time.monotonic() - started >= args.max_wall:
                        if output.exists():
                            (output / "STOP").touch()
                    samples = committed_samples(args.out)
                    progress("running", output.name)
                    time.sleep(2)
            samples = committed_samples(args.out)
            error = collection_error(output, child.returncode)
            errors = errors + 1 if error else 0
            outcomes.append(
                {
                    "episode": output.name,
                    "route_id": entry["route_id"],
                    "town": entry["town"],
                    "mode": mode,
                    "exit_code": child.returncode,
                    "collection_error": error,
                }
            )
            atomic_json(args.out / "invocation_outcomes.json", outcomes)
            if error == "partition_violation":
                fatal_error = error
            if fatal_error or errors >= 3:
                break
        status = "complete" if samples >= args.target_samples else "stopped"
        if fatal_error or errors >= 3:
            status = "failed"
        progress(status)
        result = audit(args.out)
        result.update(status=status, invocation_episodes=outcomes, fatal_error=fatal_error)
        atomic_json(args.out / "data_audit.json", result)
        print(json.dumps({"status": status, "samples": samples}, indent=2))


if __name__ == "__main__":
    main()
