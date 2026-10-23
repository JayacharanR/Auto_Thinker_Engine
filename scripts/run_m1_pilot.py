#!/usr/bin/env python3
"""Small expert data pilot: all six families, five training towns, one server at a time."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_data import audit  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402

PLAN = [
    ("Town01_Opt", "signal_stop"),
    ("Town02_Opt", "following_stopped_vehicle"),
    ("Town03_Opt", "stop_sign"),
    ("Town04_Opt", "signal_release_memory"),
    ("Town04_Opt", "route_junction"),
    ("Town10HD_Opt", "crossing_yield"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--catalog", default="configs/encounters/location_split_v1/train", type=Path
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Every pilot attempt uses a fresh output directory")
    args.out.mkdir(parents=True)
    args.out = args.out.resolve()
    args.catalog = args.catalog.resolve()
    data = (args.catalog / "catalog.json").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != (args.catalog / "catalog.sha256").read_text().split()[0]:
        raise ValueError("Training catalog hash mismatch")
    catalog = json.loads(data)
    if catalog.get("role") != "train":
        raise ValueError("Pilot only collects from the frozen training partition")
    picks = [
        next(e for e in catalog["encounters"] if e["town"] == town and e["family"] == family)
        for town, family in PLAN
    ]
    atomic_json(
        args.out / "request.json",
        {
            "catalog_sha256": digest,
            "selection": "first route id per declared town/family, without driving outcomes",
            "episodes": picks,
            "seed": 20261015,
            "mode": "expert",
        },
    )
    done, errors = 0, []

    def progress(status, town=None):
        atomic_json(
            args.out / "progress.json",
            {
                "status": status,
                "route_progress_percent": 100 * done / len(picks),
                "encounters_done": done,
                "encounters_total": len(picks),
                "current_chunk": town,
                "updated_at": time.time(),
                "errors": errors,
            },
        )

    with RunLock(ROOT / "outputs/navigation/.m1_pilot.lock"):
        for town in dict.fromkeys(e["town"] for e in picks):
            if (args.out / "STOP").exists():
                break
            group = [e for e in picks if e["town"] == town]
            output = args.out / town
            progress("running", town)
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/run_navigation.py"),
                "expert",
                "--route",
                str(args.catalog / f"{town}.xml"),
                "--collection-catalog",
                str(args.catalog),
                "--routes-subset",
                ",".join(str(e["route_id"]) for e in group),
                "--seed",
                "20261015",
                "--seed-per-route",
                "--max-wall",
                "300",
                "--out",
                str(output),
            ]
            with (args.out / f"{town}_console.log").open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if (args.out / "STOP").exists() and output.exists():
                        (output / "STOP").touch()
                    time.sleep(1)
            if child.returncode:
                errors.append({"town": town, "exit_code": child.returncode})
            else:
                done += len(group)
            progress("running")
        if (args.out / "STOP").exists():
            status = "stopped"
        else:
            status = "failed" if errors else "complete"
        progress(status)
        try:
            result = audit(args.out)
        except Exception as exc:
            atomic_json(args.out / "audit_error.json", {"error": str(exc)})
            raise
        result.update(status=status, simulator_errors=errors)
        atomic_json(args.out / "data_audit.json", result)
        print(json.dumps(result["totals"], indent=2))
        if errors:
            raise RuntimeError("Pilot had simulator errors; preserve and inspect every attempt")


if __name__ == "__main__":
    main()
