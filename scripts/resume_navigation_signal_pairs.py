#!/usr/bin/env python3
"""Resume incomplete static TRAIN towns with unchanged archived worker sources."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_navigation_signal_pairs import audit, validate_plan  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def verify_worker_sources(out):
    request = read(out / "request.json")
    if digest(out / "plan.json") != request["plan_sha256"]:
        raise ValueError("Frozen plan changed")
    if any(
        digest(ROOT / path) != sha or digest(out / "source" / path) != sha
        for path, sha in request["source_hashes"].items()
    ):
        raise ValueError("Original worker source differs; refuse mixed-code resume")


def resume(out, max_wall):
    if not 0 < max_wall <= 900 or (out / "STOP").exists():
        raise ValueError("Bounded resume required, with no pending STOP")
    verify_worker_sources(out)
    plan = read(out / "plan.json")
    validate_plan(plan)
    vendor = activate()
    config = yaml.safe_load((out / "source/configs/navigation.yaml").read_text())
    started, done, status = time.monotonic(), 0, "failed"
    ledger = {
        "started_at": time.time(),
        "maximum_wall_seconds": max_wall,
        "orchestrator_sha256": digest(Path(__file__)),
        "original_request_sha256": digest(out / "request.json"),
        "skipped_complete_towns": [],
        "retried_towns": [],
    }
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        check_resources(out, config["runtime"]["minimum_free_gb"])
        stamp = str(time.time_ns())
        history = out / "resume_attempts" / stamp
        history.mkdir(parents=True)
        shutil.copyfile(Path(__file__), history / "resume_source.py")
        if (out / "progress.json").exists():
            shutil.copyfile(out / "progress.json", history / "previous_progress.json")
        try:
            for town in sorted({c["town"] for c in plan["cases"]}):
                directory = out / town
                expected = sum(c["town"] == town for c in plan["cases"])
                summary = directory / "assessment.json"
                if summary.exists() and read(summary).get("probe_complete"):
                    complete = read(summary)
                    if len(complete["cases"]) != expected:
                        raise ValueError("Town completion count changed")
                    for metadata in (directory / "case_metadata").glob("*.json"):
                        if any(
                            digest(directory / p) != sha
                            for p, sha in read(metadata)["image_sha256"].items()
                        ):
                            raise ValueError("Committed pose image changed")
                    done += expected
                    ledger["skipped_complete_towns"].append(town)
                    continue
                if directory.exists():
                    shutil.move(str(directory), history / town)
                directory.mkdir()
                ledger["retried_towns"].append(town)
                atomic_json(history / "ledger.json", ledger)
                with CarlaServer(
                    vendor["carla_root"], directory, config["runtime"]["port"], town=town
                ):
                    with (directory / "worker.log").open("x") as log:
                        child = subprocess.Popen(
                            [
                                sys.executable,
                                "-u",
                                str(ROOT / "scripts/collect_navigation_signal_pairs.py"),
                                "collect",
                                "--worker",
                                town,
                                "--plan",
                                str(out / "plan.json"),
                                "--out",
                                str(out),
                            ],
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        stopping = None
                        while child.poll() is None:
                            if time.monotonic() - started >= max_wall:
                                (out / "STOP").touch()
                            if (out / "STOP").exists():
                                (directory / "STOP").touch()
                                stopping = stopping or time.monotonic()
                                if time.monotonic() - stopping > 20:
                                    child.terminate()
                                    child.wait(timeout=20)
                                    raise InterruptedError("STOP grace ended")
                            committed = len(list((directory / "case_metadata").glob("*.json")))
                            atomic_json(
                                out / "progress.json",
                                {
                                    "status": "running",
                                    "stage": "M2 static signal supervision",
                                    "encounters_done": done + committed,
                                    "encounters_total": len(plan["cases"]),
                                    "current_chunk": town,
                                    "updated_at": time.time(),
                                },
                            )
                            time.sleep(1)
                    if child.returncode:
                        raise RuntimeError(f"Resumed town failed: {town}")
                done += expected
            result = audit(out)
            status = "complete"
        finally:
            ledger.update(status=status, seconds=time.monotonic() - started)
            atomic_json(history / "ledger.json", ledger)
            atomic_json(
                out / "progress.json",
                {
                    "status": "stopped" if (out / "STOP").exists() else status,
                    "stage": "M2 static signal supervision",
                    "encounters_done": done,
                    "encounters_total": len(plan["cases"]),
                    "updated_at": time.time(),
                    "resume_seconds": time.monotonic() - started,
                },
            )
        print(json.dumps({k: v for k, v in result.items() if k not in ("labels", "image_sha256")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-wall", type=int, default=300)
    args = parser.parse_args()
    resume(args.out.resolve(), args.max_wall)


if __name__ == "__main__":
    main()
