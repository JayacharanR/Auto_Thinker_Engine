#!/usr/bin/env python3
"""Finish the existing RGB cache, then run the two predeclared matched fits once."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402

CACHE = ROOT / "outputs/navigation_m2_signal_policy_cache_20261018"
OUT = ROOT / "outputs/navigation_m2_signal_experiment_20261018"
LEDGER = ROOT / "reports/navigation_execution_20261018.json"


def active(path):
    ledger = json.loads(LEDGER.read_text())
    ledger["active_output"] = str(path.relative_to(ROOT))
    atomic_json(LEDGER, ledger)


def main():
    OUT.mkdir(exist_ok=True)
    with RunLock(OUT / ".runner.lock"):
        request = dict(
            cache=str(CACHE),
            cache_request_sha256=digest(CACHE / "request.json"),
            trainer_sha256=digest(ROOT / "scripts/train_navigation_signal_policy.py"),
            arms=["rgb", "null"],
            steps_per_arm=1000,
        )
        if (OUT / "request.json").exists() and json.loads(
            (OUT / "request.json").read_text()
        ) != request:
            raise ValueError("Experiment inputs changed; preserve existing attempt")
        atomic_json(OUT / "request.json", request)
        active(CACHE)
        while True:
            p = json.loads((CACHE / "progress.json").read_text())
            if (OUT / "STOP").exists():
                (CACHE / "STOP").touch()
                return
            if p["status"] == "complete":
                break
            if p["status"] not in ("starting", "running"):
                raise RuntimeError("Upstream cache did not complete; no fit launched")
            if time.time() - p["updated_at"] > 300:
                raise RuntimeError("Cache heartbeat stale; inspect ownership before continuing")
            time.sleep(2)
        reviews = {}
        for arm in request["arms"]:
            out = ROOT / f"outputs/navigation_m2_signal_residual_{arm}_20261018"
            active(out)
            progress = out / "progress.json"
            if not progress.exists():
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/train_navigation_signal_policy.py"),
                    "--arm",
                    arm,
                    "--out",
                    str(out),
                ]
                with (OUT / f"{arm}.log").open("a") as log:
                    child = subprocess.Popen(
                        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                    )
                    while child.poll() is None:
                        if (OUT / "STOP").exists():
                            out.mkdir(exist_ok=True)
                            (out / "STOP").touch()
                        time.sleep(1)
                if child.returncode:
                    raise RuntimeError(
                        f"{arm} fit exited{child.returncode}; inspect log, no automatic repeat"
                    )
            if json.loads(progress.read_text())["status"] != "complete":
                raise RuntimeError(
                    f"{arm} fit incomplete; explicit identical-source resume required"
                )
            reviews[arm] = json.loads((out / "review.json").read_text())
            if (OUT / "STOP").exists():
                return
        atomic_json(OUT / "review.json", dict(arms=reviews, driving_evidence=False))
        atomic_json(
            OUT / "progress.json",
            dict(status="complete", stage="M2 matched residual experiment", updated_at=time.time()),
        )
        active(OUT)
        print(
            json.dumps(
                {
                    a: dict(
                        selected_step=v["selected_step"], candidate_improved=v["candidate_improved"]
                    )
                    for a, v in reviews.items()
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
