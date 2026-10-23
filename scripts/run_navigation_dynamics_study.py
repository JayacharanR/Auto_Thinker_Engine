#!/usr/bin/env python3
"""Serialize all six registered dynamics members after collection and feature work."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("Preserve every dynamics study attempt")
    sources = [
        "scripts/run_navigation_dynamics_study.py",
        "scripts/train_navigation_dynamics.py",
        "src/navigation/dynamics_learning.py",
        "src/navigation/ordered_dynamics.py",
        "src/navigation/rssm_dynamics.py",
        "scripts/train_navigation_student.py",
        "reports/navigation_dynamics_fit_protocol_20261020.md",
    ]
    hashes = {s: digest(ROOT / s) for s in sources}
    args.out.mkdir(parents=True)
    for s in sources:
        target = args.out / "source" / s
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / s, target)
    atomic_json(
        args.out / "request.json",
        dict(
            after=str(args.after.resolve()),
            features=str(args.features.resolve()),
            source_hashes=hashes,
            cells=[dict(model=m, seed=s) for m in ("jepa", "rssm") for s in (42, 1051, 2060)],
            not_planner_qualification=True,
            red_positive_data_missing=True,
        ),
    )
    status, active, results = "running", None, []

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M5 factual dynamics study",
                fits_done=len(results),
                fits_total=6,
                phase=phase,
                active_output=str(active) if active else None,
                updated_at=time.time(),
            ),
        )

    try:
        for dependency in (args.after, args.features):
            while True:
                progress("Waiting for native setup/feature jobs; no concurrent GPU work")
                if (args.out / "STOP").exists():
                    raise InterruptedError("Dynamics-study STOP")
                path = dependency / "progress.json"
                if path.exists():
                    state = json.loads(path.read_text())["status"]
                    if state != "running":
                        if state != "complete":
                            raise RuntimeError(f"Dependency {dependency} ended {state}")
                        break
                time.sleep(3)
        draws = {}
        for model in ("jepa", "rssm"):
            for seed in (42, 1051, 2060):
                if (args.out / "STOP").exists():
                    raise InterruptedError("Dynamics-study STOP")
                if any(digest(ROOT / s) != h for s, h in hashes.items()):
                    raise ValueError("Registered dynamics-study sources changed")
                active = args.out / f"{model}_seed{seed}"
                progress("Factual prediction fit; no planning claim")
                with (args.out / "console.log").open("a") as log:
                    command = [
                        sys.executable,
                        "-u",
                        "scripts/train_navigation_dynamics.py",
                        "--model",
                        model,
                        "--seed",
                        str(seed),
                        "--features",
                        str(args.features),
                        "--out",
                        str(active),
                    ]
                    log.write(json.dumps(dict(command=command, time=time.time())) + "\n")
                    log.flush()
                    child = subprocess.Popen(
                        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                    )
                    while child.poll() is None:
                        if (args.out / "STOP").exists():
                            active.mkdir(parents=True, exist_ok=True)
                            (active / "STOP").touch()
                        progress("Factual prediction fit; no planning claim")
                        time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f"Factual dynamics member failed:{active}")
                rows = [
                    json.loads(line) for line in (active / "train.jsonl").read_text().splitlines()
                ]
                validation = [
                    json.loads(line)
                    for line in (active / "validation.jsonl").read_text().splitlines()
                ]
                done = json.loads((active / "progress.json").read_text())
                if (
                    done["status"] != "complete"
                    or done["steps"] != 2000
                    or len(rows) != 2000
                    or [r["step"] for r in validation] != [0, 500, 1000, 1500, 2000]
                ):
                    raise ValueError("Registered2000update/full validation budget incomplete")
                indices = [r["window_indices"] for r in rows]
                if seed in draws and draws[seed] != indices:
                    raise ValueError("World-model arms received unequal factual training draws")
                draws[seed] = indices
                selected = next(r for r in validation if r["step"] == done["selected_step"])
                results.append(
                    dict(
                        model=model,
                        seed=seed,
                        output=str(active),
                        selected_validation=selected,
                        checkpoint_sha256=digest(active / "best_model.pt"),
                        exact_paired_draws_verified=model == "rssm",
                        not_planner_qualification=True,
                    )
                )
                atomic_json(args.out / "results.json", results)
        status = "complete"
    except InterruptedError as error:
        status = "stopped"
        atomic_json(args.out / "error.json", dict(error=str(error)))
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error)))
        raise
    finally:
        progress("finished")


if __name__ == "__main__":
    main()
