#!/usr/bin/env python3
"""Run the four registered cold-start/fact candidates after the corrected study."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("Retain every prospective fact-study attempt")
    data = ROOT / "outputs/navigation_spatial_data_repaired_20261019"
    labels = ROOT / "outputs/navigation_fact_labels_20261020"
    sources = [
        "scripts/run_navigation_fact_study.py",
        "scripts/train_navigation_fact_student.py",
        "scripts/train_navigation_visual_student.py",
        "scripts/cache_navigation_grid_encoder.py",
        "src/navigation/fact_student.py",
        "src/navigation/visual_decision_student.py",
        "src/navigation/spatial_student.py",
        "src/navigation/spatial_data.py",
        "src/navigation/optical_fields.py",
        "src/navigation/grid_encoders.py",
        "scripts/train_navigation_student.py",
        "scripts/train_navigation_causal_student.py",
        "reports/navigation_fact_student_protocol_20261020.md",
    ]
    hashes = {path: digest(ROOT / path) for path in sources}
    args.out.mkdir(parents=True)
    for path in sources:
        target = args.out / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    atomic_json(
        args.out / "request.json",
        dict(
            after_study=str(args.after_study.resolve()),
            source_hashes=hashes,
            corrected_data_manifest_sha256=digest(data / "manifest.json"),
            fact_manifest_sha256=digest(labels / "manifest.json"),
            candidates=[
                dict(encoder=e, facts=f, seed=42, steps=2000, burn_in=0)
                for e in ("imagenet_resnet18", "cnn_scratch")
                for f in (False, True)
            ],
            no_m4_training_change=True,
            no_driving_claim=True,
        ),
    )
    status, active, results = "running", None, []

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M2 paired cold-start/fact study",
                fits_done=len(results),
                fits_total=4,
                active_output=str(active) if active else None,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def run(command, output):
        nonlocal active
        if (args.out / "STOP").exists():
            raise InterruptedError("Fact-study STOP")
        if any(digest(ROOT / p) != h for p, h in hashes.items()):
            raise ValueError("Queued fact-study registered source changed")
        active = output
        progress("Executing registered fresh full-label candidate")
        with (args.out / "console.log").open("a") as log:
            log.write(json.dumps(dict(command=command, time=time.time())) + "\n")
            log.flush()
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                if (args.out / "STOP").exists():
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "STOP").touch()
                progress("Executing registered fresh full-label candidate")
                time.sleep(2)
        if child.returncode:
            raise RuntimeError(f"Fact-study child failed:{output}; retain failed attempt")

    try:
        progress("Waiting for corrected67-study; serial GPU execution")
        while True:
            if (args.out / "STOP").exists():
                raise InterruptedError("Fact-study STOP while queued")
            path = args.after_study / "progress.json"
            if path.exists():
                prior = json.loads(path.read_text())["status"]
                if prior != "running":
                    if prior != "complete":
                        raise RuntimeError(f"Corrected study ended {prior}; inspect first")
                    break
            time.sleep(3)
        cache = args.out / "resnet_cache"
        run(
            [
                sys.executable,
                "-u",
                "scripts/cache_navigation_grid_encoder.py",
                "--encoder",
                "imagenet_resnet18",
                "--data",
                str(data),
                "--features-ram",
                "--out",
                str(cache),
            ],
            cache,
        )
        for encoder in ("imagenet_resnet18", "cnn_scratch"):
            paired_draws = []
            for facts in (False, True):
                output = args.out / f"{encoder}_{'facts' if facts else 'plain'}"
                command = [
                    sys.executable,
                    "-u",
                    "scripts/train_navigation_fact_student.py",
                    "--encoder",
                    encoder,
                    "--out",
                    str(output),
                ]
                if encoder == "imagenet_resnet18":
                    command.extend(["--cache", str(cache)])
                if facts:
                    command.append("--facts")
                run(command, output)
                rows = [
                    json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()
                ]
                validation = [
                    json.loads(line)
                    for line in (output / "validation.jsonl").read_text().splitlines()
                ]
                p = json.loads((output / "progress.json").read_text())
                if (
                    p["status"] != "complete"
                    or p["steps"] != 2000
                    or len(rows) != 2000
                    or [r["step"] for r in validation] != [0, 500, 1000, 1500, 2000]
                ):
                    raise ValueError("Registered fresh update/validation budget incomplete")
                draws = [r["episode_id"] for r in rows]
                if paired_draws and paired_draws != draws:
                    raise ValueError("Plain/fact arms received unequal example sequences")
                paired_draws = draws
                results.append(
                    dict(
                        encoder=encoder,
                        facts=facts,
                        output=str(output),
                        review=json.loads((output / "review.json").read_text()),
                        matched_draws_verified=True,
                    )
                )
                atomic_json(args.out / "results.json", results)
            if encoder == "imagenet_resnet18":
                feature = cache / "features.npy"
                target = feature.resolve()
                expected = json.loads((cache / "manifest.json").read_text())["features_sha256"]
                receipt = json.loads((cache / "ram_feature_storage.json").read_text())
                if (
                    not feature.is_symlink()
                    or str(target) != receipt["target"]
                    or target.parent.parent != Path("/tmp")
                    or not target.parent.name.startswith("auto_thinker_frozen_features_")
                    or digest(target) != expected
                ):
                    raise ValueError("Only owned verified regenerable RAM features may be evicted")
                atomic_json(
                    cache / "evicted_features.json",
                    dict(
                        sha256=expected,
                        completed_fits=results.copy(),
                        source_recipe_preserved=True,
                        reason="Both paired ResNet candidates complete",
                    ),
                )
                feature.unlink()
                target.unlink()
                target.parent.rmdir()
        status = "complete"
    except InterruptedError as error:
        status = "stopped"
        atomic_json(args.out / "error.json", dict(error=str(error)))
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error)))
        raise
    finally:
        progress("finished" if status != "running" else "queue interrupted")


if __name__ == "__main__":
    main()
