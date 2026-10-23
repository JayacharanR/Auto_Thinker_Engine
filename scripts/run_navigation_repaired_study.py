#!/usr/bin/env python3
"""Reproduce all67 corrected-data fits and clean JEPA continuation, serially."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_navigation_visual_study import fit_cells, fit_name, verify_fit  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402


def complete(path):
    p = path / "progress.json"
    return p.exists() and json.loads(p.read_text()).get("status") == "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--after-branches", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    data = ROOT / "outputs/navigation_spatial_data_repaired_20261019"
    cells = fit_cells() + [dict(encoder="jepa_random", fraction=1.0, seed=42, color_enabled=True)]
    sources = [
        "scripts/run_navigation_repaired_study.py",
        "scripts/train_navigation_repaired_visual.py",
        "scripts/train_navigation_visual_student.py",
        "scripts/pretrain_navigation_jepa.py",
        "scripts/cache_navigation_grid_encoder.py",
        "scripts/run_navigation_visual_study.py",
        "src/navigation/spatial_data.py",
        "src/navigation/spatial_student.py",
        "src/navigation/spatial_policy.py",
        "src/navigation/visual_decision_student.py",
        "src/navigation/grid_encoders.py",
        "src/navigation/rectangular_jepa.py",
        "src/navigation/jepa_pretraining.py",
        "src/navigation/optical_fields.py",
        "src/navigation/policy_runtime.py",
        "scripts/train_navigation_student.py",
        "scripts/train_navigation_causal_student.py",
        "src/jepa/encoder.py",
        "src/jepa/predictor.py",
        "reports/navigation_m4_geography_repair_protocol_20261019.md",
    ]
    registry_path = ROOT / "outputs/navigation_models/navigation_backbones_20261019.json"
    random_registry = ROOT / "outputs/navigation_models/random_vit_registry_20261019.json"
    request = dict(
        cells=cells,
        data_manifest_sha256=digest(data / "manifest.json"),
        registry_sha256=digest(registry_path),
        random_registry_sha256=digest(random_registry),
        source_hashes={p: digest(ROOT / p) for p in sources},
        after_branches=str(args.after_branches.resolve()),
        source_geography_repair=True,
        reuse_old_trained_heads=False,
        exact_original_training_loop=True,
        frozen_features_temporary_RAM=True,
    )
    if args.out.exists():
        if not args.resume or json.loads((args.out / "request.json").read_text()) != request:
            raise ValueError("Only identical-source repaired-study resume is allowed")
    else:
        if args.resume:
            raise ValueError("No repaired study to resume")
        args.out.mkdir(parents=True)
        for p in sources:
            target = args.out / "source" / p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / p, target)
        atomic_json(args.out / "request.json", request)
    results, active, status = [], None, "running"

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M4 corrected geography study",
                fits_done=len(results),
                fits_total=67,
                phase=phase,
                active_output=str(active) if active else None,
                updated_at=time.time(),
            ),
        )

    def run(command, output):
        nonlocal active
        if (args.out / "STOP").exists():
            raise InterruptedError("Repaired study STOP")
        if any(digest(ROOT / p) != h for p, h in request["source_hashes"].items()):
            raise ValueError("Repaired-study registered source changed")
        active = output
        progress("Executing registered corrected-data stage")
        output.parent.mkdir(parents=True, exist_ok=True)
        with (args.out / "console.log").open("a") as log:
            log.write(json.dumps(dict(command=command, time=time.time())) + "\n")
            log.flush()
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                if (args.out / "STOP").exists():
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "STOP").touch()
                progress("Executing registered corrected-data stage")
                time.sleep(2)
        if child.returncode:
            raise RuntimeError(f"Corrected stage failed:{output}; preserve and inspect")

    def evict_ram(cache, verified):
        feature = cache / "features.npy"
        if not feature.exists():
            if not (cache / "evicted_features.json").exists():
                raise ValueError("RAM cache disappeared without an eviction receipt")
            return
        receipt = json.loads((cache / "ram_feature_storage.json").read_text())
        target = Path(receipt["target"])
        if (
            not feature.is_symlink()
            or feature.resolve() != target
            or target.parent.parent != Path("/tmp")
        ):
            raise ValueError("Only owned temporary RAM feature files may be evicted")
        if not target.parent.name.startswith("auto_thinker_frozen_features_"):
            raise ValueError("Unexpected temporary feature cache directory")
        expected = json.loads((cache / "manifest.json").read_text())["features_sha256"]
        if digest(target) != expected:
            raise ValueError("Completed RAM features changed")
        atomic_json(
            cache / "evicted_features.json",
            dict(
                sha256=expected,
                bytes=target.stat().st_size,
                completed_fits=verified,
                reason="Registered encoder group complete; preserve recipe and trained artifacts",
                reproducible_from_source_shards=True,
            ),
        )
        feature.unlink()
        target.unlink()
        target.parent.rmdir()

    try:
        progress("Waiting for factual branch pilot; no concurrent simulator/trainer")
        while True:
            if (args.out / "STOP").exists():
                raise InterruptedError("Repaired study STOP while queued")
            p = args.after_branches / "progress.json"
            if p.exists():
                prior = json.loads(p.read_text())["status"]
                if prior != "running":
                    if prior != "complete":
                        raise RuntimeError(f"Prior branch stage ended {prior}; inspect first")
                    break
            time.sleep(3)
        with RunLock(ROOT / "outputs/.navigation.lock"):
            pass
        ledger_path = ROOT / "reports/navigation_execution_20261018.json"
        ledger = json.loads(ledger_path.read_text())
        ledger["active_output"] = str(args.out.relative_to(ROOT))
        ledger["milestones"]["M4"]["status"] = "repaired_study_running"
        atomic_json(ledger_path, ledger)
        pretraining = args.out / "jepa_pretraining"
        if not complete(pretraining):
            command = [
                sys.executable,
                "-u",
                "scripts/pretrain_navigation_jepa.py",
                "--data",
                str(data),
                "--out",
                str(pretraining),
            ]
            if pretraining.exists():
                command.append("--resume")
            run(command, pretraining)
        if not complete(pretraining):
            raise RuntimeError("Corrected JEPA fixed training budget incomplete")
        registry = json.loads(registry_path.read_text())
        registry["jepa_carla"] = json.loads((pretraining / "encoder_descriptor.json").read_text())
        registry.update(json.loads(random_registry.read_text()))
        derived = args.out / "backbones.json"
        atomic_json(derived, registry)
        for encoder in (
            "imagenet_resnet18",
            "dinov2_small",
            "jepa_comma2k19",
            "jepa_carla",
            "cnn_scratch",
            "nav_color_only",
            "jepa_random",
        ):
            group_cells = [c for c in cells if c["encoder"] == encoder]
            group_results = []
            all_finished = all(complete(args.out / "fits" / fit_name(c)) for c in group_cells)
            cache = None
            if encoder not in ("cnn_scratch", "nav_color_only"):
                cache = args.out / f"cache_{encoder}"
                if not all_finished and not complete(cache):
                    if cache.exists():
                        raise ValueError(
                            "Interrupted RAM cache retained; audit before reconstructing"
                        )
                    run(
                        [
                            sys.executable,
                            "-u",
                            "scripts/cache_navigation_grid_encoder.py",
                            "--encoder",
                            encoder,
                            "--registry",
                            str(derived),
                            "--data",
                            str(data),
                            "--out",
                            str(cache),
                            "--features-ram",
                        ],
                        cache,
                    )
            for cell in group_cells:
                fit = args.out / "fits" / fit_name(cell)
                if not complete(fit):
                    command = [
                        sys.executable,
                        "-u",
                        "scripts/train_navigation_repaired_visual.py",
                        "--encoder",
                        encoder,
                        "--fraction",
                        str(cell["fraction"]),
                        "--seed",
                        str(cell["seed"]),
                        "--data",
                        str(data),
                        "--out",
                        str(fit),
                    ]
                    if cache:
                        command.extend(["--cache", str(cache)])
                    if not cell["color_enabled"]:
                        command.append("--no-color")
                    if fit.exists():
                        command.append("--resume")
                    run(command, fit)
                verified = verify_fit(fit, cell)
                matched = [
                    r
                    for r in results
                    if r["seed"] == cell["seed"] and r["fraction"] == cell["fraction"]
                ]
                if any(
                    r["sampler_draws_sha256"] != verified["sampler_draws_sha256"] for r in matched
                ):
                    raise ValueError("Corrected encoder arms did not see identical episode draws")
                results.append(verified)
                group_results.append(verified)
                atomic_json(args.out / "completed" / (fit_name(cell) + ".json"), verified)
                atomic_json(args.out / "results.json", results)
                progress("Verified corrected-data fit")
            if cache:
                evict_ram(cache, group_results)
        if len(results) != 67:
            raise ValueError("Corrected matrix incomplete")
        status = "complete"
    except InterruptedError:
        status = "stopped"
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), active_output=str(active)))
        raise
    finally:
        progress("finished")
    print(json.dumps(dict(status=status, fits=len(results), expected=67)), flush=True)


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation_repaired_study.lock"):
        main()
