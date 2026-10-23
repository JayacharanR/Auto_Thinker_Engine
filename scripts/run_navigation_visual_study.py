#!/usr/bin/env python3
"""Serial66-fit visual-decision study with visible progress and cache eviction."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402

KINDS = ("imagenet_resnet18", "dinov2_small", "jepa_comma2k19", "jepa_carla", "cnn_scratch")
ALIASES = dict(
    imagenet_resnet18="resnet",
    dinov2_small="dino",
    jepa_comma2k19="comma",
    cnn_scratch="cnn",
    jepa_carla="carla",
)
FIRST = ROOT / "outputs/navigation_visual_resnet_f100_seed42_20261019"
FIRST_CACHE = ROOT / "outputs/navigation_spatial_cache_resnet_20261019"


def fit_cells():
    cells = []
    for encoder in KINDS:
        for fraction in (1.0, 0.25, 0.1, 0.05):
            for seed in (42, 123, 456):
                cells.append(
                    dict(encoder=encoder, fraction=fraction, seed=seed, color_enabled=True)
                )
        cells.append(dict(encoder=encoder, fraction=1.0, seed=42, color_enabled=False))
    cells.append(dict(encoder="nav_color_only", fraction=1.0, seed=42, color_enabled=True))
    return cells


def fit_name(cell):
    encoder = ALIASES.get(cell["encoder"], cell["encoder"])
    return f"{encoder}_f{int(cell['fraction'] * 100):03}_seed{cell['seed']}" + (
        "_no_color" if not cell["color_enabled"] else ""
    )


def verify_fit(path, cell):
    p = json.loads((path / "progress.json").read_text())
    request = json.loads((path / "request.json").read_text())
    history = [json.loads(line) for line in (path / "validation.jsonl").read_text().splitlines()]
    review = json.loads((path / "review.json").read_text())
    if p["status"] != "complete" or p["steps"] != 2000:
        raise ValueError(f"Incomplete registered fit:{path}")
    if any(request[k] != value for k, value in cell.items()):
        raise ValueError(f"Study cell identity changed:{path}")
    if request.get("neural_ego_speed_enabled") is not False:
        raise ValueError("Every corrected study cell must use the same neural speed omission")
    if [r["step"] for r in history] != [0, 500, 1000, 1500, 2000]:
        raise ValueError("Study validation schedule changed")
    if digest(path / "best_policy.pt") != review["policy_sha256"]:
        raise ValueError("Study selected weights changed")
    if any(digest(path / "source" / p) != h for p, h in request["source_hashes"].items()):
        raise ValueError("Study archived training source changed")
    draws = [json.loads(line) for line in (path / "train.jsonl").read_text().splitlines()]
    if [row["step"] for row in draws] != list(range(1, 2001)):
        raise ValueError("Study requires exactly2000 recorded applied updates")
    draw_sha256 = hashlib.sha256(
        json.dumps([(row["episode_id"], row["family"], row["frames"]) for row in draws]).encode()
    ).hexdigest()
    selected = next(row for row in history if row["step"] == review["selected_step"])
    return dict(
        cell,
        fit=str(path),
        selected_step=selected["step"],
        metrics=selected["metrics"],
        policy_sha256=review["policy_sha256"],
        labels=request["allocation"]["actual_labels"],
        elapsed_s=p["elapsed_s"],
        request_sha256=digest(path / "request.json"),
        sampler_draws_sha256=draw_sha256,
    )


def evict_features(cache, verified_fits):
    """Only a completed regenerable array; retain recipes, hashes and all trained weights."""
    array = cache / "features.npy"
    if not array.exists():
        if not (cache / "evicted_features.json").exists():
            raise ValueError("Missing encoder features without a recorded eviction")
        return
    manifest = json.loads((cache / "manifest.json").read_text())
    if digest(array) != manifest["features_sha256"] or len(verified_fits) != 13:
        raise ValueError("Complete verified encoder group required before cache eviction")
    atomic_json(
        cache / "evicted_features.json",
        dict(
            timestamp=time.time(),
            bytes=array.stat().st_size,
            sha256=manifest["features_sha256"],
            source_manifest_sha256=digest(cache / "manifest.json"),
            completed_fits=[
                dict(
                    fit=r["fit"],
                    request_sha256=r["request_sha256"],
                    policy_sha256=r["policy_sha256"],
                )
                for r in verified_fits
            ],
            reason="One frozen encoder cache at a time; preserve25GB reserve",
            regeneration=(
                "Archived source/request and preserved raw RGB shards reproduce the encoder cache. "
                "Use a fresh output and compare SHA."
            ),
        ),
    )
    array.unlink()


def main(args):
    args.out = args.out.resolve()
    data = ROOT / "outputs/navigation_spatial_data_20261019"
    registry = ROOT / "outputs/navigation_models/navigation_backbones_20261019.json"
    training_sources = [
        "scripts/train_navigation_visual_student.py",
        "src/navigation/spatial_student.py",
        "src/navigation/visual_decision_student.py",
        "src/navigation/spatial_data.py",
        "src/navigation/spatial_policy.py",
        "src/navigation/grid_encoders.py",
        "src/navigation/optical_fields.py",
        "src/navigation/policy_runtime.py",
        "scripts/train_navigation_student.py",
        "scripts/train_navigation_causal_student.py",
        "scripts/cache_navigation_grid_encoder.py",
        "scripts/profile_navigation_spatial_student.py",
        "scripts/pretrain_navigation_jepa.py",
        "src/navigation/jepa_pretraining.py",
        "src/navigation/rectangular_jepa.py",
        "src/jepa/encoder.py",
        "src/jepa/predictor.py",
        "reports/navigation_m4_spatial_protocol_20261019.md",
        "reports/navigation_visual_decision_protocol_20261019.md",
        "reports/navigation_m4_visual_protocol_20261019.md",
        "reports/navigation_jepa_continuation_protocol_20261019.md",
        "scripts/run_navigation_visual_study.py",
    ]
    request = dict(
        schema="navigation-visual-study66-v2",
        cells=fit_cells(),
        data_manifest_sha256=digest(data / "manifest.json"),
        registry_sha256=digest(registry),
        source_hashes={p: digest(ROOT / p) for p in training_sources},
        completed_first_fit=str(FIRST),
        first_cache=str(FIRST_CACHE),
        scope="Offline representation/label study; no driving or milestone gate promotion",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    request_path = args.out / "request.json"
    if request_path.exists():
        if not args.resume or json.loads(request_path.read_text()) != request:
            raise ValueError("Identical-source explicit resume required")
    else:
        if args.resume:
            raise ValueError("No study to resume")
        for source in training_sources:
            target = args.out / "source" / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, target)
        atomic_json(request_path, request)
    results = []
    active = None

    def progress(status="running", phase="fitting"):
        atomic_json(args.out / "results.json", results)
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M4 registered spatial study",
                fits_done=len(results),
                fits_total=66,
                active_output=str(active) if active else None,
                phase=phase,
                updated_at=time.time(),
            ),
        )
        ledger_path = ROOT / "reports/navigation_execution_20261018.json"
        ledger = json.loads(ledger_path.read_text())
        ledger["active_output"] = str(args.out.relative_to(ROOT))
        ledger["milestones"]["M4"]["status"] = (
            "offline_study_running" if status == "running" else "offline_study_" + status
        )
        ledger["milestones"]["M4"]["current_work"] = (
            f"{len(results)}/66 registered fits; {phase}; driving gates remain open"
        )
        atomic_json(ledger_path, ledger)

    def run(command, output):
        nonlocal active
        if (args.out / "STOP").exists():
            raise InterruptedError("Study STOP")
        active = output
        progress()
        output.parent.mkdir(parents=True, exist_ok=True)
        with (args.out / "console.log").open("a") as log:
            log.write(json.dumps(dict(command=command, time=time.time())) + "\n")
            log.flush()
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                if (args.out / "STOP").exists():
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "STOP").touch()
                progress()
                time.sleep(2)
        if child.returncode:
            raise RuntimeError(f"Stage failed ({child.returncode}):{output}; preserve and inspect")
        if (args.out / "STOP").exists():
            raise InterruptedError("Study STOP")

    def complete(path):
        p = path / "progress.json"
        return p.exists() and json.loads(p.read_text()).get("status") == "complete"

    status = "failed"
    try:
        for encoder in (*KINDS, "nav_color_only"):
            if all(
                (args.out / "completed" / (fit_name(c) + ".json")).exists()
                for c in fit_cells()
                if c["encoder"] == encoder
            ):
                group = []
                for cell in (c for c in fit_cells() if c["encoder"] == encoder):
                    record = json.loads(
                        (args.out / "completed" / (fit_name(cell) + ".json")).read_text()
                    )
                    verified = verify_fit(Path(record["fit"]), cell)
                    results.append(verified)
                    group.append(verified)
                if encoder not in ("cnn_scratch", "nav_color_only"):
                    cache = (
                        FIRST_CACHE
                        if encoder == "imagenet_resnet18"
                        else args.out / f"cache_{encoder}"
                    )
                    evict_features(cache, group)
                progress()
                continue
            current_registry = registry
            if encoder == "jepa_carla":
                pretraining = args.out / "jepa_pretraining"
                if not complete(pretraining):
                    profile = args.out / "jepa_profile"
                    if not complete(profile):
                        if profile.exists():
                            raise ValueError(
                                "Incomplete JEPA profile retained; inspect before a new attempt"
                            )
                        run(
                            [
                                sys.executable,
                                "-u",
                                "scripts/pretrain_navigation_jepa.py",
                                "--profile-only",
                                "--out",
                                str(profile),
                            ],
                            profile,
                        )
                    measured = json.loads((profile / "profile.json").read_text())
                    if measured["estimated_10000_steady_seconds"] > 18000:
                        raise ValueError(
                            "JEPA profile exceeds registered6h budget including diagnostics; "
                            "reassess before fitting"
                        )
                    command = [
                        sys.executable,
                        "-u",
                        "scripts/pretrain_navigation_jepa.py",
                        "--out",
                        str(pretraining),
                    ]
                    if pretraining.exists():
                        command.append("--resume")
                    run(command, pretraining)
                if not complete(pretraining):
                    raise RuntimeError("JEPA fixed update budget incomplete")
                registered = json.loads(registry.read_text())
                registered["jepa_carla"] = json.loads(
                    (pretraining / "encoder_descriptor.json").read_text()
                )
                current_registry = args.out / "derived_backbones.json"
                atomic_json(current_registry, registered)
            cache = None
            if encoder not in ("cnn_scratch", "nav_color_only"):
                cache = (
                    FIRST_CACHE if encoder == "imagenet_resnet18" else args.out / f"cache_{encoder}"
                )
                if not complete(cache):
                    if cache.exists():
                        raise ValueError(
                            "Incomplete encoder cache retained; inspect, do not overwrite"
                        )
                    run(
                        [
                            sys.executable,
                            "-u",
                            "scripts/cache_navigation_grid_encoder.py",
                            "--encoder",
                            encoder,
                            "--registry",
                            str(current_registry),
                            "--out",
                            str(cache),
                        ],
                        cache,
                    )
                if not (cache / "features.npy").exists():
                    raise ValueError("Encoder group unfinished but its features were evicted")
            group = []
            if encoder == "cnn_scratch":
                profile = args.out / "cnn_profile.json"
                if not profile.exists():
                    run(
                        [
                            sys.executable,
                            "-u",
                            "scripts/profile_navigation_spatial_student.py",
                            "--encoder",
                            encoder,
                            "--visual-decision",
                            "--out",
                            str(profile),
                        ],
                        args.out / "cnn_profile",
                    )
                measured = json.loads(profile.read_text())
                if measured["peak_vram_gb"] > 12 or measured["steady_mean_s"] * 2000 > 3200:
                    raise ValueError("Scratch feasibility exceeds registered resource/time budget")
            for cell in (c for c in fit_cells() if c["encoder"] == encoder):
                name = fit_name(cell)
                fit = FIRST if cell == fit_cells()[0] else args.out / "fits" / name
                if not complete(fit):
                    command = [
                        sys.executable,
                        "-u",
                        "scripts/train_navigation_visual_student.py",
                        "--encoder",
                        encoder,
                        "--fraction",
                        str(cell["fraction"]),
                        "--seed",
                        str(cell["seed"]),
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
                record = verify_fit(fit, cell)
                matched = [
                    r
                    for r in results
                    if r["seed"] == cell["seed"] and r["fraction"] == cell["fraction"]
                ]
                if any(
                    r["sampler_draws_sha256"] != record["sampler_draws_sha256"] for r in matched
                ):
                    raise ValueError("Matched encoder/ablation cells saw different label draws")
                for source, sha in json.loads((fit / "request.json").read_text())[
                    "source_hashes"
                ].items():
                    if request["source_hashes"].get(source) != sha:
                        raise ValueError("Matched study training source differs between cells")
                group.append(record)
                results.append(record)
                atomic_json(args.out / "completed" / (name + ".json"), record)
                progress()
            if cache:
                evict_features(cache, group)
        if len(results) != 66:
            raise ValueError("Incomplete registered study matrix")
        status = "complete"
    except InterruptedError:
        status = "stopped"
    except Exception as error:
        atomic_json(args.out / "error.json", dict(error=str(error), active_output=str(active)))
        raise
    finally:
        progress(status, "finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with RunLock(ROOT / "outputs/.navigation_visual_study.lock"):
        # Verify no simulator/trainer currently owns the GPU before orchestration.
        with RunLock(ROOT / "outputs/.navigation.lock"):
            pass
        main(args)
