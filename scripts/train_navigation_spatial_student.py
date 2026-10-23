#!/usr/bin/env python3
"""Fresh spatial camera heads on fixed complete-episode label budgets."""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_causal_student import score_stop_metrics  # noqa: E402
from scripts.train_navigation_student import atomic_checkpoint, restore_rng, rng_state  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.spatial_data import SpatialCorpus, validation_positions  # noqa: E402
from src.navigation.spatial_student import SpatialCameraStudent, driving_loss  # noqa: E402


def selected_pool(manifest, fraction):
    allocation = next(
        a for a in manifest["annotation_budget"]["allocations"] if a["nominal_fraction"] == fraction
    )
    ids = set(allocation["groups"])
    pool = [e for e in manifest["episodes"] if e["role"] == "train" and e["id"] in ids]
    if (
        not pool
        or {e["id"] for e in pool} != ids
        or sum(e["labels"] for e in pool) != allocation["actual_labels"]
    ):
        raise ValueError("Nonempty exact registered label allocation required")
    return pool, allocation


def initialize(encoder, dimension, seed, color_enabled=True):
    """Keep shared driving-head initialization equal despite backbone dimensions."""
    torch.random.default_generator.manual_seed(seed)
    model = SpatialCameraStudent(
        dimension, scratch=encoder == "cnn_scratch", color_enabled=color_enabled
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        template = SpatialCameraStudent(384, color_enabled=color_enabled)
    common = {
        k: v for k, v in template.state_dict().items() if not k.startswith(("low.", "scratch."))
    }
    model.load_state_dict(common, strict=False)
    return model


def inputs_for(model, corpus, episode, device, features, gradients=False):
    x, t = corpus.episode(episode, device, features)
    if model.scratch is not None:
        rgb = torch.from_numpy(np.asarray(corpus.raw.rgb(episode)).copy()).to(device)[None]
        x["features"] = model.scratch_features(rgb, gradients=gradients)
    elif features is None:
        x["features"] = torch.zeros(1, episode["end"] - episode["start"], 3, 16, 384, device=device)
    return x, t


@torch.inference_mode()
def validate(model, corpus, device, features, check=lambda: None):
    model.eval()
    sums = dict(speed=0.0, go=0.0, held_stopped=0, held_go=0, path=0.0)
    counts = dict(speed=0, go=0, stopped=0, controller_go=0)
    path_count = 0.0
    families = {}
    for e in corpus.manifest["episodes"]:
        if e["role"] != "validation":
            continue
        check()
        x, t = inputs_for(model, corpus, e, device, features)
        p, _ = model(**x)
        mask = validation_positions(e["end"] - e["start"], device)
        p = {k: v[:, mask].float() for k, v in p.items()}
        t = {k: v[:, mask] for k, v in t.items()}
        error = (p["current_speed"] - t["current_speed"]).abs()
        stopped, go = t["current_speed"] < 0.1, t["current_speed"] > 1
        held = (p["current_speed"] < 0.15) | (p["brake"] >= 0)
        controller_go = go & (t["brake"] < 0.5)
        sums["speed"] += error.sum().item()
        sums["go"] += error[go].sum().item()
        sums["held_stopped"] += (held & stopped).sum().item()
        sums["held_go"] += (held & controller_go).sum().item()
        counts["speed"] += error.numel()
        counts["go"] += int(go.sum())
        counts["stopped"] += int(stopped.sum())
        counts["controller_go"] += int(controller_go.sum())
        sums["path"] += (
            ((p["correction"] - t["correction"]).abs() * t["correction_mask"]).sum().item()
        )
        path_count += t["correction_mask"].sum().item()
        f = families.setdefault(
            e["family"], dict(stopped=0, stop_hits=0, controller_go=0, false_holds=0)
        )
        for k, v in dict(
            stopped=stopped.sum(),
            stop_hits=(held & stopped).sum(),
            controller_go=controller_go.sum(),
            false_holds=(held & controller_go).sum(),
        ).items():
            f[k] += int(v)
    if counts != dict(speed=1312, go=724, stopped=560, controller_go=582):
        raise ValueError("Registered internal validation labels/counts changed")
    return dict(
        score_stop_metrics(sums, counts),
        path_correction_mae_m=sums["path"] / max(1, path_count),
        families=families,
    )


def train(args):
    if args.steps != 2000 or not 0 < args.max_wall <= 3600:
        raise ValueError("Registered2000-update/1h downstream budget required")
    corpus = SpatialCorpus(args.data)
    pool, allocation = selected_pool(corpus.manifest, args.fraction)
    cache_manifest = None
    features = None
    descriptor = None
    if args.encoder not in ("cnn_scratch", "nav_color_only"):
        if args.cache is None:
            raise ValueError("Frozen encoder needs a completed feature cache")
        cache_manifest = json.loads((args.cache / "manifest.json").read_text())
        if cache_manifest["encoder"] != args.encoder or cache_manifest[
            "data_manifest_sha256"
        ] != digest(args.data / "manifest.json"):
            raise ValueError("Encoder cache/data identity mismatch")
        if digest(args.cache / "features.npy") != cache_manifest["features_sha256"]:
            raise ValueError("Cached frozen features changed")
        cache_request = json.loads((args.cache / "request.json").read_text())
        if digest(args.cache / "request.json") != cache_manifest["request_sha256"]:
            raise ValueError("Frozen encoder cache request changed")
        descriptor = cache_request["encoder_descriptor"]
        if any(
            digest(args.cache / "source" / p) != h
            for p, h in cache_request["source_hashes"].items()
        ):
            raise ValueError("Archived encoder extraction source changed")
        features = np.load(args.cache / "features.npy", mmap_mode="r", allow_pickle=False)
    if any(digest(args.data / p) != h for p, h in corpus.manifest["array_sha256"].items()):
        raise ValueError("Shared optical/annotation arrays changed")
    check_resources(args.out, 25)
    sources = [
        Path(__file__),
        ROOT / "src/navigation/spatial_student.py",
        ROOT / "src/navigation/spatial_data.py",
        ROOT / "src/navigation/optical_fields.py",
        ROOT / "src/navigation/spatial_policy.py",
        ROOT / "src/navigation/grid_encoders.py",
        ROOT / "src/navigation/policy_runtime.py",
        ROOT / "scripts/train_navigation_student.py",
        ROOT / "scripts/train_navigation_causal_student.py",
        ROOT / "reports/navigation_m4_spatial_protocol_20261019.md",
    ]
    request = dict(
        encoder=args.encoder,
        seed=args.seed,
        fraction=args.fraction,
        allocation=allocation,
        label_family_coverage={
            f: sum(e["labels"] for e in pool if e["family"] == f)
            for f in sorted({e["family"] for e in pool})
        },
        data_manifest_sha256=digest(args.data / "manifest.json"),
        encoder_cache_sha256=digest(args.cache / "manifest.json") if args.cache else None,
        encoder_descriptor=descriptor,
        steps=2000,
        learning_rate=0.0003,
        color_enabled=not args.no_color,
        maximum_wall_seconds=args.max_wall,
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
    )
    request_path = args.out / "request.json"
    if request_path.exists():
        if not args.resume or json.loads(request_path.read_text()) != request:
            raise ValueError(
                "Existing/source-changed fit requires a fresh output; identical-source resume only"
            )
    else:
        if args.resume:
            raise ValueError("No fit to resume")
        for p in sources:
            target = args.out / "source" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        atomic_json(request_path, request)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)
    dimension = 512 if args.encoder == "imagenet_resnet18" else 384
    model = initialize(args.encoder, dimension, args.seed, not args.no_color).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    step = selected = 0
    validation_step = -1
    best = float("inf")
    latest = None
    previous_elapsed = 0
    if args.resume:
        saved = torch.load(args.out / "latest.pt", map_location="cpu", weights_only=False)
        if saved["request_sha256"] != digest(request_path):
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        restore_rng(saved["rng"], rng)
        step, selected, validation_step = (
            saved["step"],
            saved["selected_step"],
            saved["validation_step"],
        )
        best, latest, previous_elapsed = (
            saved["best_score"],
            saved["validation"],
            saved["elapsed_total_s"],
        )
    started = time.monotonic()

    def check():
        if (
            args.out / "STOP"
        ).exists() or previous_elapsed + time.monotonic() - started >= args.max_wall:
            raise InterruptedError("Spatial student STOP/wall cap")

    def progress(status="running", phase="training"):
        elapsed = previous_elapsed + time.monotonic() - started
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M4 spatial student",
                encoder=args.encoder,
                seed=args.seed,
                fraction=args.fraction,
                labels=allocation["actual_labels"],
                steps=step,
                target_steps=2000,
                selected_step=selected,
                validation_step=validation_step,
                phase=phase,
                elapsed_s=elapsed,
                validation=latest,
                updated_at=time.time(),
            ),
        )

    def checkpoint_state():
        return dict(
            model=model.state_dict(),
            architecture="spatial-stop-go-v1",
            hidden=192,
            encoder_kind=args.encoder,
            encoder_dim=dimension,
            color_enabled=not args.no_color,
            encoder_descriptor=descriptor,
            step=step,
            request_sha256=digest(request_path),
            data_manifest_sha256=digest(args.data / "manifest.json"),
        )

    def save():
        atomic_checkpoint(
            args.out / "latest.pt",
            dict(
                checkpoint_state(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(rng),
                selected_step=selected,
                validation_step=validation_step,
                best_score=best,
                validation=latest,
                elapsed_total_s=previous_elapsed + time.monotonic() - started,
            ),
        )

    def evaluate():
        nonlocal selected, best, latest, validation_step
        progress(phase="causal validation")
        latest = validate(model, corpus, "cuda", features, check)
        validation_step = step
        improved = latest["selection_score"] < best - 1e-8
        if improved:
            selected, best = step, latest["selection_score"]
            atomic_checkpoint(args.out / "best_policy.pt", checkpoint_state())
        with (args.out / "validation.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=step, metrics=latest, selected=improved)) + "\n")
        save()
        atomic_json(
            args.out / "review.json",
            dict(
                selected_step=selected,
                best_score=best,
                latest=latest,
                policy_sha256=digest(args.out / "best_policy.pt"),
                driving_evidence=False,
                legacy_parent_promoted=False,
                annotation_labels=allocation["actual_labels"],
            ),
        )
        print(json.dumps(dict(step=step, selected=selected, metrics=latest)), flush=True)

    status = "stopped"
    try:
        progress(phase="RGB reconstruction if scratch")
        if args.encoder == "cnn_scratch":
            corpus.raw.preload_missing(check)
        if validation_step != step and step % 500 == 0:
            evaluate()
        families = sorted({e["family"] for e in pool})
        while step < 2000:
            check()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            family = families[int(rng.integers(len(families)))]
            entries = [e for e in pool if e["family"] == family]
            e = entries[int(rng.integers(len(entries)))]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, t = inputs_for(
                    model, corpus, e, "cuda", features, gradients=args.encoder == "cnn_scratch"
                )
                p, _ = model(**x)
                loss, parts = driving_loss(p, t)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite spatial student loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            with (args.out / "train.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            step=step,
                            episode_id=e["id"],
                            family=family,
                            frames=e["end"] - e["start"],
                            loss=float(loss.detach()),
                            parts={k: float(v.detach()) for k, v in parts.items()},
                        )
                    )
                    + "\n"
                )
            if step % 500 == 0:
                evaluate()
            elif step % 100 == 0:
                save()
            if step % 10 == 0:
                progress()
        status = "complete"
    except InterruptedError as error:
        print(str(error), flush=True)
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), step=step))
        raise
    finally:
        save()
        progress(status, phase="finished")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--encoder",
        choices=(
            "cnn_scratch",
            "imagenet_resnet18",
            "dinov2_small",
            "jepa_comma2k19",
            "jepa_carla",
            "nav_color_only",
        ),
        required=True,
    )
    p.add_argument("--data", type=Path, default=ROOT / "outputs/navigation_spatial_data_20261019")
    p.add_argument("--cache", type=Path)
    p.add_argument("--fraction", type=float, choices=(0.05, 0.1, 0.25, 1.0), default=1.0)
    p.add_argument("--seed", type=int, choices=(42, 123, 456), default=42)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--max-wall", type=float, default=3600)
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    with RunLock(ROOT / "outputs/.navigation.lock"):
        train(p.parse_args())
