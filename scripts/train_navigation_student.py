#!/usr/bin/env python3
"""Bounded first camera student; raw images, internal location validation, full recovery."""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.learning_data import ClipCache, prepare_cache  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, resource_shortfall  # noqa: E402
from src.navigation.student import make_student, student_loss  # noqa: E402


def atomic_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(".part")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def rng_state(rng):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "sampler": rng.bit_generator.state,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state, rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    rng.bit_generator.state = state["sampler"]
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@torch.no_grad()
def validate(model, cache, device, batch_size=4, blank_images=False):
    model.eval()
    sums = dict(
        speed=0.0,
        go=0.0,
        stopped=0.0,
        path=0.0,
        brake_correct=0,
        light_correct=0,
        red_correct=0,
        go_stalled=0,
        held_stopped=0,
        held_go=0,
    )
    counts = dict(speed=0, go=0, stopped=0, path=0, red=0, controller_go=0)
    for offset in range(0, len(cache.validation), batch_size):
        inputs, targets = cache.batch(cache.validation[offset : offset + batch_size], device)
        if blank_images:
            inputs["rgb"].zero_()
            inputs["front"].zero_()
        prediction, _ = model(**inputs)
        start = cache.length // 2
        p = {k: v[:, start:].float() for k, v in prediction.items()}
        t = {k: v[:, start:] for k, v in targets.items()}
        error = (p["current_speed"] - t["current_speed"]).abs()
        go, stopped = t["current_speed"] > 1, t["current_speed"] < 0.1
        sums["speed"] += error.sum().item()
        sums["go"] += error[go].sum().item()
        sums["stopped"] += error[stopped].sum().item()
        sums["go_stalled"] += ((p["current_speed"] < 0.5) & go).sum().item()
        counts["speed"] += error.numel()
        counts["go"] += go.sum().item()
        counts["stopped"] += stopped.sum().item()
        held = (p["current_speed"] < 0.15) | (p["brake"] >= 0)
        controller_go = go & (t["brake"] < 0.5)
        sums["held_stopped"] += (held & stopped).sum().item()
        sums["held_go"] += (held & controller_go).sum().item()
        counts["controller_go"] += controller_go.sum().item()
        mask = t["correction_mask"]
        sums["path"] += ((p["correction"] - t["correction"]).abs() * mask).sum().item()
        counts["path"] += mask.sum().item()
        sums["brake_correct"] += ((p["brake"] > 0) == (t["brake"] > 0.5)).sum().item()
        light = p["light"].argmax(-1)
        sums["light_correct"] += (light == t["light"]).sum().item()
        red = t["light"] == 1
        sums["red_correct"] += ((light == 1) & red).sum().item()
        counts["red"] += red.sum().item()
    result = {
        "speed_mae_mps": sums["speed"] / counts["speed"],
        "go_speed_mae_mps": sums["go"] / max(1, counts["go"]),
        "stopped_speed_mae_mps": sums["stopped"] / max(1, counts["stopped"]),
        "go_predicted_stop_fraction": sums["go_stalled"] / max(1, counts["go"]),
        "path_correction_mae_m": sums["path"] / max(1, counts["path"]),
        "brake_accuracy": sums["brake_correct"] / counts["speed"],
        "light_accuracy": sums["light_correct"] / counts["speed"],
        "red_recall": sums["red_correct"] / max(1, counts["red"]),
        "controller_stop_recall": sums["held_stopped"] / max(1, counts["stopped"]),
        "controller_false_hold_fraction": sums["held_go"] / max(1, counts["controller_go"]),
        "counts": counts,
    }
    result["selection_score"] = 0.5 * (
        result["go_speed_mae_mps"] + result["stopped_speed_mae_mps"]
    ) + 2 * (1 - result["brake_accuracy"])
    if hasattr(model, "stop"):
        result["selection_score"] = (
            0.5 * (1 - result["controller_stop_recall"] + result["controller_false_hold_fraction"])
            + 0.02 * result["go_speed_mae_mps"]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="outputs/navigation_m1_20k_v2_20261015")
    parser.add_argument("--cache", default="outputs/navigation_m2_cache_20k_20261015")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-wall", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--variant", choices=["legacy", "stop_go"], default="legacy")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument(
        "--init-from", type=Path, help="Own selected policy; fresh AdamW, not resume"
    )
    parser.add_argument("--corrections", type=Path, help="Passed DAgger assessment JSON")
    parser.add_argument(
        "--correction-cache", type=Path, default="outputs/navigation_m2_correction_cache_20261016"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 < args.max_wall <= 3600 or not 0 < args.steps <= 5000:
        raise ValueError("First-student jobs must be bounded to at most one hour/5000 steps")
    if args.batch_size < 1 or args.clip_length < 2:
        raise ValueError("Need a positive batch and at least two history frames")
    if not 0 < args.learning_rate <= 0.01:
        raise ValueError("Learning rate must be positive and at most 0.01")
    architecture = "camera-stop-go-v2" if args.variant == "stop_go" else "camera-student-v1"
    if args.corrections and (
        not args.init_from or args.variant != "stop_go" or args.batch_size < 2
    ):
        raise ValueError("Corrections require stop_go, --init-from and at least two batch clips")
    torch.set_num_threads(4)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    request_path = args.out / "request.json"
    if request_path.exists() and not args.resume:
        raise FileExistsError("Use --resume explicitly or a fresh output")
    files = [
        Path(__file__).resolve(),
        ROOT / "src/navigation/student.py",
        ROOT / "src/navigation/learning_data.py",
    ]
    if args.corrections:
        files += [ROOT / "src/navigation/correction_learning.py", ROOT / "src/navigation/dagger.py"]
    request = {
        "collection": str(Path(args.collection).resolve()),
        "cache": str(Path(args.cache).resolve()),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "clip_length": args.clip_length,
        "learning_rate": args.learning_rate,
        "architecture": architecture,
        "selection": (
            "Internal location validation: 0.5*(missed-stop + false-hold fraction) "
            "+ 0.02*go speed MAE; deployed controller decisions"
            if args.variant == "stop_go"
            else "Internal location validation: balanced go/stop speed MAE + 2*brake error"
        ),
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "amp": "bfloat16 on CUDA, no gradient scaler required",
    }
    initial = None
    if args.init_from:
        initial = torch.load(args.init_from, map_location="cpu", weights_only=True)
        if initial["architecture"] != architecture:
            raise ValueError("Warm-start architecture differs")
        request["initialization"] = {
            "path": str(args.init_from.resolve()),
            "sha256": hashlib.sha256(args.init_from.read_bytes()).hexdigest(),
            "selected_parent_step": initial["step"],
            "optimizer": "fresh AdamW",
            "selection_includes_parent_at_step0": True,
        }
    if args.corrections:
        request["corrections"] = {
            "assessment": str(args.corrections.resolve()),
            "sha256": hashlib.sha256(args.corrections.read_bytes()).hexdigest(),
            "cache": str(args.correction_cache.resolve()),
            "mix": "Half original clips, half corrections (odd extra correction); "
            "corrections balance stationary expert-go and required-stop endpoints",
            "future_targets": "All correction future speed masks zero",
        }
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Resume request/source changed")
    started, deadline = time.monotonic(), time.monotonic() + args.max_wall
    stop = args.out / "STOP"
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        if args.device == "cuda":
            problem = resource_shortfall(
                {"minimum_free_vram_gb": 3, "minimum_available_ram_gb": 10}
            )
            if problem:
                raise RuntimeError(problem)
        atomic_json(request_path, request)
        for source in files:
            archived = args.out / "source" / source.relative_to(ROOT)
            archived.parent.mkdir(parents=True, exist_ok=True)
            if archived.exists() and archived.read_bytes() != source.read_bytes():
                raise ValueError("Archived training source changed")
            shutil.copyfile(source, archived)

        def progress(status, phase, **values):
            atomic_json(
                args.out / "progress.json",
                {
                    "stage": "M2 camera student",
                    "status": status,
                    "phase": phase,
                    "updated_at": time.time(),
                    "elapsed_seconds": time.monotonic() - started,
                    "target_steps": args.steps,
                    **values,
                },
            )

        def preparing(done, total):
            progress("running", "preparing raw image cache", frames=done, target_frames=total)

        progress("starting", "preparing raw image cache")
        try:
            manifest = prepare_cache(args.collection, args.cache, stop, deadline, preparing)
        except InterruptedError as exc:
            progress("stopped", "preparing raw image cache", error=str(exc))
            return
        atomic_json(args.out / "dataset.json", manifest)
        correction_cache, correction_manifest = None, None
        if args.corrections:
            from src.navigation.correction_learning import CorrectionPool, prepare_corrections

            try:
                correction_manifest = prepare_corrections(
                    args.corrections, args.correction_cache, args.cache, stop, deadline, preparing
                )
            except InterruptedError as exc:
                progress("stopped", "preparing raw image cache", error=str(exc))
                return
            atomic_json(args.out / "correction_dataset.json", correction_manifest)
            correction_cache = CorrectionPool(args.correction_cache, args.clip_length)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        device = torch.device(args.device)
        cache = ClipCache(args.cache, args.clip_length)
        model = make_student(architecture).to(device)
        if initial is not None:
            model.load_state_dict(initial["model"], strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=0.0001
        )
        step, best, latest_validation = 0, float("inf"), None
        if args.resume:
            checkpoint = torch.load(args.out / "latest.pt", map_location=device, weights_only=False)
            if (
                checkpoint["request"] != request
                or checkpoint["dataset"] != manifest
                or checkpoint.get("correction_dataset") != correction_manifest
            ):
                raise ValueError("Checkpoint provenance differs from the current data/request")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            step, best = checkpoint["step"], checkpoint["best_score"]
            restore_rng(checkpoint["rng"], rng)

        def save():
            atomic_checkpoint(
                args.out / "latest.pt",
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "amp_scaler": None,
                    "rng": rng_state(rng),
                    "step": step,
                    "best_score": best,
                    "request": request,
                    "dataset": manifest,
                    "correction_dataset": correction_manifest,
                    "latest_validation": latest_validation,
                },
            )

        def export_policy():
            atomic_checkpoint(
                args.out / "best_policy.pt",
                {
                    "model": model.state_dict(),
                    "architecture": architecture,
                    "hidden": model.hidden_size,
                    "step": step,
                    "validation": latest_validation,
                    "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                },
            )

        # Baselines use training labels only; no held-out fitting or frame-level split.
        train_indices = np.concatenate(
            [np.arange(e["start"], e["end"]) for e in manifest["episodes"] if e["role"] == "train"]
        )
        val_indices = np.concatenate(
            [
                np.arange(e["start"], e["end"])
                for e in manifest["episodes"]
                if e["role"] == "validation"
            ]
        )
        train_mean = float(cache.arrays["current_speed"][train_indices].mean())
        val_speed = cache.arrays["current_speed"][val_indices]
        atomic_json(
            args.out / "baselines.json",
            {
                "train_mean_speed_mps": train_mean,
                "constant_train_mean_validation_speed_mae_mps": float(
                    np.abs(val_speed - train_mean).mean()
                ),
                "always_stopped_validation_speed_mae_mps": float(np.abs(val_speed).mean()),
                "scope": "Label prediction diagnostics on internal validation; "
                "not driving baselines",
            },
        )
        train_started, starting_step = time.monotonic(), step
        if initial is not None and not args.resume:
            latest_validation = validate(model, cache, device, args.batch_size)
            best = latest_validation["selection_score"]
            export_policy()
            with (args.out / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": 0, **latest_validation}) + "\n")
        save()
        try:
            while step < args.steps and not stop.exists() and time.monotonic() < deadline:
                model.train()
                original_size = args.batch_size // 2 if correction_cache else args.batch_size
                inputs, targets = cache.batch(cache.sample(rng, original_size), device)
                if correction_cache:
                    count = args.batch_size - original_size
                    extra_inputs, extra_targets = correction_cache.batch(
                        correction_cache.sample(rng, count), device
                    )
                    inputs = {k: torch.cat([v, extra_inputs[k]], 0) for k, v in inputs.items()}
                    targets = {k: torch.cat([v, extra_targets[k]], 0) for k, v in targets.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    predictions, _ = model(**inputs)
                    loss, losses = student_loss(predictions, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite camera student loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5, error_if_nonfinite=True
                )
                optimizer.step()
                step += 1
                row = {
                    "step": step,
                    "loss": loss.item(),
                    "grad_norm": norm.item(),
                    **{k: v.item() for k, v in losses.items()},
                }
                if step == 1 or step % 10 == 0:
                    with (args.out / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    rate = (step - starting_step) / max(1e-6, time.monotonic() - train_started)
                    progress(
                        "running",
                        "training",
                        steps=step,
                        steps_per_second=rate,
                        train_loss=loss.item(),
                        validation=latest_validation,
                    )
                if step % 100 == 0 or step == args.steps:
                    latest_validation = validate(model, cache, device, args.batch_size)
                    with (args.out / "validation.jsonl").open("a") as stream:
                        stream.write(json.dumps({"step": step, **latest_validation}) + "\n")
                    if latest_validation["selection_score"] < best:
                        best = latest_validation["selection_score"]
                        export_policy()
                    save()
                    print(json.dumps({"step": step, "validation": latest_validation}), flush=True)
            save()
            status = "complete" if step >= args.steps else "stopped"
            progress(status, "training", steps=step, validation=latest_validation)
            print(json.dumps({"status": status, "step": step}), flush=True)
        except BaseException as exc:
            save()
            progress("failed", "training", steps=step, error=str(exc))
            raise


if __name__ == "__main__":
    main()
