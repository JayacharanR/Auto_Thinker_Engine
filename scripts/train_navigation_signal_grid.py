#!/usr/bin/env python3
"""Bounded RGB-localization component; never exports a driving-policy checkpoint."""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_student import atomic_checkpoint, restore_rng, rng_state  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, resource_shortfall  # noqa: E402
from src.navigation.signal_learning import SignalDataset  # noqa: E402
from src.navigation.signal_model import SignalGrid, signal_loss  # noqa: E402


@torch.inference_mode()
def validate(model, dataset, device):
    model.eval()
    sums = dict(
        foreground=0.0,
        background=0.0,
        positives=0,
        negatives=0,
        correct=0,
        window_correct=0,
        false_half=0,
        false_nine=0,
    )
    classes = {str(cls): dict(correct=0, total=0) for cls in (1, 2, 3)}
    for offset in range(0, len(dataset.validation), 4):
        indices = dataset.validation[offset : offset + 4]
        rgb, target = dataset.batch(indices, device)
        logits = model(rgb).float()
        error = F.cross_entropy(logits, target.long(), reduction="none", ignore_index=255)
        foreground, background = (target > 0) & (target < 4), target == 0
        sums["foreground"] += error[foreground].sum().item()
        sums["background"] += error[background].sum().item()
        sums["positives"] += foreground.sum().item()
        sums["negatives"] += background.sum().item()
        probability = logits.softmax(1)
        active = probability[:, 1:].amax(1)
        sums["false_half"] += ((active >= 0.5) & background).sum().item()
        sums["false_nine"] += ((active >= 0.9) & background).sum().item()
        for index, sample in enumerate(indices):
            for cx, cy, cls in dataset.samples[sample]["points"]:
                correct = int(probability[index, :, cy, cx].argmax().item() == cls)
                sums["correct"] += correct
                classes[str(cls)]["correct"] += correct
                classes[str(cls)]["total"] += 1
                nearby = probability[index, cls, max(0, cy - 2) : cy + 3, max(0, cx - 2) : cx + 3]
                sums["window_correct"] += int(nearby.max().item() >= 0.5)
    total = sum(c["total"] for c in classes.values())
    if not total or not sums["negatives"]:
        raise ValueError("Frozen component validation lacks positive/background support")
    foreground_loss = sums["foreground"] / max(1, sums["positives"])
    background_loss = sums["background"] / sums["negatives"]
    return {
        "selection_score": foreground_loss + 0.2 * background_loss,
        "foreground_cross_entropy": foreground_loss,
        "background_cross_entropy": background_loss,
        "center_class_accuracy": sums["correct"] / total,
        "supported_points": total,
        "class_counts": classes,
        "within4px_at_probability_half": sums["window_correct"] / total,
        "background_active_fraction_at_half": sums["false_half"] / sums["negatives"],
        "background_active_fraction_at_nine": sums["false_nine"] / sums["negatives"],
        "scope": "Sparse static lamp centers/background; not detection AP or driving reliability",
    }


def train(args):
    if args.out.exists() and not args.resume:
        raise FileExistsError("Fresh component output or explicit --resume required")
    if not 0 < args.max_wall <= 1800 or args.steps < 1 or args.batch_size < 2:
        raise ValueError("Bounded component budget/batch required")
    if args.device != "cpu":
        reason = resource_shortfall({"minimum_free_vram_gb": 2, "minimum_available_ram_gb": 4})
        if reason:
            raise RuntimeError(reason)
    started, deadline = time.monotonic(), time.monotonic() + args.max_wall
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def progress(status, step, validation=None, **extra):
        atomic_json(
            args.out / "progress.json",
            {
                "status": status,
                "stage": "M2 signal localizer",
                "steps": step,
                "target_steps": args.steps,
                "updated_at": time.time(),
                "elapsed_seconds": time.monotonic() - started,
                "validation": validation,
                "invocation": args.invocation,
                **extra,
            },
        )

    dataset = SignalDataset(args.corpus, supervision=args.supervision)
    sources = [
        *sorted((ROOT / "src/navigation").glob("*.py")),
        Path(__file__).resolve(),
        ROOT / "scripts/train_navigation_student.py",
        ROOT / "scripts/collect_navigation_signal_pairs.py",
        ROOT / "scripts/probe_navigation_signal_pixels.py",
        ROOT / "scripts/audit_navigation_rule_observability.py",
    ]
    request = {
        "kind": "RGB phase-localization component only, no driving model",
        "architecture": "signal-grid-v1",
        "training_supervision": args.supervision,
        "corpus": str(args.corpus.resolve()),
        "corpus_audit_sha256": digest(args.corpus / "data_audit.json"),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "learning_rate": 0.001,
        "selection": "Frozen whole-junction validation foreground CE +0.2 background CE",
        "precision": "BF16 train on CUDA, float32 validation; no scaler required",
        "source_hashes": {str(path.relative_to(ROOT)): digest(path) for path in sources},
    }
    if args.resume:
        if json.loads((args.out / "request.json").read_text()) != request or json.loads(
            (args.out / "dataset.json").read_text()
        ) != json.loads(json.dumps(dataset.manifest)):
            raise ValueError("Component resume provenance differs")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    if not args.resume:
        atomic_json(args.out / "request.json", request)
        atomic_json(args.out / "dataset.json", dataset.manifest)
        for source in sources:
            target = args.out / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    progress("running", 0, phase="initializing component")
    model = SignalGrid().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    step, best, selected_step = 0, float("inf"), 0
    if args.resume:
        state = torch.load(args.out / "latest.pt", map_location=args.device, weights_only=False)
        if state["request"] != request:
            raise ValueError("Component checkpoint request differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], rng)
        step, best, selected_step = state["step"], state["best_score"], state["selected_step"]

    def save(validation, improved=False):
        atomic_checkpoint(
            args.out / "latest.pt",
            {
                "request": request,
                "kind": "perception_component_only",
                "architecture": "signal-grid-v1",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(rng),
                "step": step,
                "best_score": best,
                "selected_step": selected_step,
                "validation": validation,
            },
        )
        if improved:
            atomic_checkpoint(
                args.out / "best_component.pt",
                {
                    "kind": "perception_component_only",
                    "architecture": "signal-grid-v1",
                    "model": model.state_dict(),
                    "step": step,
                    "validation": validation,
                },
            )

    latest_validation = validate(model, dataset, args.device)
    if not args.resume:
        best = latest_validation["selection_score"]
        save(latest_validation, True)
        with (args.out / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": 0, **latest_validation}) + "\n")
    status = "complete"
    try:
        while step < args.steps:
            if (args.out / "STOP").exists() or time.monotonic() >= deadline:
                status = "stopped" if (args.out / "STOP").exists() else "budget_exhausted"
                break
            model.train()
            rgb, target = dataset.training_batch(rng, args.batch_size, args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch.device(args.device).type,
                dtype=torch.bfloat16,
                enabled=args.device != "cpu",
            ):
                loss = signal_loss(model(rgb), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite signal loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            step += 1
            if step % 20 == 0:
                progress(
                    "running",
                    step,
                    latest_validation,
                    phase="training component",
                    loss=float(loss.detach()),
                    steps_per_second=step / max(1, time.monotonic() - started),
                )
            if step % 200 == 0 or step == args.steps:
                latest_validation = validate(model, dataset, args.device)
                improved = latest_validation["selection_score"] < best
                if improved:
                    best, selected_step = latest_validation["selection_score"], step
                save(latest_validation, improved)
                with (args.out / "validation.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": step, **latest_validation}) + "\n")
                print(json.dumps({"step": step, "validation": latest_validation}), flush=True)
    except BaseException:
        status = "failed"
        raise
    finally:
        save(latest_validation)
        progress(
            "running" if status != "failed" else status,
            step,
            latest_validation,
            selected_step=selected_step,
            phase="finalizing component review",
        )
    selected = torch.load(args.out / "best_component.pt", map_location="cpu", weights_only=False)
    atomic_json(
        args.out / "review.json",
        {
            "status": status,
            "selected_step": selected["step"],
            "validation": selected["validation"],
            "wall_seconds": time.monotonic() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "component_sha256": digest(args.out / "best_component.pt"),
            "scope": "RGB phase-localization prototype only; no driving claim or policy promotion",
        },
    )
    progress(status, step, latest_validation, selected_step=selected_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-wall", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--supervision", choices=["center", "region"], default="center")
    args = parser.parse_args()
    args.invocation = time.time_ns()
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        try:
            train(args)
        except BaseException as exc:
            path = args.out / "progress.json"
            if path.exists():
                p = json.loads(path.read_text())
                if p.get("invocation") == args.invocation:
                    atomic_json(
                        path, {**p, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    )
            raise


if __name__ == "__main__":
    main()
