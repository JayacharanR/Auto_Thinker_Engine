#!/usr/bin/env python3
"""Matched-window diagnostics for the first student; these are not driving scores."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_student import validate  # noqa: E402
from src.navigation.learning_data import ClipCache  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402
from src.navigation.student import make_student  # noqa: E402


@torch.no_grad()
def control_diagnostics(model, cache, device):
    """Test deployed holds; positive-speed/brake overlap alone is not an error."""
    counts = dict(
        frames=0,
        positive_speed_and_brake=0,
        teacher_go=0,
        held_on_teacher_go=0,
        teacher_start_go=0,
        held_on_teacher_start_go=0,
        teacher_stopped=0,
        held_on_teacher_stopped=0,
    )
    model.eval()
    for offset in range(0, len(cache.validation), 4):
        inputs, targets = cache.batch(cache.validation[offset : offset + 4], device)
        prediction, _ = model(**inputs)
        start = cache.length // 2
        p = {k: v[:, start:] for k, v in prediction.items()}
        t = {k: v[:, start:] for k, v in targets.items()}
        held = (p["current_speed"] < 0.15) | (p["brake"] >= 0)
        go = (t["current_speed"] > 1) & (t["brake"] < 0.5)
        start_go = go & (inputs["ego"][:, start:, 0] < 0.2)
        stopped = t["current_speed"] < 0.1
        counts["frames"] += held.numel()
        counts["positive_speed_and_brake"] += (
            ((p["current_speed"] >= 0.15) & (p["brake"] >= 0)).sum().item()
        )
        for name, mask in (
            ("teacher_go", go),
            ("teacher_start_go", start_go),
            ("teacher_stopped", stopped),
        ):
            counts[name] += mask.sum().item()
            counts["held_on_" + name] += (held & mask).sum().item()
    return {
        "counts": counts,
        "held_on_teacher_go_fraction": counts["held_on_teacher_go"] / max(1, counts["teacher_go"]),
        "held_on_teacher_start_go_fraction": (
            counts["held_on_teacher_start_go"] / counts["teacher_start_go"]
            if counts["teacher_start_go"]
            else None
        ),
        "held_on_teacher_stopped_fraction": counts["held_on_teacher_stopped"]
        / max(1, counts["teacher_stopped"]),
        "scope": "Same internal validation windows; deployed hold = speed<0.15 or brake>=0.5. "
        "Teacher-go excludes teacher braking; not a closed-loop success metric. "
        "Positive target and braking can be valid deceleration above target speed; "
        "their co-occurrence count is not an error count.",
    }


def assess(run, device, clip_length=None):
    run = Path(run)
    request = json.loads((run / "request.json").read_text())
    cache = ClipCache(request["cache"], clip_length or request["clip_length"])
    checkpoint = torch.load(run / "best_policy.pt", map_location="cpu", weights_only=False)
    model = make_student(checkpoint["architecture"], checkpoint["hidden"]).to(device)
    model.load_state_dict(checkpoint["model"])
    original = validate(model, cache, device)
    blank = validate(model, cache, device, blank_images=True)
    controls = control_diagnostics(model, cache, device)
    indices = (
        np.asarray(cache.validation)[:, None] + np.arange(cache.length // 2, cache.length)[None]
    ).flatten()
    train = np.concatenate(
        [
            np.arange(e["start"], e["end"])
            for e in cache.manifest["episodes"]
            if e["role"] == "train"
        ]
    )
    truth = cache.arrays["current_speed"][indices]
    train_mean = float(cache.arrays["current_speed"][train].mean())
    train_brake = bool(cache.arrays["brake"][train].mean() > 0.5)
    train_light = int(np.bincount(cache.arrays["light"][train], minlength=5).argmax())
    mask = cache.arrays["correction_mask"][indices]
    baselines = {
        "always_stop_speed_mae_mps": float(truth.mean()),
        "train_mean_speed_mae_mps": float(np.abs(truth - train_mean).mean()),
        "route_without_correction_mae_m": float(
            (np.abs(cache.arrays["correction"][indices]) * mask).sum() / max(1, mask.sum())
        ),
        "train_majority_brake": train_brake,
        "train_majority_brake_accuracy": float(
            (cache.arrays["brake"][indices] == train_brake).mean()
        ),
        "train_majority_light": train_light,
        "train_majority_light_accuracy": float(
            (cache.arrays["light"][indices] == train_light).mean()
        ),
        "evaluated_frames": len(indices),
        "scope": "Exactly the same held-out-internal clip positions and burn-in as model metrics",
    }
    return {
        "run": str(run.resolve()),
        "best_step": checkpoint["step"],
        "architecture": checkpoint["architecture"],
        "evaluation_clip_length": cache.length,
        "checkpoint_sha256": hashlib.sha256((run / "best_policy.pt").read_bytes()).hexdigest(),
        "internal_validation": original,
        "all_images_blanked": blank,
        "baselines": baselines,
        "controller_diagnostics": controls,
        "teacher_braking_audit": json.loads((run / "teacher_braking_audit.json").read_text())
        if (run / "teacher_braking_audit.json").exists()
        else None,
        "label_observability": json.loads((run / "light_label_diagnostic.json").read_text())
        if (run / "light_label_diagnostic.json").exists()
        else None,
        "frozen_training_source_changes": [
            p
            for p, h in request["source_hashes"].items()
            if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h
        ],
        "archived_training_sources_verified": all(
            (run / "source" / p).exists()
            and hashlib.sha256((run / "source" / p).read_bytes()).hexdigest() == h
            for p, h in request["source_hashes"].items()
        ),
        "scope": "Internal location validation only. Blanking is an input-sensitivity diagnostic, "
        "not a trained navigation-only baseline or evidence of driving performance. "
        "No heldout_dev or Town05 samples used.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clip-length", type=int)
    parser.add_argument(
        "--reference", type=Path, help="Frozen policy, assessed on identical windows"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)

    def assessments():
        result = assess(args.out, args.device, args.clip_length)
        if args.reference:
            request = json.loads((args.out / "request.json").read_text())
            reference_request = json.loads((args.reference / "request.json").read_text())
            if request["cache"] != reference_request["cache"]:
                raise ValueError("Matched comparison requires the same immutable data cache")
            result["reference"] = assess(
                args.reference, args.device, result["evaluation_clip_length"]
            )
            result["comparison_note"] = (
                "Frozen checkpoints on identical windows/burn-in. The saved requests record "
                "which architecture/data/optimization choices differ; this comparison does "
                "not establish an isolated architecture effect or equal training compute."
            )
        return result

    if args.device == "cuda":
        with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
            result = assessments()
    else:
        result = assessments()
    atomic_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
