#!/usr/bin/env python3
"""One TRAIN forward/backward feasibility profile, with no optimizer update."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_spatial_student import (  # noqa: E402
    initialize,
    inputs_for,
    selected_pool,
)
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402
from src.navigation.spatial_data import SpatialCorpus  # noqa: E402
from src.navigation.spatial_student import driving_loss  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--data", type=Path, default=ROOT / "outputs/navigation_spatial_data_20261019")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--visual-decision", action="store_true")
    args = p.parse_args()
    corpus = SpatialCorpus(args.data)
    pool, _ = selected_pool(corpus.manifest, 1.0)
    episode = max(pool, key=lambda e: e["end"] - e["start"])
    features = (
        np.load(args.cache / "features.npy", mmap_mode="r", allow_pickle=False)
        if args.cache
        else None
    )
    if args.encoder == "cnn_scratch":
        corpus.raw.preload_missing()
    dimension = 512 if args.encoder == "imagenet_resnet18" else 384
    torch.set_num_threads(2)
    initializer = initialize
    if args.visual_decision:
        from scripts.train_navigation_visual_student import initialize as visual_initialize

        initializer = visual_initialize
    model = initializer(args.encoder, dimension, 42).cuda()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, t = inputs_for(
                model, corpus, episode, "cuda", features, gradients=args.encoder == "cnn_scratch"
            )
            prediction, _ = model(**x)
            loss, _ = driving_loss(prediction, t)
        loss.backward()
        torch.cuda.synchronize()
        timings.append(time.monotonic() - started)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients) and torch.isfinite(loss)
    result = dict(
        encoder=args.encoder,
        neural_ego_speed_enabled=not args.visual_decision,
        episode_id=episode["id"],
        frames=episode["end"] - episode["start"],
        forward_backward_s=timings[0],
        repeated_forward_backward_s=timings,
        steady_mean_s=sum(timings[1:]) / 2,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        finite_gradients=True,
        optimizer_updates=0,
        note=(
            "Longest TRAIN episode feasibility, includes first-call warmup; "
            "no trained checkpoint or validation result"
        ),
    )
    atomic_json(args.out, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation.lock"):
        main()
