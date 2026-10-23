#!/usr/bin/env python3
"""Bounded, source-frozen TRAIN-only JEPA continuation with exact recovery."""

import argparse
import copy
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_student import atomic_checkpoint, restore_rng, rng_state  # noqa: E402
from src.jepa.encoder import ViTEncoder  # noqa: E402
from src.jepa.predictor import JEPAPredictor  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.jepa_pretraining import (  # noqa: E402
    ema_update,
    normalized_video,
    prediction_loss,
    sample_clips,
    tube_masks,
    unlabeled_split,
    variance_diagnostics,
)
from src.navigation.rectangular_jepa import RectangularJEPA, grid_positions  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.spatial_data import DOMAINS, RawContext  # noqa: E402


def models(initial):
    original = torch.load(initial, map_location="cpu", weights_only=False)
    backbone = ViTEncoder(
        img_size=128,
        num_frames=8,
        patch_size=16,
        tubelet_size=2,
        embed_dim=384,
        depth=12,
        num_heads=6,
    )
    backbone.load_state_dict(original["context_encoder_state_dict"], strict=True)
    context = RectangularJEPA(backbone, (4, 8, 8)).cuda()
    predictor = JEPAPredictor(num_patches=576, action_conditioning=False).cuda()
    state = dict(original["predictor_state_dict"])
    if any(k.startswith("action_proj.") for k in state):
        raise ValueError("M4 source predictor must be action-unconditioned")
    state["pos_embed"] = grid_positions(state["pos_embed"], (4, 8, 8), (4, 9, 16))
    predictor.load_state_dict(state, strict=True)
    teacher = copy.deepcopy(context).eval().requires_grad_(False)
    return context, predictor, teacher


def main(args):
    registry = json.loads(
        (ROOT / "outputs/navigation_models/navigation_backbones_20261019.json").read_text()
    )
    initial = Path(registry["jepa_comma2k19"]["checkpoint"])
    if digest(initial) != registry["jepa_comma2k19"]["sha256"]:
        raise ValueError("Source JEPA changed")
    manifest = json.loads((args.data / "manifest.json").read_text())
    train, validation = unlabeled_split(manifest)
    check_resources(args.out, 26.0)
    if any(digest(p) != h for p, h in manifest["source_shard_sha256"].items()):
        raise ValueError("Preserved raw source shard changed")
    raw_rgb_sha256 = {
        name: digest(root / "rgb.npy")
        for name, root in DOMAINS.items()
        if (root / "rgb.npy").exists()
    }
    sources = [
        Path(__file__),
        ROOT / "src/navigation/jepa_pretraining.py",
        ROOT / "src/navigation/rectangular_jepa.py",
        ROOT / "src/navigation/spatial_data.py",
        ROOT / "src/jepa/encoder.py",
        ROOT / "src/jepa/predictor.py",
        ROOT / "scripts/train_navigation_student.py",
        ROOT / "reports/navigation_m4_spatial_protocol_20261019.md",
        ROOT / "reports/navigation_jepa_continuation_protocol_20261019.md",
    ]
    request = dict(
        initial_sha256=digest(initial),
        data_manifest_sha256=digest(args.data / "manifest.json"),
        train_episodes=[e["id"] for e in train],
        validation_episodes=[e["id"] for e in validation],
        train_frames=sum(e["labels"] for e in train),
        validation_frames=sum(e["labels"] for e in validation),
        action_labels_consumed=0,
        raw_rgb_sha256=raw_rgb_sha256,
        steps=10000,
        clips_per_update=8,
        views=3,
        microbatch_views=6,
        seed=42,
        optimizer="AdamW lr1e-4 weight_decay.05",
        ema=0.996,
        maximum_wall_seconds=21600,
        profile_only=args.profile_only,
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
    )
    path = args.out / "request.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != request:
            raise ValueError("Fresh output or identical-source explicit resume required")
    else:
        if args.resume:
            raise ValueError("No checkpoint to resume")
        for source in sources:
            target = args.out / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        atomic_json(path, request)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    context, predictor, teacher = models(initial)
    parameters = list(context.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.05)
    raw = RawContext(manifest)
    step, previous_elapsed, diagnostics = 0, 0.0, None
    validation_step = -1
    pending_rng = None
    if args.resume:
        saved = torch.load(args.out / "latest.pt", map_location="cpu", weights_only=False)
        if saved["request_sha256"] != digest(path):
            raise ValueError("Pretraining recovery identity mismatch")
        context.backbone.load_state_dict(saved["context_encoder_state_dict"], strict=True)
        teacher.backbone.load_state_dict(saved["target_encoder_state_dict"], strict=True)
        predictor.load_state_dict(saved["predictor_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        restore_rng(saved["rng"], rng)
        step, previous_elapsed, diagnostics = (
            saved["step"],
            saved["elapsed_s"],
            saved["diagnostics"],
        )
        validation_step = saved["validation_step"]
    started = time.monotonic()

    def check():
        if (args.out / "STOP").exists() or previous_elapsed + time.monotonic() - started > 21600:
            raise InterruptedError("JEPA STOP/6h cap")
        if shutil.disk_usage(args.out).free < 25e9:
            raise InterruptedError("JEPA25GB disk reserve")

    def progress(status="running", phase="unlabeled training"):
        elapsed = previous_elapsed + time.monotonic() - started
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M4 TRAIN-only JEPA",
                steps=step,
                target_steps=10000,
                phase=phase,
                elapsed_s=elapsed,
                diagnostics=diagnostics,
                updated_at=time.time(),
            ),
        )

    def save(final=False):
        state = dict(
            context_encoder_state_dict=context.backbone.state_dict(),
            predictor_state_dict=predictor.state_dict(),
            target_encoder_state_dict=teacher.backbone.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=rng_state(rng),
            step=step,
            diagnostics=diagnostics,
            request_sha256=digest(path),
            elapsed_s=previous_elapsed + time.monotonic() - started,
            position_grids=dict(encoder_source=[4, 8, 8], actual=[4, 9, 16], predictor=[4, 9, 16]),
            validation_step=validation_step,
        )

        def byte_count(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(byte_count(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return sum(byte_count(v) for v in value)
            return 0

        needed = byte_count(state) + 10_000_000
        if final:
            needed += byte_count(state["context_encoder_state_dict"])
            needed += byte_count(state["predictor_state_dict"])
        if shutil.disk_usage(args.out).free < 25e9 + needed:
            atomic_json(
                args.out / "checkpoint_deferred.json",
                dict(
                    step=step,
                    reason="Atomic checkpoint would breach25GB reserve",
                    prior_checkpoint_preserved=True,
                ),
            )
            return False
        atomic_checkpoint(args.out / "latest.pt", state)
        if final:
            atomic_checkpoint(
                args.out / "final.pt",
                {
                    k: v
                    for k, v in state.items()
                    if k not in ("optimizer", "rng", "target_encoder_state_dict")
                },
            )
        return True

    @torch.inference_mode()
    def evaluate():
        nonlocal validation_step
        context.eval()
        predictor.eval()
        fixed = np.random.default_rng(20261019)
        clips, _ = sample_clips(raw, validation, fixed, 16)
        visible, masked = tube_masks(len(clips), fixed, "cuda")
        losses, tokens = [], []
        for start in range(0, len(clips), 6):
            check()
            video = normalized_video(clips[start : start + 6], "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = prediction_loss(
                    context,
                    teacher,
                    predictor,
                    video,
                    visible[start : start + 6],
                    masked[start : start + 6],
                )
                tokens.append(context(video).float().cpu())
            losses.append(float(loss))
        metrics = dict(loss=float(np.mean(losses)), **variance_diagnostics(torch.cat(tokens)))
        with (args.out / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=step, **metrics)) + "\n")
        print(json.dumps(dict(step=step, **metrics)), flush=True)
        validation_step = step
        return metrics

    status = "stopped"
    try:
        progress(phase="reconstructing supplement RGB in RAM")
        raw.preload_missing(check)
        if args.profile_only:
            times = []
            clips, _ = sample_clips(raw, train, rng)
            visible, masked = tube_masks(len(clips), rng, "cuda")
            for _ in range(3):
                check()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                before = time.monotonic()
                for start in range(0, len(clips), 6):
                    video = normalized_video(clips[start : start + 6], "cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, _ = prediction_loss(
                            context,
                            teacher,
                            predictor,
                            video,
                            visible[start : start + 6],
                            masked[start : start + 6],
                        )
                    (loss / 4).backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                torch.cuda.synchronize()
                times.append(time.monotonic() - before)
            atomic_json(
                args.out / "profile.json",
                dict(
                    seconds_per_24view_update=times,
                    peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
                    finite_gradients=True,
                    optimizer_updates=0,
                    estimated_10000_steady_seconds=float(np.mean(times[1:]) * 10000),
                ),
            )
            status = "complete"
            return
        if step == 0 and validation_step != 0:
            diagnostics = evaluate()
            atomic_json(args.out / "initial_diagnostics.json", diagnostics)
            if not save():
                raise InterruptedError("JEPA checkpoint disk reserve")
        elif step % 1000 == 0 and validation_step != step:
            diagnostics = evaluate()
        initial_diagnostics = json.loads((args.out / "initial_diagnostics.json").read_text())
        while step < 10000:
            check()
            context.train()
            predictor.train()
            optimizer.zero_grad(set_to_none=True)
            pending_rng = rng_state(rng)
            clips, samples = sample_clips(raw, train, rng)
            visible, masked = tube_masks(len(clips), rng, "cuda")
            loss_sum = 0.0
            for start in range(0, len(clips), 6):
                check()
                video = normalized_video(clips[start : start + 6], "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = prediction_loss(
                        context,
                        teacher,
                        predictor,
                        video,
                        visible[start : start + 6],
                        masked[start : start + 6],
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite JEPA loss")
                (loss / 4).backward()
                loss_sum += float(loss.detach()) / 4
            torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
            # Fixed warmup/cosine schedule, never selected by driving outcomes.
            scale = min(1.0, (step + 1) / 500) * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / 10000))
            )
            for group in optimizer.param_groups:
                group["lr"] = 1e-4 * scale
            optimizer.step()
            ema_update(context, teacher)
            step += 1
            pending_rng = None
            with (args.out / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=step, loss=loss_sum, clips=samples)) + "\n")
            if step % 1000 == 0:
                diagnostics = evaluate()
                for key in ("within_clip_variance", "between_scene_variance"):
                    if diagnostics[key] < max(1e-6, initial_diagnostics[key] * 0.01):
                        raise FloatingPointError(f"JEPA representation collapse: {key}")
            if step % 250 == 0:
                if not save():
                    raise InterruptedError("JEPA checkpoint disk reserve")
            if step % 10 == 0:
                progress()
        if not save(final=True):
            raise InterruptedError("JEPA export disk reserve")
        atomic_json(
            args.out / "encoder_descriptor.json",
            dict(
                kind="jepa_carla",
                checkpoint=str((args.out / "final.pt").resolve()),
                sha256=digest(args.out / "final.pt"),
                local_carla_labels=0,
                training_frames=request["train_frames"],
                updates=step,
                source_request_sha256=digest(path),
            ),
        )
        status = "complete"
    except InterruptedError as error:
        if pending_rng is not None:
            restore_rng(pending_rng, rng)
        print(str(error), flush=True)
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), step=step))
        raise
    finally:
        if not args.profile_only:
            save()
        progress(status, "finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "outputs/navigation_spatial_data_20261019"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    with RunLock(ROOT / "outputs/.navigation.lock"):
        main(parser.parse_args())
