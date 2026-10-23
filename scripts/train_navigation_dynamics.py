#!/usr/bin/env python3
"""Source-frozen factual dynamics members; offline evidence cannot qualify a planner."""

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

from scripts.train_navigation_student import atomic_checkpoint, restore_rng, rng_state  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dynamics_learning import (  # noqa:E402
    member_population,
    sample_batch,
    tensor_batch,
)
from src.navigation.ordered_dynamics import OrderedJEPADynamics  # noqa:E402
from src.navigation.rssm_dynamics import RSSMDynamics, prediction_objective  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402


@torch.inference_mode()
def validate(model, windows, features, check=lambda: None):
    model.eval()
    ids = np.flatnonzero(windows["is_validation"])
    counts = np.zeros((8, 3), np.int64)
    error = np.zeros((8, 3), float)
    bce = np.zeros(8, float)
    positive = np.zeros(8, np.int64)
    missed = np.zeros(8, np.int64)
    latent_sum, latent_count = 0.0, 0
    progress_objective_sum = 0.0
    calibration = np.zeros((10, 3), float)
    for start in range(0, len(ids), 16):
        check()
        inputs, targets = tensor_batch(windows, features, ids[start : start + 16], "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(**inputs)
        logits = prediction["outcome_logits"].float()
        prob = logits[..., :2].sigmoid()
        mask = targets["cost_valid"]
        outcomes = torch.where(mask > 0, targets["outcomes"], 0)
        valid = targets["latent_valid"]
        token_error = torch.nn.functional.smooth_l1_loss(
            prediction["future_tokens"].float(),
            torch.where(valid[..., None, None] > 0, targets["future_tokens"], 0),
            reduction="none",
        ).mean((-1, -2))
        latent_sum += float((token_error * valid).sum())
        latent_count += int(valid.sum())
        collision_error = (prob[..., 0] - outcomes[..., 0]).square()
        progress_error = (logits[..., 2] - outcomes[..., 2]).abs()
        ce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[..., 0], outcomes[..., 0], reduction="none"
        )
        counts += mask.sum(0).long().cpu().numpy()
        progress_objective_sum += float(
            (
                torch.nn.functional.smooth_l1_loss(
                    logits[..., 2] / 20, outcomes[..., 2] / 20, reduction="none"
                )
                * mask[..., 2]
            ).sum()
        )
        error[:, 0] += (collision_error * mask[..., 0]).sum(0).cpu().numpy()
        error[:, 2] += (progress_error * mask[..., 2]).sum(0).cpu().numpy()
        bce += (ce * mask[..., 0]).sum(0).cpu().numpy()
        positive += ((outcomes[..., 0] == 1) * mask[..., 0]).sum(0).long().cpu().numpy()
        missed += (
            ((outcomes[..., 0] == 1) * (prob[..., 0] < 0.5) * mask[..., 0])
            .sum(0)
            .long()
            .cpu()
            .numpy()
        )
        pp = prob[..., 0][mask[..., 0] > 0].cpu().numpy()
        yy = outcomes[..., 0][mask[..., 0] > 0].cpu().numpy()
        for bin_id in range(10):
            chosen = np.minimum((pp * 10).astype(int), 9) == bin_id
            calibration[bin_id] += [chosen.sum(), pp[chosen].sum(), yy[chosen].sum()]
    latent = latent_sum / max(1, latent_count)
    collision_bce = float(bce.sum() / max(1, counts[:, 0].sum()))
    objective = latent + collision_bce + progress_objective_sum / max(1, counts[:, 2].sum())
    return dict(
        selection_objective=objective,
        latent_smooth_l1=latent,
        collision_bce=collision_bce,
        collision_brier_by_horizon=(error[:, 0] / np.maximum(1, counts[:, 0])).tolist(),
        progress_mae_m_by_horizon=(error[:, 2] / np.maximum(1, counts[:, 2])).tolist(),
        collision_labeled_by_horizon=counts[:, 0].tolist(),
        positive_by_horizon=positive.tolist(),
        missed_positive_at_half_by_horizon=missed.tolist(),
        calibration_bins=[
            dict(
                count=int(n),
                mean_predicted=float(p / max(1, n)),
                fraction_positive=float(y / max(1, n)),
            )
            for n, p, y in calibration
        ],
        validation_windows=len(ids),
        red_risk_qualified=False,
        planner_qualified=False,
    )


def train(args):
    if args.seed not in (42, 1051, 2060):
        raise ValueError("Registered3independent seeds required")
    check_resources(args.out, 25)
    data = json.loads((args.data / "manifest.json").read_text())
    if (data["windows"], data["train_windows"], data["validation_windows"]) != (3800, 2845, 955):
        raise ValueError("Registered corrected factual corpus required")
    if digest(args.data / "windows.npz") != data["window_arrays_sha256"]:
        raise ValueError("Factual windows changed")
    cache = json.loads((args.features / "manifest.json").read_text())
    if (
        cache["data_manifest_sha256"] != digest(args.data / "manifest.json")
        or digest(args.features / "features.npy") != cache["features_sha256"]
    ):
        raise ValueError("Frozen factual encoder identity mismatch")
    windows = dict(np.load(args.data / "windows.npz", allow_pickle=False))
    features = np.load(args.features / "features.npy", mmap_mode="r", allow_pickle=False)
    groups, positives, chosen = member_population(windows, args.seed)
    sources = [
        "scripts/train_navigation_dynamics.py",
        "src/navigation/dynamics_learning.py",
        "src/navigation/ordered_dynamics.py",
        "src/navigation/rssm_dynamics.py",
        "scripts/train_navigation_student.py",
        "reports/navigation_dynamics_fit_protocol_20261020.md",
    ]
    request = dict(
        model=args.model,
        seed=args.seed,
        steps=2000,
        batch=16,
        hidden=128,
        spatial_tokens=48,
        optimizer="AdamW lr3e-4 wd.01",
        maximum_wall_seconds=7200,
        source_hashes={p: digest(ROOT / p) for p in sources},
        window_manifest_sha256=digest(args.data / "manifest.json"),
        encoder_cache_manifest_sha256=digest(args.features / "manifest.json"),
        bootstrap_episodes=chosen.tolist(),
        positive_pool_windows=len(positives),
        red_training_masked=True,
        not_planning_qualification=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    request_path = args.out / "request.json"
    if request_path.exists():
        if not args.resume or json.loads(request_path.read_text()) != request:
            raise ValueError("Only identical-source explicit member resume is allowed")
    else:
        if args.resume:
            raise ValueError("No member checkpoint to resume")
        atomic_json(request_path, request)
        for p in sources:
            target = args.out / "source" / p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / p, target)
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 17)
    model = (OrderedJEPADynamics if args.model == "jepa" else RSSMDynamics)(
        spatial_tokens=48
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    step, selected, validation_step, best, elapsed_before = 0, 0, -1, float("inf"), 0.0
    metrics = None
    if args.resume:
        saved = torch.load(args.out / "latest.pt", map_location="cpu", weights_only=False)
        if saved["request_sha256"] != digest(request_path):
            raise ValueError("Checkpoint identity changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        restore_rng(saved["rng"], rng)
        step, selected, validation_step, best, elapsed_before = [
            saved[k] for k in ("step", "selected_step", "validation_step", "best", "elapsed_s")
        ]
        metrics = saved["validation"]
    started = time.monotonic()

    def check():
        if (args.out / "STOP").exists() or elapsed_before + time.monotonic() - started > 7200:
            raise InterruptedError("Dynamics member STOP/2h cap")
        if shutil.disk_usage(args.out).free < 25e9 + 20e6:
            raise InterruptedError("Dynamics member25GB atomic-checkpoint reserve")

    def progress(status="running", phase="factual prediction training"):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M5 factual dynamics member",
                steps=step,
                target_steps=2000,
                model=args.model,
                seed=args.seed,
                phase=phase,
                selected_step=selected,
                validation_step=validation_step,
                validation=metrics,
                elapsed_s=elapsed_before + time.monotonic() - started,
                updated_at=time.time(),
            ),
        )

    def payload():
        return dict(
            model=model.state_dict(),
            architecture="ordered-jepa" if args.model == "jepa" else "rssm",
            spatial_tokens=48,
            hidden=128,
            step=step,
            request_sha256=digest(request_path),
            red_risk_qualified=False,
            planner_qualified=False,
        )

    def save():
        atomic_checkpoint(
            args.out / "latest.pt",
            dict(
                payload(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(rng),
                selected_step=selected,
                validation_step=validation_step,
                best=best,
                validation=metrics,
                elapsed_s=elapsed_before + time.monotonic() - started,
            ),
        )

    def evaluate():
        nonlocal metrics, selected, best, validation_step
        progress(phase="all955whole-location validation windows")
        metrics = validate(model, windows, features, check)
        validation_step = step
        if metrics["selection_objective"] < best:
            selected, best = step, metrics["selection_objective"]
            atomic_checkpoint(args.out / "best_model.pt", payload())
        with (args.out / "validation.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=step, metrics=metrics, selected_step=selected)) + "\n")
        save()
        print(json.dumps(dict(step=step, selected=selected, validation=metrics)), flush=True)

    status = "stopped"
    try:
        if validation_step != step and step % 500 == 0:
            evaluate()
        while step < 2000:
            check()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            ids = sample_batch(groups, positives, rng)
            inputs, targets = tensor_batch(windows, features, ids, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = (
                    model(**inputs, sample=True) if args.model == "rssm" else model(**inputs)
                )
                loss, parts = prediction_objective(prediction, **targets)
                if args.model == "rssm":
                    kl = model.posterior_kl(
                        prediction, targets["future_tokens"], targets["latent_valid"]
                    )
                    loss = loss + 0.01 * kl
                    parts["kl"] = kl
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite factual dynamics loss")
            loss.backward()
            # An unobserved red class cannot acquire weight-decay-only updates.
            with torch.no_grad():
                model.cost.weight.grad[1].zero_()
                model.cost.bias.grad[1].zero_()
                red_weight = model.cost.weight[1].clone()
                red_bias = model.cost.bias[1].clone()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                model.cost.weight[1].copy_(red_weight)
                model.cost.bias[1].copy_(red_bias)
            step += 1
            with (args.out / "train.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            step=step,
                            window_indices=ids.tolist(),
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
        progress(status, "finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("jepa", "rssm"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "outputs/navigation_dynamics_windows_20261019"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    with RunLock(ROOT / "outputs/.navigation.lock"):
        train(parser.parse_args())
