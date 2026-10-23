#!/usr/bin/env python3
"""One protected continuous-data/whole-history camera countermeasure, no new rules."""

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

from scripts.train_navigation_student import (  # noqa: E402
    atomic_checkpoint,
    restore_rng,
    rng_state,
    validate,
)
from src.navigation.causal_learning import (  # noqa: E402
    EpisodeCache,
    original_training_episodes,
    prepare_approach_cache,
    stream_forward,
)
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.learning_data import ClipCache  # noqa: E402
from src.navigation.runtime import (  # noqa: E402
    RunLock,
    atomic_json,
    check_resources,
    resource_shortfall,
)
from src.navigation.student import make_student, student_loss  # noqa: E402

PARENT = ROOT / "outputs/navigation_m2_stop_go_20261016/best_policy.pt"
PARENT_SHA = "80bee4e338eec8f55f57d96ad4cf0bf27c736e47bc641e2b1dd4e8bb0cfdc3e4"
BASE = ROOT / "outputs/navigation_m2_cache_20k_20261015"
BASE_SHA = "b9e09499fa0c0158a62849369f44a2c942d90aa9c2dcdff1b7dffdd73e6710ae"


def score_stop_metrics(sums, counts):
    metrics = dict(
        speed_mae_mps=sums["speed"] / max(1, counts["speed"]),
        go_speed_mae_mps=sums["go"] / max(1, counts["go"]),
        controller_stop_recall=sums["held_stopped"] / max(1, counts["stopped"]),
        controller_false_hold_fraction=sums["held_go"] / max(1, counts["controller_go"]),
        counts=counts,
    )
    metrics["selection_score"] = (
        0.5 * (1 - metrics["controller_stop_recall"] + metrics["controller_false_hold_fraction"])
        + 0.02 * metrics["go_speed_mae_mps"]
    )
    return metrics


@torch.no_grad()
def causal_validation(model, cache, check):
    model.eval()
    sums = dict(speed=0.0, go=0.0, held_stopped=0, held_go=0)
    counts = dict(speed=0, go=0, stopped=0, controller_go=0)
    for episode in cache.manifest["episodes"]:
        if episode["role"] != "validation":
            continue
        check()
        inputs, targets = cache.episode(episode, next(model.parameters()).device)
        prediction = stream_forward(model, inputs, chunk=8)
        length = episode["end"] - episode["start"]
        mask = torch.zeros(length, dtype=torch.bool, device=inputs["rgb"].device)
        for start in range(0, length - 31, 32):
            mask[start + 16 : start + 32] = True
        if not mask.any():
            continue
        p = {k: v[:, mask].float() for k, v in prediction.items()}
        t = {k: v[:, mask] for k, v in targets.items()}
        error = (p["current_speed"] - t["current_speed"]).abs()
        go, stopped = t["current_speed"] > 1, t["current_speed"] < 0.1
        held = (p["current_speed"] < 0.15) | (p["brake"] >= 0)
        controller_go = go & (t["brake"] < 0.5)
        sums["speed"] += error.sum().item()
        sums["go"] += error[go].sum().item()
        sums["held_stopped"] += (held & stopped).sum().item()
        sums["held_go"] += (held & controller_go).sum().item()
        counts["speed"] += error.numel()
        counts["go"] += go.sum().item()
        counts["stopped"] += stopped.sum().item()
        counts["controller_go"] += controller_go.sum().item()
    return score_stop_metrics(sums, counts)


def passes(candidate, parent):
    a, b = candidate["causal"], parent["causal"]
    x, y = candidate["clipped"], parent["clipped"]
    return bool(
        a["counts"] == b["counts"]
        and a["controller_stop_recall"] >= b["controller_stop_recall"] - 1e-10
        and a["controller_false_hold_fraction"] <= b["controller_false_hold_fraction"] + 0.02
        and a["go_speed_mae_mps"] <= b["go_speed_mae_mps"] + 0.2
        and a["selection_score"] < b["selection_score"] - 1e-8
        and x["controller_stop_recall"] >= y["controller_stop_recall"] - 0.02
        and x["controller_false_hold_fraction"] <= y["controller_false_hold_fraction"] + 0.02
    )


def train(args):
    args.out = args.out.resolve()
    args.collection = args.collection.resolve()
    args.cache = args.cache.resolve()
    if not 0 < args.max_wall <= 3600 or not 0 < args.steps <= 1000:
        raise ValueError("Bounded1000-update/1h fit")
    if digest(PARENT) != PARENT_SHA or digest(BASE / "manifest.json") != BASE_SHA:
        raise ValueError("Protected parent/original split identity changed")
    position_audit = json.loads(args.positions_report.read_text())
    if (
        not position_audit["audit_passed"]
        or not position_audit["all_actual_20hz_positions_train"]
        or position_audit["collection_audit_sha256"] != digest(args.collection / "data_audit.json")
        or position_audit["guard_sha256"] != digest(args.collection / "guard.json")
    ):
        raise ValueError("Independent20Hz position audit differs from this collection")
    check_resources(args.out, 25)
    if args.device != "cpu":
        reason = resource_shortfall(dict(minimum_free_vram_gb=3, minimum_available_ram_gb=4))
        if reason:
            raise RuntimeError(reason)
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/train_navigation_student.py",
        ROOT / "src/navigation/causal_learning.py",
        ROOT / "src/navigation/student.py",
        ROOT / "src/navigation/learning_data.py",
        ROOT / "reports/navigation_m2_causal_memory_protocol_20261018.md",
    ]
    request = dict(
        parent=str(PARENT),
        parent_sha256=PARENT_SHA,
        base_manifest_sha256=BASE_SHA,
        collection=str(args.collection),
        collection_audit_sha256=digest(args.collection / "data_audit.json"),
        positions_report_sha256=digest(args.positions_report),
        cache=str(args.cache),
        steps=args.steps,
        seed=42,
        learning_rate=5e-5,
        maximum_total_wall_seconds=args.max_wall,
        training="Half original family-balanced episodes, half accepted varied-wait approaches",
        history="Episode start, full GRU gradient, visual checkpoint chunks8, burnin8",
        precision="BF16 training without scaler; FP32 causal and clipped scoring",
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
    )
    request_path = args.out / "request.json"
    if request_path.exists():
        if not args.resume or json.loads(request_path.read_text()) != request:
            raise ValueError(
                "Existing attempt/source mismatch; explicit identical-source resume required"
            )
    else:
        if args.resume:
            raise ValueError("No attempt to resume")
        for p in sources:
            target = args.out / "source" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        atomic_json(request_path, request)
    resume_state = (
        torch.load(args.out / "latest.pt", map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    prior_elapsed = resume_state.get("elapsed_total_s", 0) if resume_state else 0
    started = time.monotonic()
    deadline = started + max(0, args.max_wall - prior_elapsed)
    stop = args.out / "STOP"
    step = 0
    validation_step = -1
    best_step = 0
    best_score = float("inf")
    latest_validation = None

    def check():
        if stop.exists() or time.monotonic() >= deadline:
            raise InterruptedError("STOP or fit wall cap")

    def progress(phase, **values):
        atomic_json(
            args.out / "progress.json",
            dict(
                status="running",
                stage="M2 camera student",
                phase=phase,
                steps=step,
                target_steps=args.steps,
                selected_step=best_step,
                updated_at=time.time(),
                **values,
            ),
        )

    progress("preparing raw image cache")
    prepare_approach_cache(
        args.collection,
        args.cache,
        BASE,
        stop,
        deadline,
        lambda n, total: progress("preparing raw image cache", frames=n, target_frames=total),
    )
    original, extra = EpisodeCache(BASE), EpisodeCache(args.cache)
    guard = json.loads((args.collection / "guard.json").read_text())
    pool, excluded = original_training_episodes(original.manifest, guard["allowed_junctions"])
    families = sorted({e["family"] for e in pool})
    atomic_json(
        args.out / "training_domains.json",
        dict(
            original=[e["data"] for e in pool],
            excluded_original_component_locations=[e["data"] for e in excluded],
            supplement=[e["data"] for e in extra.manifest["episodes"]],
            validation_unchanged=BASE_SHA,
        ),
    )
    device = torch.device(args.device)
    torch.set_num_threads(2)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = make_student(parent["architecture"], parent["hidden"]).to(device)
    model.load_state_dict(parent["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    clipped_cache = ClipCache(BASE, length=32)
    baseline = None
    last = args.out / "latest.pt"
    if args.resume:
        saved = resume_state
        if saved["request_sha256"] != digest(request_path):
            raise ValueError("Resume request differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        step = saved["step"]
        validation_step = saved["validation_step"]
        best_step = saved["best_step"]
        best_score = saved["best_score"]
        latest_validation = saved["validation"]
        baseline = saved["baseline"]
        restore_rng(saved["rng"], rng)
    else:
        shutil.copyfile(PARENT, args.out / "best_policy.pt")

    def save():
        atomic_checkpoint(
            last,
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                architecture=parent["architecture"],
                hidden=parent["hidden"],
                step=step,
                validation_step=validation_step,
                best_step=best_step,
                best_score=best_score,
                validation=latest_validation,
                baseline=baseline,
                rng=rng_state(rng),
                request_sha256=digest(request_path),
                cache_manifest_sha256=digest(args.cache / "manifest.json"),
                elapsed_total_s=prior_elapsed + time.monotonic() - started,
            ),
        )

    def evaluate():
        nonlocal latest_validation, validation_step, best_score, best_step, baseline
        progress("causal and clipped validation")
        causal = causal_validation(model, original, check)
        clipped = validate(model, clipped_cache, device, batch_size=4)
        if causal["counts"]["speed"] != clipped["counts"]["speed"]:
            raise ValueError("Causal/clipped validation denominators differ")
        latest_validation = dict(causal=causal, clipped=clipped)
        validation_step = step
        if baseline is None:
            baseline = latest_validation
            best_score = causal["selection_score"]
            atomic_json(args.out / "baseline.json", baseline)
        accepted = (
            step > 0
            and passes(latest_validation, baseline)
            and causal["selection_score"] < best_score
        )
        if accepted:
            best_score = causal["selection_score"]
            best_step = step
            atomic_checkpoint(
                args.out / "best_policy.pt",
                dict(
                    model=model.state_dict(),
                    architecture=parent["architecture"],
                    hidden=parent["hidden"],
                    step=step,
                ),
            )
        with (args.out / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=step, **latest_validation, accepted=accepted)) + "\n")
        save()
        atomic_json(
            args.out / "review.json",
            dict(
                selected_step=best_step,
                candidate_improved=best_step > 0,
                best_score=best_score,
                parent=baseline,
                latest=latest_validation,
                policy_sha256=digest(args.out / "best_policy.pt"),
                driving_evidence=False,
            ),
        )
        print(
            json.dumps(dict(step=step, selected=best_step, causal=causal, accepted=accepted)),
            flush=True,
        )

    status = "stopped"
    try:
        if validation_step != step:
            evaluate()
        training_started = time.monotonic()
        initial_step = step
        while step < args.steps:
            check()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            supplement = bool(step % 2)
            if supplement:
                selected = extra.manifest["episodes"]
                e = selected[int(rng.integers(len(selected)))]
                inputs, targets = extra.episode(e, device)
            else:
                family = families[int(rng.integers(len(families)))]
                selected = [e for e in pool if e["family"] == family]
                e = selected[int(rng.integers(len(selected)))]
                inputs, targets = original.episode(e, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                predicted = stream_forward(model, inputs, chunk=8, gradients=True)
                loss, parts = student_loss(
                    predicted, targets, burn_in=8, perception_supervision=not supplement
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite causal-student loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            with (args.out / "train.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            step=step,
                            loss=float(loss),
                            domain="approach" if supplement else "original",
                            route_id=e["route_id"],
                            frames=e["end"] - e["start"],
                        )
                    )
                    + "\n"
                )
            if step % 10 == 0:
                progress(
                    "training",
                    train_loss=float(loss),
                    steps_per_second=(step - initial_step)
                    / max(1e-5, time.monotonic() - training_started),
                    validation=latest_validation["causal"],
                )
            if step % 50 == 0:
                save()
            if step % 250 == 0 or step == args.steps:
                evaluate()
        status = "complete"
    except InterruptedError as error:
        print(str(error), flush=True)
    finally:
        save()
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M2 camera student",
                phase="causal history fit",
                steps=step,
                target_steps=args.steps,
                selected_step=best_step,
                validation=latest_validation["causal"] if latest_validation else None,
                updated_at=time.time(),
            ),
        )
    print(json.dumps(dict(status=status, step=step, selected_step=best_step)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--collection", type=Path, default=ROOT / "outputs/navigation_m2_varied_waits_20261018"
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "outputs/navigation_m2_causal_approach_cache_20261018"
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--positions-report",
        type=Path,
        default=ROOT / "reports/navigation_m2_varied_wait_positions_20261018.json",
    )
    parser.add_argument("--max-wall", type=float, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with RunLock(ROOT / "outputs/.navigation.lock"):
        try:
            train(args)
        except BaseException as error:
            if args.out.exists():
                atomic_json(args.out / "failure.json", dict(error=repr(error)))
                path = args.out / "progress.json"
                p = json.loads(path.read_text()) if path.exists() else {}
                atomic_json(
                    path,
                    dict(
                        p,
                        status="stopped" if isinstance(error, InterruptedError) else "failed",
                        error=repr(error),
                        updated_at=time.time(),
                    ),
                )
            raise


if __name__ == "__main__":
    main()
