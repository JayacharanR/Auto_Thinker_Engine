#!/usr/bin/env python3
"""Matched RGB-fact/null-token stop residuals; original driver weights immutable."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cache_navigation_signal_policy import BUNDLE, PROTOCOL  # noqa: E402
from scripts.train_navigation_causal_student import (  # noqa: E402
    PARENT,
    PARENT_SHA,
    score_stop_metrics,
)
from scripts.train_navigation_student import atomic_checkpoint, restore_rng, rng_state  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.signal_policy import StopResidual  # noqa: E402


class FactCache:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.arrays = {
            p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in self.root.glob("*.npy")
        }

    def episode(self, e, arm, device):
        a = {
            k: torch.from_numpy(np.asarray(v[e["start"] : e["end"]]).copy()).to(device)[None]
            for k, v in self.arrays.items()
        }
        if arm == "null":
            a["signal_facts"].zero_()
            a["signal_valid"].zero_()
            a["signal_valid"][..., 0] = True
        return a


def predict(model, a):
    return model(a["recurrent"], a["navigation"], a["ego"], a["signal_facts"], a["signal_valid"])[0]


def stop_loss(residual, a, original):
    stop = a["stop_logits"] + residual
    speed = torch.cat([a["current_speed"].unsqueeze(-1), a["speeds"]], -1)
    target = (speed < 0.1).float()
    valid = torch.cat([torch.ones_like(a["current_speed"].unsqueeze(-1)), a["speed_mask"]], -1)
    error = F.binary_cross_entropy_with_logits(stop[:, 8:], target[:, 8:], reduction="none")
    mask = valid[:, 8:, 1:]
    loss = 2 * error[..., 0].mean() + (error[..., 1:] * mask).sum() / mask.sum().clamp_min(1)
    if original:
        # Soft parent targets constrain changes without a hardcoded signal rule.
        distill = F.binary_cross_entropy_with_logits(
            stop[:, 8:], a["stop_logits"][:, 8:].sigmoid(), reduction="none"
        )
        loss = loss + 0.1 * (distill * valid[:, 8:]).sum() / valid[:, 8:].sum().clamp_min(1)
    return loss


def validation_mask(length, device):
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    for start in range(0, length - 31, 32):
        mask[start + 16 : start + 32] = True
    return mask


@torch.inference_mode()
def validate_residual(model, cache, arm, device, check=lambda: None):
    model.eval()
    sums = dict(speed=0.0, go=0.0, held_stopped=0, held_go=0)
    counts = dict(speed=0, go=0, stopped=0, controller_go=0)
    families = {}
    for e in cache.manifest["episodes"]:
        if e["role"] != "validation":
            continue
        check()
        a = cache.episode(e, arm, device)
        stop = a["stop_logits"] + predict(model, a)
        speed = torch.where(
            stop[..., 0] >= 0, torch.zeros_like(stop[..., 0]), a["cruise_speeds"][..., 0]
        )
        mask = validation_mask(speed.shape[1], device)
        p, truth = speed[:, mask], a["current_speed"][:, mask]
        held = (p < 0.15) | (stop[:, mask, 0] >= 0)
        stopped, go = truth < 0.1, truth > 1
        controller_go = go & (a["brake"][:, mask] < 0.5)
        error = (p - truth).abs()
        sums["speed"] += error.sum().item()
        sums["go"] += error[go].sum().item()
        sums["held_stopped"] += (held & stopped).sum().item()
        sums["held_go"] += (held & controller_go).sum().item()
        counts["speed"] += error.numel()
        counts["go"] += go.sum().item()
        counts["stopped"] += stopped.sum().item()
        counts["controller_go"] += controller_go.sum().item()
        f = families.setdefault(
            e["family"], dict(stopped=0, stop_hits=0, controller_go=0, false_holds=0)
        )
        for k, value in dict(
            stopped=stopped.sum(),
            stop_hits=(held & stopped).sum(),
            controller_go=controller_go.sum(),
            false_holds=(held & controller_go).sum(),
        ).items():
            f[k] += int(value)
    return dict(score_stop_metrics(sums, counts), families=families)


def passes(candidate, parent):
    if candidate["counts"] != parent["counts"] or set(candidate["families"]) != set(
        parent["families"]
    ):
        return False
    for key, old in parent["families"].items():
        new = candidate["families"][key]
        if new["stopped"] != old["stopped"] or new["controller_go"] != old["controller_go"]:
            return False
        if (
            old["stopped"]
            and new["stop_hits"] / new["stopped"] < old["stop_hits"] / old["stopped"] - 0.02
        ):
            return False
    return bool(
        candidate["controller_stop_recall"] >= parent["controller_stop_recall"]
        and candidate["controller_false_hold_fraction"]
        <= parent["controller_false_hold_fraction"] + 0.02
        and candidate["go_speed_mae_mps"] <= parent["go_speed_mae_mps"] + 0.2
        and candidate["selection_score"] < parent["selection_score"] - 1e-8
    )


def train(args):
    if digest(PARENT) != PARENT_SHA:
        raise ValueError("Parent identity changed")
    manifest = json.loads((args.cache / "manifest.json").read_text())
    cache_request = json.loads((args.cache / "request.json").read_text())
    if digest(args.cache / "request.json") != manifest["request_sha256"] or cache_request[
        "protocol_sha256"
    ] != digest(PROTOCOL):
        raise ValueError("Cache/protocol provenance changed")
    if any(
        digest(args.cache / "source" / p) != h for p, h in cache_request["source_hashes"].items()
    ):
        raise ValueError("Archived cache source changed")
    for name, data in manifest["domains"].items():
        root = args.cache / name
        if digest(root / "manifest.json") != data["manifest_sha256"] or any(
            digest(root / p) != h for p, h in data["array_sha256"].items()
        ):
            raise ValueError("Cached feature/label arrays changed")
    check_resources(args.out, 25)
    sources = [
        Path(__file__),
        PROTOCOL,
        ROOT / "scripts/cache_navigation_signal_policy.py",
        ROOT / "scripts/train_navigation_causal_student.py",
        ROOT / "scripts/train_navigation_student.py",
        ROOT / "src/navigation/signal_policy.py",
        ROOT / "src/navigation/student.py",
        ROOT / "src/navigation/policy_runtime.py",
        ROOT / "src/navigation/signal_perception.py",
    ]
    request = dict(
        arm=args.arm,
        parent_sha256=PARENT_SHA,
        cache=str(args.cache.resolve()),
        cache_manifest_sha256=digest(args.cache / "manifest.json"),
        bundle=str(BUNDLE),
        bundle_sha256=digest(BUNDLE),
        steps=1000,
        seed=42,
        learning_rate=0.001,
        maximum_wall_seconds=1800,
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
    )
    request_path = args.out / "request.json"
    if request_path.exists():
        if not args.resume or json.loads(request_path.read_text()) != request:
            raise ValueError("Explicit identical-source resume required")
    else:
        if args.resume:
            raise ValueError("No attempt to resume")
        for p in sources:
            target = args.out / "source" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        atomic_json(request_path, request)
        shutil.copyfile(PARENT, args.out / "best_policy.pt")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    model = StopResidual().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    original, approach = FactCache(args.cache / "original"), FactCache(args.cache / "approach")
    pool = [e for e in original.manifest["episodes"] if e["role"] == "train"]
    families = sorted({e["family"] for e in pool})
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    step = selected = 0
    validation_step = -1
    best_score = float("inf")
    baseline = latest = None
    previous_elapsed = 0
    if args.resume:
        saved = torch.load(args.out / "latest.pt", map_location="cuda", weights_only=False)
        if saved["request_sha256"] != digest(request_path):
            raise ValueError("Resume request mismatch")
        model.load_state_dict(saved["residual"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], rng)
        step, selected, validation_step = (
            saved["step"],
            saved["selected_step"],
            saved["validation_step"],
        )
        baseline, latest, best_score = saved["baseline"], saved["validation"], saved["best_score"]
        previous_elapsed = saved["elapsed_total_s"]
    started = time.monotonic()

    def check():
        if (args.out / "STOP").exists() or previous_elapsed + time.monotonic() - started >= 1800:
            raise InterruptedError("Residual STOP/wall cap")

    def progress(status="running", phase="training"):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M2 stop residual",
                arm=args.arm,
                steps=step,
                target_steps=1000,
                selected_step=selected,
                phase=phase,
                validation=latest,
                updated_at=time.time(),
            ),
        )

    def save():
        atomic_checkpoint(
            args.out / "latest.pt",
            dict(
                residual=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(rng),
                step=step,
                selected_step=selected,
                validation_step=validation_step,
                baseline=baseline,
                validation=latest,
                best_score=best_score,
                request_sha256=digest(request_path),
                elapsed_total_s=previous_elapsed + time.monotonic() - started,
            ),
        )

    def evaluate():
        nonlocal latest, baseline, validation_step, selected, best_score
        progress(phase="validation")
        latest = validate_residual(model, original, args.arm, "cuda", check)
        if latest["counts"] != dict(speed=1312, go=724, stopped=560, controller_go=582):
            raise ValueError("Frozen validation denominator changed")
        validation_step = step
        if baseline is None:
            baseline = latest
            best_score = latest["selection_score"]
            old = json.loads(
                (ROOT / "outputs/navigation_m2_causal_history_20261018/baseline.json").read_text()
            )["causal"]
            for key in (
                "controller_stop_recall",
                "controller_false_hold_fraction",
                "go_speed_mae_mps",
                "selection_score",
            ):
                if abs(old[key] - baseline[key]) > 1e-5:
                    raise ValueError("Zero-residual cache baseline differs from original parent")
            atomic_json(args.out / "baseline.json", baseline)
        accepted = step > 0 and passes(latest, baseline) and latest["selection_score"] < best_score
        if accepted:
            selected, best_score = step, latest["selection_score"]
            state = dict(parent["model"])
            state.update({"residual." + k: v.detach().cpu() for k, v in model.state_dict().items()})
            atomic_checkpoint(
                args.out / "best_policy.pt",
                dict(
                    model=state,
                    architecture="camera-signal-residual-v3",
                    hidden=192,
                    step=step,
                    fact_arm=args.arm,
                    signal_bundle=str(BUNDLE),
                    signal_bundle_sha256=digest(BUNDLE),
                    parent_sha256=PARENT_SHA,
                    request_sha256=digest(request_path),
                ),
            )
        with (args.out / "validation.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=step, metrics=latest, accepted=accepted)) + "\n")
        save()
        atomic_json(
            args.out / "review.json",
            dict(
                selected_step=selected,
                candidate_improved=selected > 0,
                latest=latest,
                parent=baseline,
                best_score=best_score,
                policy_sha256=digest(args.out / "best_policy.pt"),
                driving_evidence=False,
            ),
        )
        print(
            json.dumps(dict(step=step, selected=selected, accepted=accepted, metrics=latest)),
            flush=True,
        )

    status = "stopped"
    try:
        if validation_step != step:
            evaluate()
        while step < 1000:
            check()
            if step % 2:
                domain, cache = "approach", approach
                entries = approach.manifest["episodes"]
            else:
                domain, cache = "original", original
                family = families[int(rng.integers(len(families)))]
                entries = [e for e in pool if e["family"] == family]
            e = entries[int(rng.integers(len(entries)))]
            a = cache.episode(e, args.arm, "cuda")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = stop_loss(predict(model, a), a, domain == "original")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite stop-residual loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            with (args.out / "train.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            step=step,
                            domain=domain,
                            route_id=e["route_id"],
                            frames=e["end"] - e["start"],
                            loss=float(loss.detach()),
                        )
                    )
                    + "\n"
                )
            if step % 250 == 0:
                evaluate()
            elif step % 25 == 0:
                save()
            if step % 10 == 0:
                progress()
        status = "complete"
    except InterruptedError as error:
        print(str(error), flush=True)
    finally:
        save()
        progress(status=status, phase="finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("rgb", "null"), required=True)
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "outputs/navigation_m2_signal_policy_cache_20261018"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    with RunLock(ROOT / "outputs/.navigation.lock"):
        train(parser.parse_args())
