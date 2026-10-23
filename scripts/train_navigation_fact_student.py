#!/usr/bin/env python3
"""Prospective paired cold-start/fact-supervision fits on corrected full-label data."""

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
from scripts.train_navigation_visual_student import (  # noqa:E402
    initialize,
    inputs_for,
    selected_pool,
    validate,
)
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.fact_student import FactStudent, fact_loss  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.spatial_data import SpatialCorpus, validation_positions  # noqa:E402
from src.navigation.spatial_student import driving_loss  # noqa:E402

FACTS = ("light", "presence", "distance", "hazards")


class FactCorpus(SpatialCorpus):
    def __init__(self, data, labels, enabled):
        super().__init__(data)
        self.enabled = enabled
        request = json.loads((labels / "request.json").read_text())
        manifest = json.loads((labels / "manifest.json").read_text())
        if request["corrected_data_manifest_sha256"] != digest(data / "manifest.json") or manifest[
            "request_sha256"
        ] != digest(labels / "request.json"):
            raise ValueError("Fact labels do not align with corrected driving data")
        if any(digest(labels / p) != h for p, h in manifest["arrays_sha256"].items()):
            raise ValueError("Separate semantic targets changed")
        self.facts = {
            name: np.load(labels / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in FACTS
        }
        if any(len(array) != self.manifest["samples"] for array in self.facts.values()):
            raise ValueError("One semantic target per corrected timestamp required")

    def episode(self, episode, device, features=None):
        inputs, targets = super().episode(episode, device, features)
        if self.enabled:
            targets.update(
                {
                    name: torch.from_numpy(
                        np.asarray(array[episode["start"] : episode["end"]]).copy()
                    ).to(device)[None]
                    for name, array in self.facts.items()
                }
            )
        return inputs, targets


def initialize_candidate(encoder, dimension, seed, facts):
    plain = initialize(encoder, dimension, seed, color_enabled=True)
    if not facts:
        return plain
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FactStudent(dimension, scratch=encoder == "cnn_scratch")
    missing, unexpected = model.load_state_dict(plain.state_dict(), strict=False)
    if unexpected or not missing or any(not key.startswith("facts.") for key in missing):
        raise ValueError("Fact arm must inherit only identical fresh paired driving initialization")
    return model


@torch.inference_mode()
def validate_candidate(model, corpus, device, features, check=lambda: None):
    metrics = validate(model, corpus, device, features, check)
    if not corpus.enabled:
        return metrics
    sums = dict(
        light_correct=0,
        light_count=0,
        presence_correct=0,
        presence_count=0,
        distance_error=0.0,
        distance_count=0,
        hazard_correct=0,
        hazard_count=0,
    )
    for episode in corpus.manifest["episodes"]:
        if episode["role"] != "validation":
            continue
        check()
        inputs, target = inputs_for(model, corpus, episode, device, features)
        prediction, _ = model(**inputs)
        mask = validation_positions(episode["end"] - episode["start"], device)
        p = {k: v[:, mask].float() for k, v in prediction.items()}
        t = {k: v[:, mask] for k, v in target.items()}
        known = t["light"] != 4
        sums["light_correct"] += int(((p["light"].argmax(-1) == t["light"]) & known).sum())
        sums["light_count"] += int(known.sum())
        for name, prefix in (("presence", "presence"), ("hazards", "hazard")):
            sums[prefix + "_correct"] += int(((p[name] >= 0) == (t[name] >= 0.5)).sum())
            sums[prefix + "_count"] += t[name].numel()
        sums["distance_error"] += float(
            ((p["distance"] - t["distance"]).abs() * t["presence"]).sum()
        )
        sums["distance_count"] += int(t["presence"].sum())
    metrics["auxiliary_facts"] = dict(
        sums,
        distances_mae_m=sums["distance_error"] / max(1, sums["distance_count"]),
        visibility_certified=False,
        participates_in_checkpoint_selection=False,
    )
    return metrics


def train(args):
    if args.steps != 2000 or not 0 < args.max_wall <= 3600:
        raise ValueError("Registered2000-update/1h downstream budget required")
    corpus = FactCorpus(args.data, args.fact_labels, args.facts)
    if (
        corpus.manifest.get("schema") != "navigation-spatial-study-geography-repair-v3"
        or corpus.manifest["samples"] != 18534
        or args.fraction != 1.0
        or args.seed != 42
    ):
        raise ValueError(
            "Only the prospective corrected full-label/seed42 comparison is registered"
        )
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
        ROOT / "reports/navigation_visual_decision_protocol_20261019.md",
        ROOT / "src/navigation/visual_decision_student.py",
        ROOT / "src/navigation/fact_student.py",
        ROOT / "scripts/train_navigation_visual_student.py",
        ROOT / "reports/navigation_fact_student_protocol_20261020.md",
    ]
    request = dict(
        encoder=args.encoder,
        neural_ego_speed_enabled=False,
        fact_supervision=args.facts,
        burn_in=0,
        corrected_fact_manifest_sha256=digest(args.fact_labels / "manifest.json"),
        additional_semantic_label_timestamps=15712 if args.facts else 0,
        comparison="Full-label engineering candidates; separate from M4 label-efficiency matrix",
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
    model = initialize_candidate(args.encoder, dimension, args.seed, args.facts).cuda()
    torch.manual_seed(args.seed)  # Equal training RNG after optional head construction.
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
                stage="M2 cold-start and fact supervision",
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
            architecture="spatial-visual-facts-v3" if args.facts else "spatial-visual-decision-v2",
            fact_supervision=args.facts,
            burn_in=0,
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
        latest = validate_candidate(model, corpus, "cuda", features, check)
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
                loss, parts = (fact_loss if args.facts else driving_loss)(p, t, burn_in=0)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("cnn_scratch", "imagenet_resnet18"), required=True)
    parser.add_argument("--facts", action="store_true")
    parser.add_argument(
        "--data", type=Path, default=ROOT / "outputs/navigation_spatial_data_repaired_20261019"
    )
    parser.add_argument(
        "--fact-labels", type=Path, default=ROOT / "outputs/navigation_fact_labels_20261020"
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.steps, args.max_wall, args.fraction, args.seed, args.no_color = (
        2000,
        3600.0,
        1.0,
        42,
        False,
    )
    with RunLock(ROOT / "outputs/.navigation.lock"):
        train(args)
