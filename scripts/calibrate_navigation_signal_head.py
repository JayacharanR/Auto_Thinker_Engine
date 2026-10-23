#!/usr/bin/env python3
"""One TRAIN-derived foreground-bias rule; no validation threshold fitting."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.assess_navigation_signal_detector import audit_predictions  # noqa: E402
from scripts.train_navigation_signal_detector import evaluate  # noqa: E402
from scripts.train_navigation_student import atomic_checkpoint  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.signal_detection_data import SignalDetectionData  # noqa: E402
from src.navigation.signal_detector import make_signal_detector  # noqa: E402


def train_recall_bias(parent_margin, candidate_margin):
    parent, candidate = np.asarray(parent_margin), np.asarray(candidate_margin)
    if (
        parent.shape != candidate.shape
        or parent.ndim != 1
        or not np.isfinite(parent).all()
        or not np.isfinite(candidate).all()
    ):
        raise ValueError("Paired finite TRAIN margins required")
    target = int((parent > 0).sum())
    if int((candidate > 0).sum()) >= target:
        return 0.0
    cutoff = np.sort(candidate)[::-1][target - 1]
    return max(0.0, float(-cutoff + 1e-6))


@torch.no_grad()
def calibrate(args):
    if args.out.exists():
        raise FileExistsError("Fresh calibration output required")
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20261016)
    rng = np.random.default_rng(20261016)
    parent_path = args.parent / "best_component.pt"
    candidate_path = args.candidate / "latest.pt"
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if parent["step"] != 50 or candidate["step"] != 200 or candidate["validation_step"] != 200:
        raise ValueError("Fixed parent50 and completed candidate200 required")
    prefix = "roi_heads.box_predictor.cls_score."
    if not all(
        torch.equal(t, parent["model"][name])
        for name, t in candidate["model"].items()
        if not name.startswith(prefix)
    ):
        raise ValueError("Candidate changed frozen ROI/proposal features")
    data = SignalDetectionData(args.corpus)
    data.attach_hard_negatives(args.negatives, digest(parent_path), args.release)
    check_resources(args.out, 25)
    sources = [
        Path(__file__).resolve(),
        ROOT / "src/navigation/signal_detector.py",
        ROOT / "scripts/train_navigation_signal_detector.py",
        ROOT / "src/navigation/signal_detection_data.py",
        ROOT / "reports/navigation_m2_signal_train_calibration_protocol_20261016.md",
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            kind="Single TRAIN-derived bias diagnostic, not driving",
            parent_sha256=digest(parent_path),
            candidate_sha256=digest(candidate_path),
            release_sha256=digest(args.release),
            maximum_wall_seconds=300,
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        ),
    )
    model = make_signal_detector()
    model.load_state_dict(candidate["model"], strict=True)
    model = model.eval().to("cuda")

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started >= 300:
            raise InterruptedError("STOP or diagnostic cap")

    indices = [
        i
        for i in data.base.train
        if len(data.targets[i]["boxes"]) or len(data.targets[i]["hard_negative_boxes"])
    ]
    features, labels = [], []
    for offset in range(0, len(indices), 3):
        check()
        batch = indices[offset : offset + 3]
        assert all(data.base.samples[i]["role"] == "train" for i in batch)
        images, targets = data.batch(batch, "cuda")
        images, targets = model.transform(images, targets)
        encoded = model.backbone(images.tensors)
        boxes = [torch.cat([t["boxes"], t["hard_negative_boxes"]]) for t in targets]
        pooled = model.roi_heads.box_roi_pool(encoded, boxes, images.image_sizes)
        features.append(model.roi_heads.box_head(pooled).cpu())
        labels.extend(
            [
                value
                for t in targets
                for value in ([1] * len(t["boxes"]) + [0] * len(t["hard_negative_boxes"]))
            ]
        )
        atomic_json(
            args.out / "progress.json",
            dict(
                status="running",
                stage="M2 TRAIN calibration",
                images_done=min(offset + 3, len(indices)),
                images_total=len(indices),
                phase="TRAIN ROI feature audit",
                updated_at=time.time(),
            ),
        )
    x = torch.cat(features).numpy()
    y = np.asarray(labels)
    assert int((y == 1).sum()) == 179 and int((y == 0).sum()) == 440

    def margin(state):
        w = state[prefix + "weight"].numpy()
        b = state[prefix + "bias"].numpy()
        return x @ (w[1] - w[0]) + (b[1] - b[0])

    old, new = margin(parent["model"]), margin(candidate["model"])
    delta = train_recall_bias(old[y == 1], new[y == 1])
    result = dict(
        parent_train_positive_accepted=int((old[y == 1] > 0).sum()),
        candidate_train_positive_accepted=int((new[y == 1] > 0).sum()),
        calibrated_train_positive_accepted=int(((new[y == 1] + delta) > 0).sum()),
        parent_train_negative_accepted=int((old[y == 0] > 0).sum()),
        candidate_train_negative_accepted=int((new[y == 0] > 0).sum()),
        calibrated_train_negative_accepted=int(((new[y == 0] + delta) > 0).sum()),
        train_positive_total=179,
        train_negative_total=440,
        foreground_bias_addition=delta,
    )
    if delta == 0:
        validation = candidate["validation"]
        result["identical_candidate_evaluation_reused"] = True
    else:
        model.roi_heads.box_predictor.cls_score.bias[1].add_(delta)
        linear_path = args.classifier / "robust_rgb_linear.npz"
        if digest(linear_path) != candidate["request"]["crop_classifier_sha256"]:
            raise ValueError("Fixed crop classifier changed")
        with np.load(linear_path, allow_pickle=False) as saved:
            linear = {k: saved[k].copy() for k in ("coef", "intercept", "classes")}
        validation, predictions = evaluate(model, data, linear, rng, check, lambda *a: None)
        one_to_one, _ = audit_predictions(predictions, data.labels)
        validation["one_to_one"] = one_to_one
        atomic_json(args.out / "predictions.json", dict(rows=predictions))
        atomic_checkpoint(
            args.out / "calibrated_component.pt",
            dict(
                kind="perception_candidate_only",
                architecture="signal-detector-v1",
                model=model.state_dict(),
                training_bias=delta,
                source_candidate_sha256=digest(candidate_path),
                validation=validation,
            ),
        )
    result["validation"] = validation
    exact = validation["one_to_one"]
    counts = validation["counts"]
    result["eligible_vs_parent"] = (
        exact["iou3"] >= 30
        and exact["iou3_state_correct"] >= 24
        and counts["approximate_background_candidates"] <= 681
        and validation["selection_score"] < parent["validation"]["selection_score"]
    )
    result["wall_seconds"] = time.monotonic() - started
    atomic_json(args.out / "assessment.json", result)
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            stage="M2 TRAIN calibration",
            images_done=len(indices),
            images_total=len(indices),
            phase="single rule complete",
            updated_at=time.time(),
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "candidate", "corpus", "negatives", "release", "classifier", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        fresh = not args.out.exists()
        try:
            calibrate(args)
        except BaseException as exc:
            if fresh and (args.out / "request.json").exists():
                atomic_json(
                    args.out / "progress.json",
                    dict(
                        status="stopped" if isinstance(exc, InterruptedError) else "failed",
                        stage="M2 TRAIN calibration",
                        error=f"{type(exc).__name__}: {exc}",
                        updated_at=time.time(),
                    ),
                )
            raise
