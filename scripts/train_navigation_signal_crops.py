#!/usr/bin/env python3
"""TRAIN-only outward-crop augmentation; score fixed detector boxes, no controls."""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.assess_navigation_signal_crops import crop_feature  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_detection_data import SignalDetectionData  # noqa: E402


def expand_box(box, rng):
    x0, y0, x1, y1 = box
    dx, dy = max(1, math.ceil(0.3 * (x1 - x0))), max(1, math.ceil(0.2 * (y1 - y0)))
    return [
        max(0, x0 - int(rng.integers(dx + 1))),
        max(0, y0 - int(rng.integers(dy + 1))),
        min(512, x1 + int(rng.integers(dx + 1))),
        min(288, y1 + int(rng.integers(dy + 1))),
    ]


def predict(linear, features):
    return linear["classes"][(features @ linear["coef"].T + linear["intercept"]).argmax(1)]


def train(args):
    if args.out.exists():
        raise FileExistsError("Fresh output required")
    started = time.monotonic()
    data = SignalDetectionData(args.corpus)
    review = json.loads((args.detector / "review.json").read_text())
    audit = json.loads((args.detector / "one_to_one_audit.json").read_text())
    assert review["selected_step"] == audit["selected_step"] == 50
    path = args.detector / "predictions_step50.json"
    if digest(path) != audit["evidence_sha256"][str(path.resolve().relative_to(ROOT))]:
        raise ValueError("Fixed detector predictions changed")
    if digest(args.detector / "best_component.pt") != review["component_sha256"]:
        raise ValueError("Fixed detector component changed")
    detector_request = json.loads((args.detector / "request.json").read_text())
    if detector_request["corpus_audit_sha256"] != digest(args.corpus / "data_audit.json"):
        raise ValueError("Detector and classifier corpora differ")
    rows = json.loads(path.read_text())["rows"]
    old_path = ROOT / "outputs/navigation_m2_signal_oracle_crops_20261016/oracle_rgb_linear.npz"
    old_report = json.loads((old_path.parent / "assessment.json").read_text())
    if digest(old_path) != old_report["artifact_sha256"]:
        raise ValueError("Original classifier artifact changed")
    with np.load(old_path, allow_pickle=False) as f:
        old = {k: f[k].copy() for k in ("coef", "intercept", "classes")}
    phases = {"red": 0, "red_repeat": 0, "yellow": 1, "green": 2}
    rng = np.random.default_rng(20261016)
    features, truth, manifest = [], [], []
    gt_features, gt_truth = [], []
    for index, sample in enumerate(data.base.samples):
        rgb = data.base.images[index].transpose(1, 2, 0)
        for label in data.annotation_rows[index]:
            box = label["crop_xyxy"]
            if sample["role"] == "train":
                for n in range(16):
                    augmented = list(box) if n == 0 else expand_box(box, rng)
                    features.append(crop_feature(rgb, augmented))
                    truth.append(phases[sample["phase"]])
                    manifest.append(dict(sample=index, head=label["head"], box=augmented))
            else:
                gt_features.append(crop_feature(rgb, box))
                gt_truth.append(phases[sample["phase"]])
    assert len(features) == 179 * 16 and len(gt_truth) == 72
    index_by_path = {s["path"]: i for i, s in enumerate(data.base.samples)}
    detected_x, detected_y = [], []
    for match in audit["assignments"]:
        row = rows[match["row"]]
        sample = data.base.samples[index_by_path[row["sample"]["path"]]]
        if sample["role"] != "validation":
            raise ValueError("Only frozen validation boxes may be scored")
        box = row["predictions"][match["prediction"]]["box"]
        integer = [
            max(0, math.floor(box[0])),
            max(0, math.floor(box[1])),
            min(512, math.ceil(box[2])),
            min(288, math.ceil(box[3])),
        ]
        image = data.base.images[index_by_path[sample["path"]]].transpose(1, 2, 0)
        detected_x.append(crop_feature(image, integer))
        detected_y.append(match["truth"])
    x, y, test_x, test_y = map(np.asarray, (features, truth, detected_x, detected_y))
    gt_x, gt_y = np.asarray(gt_features), np.asarray(gt_truth)
    old_prediction = predict(old, test_x)
    assert len(test_y) == 30 and int((old_prediction == test_y).sum()) == 21
    args.out.mkdir(parents=True)
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/assess_navigation_signal_crops.py",
        ROOT / "src/navigation/signal_detection_data.py",
        ROOT / "src/navigation/signal_learning.py",
        ROOT / "reports/navigation_m2_signal_crop_alignment_protocol_20261016.md",
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            kind="Crop augmentation component only; no driving",
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
            corpus_sha256=digest(args.corpus / "data_audit.json"),
            fixed_detector_predictions_sha256=digest(path),
            fixed_assignments_sha256=digest(args.detector / "one_to_one_audit.json"),
            original_classifier_sha256=digest(old_path),
            model="C1 balanced L2 lbfgs LogisticRegression max_iter1000; total sample weight179",
            seed=20261016,
            source_crops=179,
            augmented_crops=2864,
        ),
    )
    atomic_json(args.out / "training_crop_manifest.json", dict(rows=manifest))
    fit_started = time.monotonic()
    with threadpool_limits(limits=2):
        model = LogisticRegression(
            C=1, max_iter=1000, class_weight="balanced", random_state=20261016
        )
        model.fit(x, y, sample_weight=np.full(len(y), 1 / 16))
    fit_seconds = time.monotonic() - fit_started
    new = dict(coef=model.coef_, intercept=model.intercept_, classes=model.classes_)
    saved_path = args.out / "robust_rgb_linear.npz"
    np.savez(saved_path, **new)
    prediction = predict(new, test_x)
    if not np.array_equal(prediction, model.predict(test_x)):
        raise ValueError("Exported coefficients fail prediction parity")
    with np.load(saved_path, allow_pickle=False) as saved:
        assert np.array_equal(predict(saved, test_x), prediction)
    result = dict(
        diagnostic_complete=True,
        matched_detector_boxes=30,
        denominator=72,
        original_detected_state_correct=int((old_prediction == test_y).sum()),
        augmented_detected_state_correct=int((prediction == test_y).sum()),
        original_oracle_gt_correct=int((predict(old, gt_x) == gt_y).sum()),
        augmented_oracle_gt_correct=int((predict(new, gt_x) == gt_y).sum()),
        predicted_roi_confusion_matrix=confusion_matrix(
            test_y, prediction, labels=[0, 1, 2]
        ).tolist(),
        class_order=["Red", "Yellow", "Green"],
        optimizer_iterations=model.n_iter_.tolist(),
        fit_seconds=fit_seconds,
        wall_seconds=time.monotonic() - started,
        artifact_sha256=digest(saved_path),
        saved_prediction_parity=True,
        scope="Fixed predicted-box RGB classification; predicted ROIs, no driving claim",
    )
    atomic_json(args.out / "assessment.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    train(parser.parse_args())
