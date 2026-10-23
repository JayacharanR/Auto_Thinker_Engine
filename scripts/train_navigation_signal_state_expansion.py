#!/usr/bin/env python3
"""One expanded TRAIN state fit; fixed detector boxes and safeguarded old ROIs."""

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.assess_navigation_signal_crops import crop_feature  # noqa: E402
from scripts.assess_navigation_signal_detector import audit_predictions  # noqa: E402
from scripts.train_navigation_signal_crops import expand_box, predict  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.signal_detection_data import SignalDetectionData  # noqa: E402
from src.navigation.signal_roi_learning import validation_payload_digest  # noqa: E402


def integer_box(box):
    return [
        max(0, math.floor(box[0])),
        max(0, math.floor(box[1])),
        min(512, math.ceil(box[2])),
        min(288, math.ceil(box[3])),
    ]


def train(args):
    if args.out.exists():
        raise FileExistsError("Fresh state experiment required")
    started = time.monotonic()
    data = SignalDetectionData(args.corpus)
    # Empty verified-negative arrays are part of the frozen validation signature.
    data.attach_hard_negatives(
        ROOT / "outputs/navigation_m2_signal_hard_negatives_20261016",
        "8df27ee218056e392cc5ddb0f50fef9263a1e8b81517b1d5ecbeba8038a6e49b",
        ROOT / "outputs/navigation_m2_signal_negative_release_20261016/release.json",
    )
    data.append_training_corpus(args.supplement)
    validation_sha = validation_payload_digest(data)
    if validation_sha != "9ab66584c129e8a84a676512bad8b693dd9f82d2df5de2f1a8c74a3edcb0bd2f":
        raise ValueError("Validation identity changed")
    review = json.loads((args.detector / "review.json").read_text())
    detector_path = args.detector / "best_candidate.pt"
    if review["selected_step"] != 500 or not review["candidate_improved"]:
        raise ValueError("Fixed improved ROI adapter500 required")
    if digest(detector_path) != "2dbd3e56c7de29c5f1b79c284f7fe0f704ec659e43b0a4885583ca2f108ee507":
        raise ValueError("Fixed selected detector changed")
    prediction_path = args.detector / "predictions_step500.json"
    old_detector_path = (
        ROOT / "outputs/navigation_m2_signal_detector_20261016/predictions_step50.json"
    )
    old_audit_path = old_detector_path.parent / "one_to_one_audit.json"
    old_evidence = json.loads(old_audit_path.read_text())["evidence_sha256"]
    if digest(old_detector_path) != old_evidence[str(old_detector_path.relative_to(ROOT))]:
        raise ValueError("Original safeguard predictions changed")
    rows = json.loads(prediction_path.read_text())["rows"]
    old_path = ROOT / "outputs/navigation_m2_signal_crop_alignment_20261016/robust_rgb_linear.npz"
    if digest(old_path) != "ca08ac5a7c301c76489a9fc8cde692a9da213c0fa679fd7307e7dc0d2a6b5de2":
        raise ValueError("Old classifier changed")
    with np.load(old_path, allow_pickle=False) as saved:
        old = {k: saved[k].copy() for k in ("coef", "intercept", "classes")}
    exact, assigned = audit_predictions(rows, data.labels)
    assert exact["iou3"] == 44 and exact["iou3_state_correct"] == 29
    check_resources(args.out, 25)
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/train_navigation_signal_crops.py",
        ROOT / "scripts/assess_navigation_signal_crops.py",
        ROOT / "scripts/assess_navigation_signal_detector.py",
        *sorted((ROOT / "src/navigation").glob("*.py")),
        ROOT / "reports/navigation_m2_signal_state_expansion_protocol_20261016.md",
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
            fixed_detector_sha256=digest(detector_path),
            fixed_prediction_sha256=digest(prediction_path),
            old_detector_prediction_sha256=digest(old_detector_path),
            old_classifier_sha256=digest(old_path),
            validation_payload_sha256=validation_sha,
            original_audit_sha256=digest(args.corpus / "data_audit.json"),
            supplement_audit_sha256=digest(args.supplement / "data_audit.json"),
            model="C1 balanced L2 lbfgs LogisticRegression max_iter1000;16 variants/source",
            source_weight=642,
            original_weight=321,
            supplement_weight=321,
            maximum_wall_seconds=300,
            seed=20261016,
        ),
    )

    def check(phase, done=0, total=1):
        if (args.out / "STOP").exists() or time.monotonic() - started >= 300:
            raise InterruptedError("STOP or state-fit cap")
        atomic_json(
            args.out / "progress.json",
            dict(
                status="running",
                phase=phase,
                images_done=done,
                images_total=total,
                updated_at=time.time(),
            ),
        )

    phases = {"red": 0, "red_repeat": 0, "yellow": 1, "green": 2}
    rng = np.random.default_rng(20261016)
    x, y, weights, manifest = [], [], [], []
    for offset, index in enumerate(data.base.train):
        if offset % 24 == 0:
            check("Preparing TRAIN state crops", offset, len(data.base.train))
        sample = data.base.samples[index]
        assert sample["role"] == "train"
        image = data.base.images[index].transpose(1, 2, 0)
        original = "supplement_corpus" not in sample
        for label in data.annotation_rows[index]:
            for n in range(16):
                box = label["crop_xyxy"] if n == 0 else expand_box(label["crop_xyxy"], rng)
                x.append(crop_feature(image, box))
                y.append(phases[sample["phase"]])
                weights.append(321 / (179 if original else 463) / 16)
                manifest.append(dict(sample=index, head=label["head"], box=box, original=original))
    x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)
    assert len(x) == 642 * 16 and abs(weights.sum() - 642) < 1e-6
    atomic_json(args.out / "training_crop_manifest.json", dict(rows=manifest))
    check("Fitting one expanded state classifier")
    fit_start = time.monotonic()
    with threadpool_limits(limits=2):
        model = LogisticRegression(
            C=1, max_iter=1000, class_weight="balanced", random_state=20261016
        )
        model.fit(x, y, sample_weight=weights)
    fit_seconds = time.monotonic() - fit_start
    new = dict(coef=model.coef_, intercept=model.intercept_, classes=model.classes_)
    artifact = args.out / "expanded_rgb_linear.npz"
    np.savez(artifact, **new)
    by_path = {sample["path"]: i for i, sample in enumerate(data.base.samples)}

    def matched_score(predicted_rows, linear):
        _, pairs = audit_predictions(predicted_rows, data.labels)
        features, truth = [], []
        for item in pairs:
            row = predicted_rows[item["row"]]
            index = by_path[row["sample"]["path"]]
            assert data.base.samples[index]["role"] == "validation"
            box = integer_box(row["predictions"][item["prediction"]]["box"])
            features.append(crop_feature(data.base.images[index].transpose(1, 2, 0), box))
            truth.append(item["truth"])
        features, truth = np.asarray(features), np.asarray(truth)
        prediction = predict(linear, features)
        return int((prediction == truth).sum()), features

    old_correct, actual_x = matched_score(rows, old)
    assert old_correct == 29
    assert np.array_equal(predict(new, actual_x), model.predict(actual_x))
    with np.load(artifact, allow_pickle=False) as loaded:
        assert np.array_equal(predict(loaded, actual_x), predict(new, actual_x))
    # Every new state is inferred from RGB and a predicted ROI, with no label.
    updated = copy.deepcopy(rows)
    for offset, row in enumerate(updated):
        check("Classifying unchanged predicted ROIs", offset, len(updated))
        index = by_path[row["sample"]["path"]]
        image = data.base.images[index].transpose(1, 2, 0)
        for p in row["predictions"]:
            vector = crop_feature(image, integer_box(p["box"]))
            p["state"] = int(predict(new, vector[None])[0])
    for before, after in zip(rows, updated):
        assert before["sample"] == after["sample"]
        assert [(p["box"], p["score"]) for p in before["predictions"]] == [
            (p["box"], p["score"]) for p in after["predictions"]
        ]
        after.pop("supported_matches", None)  # Old state-specific diagnostics are stale.
    combined, _ = audit_predictions(updated, data.labels)
    assert combined["iou3"] == 44
    old_rows = json.loads(old_detector_path.read_text())["rows"]
    old_compatibility, _ = matched_score(old_rows, old)
    new_compatibility, _ = matched_score(old_rows, new)
    assert old_compatibility == 24
    oracle_x, oracle_y = [], []
    for index in data.base.validation:
        image = data.base.images[index].transpose(1, 2, 0)
        for label in data.annotation_rows[index]:
            oracle_x.append(crop_feature(image, label["crop_xyxy"]))
            oracle_y.append(phases[data.base.samples[index]["phase"]])
    oracle_x, oracle_y = np.asarray(oracle_x), np.asarray(oracle_y)
    accepted = combined["iou3_state_correct"] > 29 and new_compatibility >= 24
    result = dict(
        complete=True,
        candidate_accepted=accepted,
        fixed_detections=44,
        denominator=72,
        old_joint_state=29,
        new_joint_state=combined["iou3_state_correct"],
        old_original_detector_roi_correct=24,
        new_original_detector_roi_correct=new_compatibility,
        old_privileged_oracle_correct=int((predict(old, oracle_x) == oracle_y).sum()),
        new_privileged_oracle_correct=int((predict(new, oracle_x) == oracle_y).sum()),
        one_to_one=combined,
        exported_prediction_parity=True,
        box_score_identity_preserved=True,
        source_labels=642,
        augmented_crops=len(x),
        source_weight=float(weights.sum()),
        fit_seconds=fit_seconds,
        wall_seconds=time.monotonic() - started,
        optimizer_iterations=model.n_iter_.tolist(),
        artifact_sha256=digest(artifact),
        scope="Fixed predicted-ROI state component; oracle separate; no driving promotion",
    )
    atomic_json(args.out / "predictions_expanded_state.json", dict(rows=updated))
    atomic_json(args.out / "assessment.json", result)
    selected_classifier = artifact if accepted else old_path
    atomic_json(
        args.out / "perception_bundle.json",
        dict(
            kind="Selected development perception components only, not a driving policy",
            detector_architecture="signal-detector-roi-residual-v1",
            detector_checkpoint=str(detector_path.resolve()),
            detector_sha256=digest(detector_path),
            state_classifier=str(selected_classifier.resolve()),
            state_sha256=digest(selected_classifier),
            expanded_state_accepted=accepted,
            selected_joint_state=combined["iou3_state_correct"] if accepted else 29,
            fixed_detection_count=44,
            driving_policy_changed=False,
            evaluation_sha256=digest(args.out / "assessment.json"),
        ),
    )
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            phase="Fixed state fit complete",
            images_done=144,
            images_total=144,
            updated_at=time.time(),
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("corpus", "supplement", "detector", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        fresh = not args.out.exists()
        try:
            train(args)
        except BaseException as exc:
            if fresh and (args.out / "request.json").exists():
                atomic_json(
                    args.out / "progress.json",
                    dict(
                        status="stopped" if isinstance(exc, InterruptedError) else "failed",
                        error=f"{type(exc).__name__}: {exc}",
                        updated_at=time.time(),
                    ),
                )
            raise
