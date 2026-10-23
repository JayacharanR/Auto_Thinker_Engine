#!/usr/bin/env python3
"""CPU oracle-ROI appearance diagnostic; no learned detection or driving claim."""

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_learning import SignalDataset  # noqa: E402


def crop_feature(rgb, box):
    if rgb.dtype != np.uint8 or rgb.shape != (288, 512, 3):
        raise ValueError("Native RGB required")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 288):
        raise ValueError("Valid in-image annotation crop required")
    crop = Image.fromarray(rgb).crop(box).resize((16, 32), Image.Resampling.BILINEAR)
    return np.asarray(crop).astype(np.float32).ravel() / 255 - 0.5


def assess(corpus, out):
    if out.exists():
        raise FileExistsError("Fresh oracle diagnostic output required")
    started = time.monotonic()
    dataset = SignalDataset(corpus)
    audit = json.loads((corpus / "data_audit.json").read_text())
    labels = defaultdict(list)
    for row in audit["labels"]:
        if row["static_phase_response_supported"]:
            labels[(row["town"], row["case_id"], row["view"])].append(row)
    features, truth, rows = [], [], []
    for index, sample in enumerate(dataset.samples):
        for label in labels[(sample["town"], sample["case_id"], sample["view"])]:
            if sample["phase"] == "yellow" and not label["yellow_color_supported"]:
                continue
            features.append(
                crop_feature(dataset.images[index].transpose(1, 2, 0), label["crop_xyxy"])
            )
            truth.append({"red": 0, "red_repeat": 0, "yellow": 1, "green": 2}[sample["phase"]])
            rows.append({**sample, "head": label["head"], "crop_xyxy": label["crop_xyxy"]})
    x, y = np.asarray(features), np.asarray(truth)
    train = np.array([row["role"] == "train" for row in rows])
    if sum(~train) != sum(len(dataset.samples[i]["points"]) for i in dataset.validation):
        raise ValueError("Oracle crop count differs from frozen center evaluation")
    out.mkdir(parents=True)
    sources = [Path(__file__).resolve(), ROOT / "src/navigation/signal_learning.py"]
    for source in sources:
        destination = out / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    atomic_json(
        out / "request.json",
        {
            "kind": "PRIVILEGED ORACLE CROP diagnostic, not a deployable perception model",
            "feature": "Native annotated ROI resized16x32, bilinear RGB then flatten/255-.5",
            "model": "LogisticRegression C1,L2,lbfgs,1000 iterations,class-balanced,seed20261016",
            "sklearn_version": sklearn.__version__,
            "corpus_audit_sha256": digest(corpus / "data_audit.json"),
            "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        },
    )
    model = LogisticRegression(C=1, max_iter=1000, class_weight="balanced", random_state=20261016)
    model.fit(x[train], y[train])
    prediction = model.predict(x[~train])
    np.savez(
        out / "oracle_rgb_linear.npz",
        coef=model.coef_,
        intercept=model.intercept_,
        classes=model.classes_,
    )
    with np.load(out / "oracle_rgb_linear.npz", allow_pickle=False) as saved:
        restored = saved["classes"][(x[~train] @ saved["coef"].T + saved["intercept"]).argmax(1)]
    if not np.array_equal(prediction, restored):
        raise ValueError("Saved linear model predictions differ")
    held = [row for row in rows if row["role"] == "validation"]
    groups = defaultdict(lambda: [0, 0])
    for row, correct in zip(held, prediction == y[~train]):
        key = f"{row['town']}:{row['junction_id']}"
        groups[key][0] += int(correct)
        groups[key][1] += 1
    result = {
        "diagnostic_complete": True,
        "oracle_crops": True,
        "training_crops": int(train.sum()),
        "validation_crops": int((~train).sum()),
        "validation_correct": int((prediction == y[~train]).sum()),
        "validation_accuracy": float((prediction == y[~train]).mean()),
        "training_accuracy": float((model.predict(x[train]) == y[train]).mean()),
        "classes": ["Red", "Yellow", "Green"],
        "confusion_matrix": confusion_matrix(y[~train], prediction, labels=[0, 1, 2]).tolist(),
        "by_junction_correct_total": dict(groups),
        "iterations": model.n_iter_.tolist(),
        "wall_seconds": time.monotonic() - started,
        "artifact_predictions_verified": True,
        "artifact_sha256": digest(out / "oracle_rgb_linear.npz"),
        "scope": "Oracle ROI appearance diagnostic; no localization, lane or driving result",
        "limitations": [
            "Only4 TRAIN validation junctions and72 correlated crops",
            "Annotation development inspected some of those TRAIN locations",
            "Different model/normalization; not an isolated architecture or encoder ablation",
        ],
    }
    atomic_json(out / "assessment.json", result)
    atomic_json(out / "crop_manifest.json", {"rows": rows, "truth": truth})
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assess(args.corpus, args.out)


if __name__ == "__main__":
    main()
