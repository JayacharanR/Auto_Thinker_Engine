#!/usr/bin/env python3
"""TRAIN-only visual support audit of frozen out-of-fold RGB classifier errors."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa:E402


def main():
    def read(path):
        return json.loads(path.read_text())

    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    study = ROOT / "outputs/navigation_signal_phase_boost_20261021"
    rows = read(cache / "train_rows.json")
    with np.load(ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz") as a:
        folds, predicted = a["fold"], a["predicted"]
    probabilities = np.zeros((len(rows), 4))
    for fold in range(3):
        probabilities[folds == fold] = np.load(
            study / f"joint100_fold{fold}/train_class_probability.npy"
        )[folds == fold]
    data = DiverseSignalDataset(
        ROOT,
        ROOT / "outputs/navigation_signal_pixel_audit_20261021",
        ROOT / "outputs/navigation_signal_diversity_audit_20261021",
    )
    keys = {"red": 1, "red_repeat": 1, "yellow": 2, "green": 3}
    counts, details = defaultdict(lambda: dict(points=0, foreground_kept=0, joint=0)), []
    for index, row in enumerate(rows):
        if not predicted[index] or row["label"] != 1:
            continue
        sample_id = row["sample"]
        if sample_id not in data.train or row["role"] != "train":
            raise ValueError("No component VAL or reserved data may enter TRAIN support audit")
        sample = data.samples[sample_id]
        cls = keys[sample["phase"]]
        x, y = row["x"], row["y"]
        mask = data.targets["bulb"][sample_id]
        matches = [
            b
            for b in data.boxes[sample_id]
            if b[0] <= x < b[2] and b[1] <= y < b[3] and b[4] == cls
        ]
        if not matches:
            raise ValueError("Known positive proposal lost its TRAIN housing")
        box = matches[0]
        # Restrict support to the matched housing, not a neighboring light.
        yy, xx = np.where(mask[box[1] : box[3], box[0] : box[2]] == cls)
        distances = np.maximum(np.abs(xx + box[0] - x), np.abs(yy + box[1] - y))
        radius = int(distances.min()) if len(distances) else None
        support = "active_in_5x5" if radius is not None and radius <= 2 else "outside_5x5"
        p = probabilities[index]
        kept = bool(p[1:].sum() >= 0.5)
        joint = kept and int(p[1:].argmax() + 1) == cls
        location = f"{sample['town']}/{sample['junction_id']}"
        for key in ("all", support, f"state{cls}/{support}", f"location/{location}"):
            counts[key]["points"] += 1
            counts[key]["foreground_kept"] += int(kept)
            counts[key]["joint"] += int(joint)
        details.append(
            dict(
                row=index,
                sample=sample_id,
                location=location,
                cls=cls,
                active_pixel_chebyshev_radius=radius,
                support=support,
                foreground_kept=kept,
                joint=bool(joint),
            )
        )
    if len(details) != 3371:
        raise ValueError("Original predicted-positive TRAIN point denominator changed")
    result = dict(
        TRAIN_only=True,
        posthoc_no_fit_or_reselection=True,
        excludes_GT_only_anchors=True,
        original_model_unchanged=True,
        counts=dict(counts),
        details=details,
        source_sha256=digest(__file__),
        rows_sha256=digest(cache / "train_rows.json"),
        interpretation="Point counts are correlated; active-mask support is conservative",
    )
    atomic_json(ROOT / "reports/navigation_signal_train_support_20261021.json", result)
    print(json.dumps({k: v for k, v in counts.items() if not k.startswith("location/")}, indent=2))


if __name__ == "__main__":
    main()
