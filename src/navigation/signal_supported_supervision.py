"""Offline TRAIN-only eligibility and sampled proposal diagnostics."""

import numpy as np


def supported_training_rows(rows, support, train_indices, predicted):
    """Unknown local phase is excluded from fitting, never relabeled background."""
    train_indices = set(train_indices)
    keep = np.ones(len(rows), dtype=bool)
    positives = {i for i, row in enumerate(rows) if predicted[i] and row["label"] == 1}
    seen = set()
    for row in rows:
        if row["sample"] not in train_indices or row["role"] != "train":
            raise ValueError("Only original TRAIN rows may enter supervision")
    for detail in support:
        i = detail["row"]
        if i not in positives or i in seen or rows[i]["sample"] != detail["sample"]:
            raise ValueError("Exactly one support record per predicted positive required")
        seen.add(i)
        radius = detail["nearest_matching_radius"]
        if radius is None or radius < 0:
            raise ValueError("Audited same-housing active support required")
        keep[i] = radius <= 2
    if seen != positives:
        raise ValueError("Missing predicted-positive support")
    return keep


def sampled_housing_score(probabilities, rows, predicted, samples, boxes, train_indices):
    """Same point-in-housing matching, but TRAIN sampled proposals, not a benchmark."""
    grouped = {i: [] for i in train_indices}
    for j in np.flatnonzero(predicted):
        row, p = rows[j], probabilities[j]
        if row["sample"] not in grouped or row["role"] != "train":
            raise ValueError("TRAIN-only scoring")
        if p.shape != (4,) or not np.isfinite(p).all() or abs(p.sum() - 1) > 1e-6:
            raise ValueError("Complete four-class probabilities required")
        if p[1:].sum() >= 0.5:
            grouped[row["sample"]].append(
                dict(
                    x=row["x"],
                    y=row["y"],
                    cls=int(p[1:].argmax() + 1),
                    score=float(p[1:].sum()),
                    background=row["label"] == 0,
                )
            )
    result = dict(labels=0, localized=0, joint=0, sampled_background=0, towns={}, classes={})
    for i in train_indices:
        available, matched = set(range(len(boxes[i]))), {}
        for peak in sorted(grouped[i], key=lambda p: (-p["score"], p["y"], p["x"])):
            eligible = [
                k
                for k in sorted(available)
                if boxes[i][k][0] <= peak["x"] < boxes[i][k][2]
                and boxes[i][k][1] <= peak["y"] < boxes[i][k][3]
            ]
            if eligible:
                k = eligible[0]
                available.remove(k)
                matched[k] = peak
            elif peak["background"]:
                result["sampled_background"] += 1
        for k, box in enumerate(boxes[i]):
            localized = k in matched
            joint = localized and matched[k]["cls"] == box[4]
            for count in (
                result,
                result["towns"].setdefault(
                    samples[i]["town"], dict(labels=0, localized=0, joint=0)
                ),
                result["classes"].setdefault(str(box[4]), dict(labels=0, localized=0, joint=0)),
            ):
                count["labels"] += 1
                count["localized"] += int(localized)
                count["joint"] += int(joint)
    return result
