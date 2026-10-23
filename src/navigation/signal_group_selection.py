"""TRAIN-only location folds and selection for frozen signal features."""

import hashlib

import numpy as np


def location_folds(rows, dataset):
    train, validation = set(dataset["train"]), set(dataset["validation"])
    if train & validation:
        raise ValueError("TRAIN and component VAL overlap")
    groups = []
    for row in rows:
        index = row["sample"]
        sample = dataset["samples"][index]
        if index not in train or row["role"] != "train" or sample["role"] != "train":
            raise ValueError("Only TRAIN rows may enter internal folds")
        groups.append(f"{sample['town']}/{sample['junction_id']}")
    ordered = sorted(set(groups), key=lambda g: hashlib.sha256(f"42:{g}".encode()).hexdigest())
    assignment = {group: n % 3 for n, group in enumerate(ordered)}
    return np.array([assignment[g] for g in groups]), np.array(groups), assignment


def macro_location_logloss(probability, labels, groups, mask):
    p = np.clip(np.asarray(probability)[mask], 1e-12, 1 - 1e-12)
    y, g = np.asarray(labels)[mask], np.asarray(groups)[mask]
    if not np.isfinite(p).all():
        raise ValueError("Every held-out TRAIN point needs a finite probability")
    scores = {}
    for location in sorted(set(g)):
        at = g == location
        if not (y[at] == 0).any() or not (y[at] == 1).any():
            raise ValueError("Each TRAIN selection location needs both classes")
        scores[str(location)] = float(
            -0.5 * (np.log(p[at & (y == 1)]).mean() + np.log1p(-p[at & (y == 0)]).mean())
        )
    return float(np.mean(list(scores.values()))), scores
