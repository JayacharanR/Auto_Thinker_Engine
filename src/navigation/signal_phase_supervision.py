"""TRAIN-only phase labels and balanced foreground weights for RGB classifiers."""

import numpy as np


def phase_targets(rows, dataset, binary_labels):
    train = set(dataset["train"])
    labels = np.asarray(binary_labels)
    if labels.shape != (len(rows),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Original binary TRAIN labels required")
    result = labels.copy()
    states = {"red": 1, "red_repeat": 1, "yellow": 2, "green": 3}
    for i, row in enumerate(rows):
        sample = dataset["samples"][row["sample"]]
        if row["sample"] not in train or row["role"] != "train" or sample["role"] != "train":
            raise ValueError("Only TRAIN labels can receive phase supervision")
        if row["label"] != labels[i]:
            raise ValueError("Original binary labels changed")
        if labels[i]:
            result[i] = states[sample["phase"]]
    return result


def phase_balanced_weights(labels):
    labels = np.asarray(labels)
    if set(np.unique(labels)) != {0, 1, 2, 3}:
        raise ValueError("Every fit requires background and all three TRAIN phases")
    result = np.empty(len(labels), np.float64)
    for state in range(4):
        at = labels == state
        result[at] = len(labels) / ((2 if state == 0 else 6) * at.sum())
    return result


def joint_candidates(peaks, probabilities):
    p = np.asarray(probabilities)
    if p.shape != (len(peaks), 4) or not np.isfinite(p).all():
        raise ValueError("Four finite probabilities for every original RGB proposal required")
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError("Valid normalized class probabilities required")
    foreground = p[:, 1:].sum(1)
    result = [
        dict(x=point["x"], y=point["y"], cls=int(value[1:].argmax()) + 1, score=float(probability))
        for point, value, probability in zip(peaks, p, foreground)
        if probability >= 0.5
    ]
    return sorted(result, key=lambda point: (-point["score"], point["y"], point["x"]))
