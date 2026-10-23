"""Staged numeric-tree inference; no deployed policy imports this module.

Exports the existing classifier without fitting, threshold changes or new inputs.
All tree paths and leaf values are retained. Parallel summation may differ at
floating-point roundoff; parity must be measured before any deployment.
"""

import numpy as np
import torch
from torch import nn

SCHEMA = "numeric-hgb-tensor-v1-staged"


def export_numeric_classifier(model):
    if model.is_categorical_ is not None and np.any(model.is_categorical_):
        raise ValueError("Only numeric features supported")
    if not np.array_equal(model.classes_, [0, 1, 2, 3]) or model.n_features_in_ != 87:
        raise ValueError("Exact BG/red/yellow/green RGB87 classifier required")
    rounds = model._predictors
    if len(rounds) != 100 or any(len(row) != 4 for row in rounds):
        raise ValueError("Exact joint100 four-class tree layout required")
    trees = [t for row in rounds for t in row]
    nodes = max(len(t.nodes) for t in trees)
    arrays = {
        name: np.zeros((400, nodes), dtype=dtype)
        for name, dtype in (
            ("feature", np.int64),
            ("threshold", np.float64),
            ("left", np.int64),
            ("right", np.int64),
            ("value", np.float64),
            ("leaf", np.bool_),
        )
    }
    maximum_depth = 0
    for i, tree in enumerate(trees):
        n = tree.nodes
        if n["is_categorical"].any():
            raise ValueError("Categorical node cannot be exported")
        maximum_depth = max(maximum_depth, int(n["depth"].max()))
        for source, target in (
            ("feature_idx", "feature"),
            ("num_threshold", "threshold"),
            ("left", "left"),
            ("right", "right"),
            ("value", "value"),
            ("is_leaf", "leaf"),
        ):
            arrays[target][i, : len(n)] = n[source]
        leaf = arrays["leaf"][i]
        arrays["left"][i, leaf] = np.flatnonzero(leaf)
        arrays["right"][i, leaf] = np.flatnonzero(leaf)
        arrays["feature"][i, leaf] = 0
    return dict(
        schema=SCHEMA,
        arrays=arrays,
        baseline=model._baseline_prediction.copy(),
        maximum_depth=maximum_depth,
        classes=model.classes_.copy(),
        feature_count=87,
    )


class NumericSignalTrees(nn.Module):
    def __init__(self, bundle):
        super().__init__()
        if (
            bundle["schema"] != SCHEMA
            or bundle["feature_count"] != 87
            or not np.array_equal(bundle["classes"], [0, 1, 2, 3])
        ):
            raise ValueError("Exact registered numeric signal tree bundle required")
        self.maximum_depth = bundle["maximum_depth"]
        for name, values in bundle["arrays"].items():
            self.register_buffer(name, torch.from_numpy(values.copy()))
        self.register_buffer("baseline", torch.from_numpy(bundle["baseline"].copy()))
        self.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, features):
        if (
            features.ndim != 2
            or features.shape[1] != 87
            or features.dtype != torch.float32
            or not torch.isfinite(features).all()
        ):
            raise ValueError("Finite FP32 RGB87 feature rows required")
        if features.device != self.value.device:
            raise ValueError("Features and tree arrays must be on the same device")
        n = len(features)
        tree = torch.arange(400, device=features.device)[None, :]
        sample = torch.arange(n, device=features.device)[:, None]
        current = torch.zeros((n, 400), dtype=torch.long, device=features.device)
        for _ in range(self.maximum_depth):
            column = self.feature[tree, current]
            left = features[sample, column].double() <= self.threshold[tree, current]
            current = torch.where(left, self.left[tree, current], self.right[tree, current])
        if not self.leaf[tree, current].all():
            raise ValueError("Exported depth did not reach all leaves")
        values = self.value[tree, current].reshape(n, 100, 4)
        return (self.baseline + values.sum(1)).softmax(-1)
