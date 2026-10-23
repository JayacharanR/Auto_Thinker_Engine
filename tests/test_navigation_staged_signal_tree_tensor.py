import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

PATH = Path(__file__).resolve().parents[1] / "patches/navigation_signal_tree_tensor_candidate.py"
SPEC = importlib.util.spec_from_file_location("staged_numeric_trees", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_export_preserves_real_tree_paths_threshold_boundaries_and_class_probabilities():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(160, 87)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int) + 2 * (x[:, 1] > 0).astype(int)
    with threadpool_limits(limits=2):
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=5,
            min_samples_leaf=5,
            early_stopping=False,
            random_state=42,
        ).fit(x, y)
        bundle = MODULE.export_numeric_classifier(model)
        torch.set_num_threads(2)
        exported = MODULE.NumericSignalTrees(bundle)
        checks = [x]
        for tree in model._predictors[0]:
            for node in tree.nodes:
                if not node["is_leaf"]:
                    threshold = np.float32(node["num_threshold"])
                    probes = np.zeros((3, 87), np.float32)
                    probes[:, node["feature_idx"]] = [
                        np.nextafter(threshold, -np.inf, dtype=np.float32),
                        threshold,
                        np.nextafter(threshold, np.inf, dtype=np.float32),
                    ]
                    checks.append(probes)
        rows = np.concatenate(checks)
        np.testing.assert_allclose(
            exported(torch.from_numpy(rows)).numpy(), model.predict_proba(rows), atol=2e-13, rtol=0
        )
        assert exported(torch.empty((0, 87))).shape == (0, 4)
        with pytest.raises(ValueError, match="Finite"):
            exported(torch.full((1, 87), float("nan")))
