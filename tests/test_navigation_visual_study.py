import json

import numpy as np
import pytest
import torch

from scripts.run_navigation_visual_study import evict_features, fit_cells, fit_name
from src.navigation.dagger import digest


def test_corrected_study_matrix_retains_all_baselines_budgets_and_ablations():
    cells = fit_cells()
    assert len(cells) == 66 and len(set(map(fit_name, cells))) == 66
    main = [c for c in cells if c["encoder"] != "nav_color_only" and c["color_enabled"]]
    assert len(main) == 60
    for encoder in {c["encoder"] for c in main}:
        assert len([c for c in main if c["encoder"] == encoder]) == 12
        assert len([c for c in cells if c["encoder"] == encoder and not c["color_enabled"]]) == 1


def test_regenerable_cache_eviction_requires_all_thirteen_verified_fits(tmp_path):
    array = tmp_path / "features.npy"
    np.save(array, np.ones((3, 2), np.float16))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(features_sha256=digest(array))))
    with pytest.raises(ValueError, match="Complete verified"):
        evict_features(tmp_path, [])
    assert array.exists() and not (tmp_path / "evicted_features.json").exists()


def test_visual_input_ablation_starts_with_identical_weights():
    from scripts.train_navigation_spatial_student import initialize as original
    from scripts.train_navigation_visual_student import initialize as visual

    a, b = original("imagenet_resnet18", 512, 42), visual("imagenet_resnet18", 512, 42)
    assert all(torch.equal(v, b.state_dict()[key]) for key, v in a.state_dict().items())
