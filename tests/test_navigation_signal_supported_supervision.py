import numpy as np
import pytest

from src.navigation.signal_supported_supervision import (
    sampled_housing_score,
    supported_training_rows,
)


def test_final_fit_requires_complete_unchanged_full_train_audit(tmp_path):
    import json

    from scripts.train_navigation_signal_rgb_boost import require_supported_final_audit
    from src.navigation.dagger import digest

    with pytest.raises(ValueError, match="requires a passing"):
        require_supported_final_audit({})
    assessment = tmp_path / "assessment.json"
    assessment.write_text(
        json.dumps(
            dict(
                train_audit_pass=True,
                exact_sampled_feature_and_probability_rows=48865,
                independent_sampled_recount_exact=True,
                no_component_val=True,
            )
        )
    )
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps(dict(status="running")))
    request = dict(
        supported_full_audit=str(tmp_path), supported_full_audit_sha256=digest(assessment)
    )
    with pytest.raises(ValueError, match="Completed matching"):
        require_supported_final_audit(request)
    progress.write_text(json.dumps(dict(status="complete")))
    require_supported_final_audit(request)
    assessment.write_text(assessment.read_text().replace("true", "false", 1))
    with pytest.raises(ValueError, match="Completed matching"):
        require_supported_final_audit(request)


def test_only_unsupported_predicted_positive_is_excluded():
    rows = [dict(sample=0, role="train", label=label) for label in (1, 1, 0, 1)]
    predicted = np.array([True, True, True, False])
    support = [dict(row=i, sample=0, nearest_matching_radius=r) for i, r in ((0, 2), (1, 3))]
    assert supported_training_rows(rows, support, [0], predicted).tolist() == [
        True,
        False,
        True,
        True,
    ]
    with pytest.raises(ValueError, match="Missing"):
        supported_training_rows(rows, support[:1], [0], predicted)
    with pytest.raises(ValueError, match="Only original TRAIN"):
        supported_training_rows(rows, support, [], predicted)


def test_scoring_keeps_original_boxes_and_excludes_gt_only_anchors():
    rows = [dict(sample=0, role="train", label=1, x=x, y=1) for x in (1, 2, 6)]
    rows.append(dict(sample=0, role="train", label=0, x=9, y=9))
    # Higher-confidence wrong-phase point consumes the first housing, as in benchmark.
    p = np.array(
        [
            [0.1, 0.8, 0.05, 0.05],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.1, 0.8, 0.05, 0.05],
        ]
    )
    result = sampled_housing_score(
        p,
        rows,
        np.array([True, True, False, True]),
        [dict(town="TRAIN")],
        [[[0, 0, 4, 4, 1], [5, 0, 8, 4, 1]]],
        [0],
    )
    assert (result["labels"], result["localized"], result["joint"]) == (2, 1, 0)
    assert result["sampled_background"] == 1
