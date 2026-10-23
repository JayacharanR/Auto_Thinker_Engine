import numpy as np
import pytest

from src.navigation.signal_group_selection import location_folds, macro_location_logloss


def test_whole_locations_never_cross_internal_folds_and_val_is_rejected():
    samples = [dict(town="Town01", junction_id=i // 2, role="train") for i in range(12)]
    data = dict(samples=samples, train=list(range(12)), validation=[12])
    rows = [dict(sample=i, role="train") for i in range(12)]
    folds, groups, assignment = location_folds(rows, data)
    assert set(folds) == {0, 1, 2}
    for group in assignment:
        assert len(set(folds[groups == group])) == 1
    with pytest.raises(ValueError, match="overlap"):
        location_folds(rows, dict(data, validation=[0]))
    with pytest.raises(ValueError, match="Only TRAIN"):
        location_folds([dict(sample=0, role="validation")], data)


def test_selection_weights_locations_and_classes_not_repeated_background():
    p = np.array([0.8, 0.2, 0.6, 0.4, 0.4, 0.4])
    y = np.array([1, 0, 1, 0, 0, 0])
    groups = np.array(["a", "a", "b", "b", "b", "b"])
    result, _ = macro_location_logloss(p, y, groups, np.ones(6, bool))
    assert result == pytest.approx((-np.log(0.8) - np.log(0.6)) / 2)
    p[-1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        macro_location_logloss(p, y, groups, np.ones(6, bool))
