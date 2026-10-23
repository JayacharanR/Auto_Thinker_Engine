import numpy as np
import pytest

from scripts.audit_navigation_signal_proposer import validate_arrays


def test_cache_audit_checks_all_shapes_offsets_values_and_class_probabilities():
    train = np.zeros((3, 2, 36), np.float16)
    val = np.zeros((4, 36), np.float16)
    train[..., 32:] = 0.25
    val[:, 32:] = 0.25
    offsets = np.array([0, 1, 4])
    validate_arrays(train, val, offsets, [0, 1, 4], 2)
    with pytest.raises(ValueError, match="offsets"):
        validate_arrays(train, val, np.array([0, 2, 4]), [0, 1, 4], 2)
    with pytest.raises(ValueError, match="denominators"):
        validate_arrays(train[:, :1], val, offsets, offsets, 2)
    val[3, 0] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        validate_arrays(train, val, offsets, offsets, 2)
    val[3, 0] = 0
    train[2, 1, 32] = 0.8
    with pytest.raises(ValueError, match="normalized"):
        validate_arrays(train, val, offsets, offsets, 2)
