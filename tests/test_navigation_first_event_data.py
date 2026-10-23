import numpy as np
import pytest

from src.navigation.first_event_data import first_contact_targets, latent_target_mask


def test_contact_index_and_unobserved_latent_tail():
    event, valid = first_contact_targets(65, 65)
    assert np.flatnonzero(event).tolist() == [64]
    assert valid.sum() == 65
    assert latent_target_mask(65, 65).tolist() == [True] * 6 + [False] * 2
    assert latent_target_mask(80, None).tolist() == [True] * 7 + [False]


def test_unobserved_event_and_preexisting_contact_do_not_become_safe_labels():
    event, valid = first_contact_targets(30, 65)
    assert event.sum() == 0 and valid.sum() == 30
    event, valid = first_contact_targets(80, 0)
    assert not event.any() and not valid.any()
    with pytest.raises(ValueError):
        first_contact_targets(81, None)
