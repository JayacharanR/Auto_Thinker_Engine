import numpy as np
import pytest

from src.navigation.signal_pixel_labels import (
    active_bulb_masks,
    current_training_locations,
)


def test_response_is_spatial_not_static_color_and_requires_repeat_stability():
    blank = np.full((6, 4, 3), 15, np.uint8)
    blank[:, 3] = [255, 0, 0]  # A permanently red object is not a phase response.
    red, green, yellow, repeat = [blank.copy() for _ in range(4)]
    red[0, 0] = repeat[0, 0] = [255, 0, 0]
    yellow[2, 0] = [255, 255, 0]
    green[4, 0] = [0, 255, 0]
    red[0, 1] = [255, 0, 0]  # Absent in red-repeat: must remain unknown.
    masks = active_bulb_masks(red, green, yellow, repeat)
    assert masks.sum(axis=(1, 2)).tolist() == [1, 1, 1]
    assert masks[0, 0, 0] and masks[1, 2, 0] and masks[2, 4, 0]
    assert not masks[:, :, 3].any()
    assert not masks[:, 0, 1].any()


def test_unknown_response_and_invalid_alignment():
    image = np.zeros((2, 2, 3), np.uint8)
    assert not active_bulb_masks(image, image, image, image).any()
    with pytest.raises(ValueError):
        active_bulb_masks(image, image[:1], image, image)
    with pytest.raises(ValueError):
        active_bulb_masks(image.astype(float), image, image, image)


def test_all_geographic_reservations_override_old_train_role():
    allowed = {"Town01_Opt": [1, 2, 3, 4, 5], "Town05": [1]}
    assert current_training_locations(
        allowed, {"Town01_Opt": [2]}, {"Town01_Opt": [3]}, {"Town01_Opt": [4]}
    ) == {"Town01_Opt": [1, 5]}
