"""Independent crop/geometry checks for the component diagnostics."""

import numpy as np
import pytest

from scripts.assess_navigation_reference_detector import iou
from scripts.assess_navigation_signal_crops import crop_feature


def test_crop_feature_reads_only_native_rgb_inside_supplied_roi():
    first = np.zeros((288, 512, 3), dtype=np.uint8)
    second = np.full_like(first, 255)
    first[20:40, 10:30] = second[20:40, 10:30] = [255, 0, 128]
    a, b = crop_feature(first, [10, 20, 30, 40]), crop_feature(second, [10, 20, 30, 40])
    assert a.shape == (16 * 32 * 3,) and np.array_equal(a, b)
    assert np.allclose(
        a.reshape(32, 16, 3)[0, 0], [0.5, -0.5, 128 / 255 - 0.5], atol=1e-7, rtol=0
    )


def test_crop_feature_rejects_outside_frame_and_non_native_inputs():
    image = np.zeros((288, 512, 3), dtype=np.uint8)
    for box in ([-1, 0, 3, 4], [10, 5, 9, 6], [0, 0, 513, 288]):
        with pytest.raises(ValueError, match="in-image"):
            crop_feature(image, box)
    with pytest.raises(ValueError, match="Native RGB"):
        crop_feature(image.astype(np.float32), [0, 0, 2, 2])


def test_box_overlap_has_correct_union_and_handles_empty_or_disjoint_boxes():
    assert iou([0, 0, 4, 4], [2, 0, 6, 4]) == pytest.approx(1 / 3)
    assert iou([0, 0, 4, 4], [0, 0, 4, 4]) == 1
    assert iou([0, 0, 4, 4], [4, 0, 8, 4]) == 0
    assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0
