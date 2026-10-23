"""Zero-adapter parity, label-safe augmentation and protected prospective selection."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.navigation.policy_worker import PrivilegeGuard
from src.navigation.signal_detector import (
    ResidualSignalPredictor,
    install_roi_adapter,
    make_signal_detector,
)
from src.navigation.signal_roi_learning import operational_key, roi_variants, supervised_rois


def test_zero_residual_preserves_logits_boxes_and_only_adapter_can_learn():
    torch.manual_seed(42)
    base = FastRCNNPredictor(1024, 2)
    head = ResidualSignalPredictor(base)
    x = torch.randn(17, 1024)
    old_logits, old_boxes = base(x)
    new_logits, new_boxes = head(x)
    assert torch.equal(old_logits, new_logits)
    assert torch.equal(old_boxes, new_boxes)
    new_logits.sum().backward()
    assert all(p.grad is None for p in head.base.parameters())
    assert head.adapter[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        head.adapter[-1].bias.fill_(1000)
    changed, boxes = head(x)
    assert torch.equal(changed[:, 0], old_logits[:, 0]) and torch.equal(boxes, old_boxes)
    assert torch.allclose(changed[:, 1] - old_logits[:, 1], torch.full((17,), 2.0), atol=1e-6)


def test_adapter_architecture_roundtrip_and_exact_trainable_scope():
    model = make_signal_detector(roi_adapter=True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 65665
    assert all(
        name.startswith("roi_heads.box_predictor.adapter.")
        for name, p in model.named_parameters()
        if p.requires_grad
    )
    restored = make_signal_detector(roi_adapter=True)
    restored.load_state_dict(model.state_dict(), strict=True)
    with pytest.raises(ValueError, match="already installed"):
        install_roi_adapter(model)


def test_positive_context_contains_housing_negative_context_never_leaves_certified_box():
    rng = np.random.default_rng(13)
    for box in ([0.0, 0.0, 2.0, 3.0], [510.0, 282.0, 512.0, 288.0], [10.0, 20.0, 21.0, 40.0]):
        for _ in range(30):
            for x0, y0, x1, y1 in roi_variants(box, rng, positive=True):
                assert 0 <= x0 <= box[0] < box[2] <= x1 <= 512
                assert 0 <= y0 <= box[1] < box[3] <= y1 <= 288
                assert (x1 - x0) * (y1 - y0) <= 2.25001 * (box[2] - box[0]) * (box[3] - box[1])
            for x0, y0, x1, y1 in roi_variants(box, rng, positive=False):
                assert box[0] <= x0 < x1 <= box[2] and box[1] <= y0 < y1 <= box[3]


def test_validation_image_refused_before_cached_features_and_training_module_blocked():
    data = SimpleNamespace(base=SimpleNamespace(samples=[dict(role="validation")], train=[]))
    with pytest.raises(ValueError, match="Only TRAIN"):
        supervised_rois(data, 0, np.random.default_rng(1))
    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("src.navigation.signal_roi_learning")


def baseline():
    return dict(
        one_to_one=dict(
            denominator=72, iou3=36, iou3_state_correct=26, by_town={"A": [6, 20], "B": [30, 52]}
        ),
        counts=dict(approximate_background_candidates=209),
        by_native_height=dict(under8px=[15, 40]),
    )


def test_new_selector_refuses_background_only_town_tiny_and_denominator_regressions():
    old = baseline()
    bg = copy.deepcopy(old)
    bg["counts"]["approximate_background_candidates"] = 1
    assert operational_key(bg, old) is None
    candidate = copy.deepcopy(old)
    candidate["one_to_one"]["iou3_state_correct"] += 1
    assert operational_key(candidate, old) == (27, 36, -209)
    for field in ("town", "tiny", "background", "denominator"):
        failed = copy.deepcopy(candidate)
        if field == "town":
            failed["one_to_one"]["by_town"]["A"][0] -= 1
        elif field == "tiny":
            failed["by_native_height"]["under8px"][0] -= 1
        elif field == "background":
            failed["counts"]["approximate_background_candidates"] += 1
        else:
            failed["one_to_one"]["denominator"] -= 1
        assert operational_key(failed, old) is None
    assert old == baseline()
