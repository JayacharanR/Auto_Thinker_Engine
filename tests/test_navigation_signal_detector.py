"""Independent assignment, transform and isolation tests for sparse detector fitting."""

import numpy as np
import pytest
import torch

from src.navigation.signal_detection_data import SignalDetectionData
from src.navigation.signal_detector import (
    IgnoreTransform,
    detector_training,
    ignore_negatives,
    make_signal_detector,
)


def test_unknown_housing_excludes_negative_anchors_but_preserves_positives():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 100.0, 100.0],
            [10.0, 10.0, 14.0, 20.0],
            [40.0, 40.0, 45.0, 50.0],
            [10.0, 10.0, 14.0, 20.0],
        ]
    )
    labels = torch.tensor([0.0, 0.0, 0.0, 1.0])
    ignore = torch.tensor([[9.0, 9.0, 15.0, 21.0]])
    result = ignore_negatives(boxes, labels, ignore)
    assert result.tolist() == [-1, -1, 0, 1]
    assert labels.tolist() == [0, 0, 0, 1]  # No mutation of upstream matcher labels.
    assert torch.equal(ignore_negatives(boxes, labels, torch.empty(0, 4)), labels)


def test_ignore_regions_resize_with_rgb_and_supported_boxes():
    transform = IgnoreTransform(64, 128, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    target = {
        "boxes": torch.tensor([[4.0, 2.0, 8.0, 6.0]]),
        "ignore_boxes": torch.tensor([[2.0, 1.0, 10.0, 7.0]]),
    }
    rgb, scaled = transform.resize(torch.zeros(3, 32, 64), target)
    assert rgb.shape == (3, 64, 128)
    assert torch.equal(scaled["boxes"], torch.tensor([[8.0, 4.0, 16.0, 12.0]]))
    assert torch.equal(scaled["ignore_boxes"], torch.tensor([[4.0, 2.0, 20.0, 14.0]]))
    _, inference = transform.resize(torch.zeros(3, 32, 64))
    assert inference is None  # Inference needs RGB only.


@pytest.fixture(scope="module")
def model():
    return make_signal_detector()


def test_both_detector_samplers_ignore_unknown_lights_and_keep_supported_positives(model):
    boxes = torch.tensor(
        [[10.0, 10.0, 14.0, 20.0], [50.0, 50.0, 54.0, 60.0], [80.0, 80.0, 84.0, 90.0]]
    )
    target = {
        "boxes": boxes[:1],
        "labels": torch.ones(1, dtype=torch.long),
        "ignore_boxes": boxes[:2],
    }
    labels, _ = model.rpn.assign_targets_to_anchors([boxes], [target])
    assert labels[0].tolist() == [1, -1, 0]
    proposals, _, sampled, _ = model.roi_heads.select_training_samples([boxes], [target])
    assert -1 not in sampled[0].tolist()
    assert (sampled[0] == 1).any() and (sampled[0] == 0).any()
    assert not any(torch.equal(p, boxes[1]) for p in proposals[0])
    assert not hasattr(model.roi_heads, "_ignore_regions")


def test_features_and_normalization_stay_frozen_with_only_rpn_and_predictor_trainable(model):
    detector_training(model)
    assert model.training and not model.backbone.training and not model.roi_heads.box_head.training
    assert all(
        name.startswith(("rpn.", "roi_heads.box_predictor."))
        for name, p in model.named_parameters()
        if p.requires_grad
    )
    assert model.rpn.anchor_generator.sizes == ((8,), (16,), (32,), (64,), (128,))
    assert model.rpn.anchor_generator.aspect_ratios == ((1.0, 2.0, 4.0),) * 5


def test_detector_sampler_never_reads_validation_and_privileged_data_import_is_blocked():
    data = SignalDetectionData.__new__(SignalDetectionData)
    data.base = type("Base", (), {"train": [0, 1, 2]})()
    data.positive_train = [1]
    rng = np.random.default_rng(123)
    for _ in range(100):
        a, b = data.sample_train(rng)
        assert a == 1 and b in (0, 1, 2)
    from src.navigation.policy_worker import PrivilegeGuard

    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("src.navigation.signal_detection_data")


def test_expanded_sampler_balances_domains_without_reading_validation():
    data = SignalDetectionData.__new__(SignalDetectionData)
    data.training_domains = ([0, 1, 2], [10, 11])
    data.positive_domains = ([1], [11])
    rng = np.random.default_rng(123)
    first, second = [], []
    for _ in range(1000):
        a, b = data.sample_train(rng)
        assert a in (1, 11) and b in (0, 1, 2, 10, 11)
        first.append(a == 11)
        second.append(b >= 10)
    assert 400 < sum(first) < 600 and 400 < sum(second) < 600


def test_append_refuses_validation_group_before_mutating_original_data(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import src.navigation.signal_detection_data as module
    from src.navigation.dagger import digest

    base = tmp_path / "base"
    extra = tmp_path / "extra"
    base.mkdir()
    extra.mkdir()
    (base / "data_audit.json").write_text("{}")
    (extra / "plan.json").write_text(
        json.dumps(dict(expansion=dict(base_audit_sha256=digest(base / "data_audit.json"))))
    )
    data = SignalDetectionData.__new__(SignalDetectionData)
    data.root = base
    data.base = SimpleNamespace(roles={("Town03_Opt", 1): "validation"}, images=np.zeros((1, 3)))
    original = data.base.images.copy()
    extra_data = SimpleNamespace(base=SimpleNamespace(roles={("Town03_Opt", 1): "train"}))
    monkeypatch.setattr(module, "SignalDetectionData", lambda *a, **kw: extra_data)
    with pytest.raises(ValueError, match="overlaps validation"):
        data.append_training_corpus(extra)
    assert np.array_equal(data.base.images, original)


def test_one_prediction_cannot_count_as_two_lights_and_matching_recovers_alternatives():
    from scripts.assess_navigation_signal_detector import match_boxes

    truth = [[0, 0, 4, 4], [2, 0, 6, 4]]
    shared = [1, 0, 5, 4]
    assert len(match_boxes(truth, [shared], 0.3)) == 1
    pairs = match_boxes(truth, [shared, [0, 0, 3, 4]], 0.3)
    assert len(pairs) == 2 and len({p for _, p in pairs}) == 2
    assert match_boxes(truth, [], 0.3) == []


def test_crop_expansions_retain_all_source_pixels_and_are_seed_reproducible():
    from scripts.train_navigation_signal_crops import expand_box

    first, second = np.random.default_rng(3), np.random.default_rng(3)
    for box in ([2, 4, 4, 8], [0, 0, 3, 5], [510, 283, 512, 288]):
        for _ in range(30):
            expanded = expand_box(box, first)
            assert expanded == expand_box(box, second)
            assert 0 <= expanded[0] <= box[0] < box[2] <= expanded[2] <= 512
            assert 0 <= expanded[1] <= box[1] < box[3] <= expanded[3] <= 288


def test_interrupted_validation_restores_all_training_rng_states():
    from scripts.train_navigation_signal_detector import BudgetStop, evaluate

    rng = np.random.default_rng(7)
    torch.manual_seed(88)
    expected_torch = torch.get_rng_state().clone()
    expected_numpy = rng.bit_generator.state.copy()
    dataset = type("Dataset", (), {"base": type("Base", (), {"validation": [0, 1]})()})()

    class Model:
        def train(self):
            return None

        backbone = type("Features", (), {"eval": lambda self: None})()
        roi_heads = type("Heads", (), {"box_head": backbone})()

    def stop():
        raise BudgetStop()

    with pytest.raises(BudgetStop):
        evaluate(Model(), dataset, None, rng, stop, lambda *args: None)
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert rng.bit_generator.state == expected_numpy


def test_mined_background_rejects_any_small_housing_inside_a_large_candidate():
    from scripts.mine_navigation_signal_negatives import outside_all_housings

    assert not outside_all_housings([0, 0, 100, 100], [[10, 10, 12, 14]])
    assert outside_all_housings([0, 0, 5, 5], [[10, 10, 12, 14]])
    assert outside_all_housings([0, 0, 5, 5], np.empty((0, 4)))


def test_phase_invariant_background_filter_rejects_controlled_bulb_changes():
    from scripts.mine_navigation_signal_negatives import phase_invariant

    images = {
        p: np.zeros((3, 8, 8), dtype=np.uint8) for p in ("red", "green", "yellow", "red_repeat")
    }
    assert phase_invariant(images, [0, 0, 8, 8])["passed"]
    images["green"][:, 2:6, 2:6] = 100
    assert not phase_invariant(images, [0, 0, 8, 8])["passed"]


def test_semantic_filter_preserves_other_signs_but_excludes_lights_and_changed_pixels():
    from scripts.verify_navigation_signal_negatives import semantic_crop_check

    tags = np.full((288, 512), 8, np.uint8)  # A sign is negative for LIGHT detection only.
    rgb = np.zeros((288, 512, 3), np.uint8)
    assert semantic_crop_check(tags, rgb, rgb, [10, 10, 15, 20])["approved"]
    tags[8, 8] = 7  # Two-pixel safety margin must catch the adjacent real light.
    assert not semantic_crop_check(tags, rgb, rgb, [10, 10, 15, 20])["approved"]
    tags.fill(3)
    assert not semantic_crop_check(tags, rgb, np.full_like(rgb, 100), [10, 10, 15, 20])["approved"]


def test_semantic_release_veto_is_checked_before_mutating_replay_targets(tmp_path):
    import json

    for name, data in (
        ("manifest.json", {}),
        ("request.json", {}),
        ("training_release.json", {"approved": False}),
    ):
        (tmp_path / name).write_text(json.dumps(data))
    dataset = SignalDetectionData.__new__(SignalDetectionData)
    dataset.targets = [{"sentinel": 42}]
    with pytest.raises(ValueError, match="Semantic training release"):
        dataset.attach_hard_negatives(tmp_path, "unused")
    assert dataset.targets == [{"sentinel": 42}]


def test_numeric_lens_metadata_does_not_use_attribute_display_string():
    from src.navigation.signal_label_probe import lens_values

    class Attribute:
        def as_float(self):
            return -1.0

        def __str__(self):
            return "ActorAttribute(id=lens_k,type=float,value=-1)"

    class Blueprint:
        def has_attribute(self, key):
            return key == "lens_k"

        def get_attribute(self, key):
            return Attribute()

    assert lens_values(Blueprint()) == {"lens_k": -1.0}
    from src.navigation.policy_worker import PrivilegeGuard

    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("src.navigation.signal_label_probe")


def test_explicit_roi_replay_retains_positive_labels_and_zero_gt_regression_targets(model):
    proposals = [torch.tensor([[10.0, 10.0, 14.0, 20.0], [80.0, 80.0, 84.0, 90.0]])]
    target = {
        "boxes": proposals[0][:1].clone(),
        "labels": torch.ones(1, dtype=torch.long),
        "ignore_boxes": proposals[0][:1].clone(),
        "hard_negative_boxes": torch.tensor([[40.0, 40.0, 45.0, 50.0]]),
    }
    output, _, labels, regression = model.roi_heads.select_training_samples(proposals, [target])
    assert len(output[0]) == 11  # Three ordinary sampled ROIs + eight explicit replay ROIs.
    assert labels[0][-8:].tolist().count(0) == labels[0][-8:].tolist().count(1) == 4
    assert not regression[0][-8:].any()


def test_train_only_bias_is_minimal_for_parent_acceptance_and_zero_if_unneeded():
    from scripts.calibrate_navigation_signal_head import train_recall_bias

    parent, candidate = np.array([1.0, 2.0, -1.0, -2.0]), np.array([-2.0, -3.0, -4.0, -5.0])
    bias = train_recall_bias(parent, candidate)
    assert ((candidate + bias) > 0).sum() == (parent > 0).sum()
    assert ((candidate + bias - 0.0001) > 0).sum() < (parent > 0).sum()
    assert train_recall_bias(parent, parent + 1) == 0
    with pytest.raises(ValueError, match="finite TRAIN"):
        train_recall_bias(parent, [1.0, 2.0, float("nan"), 3.0])
