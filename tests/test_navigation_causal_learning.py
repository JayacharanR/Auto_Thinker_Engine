"""Whole-episode gradient parity and fixed-burn-in/label-boundary protections."""

import copy

import pytest
import torch

from scripts.train_navigation_causal_student import causal_validation, passes
from src.navigation.causal_learning import original_training_episodes, stream_forward
from src.navigation.policy_worker import PrivilegeGuard
from src.navigation.student import CameraStopGoStudent, student_loss


def test_checkpointed_visual_chunks_preserve_complete_history_and_gradients():
    torch.manual_seed(1)
    torch.set_num_threads(2)
    original = CameraStopGoStudent(hidden=32)
    chunked = copy.deepcopy(original)
    inputs = dict(
        rgb=torch.randint(256, (1, 6, 3, 3, 48, 64), dtype=torch.uint8),
        front=torch.randint(256, (1, 6, 3, 96, 128), dtype=torch.uint8),
        navigation=torch.randn(1, 6, 16, 2),
        ego=torch.randn(1, 6, 3),
    )
    full, _ = original(**inputs)
    streamed = stream_forward(chunked, inputs, chunk=2, gradients=True)
    for k in full:
        assert torch.allclose(full[k], streamed[k], atol=2e-5), k
    full["cruise_speeds"].sum().backward()
    streamed["cruise_speeds"].sum().backward()
    for (name, a), (_, b) in zip(original.named_parameters(), chunked.named_parameters()):
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=2e-4, rtol=2e-4), name


def test_fixed_burnin_and_unapproved_supplement_perception_do_not_create_false_labels():
    torch.manual_seed(2)
    model = CameraStopGoStudent(hidden=32)
    inputs = dict(
        rgb=torch.randint(256, (1, 6, 3, 3, 48, 64), dtype=torch.uint8),
        front=torch.randint(256, (1, 6, 3, 96, 128), dtype=torch.uint8),
        navigation=torch.randn(1, 6, 16, 2),
        ego=torch.randn(1, 6, 3),
    )
    p, _ = model(**inputs)
    targets = dict(
        correction=torch.zeros(1, 6, 8),
        correction_mask=torch.ones(1, 6, 8),
        current_speed=torch.ones(1, 6) * 4,
        speeds=torch.ones(1, 6, 8) * 4,
        speed_mask=torch.ones(1, 6, 8),
        brake=torch.zeros(1, 6),
        light=torch.zeros(1, 6, dtype=torch.long),
        hazards=torch.zeros(1, 6, 4),
        distance=torch.zeros(1, 6, 2),
        presence=torch.zeros(1, 6, 2),
    )
    default, _ = student_loss(p, targets)
    explicit, _ = student_loss(p, targets, burn_in=3)
    assert torch.equal(default, explicit)
    loss, parts = student_loss(p, targets, burn_in=1, perception_supervision=False)
    assert not {"light", "distance", "presence", "hazards"} & parts.keys()
    loss.backward()
    assert model.light.weight.grad is None and model.speed.weight.grad is not None
    with pytest.raises(ValueError, match="Burn-in"):
        student_loss(p, targets, burn_in=6)


def test_original_sampling_filter_preserves_validation_and_excludes_component_locations():
    entries = [dict(role="train", family=str(i), town="Town03", junction_ids=[i]) for i in range(6)]
    entries += [
        dict(role="train", family="0", town="Town03", junction_ids=[655]),
        dict(role="validation", family="0", town="Town03", junction_ids=[777]),
    ]
    original = copy.deepcopy(entries)
    good, bad = original_training_episodes(dict(episodes=entries), {"Town03": list(range(6))})
    assert len(good) == 6 and len(bad) == 1 and entries == original
    with pytest.raises(ImportError):
        PrivilegeGuard().find_spec("src.navigation.causal_learning")


def test_prospective_selector_requires_same_labels_and_protects_both_history_modes():
    parent = dict(
        causal=dict(
            controller_stop_recall=0.5,
            controller_false_hold_fraction=0.05,
            go_speed_mae_mps=2,
            selection_score=0.315,
            counts=dict(speed=100),
        ),
        clipped=dict(controller_stop_recall=0.425, controller_false_hold_fraction=0.048),
    )
    good = copy.deepcopy(parent)
    good["causal"].update(controller_stop_recall=0.6, selection_score=0.265)
    assert passes(good, parent)
    for field, value in [
        ("controller_stop_recall", 0.49),
        ("controller_false_hold_fraction", 0.08),
        ("go_speed_mae_mps", 2.3),
        ("counts", dict(speed=99)),
    ]:
        bad = copy.deepcopy(good)
        bad["causal"][field] = value
        assert not passes(bad, parent)
    for field, value in [("controller_stop_recall", 0.4), ("controller_false_hold_fraction", 0.08)]:
        bad = copy.deepcopy(good)
        bad["clipped"][field] = value
        assert not passes(bad, parent)


def test_causal_validation_uses_original_label_positions_after_full_prefix_history():
    from types import SimpleNamespace

    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, rgb, front, navigation, ego, hidden=None):
            n = rgb.shape[1]
            start = 0 if hidden is None else hidden
            values = start + torch.arange(1, n + 1, dtype=torch.float32)[None] + self.anchor * 0
            return dict(current_speed=values, brake=torch.full_like(values, -1)), values[:, -1:]

    inputs = {k: torch.zeros(1, 64, 1) for k in ["rgb", "front", "navigation", "ego"]}
    targets = dict(current_speed=torch.full((1, 64), 2.0), brake=torch.zeros(1, 64))
    cache = SimpleNamespace(
        manifest=dict(episodes=[dict(role="validation", start=0, end=64)]),
        episode=lambda e, d: (inputs, targets),
    )
    result = causal_validation(CountingModel(), cache, lambda: None)
    assert result["counts"]["speed"] == 32
    # Original positions16..31 and48..63 see the full episode prefix, not a reset.
    assert result["speed_mae_mps"] == pytest.approx(38.5)
