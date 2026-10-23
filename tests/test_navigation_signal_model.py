"""Spatial phase learning: equivariance, ignored labels and protected geography."""

import json

import numpy as np
import pytest
import torch

from src.navigation.signal_learning import SignalDataset, junction_roles, phase_targets
from src.navigation.signal_model import SignalGrid, signal_loss


def test_localizer_has_no_absolute_position_prior_and_preserves_two_pixel_translation():
    torch.manual_seed(3)
    model = SignalGrid().eval()
    image = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    with torch.no_grad():
        first = model(image)
        moved = model(image.roll(4, -1)).roll(-2, -1)
    assert first.shape == (1, 4, 32, 32)
    torch.testing.assert_close(first[..., 8:-8, 8:-8], moved[..., 8:-8, 8:-8])
    with pytest.raises(ValueError, match="Even-sized"):
        model(torch.zeros(1, 3, 63, 64))


def test_ignored_oracle_boxes_have_no_training_gradient():
    prediction = torch.randn(1, 4, 4, 4, requires_grad=True)
    targets = torch.full((1, 4, 4), 255, dtype=torch.long)
    targets[0, 0, 0], targets[0, 1, 1] = 0, 1
    loss = signal_loss(prediction, targets)
    loss.backward()
    assert prediction.grad[0, :, 0, 0].abs().sum() > 0
    assert prediction.grad[0, :, 1, 1].abs().sum() > 0
    assert prediction.grad[0, :, 3, 3].abs().sum() == 0


def test_failed_static_gate_cannot_supply_training_samples(tmp_path):
    (tmp_path / "data_audit.json").write_text(
        json.dumps({"static_label_data_gate_passed": False, "all_cases_complete": True})
    )
    with pytest.raises(ValueError, match="label-data gate"):
        SignalDataset(tmp_path)


def test_all_poses_views_and_phases_of_a_junction_stay_in_the_same_split():
    cases = [
        {"town": "Town03_Opt", "junction_id": i, "phase": phase}
        for i in range(17)
        for phase in ("red", "green", "yellow")
    ]
    roles = junction_roles(cases)
    assert roles == junction_roles(list(reversed(cases)))
    assert sum(role == "validation" for role in roles.values()) == 4
    assert len(roles) == 17
    with pytest.raises(ValueError, match="Town05"):
        junction_roles(cases + [{"town": "Town05_Opt", "junction_id": 7}])


def test_crop_sampler_preserves_foreground_and_never_reads_a_validation_image():
    dataset = SignalDataset.__new__(SignalDataset)
    dataset.images = np.zeros((4, 3, 288, 512), np.uint8)
    dataset.images[3] = 255  # A sentinel validation image.
    dataset.targets = np.zeros((4, 144, 256), np.uint8)
    dataset.train = [0, 1, 2]
    dataset.positive = {label: [(label - 1, 128, 72)] for label in (1, 2, 3)}
    for label in (1, 2, 3):
        dataset.targets[label - 1, 72, 128] = label
    dataset.training_targets = dataset.targets.copy()
    images, targets = dataset.training_batch(np.random.default_rng(4), 16, "cpu")
    assert images.shape == (16, 3, 128, 128) and not images.any()
    assert targets.shape == (16, 64, 64)
    assert all((targets[index] > 0).any() for index in range(0, 16, 2))


def test_resume_refuses_changed_archived_worker_before_mutating_any_town(tmp_path):
    from scripts.resume_navigation_signal_pairs import ROOT, verify_worker_sources
    from src.navigation.dagger import digest

    relative = "src/navigation/contracts.py"
    archived = tmp_path / "source" / relative
    archived.parent.mkdir(parents=True)
    archived.write_bytes((ROOT / relative).read_bytes())
    (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "request.json").write_text(
        json.dumps(
            {
                "plan_sha256": digest(tmp_path / "plan.json"),
                "source_hashes": {relative: digest(archived)},
            }
        )
    )
    verify_worker_sources(tmp_path)
    archived.write_bytes(archived.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="mixed-code"):
        verify_worker_sources(tmp_path)
    assert not (tmp_path / "resume_attempts").exists()


def test_region_training_preserves_every_evaluation_center_and_background_mask():
    ignore = np.zeros((144, 256), np.uint8)
    ignore[2:18, 2:14] = 255
    labels = [{"crop_xyxy": [10, 10, 16, 22], "yellow_color_supported": True}]
    a, train_a, points_a = phase_targets(ignore, labels, "red", 1, "center")
    b, train_b, points_b = phase_targets(ignore, labels, "red", 1, "region")
    assert np.array_equal(a, b) and points_a == points_b
    assert np.array_equal(a == 0, b == 0)
    assert (train_b == 1).sum() > (train_a == 1).sum()
    for cx, cy, label in points_a:
        assert train_a[cy, cx] == train_b[cy, cx] == label
    from src.navigation.policy_worker import PrivilegeGuard

    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("src.navigation.signal_learning")
