from pathlib import Path

import pytest
import torch

from src.navigation.signal_proposer_features import (
    FrozenBulbFeatures,
    point_features,
    verify_ordered_dataset,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_identity_accepts_json_tuple_encoding_but_rejects_order_role_and_point_changes():
    import copy
    import json

    samples = [dict(role="train", points=[(1, 2, 3)]), dict(role="validation", points=[])]
    saved = json.loads(json.dumps(dict(samples=samples, train=[0], validation=[1])))
    verify_ordered_dataset(samples, [0], [1], saved)
    with pytest.raises(ValueError, match="identity"):
        verify_ordered_dataset(samples[::-1], [0], [1], saved)
    with pytest.raises(ValueError, match="identity"):
        verify_ordered_dataset(samples, [1], [0], saved)
    altered = copy.deepcopy(samples)
    altered[0]["points"] = [(2, 1, 3)]
    with pytest.raises(ValueError, match="identity"):
        verify_ordered_dataset(altered, [0], [1], saved)


def test_point_gather_preserves_xy_order_edges_duplicates_and_every_point():
    f = torch.arange(32 * 3 * 5).reshape(32, 3, 5).float()
    p = torch.arange(4 * 3 * 5).reshape(4, 3, 5).float()
    points = torch.tensor([[4, 2], [0, 0], [4, 2], [1, 2]])
    result = point_features(f, p, points)
    assert result.shape == (4, 36)
    for n, (x, y) in enumerate(points):
        torch.testing.assert_close(result[n], torch.cat([f[:, y, x], p[:, y, x]]))
    assert point_features(f, p, torch.empty(0, 2, dtype=torch.long)).shape == (0, 36)
    with pytest.raises(ValueError, match="outside"):
        point_features(f, p, torch.tensor([[5, 0]]))
    with pytest.raises(ValueError, match="Integer XY"):
        point_features(f, p, points.float())


def test_real_frozen_representation_matches_original_full_image_classifier():
    torch.set_num_threads(2)
    checkpoint = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb/step1000.pt"
    model = FrozenBulbFeatures(checkpoint)
    model.train(True)
    assert not any(m.training for m in model.modules())
    assert not any(p.requires_grad for p in model.parameters())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    # Deterministic spatial pattern exercises image edges and spatial variation.
    rgb = (torch.arange(3 * 288 * 512).reshape(1, 3, 288, 512) % 256).to(torch.uint8)
    with torch.inference_mode():
        expected = model.model(rgb).float().softmax(1)
        features, probabilities = model(rgb)
    assert torch.equal(expected, probabilities)
    assert features.shape == (1, 32, 288, 512)
    assert torch.isfinite(features).all() and torch.isfinite(probabilities).all()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    with pytest.raises(ValueError, match="Only native"):
        model(rgb.float())
    with pytest.raises(ValueError, match="Only native"):
        model(torch.zeros(1, 4, 288, 512, dtype=torch.uint8))


def test_checkpoint_identity_is_verified_before_loading(tmp_path):
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not an approved checkpoint")
    with pytest.raises(ValueError, match="Exact frozen"):
        FrozenBulbFeatures(wrong)
