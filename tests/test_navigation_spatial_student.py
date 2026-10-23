"""Tiny-pixel retention, causal episode context and actual learned control boundary."""

import copy

import pytest
import torch

from src.navigation.grid_encoders import FrozenGridEncoder
from src.navigation.optical_fields import optical_fields, past_average
from src.navigation.policy_worker import PrivilegeGuard
from src.navigation.spatial_data import study_manifest, validation_positions
from src.navigation.spatial_student import SpatialCameraStudent, driving_loss


def test_fixed_fields_retain_four_saturated_pixels_and_declare_no_rule():
    image = torch.zeros(3, 3, 288, 512, dtype=torch.uint8)
    image[0, 0, 34:36, 66:68] = 255
    fields = optical_fields(image)
    assert fields.shape == (3, 144, 11)
    cell = 1 * 16 + 2
    assert fields[0, cell, 3] == 1 and fields[0, cell, 8] == pytest.approx(4 / 1024)
    assert fields[0, :, 3].count_nonzero() == 1 and not fields[1:].any()
    with pytest.raises(ValueError, match="Native"):
        optical_fields(image.float())


def test_past_average_matches_literal_past_clips_and_cannot_see_future_gradients():
    x = torch.randn(12, 3, 16, 4, requires_grad=True)
    averaged = past_average(x)
    expected = torch.stack(
        [x[[max(0, t - i) for i in range(7, -1, -1)]].mean(0) for t in range(12)]
    )
    assert torch.allclose(averaged, expected, atol=1e-6)
    averaged[3].sum().backward()
    assert x.grad[:4].abs().sum() > 0 and not x.grad[4:].any()


def inputs(steps=5):
    return dict(
        features=torch.randn(1, steps, 3, 16, 24),
        optical=torch.randn(1, steps, 3, 144, 11),
        navigation=torch.randn(1, steps, 16, 2),
        ego=torch.randn(1, steps, 3),
    )


def test_spatial_student_streaming_is_causal_and_stops_cannot_request_speed():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = SpatialCameraStudent(encoder_dim=24, hidden=32).eval()
    x = inputs()
    all_output, _ = model(**x)
    hidden = None
    pieces = []
    for i in range(5):
        output, hidden = model(**{k: v[:, i : i + 1] for k, v in x.items()}, hidden=hidden)
        pieces.append(output)
    for k, v in all_output.items():
        assert torch.allclose(v, torch.cat([p[k] for p in pieces], 1), atol=3e-6), k
    changed = {k: v.clone() for k, v in x.items()}
    changed["optical"][:, 4] += 100
    altered, _ = model(**changed)
    assert torch.equal(all_output["current_speed"][:, :4], altered["current_speed"][:, :4])
    assert not ((all_output["current_speed"] > 0) & (all_output["brake"] >= 0)).any()


def test_driving_loss_cannot_consume_unselected_future_or_unverified_perception_labels():
    torch.manual_seed(12)
    model = SpatialCameraStudent(encoder_dim=24, hidden=32)
    p, _ = model(**inputs(12))
    t = dict(
        current_speed=torch.ones(1, 12),
        speeds=torch.ones(1, 12, 8),
        speed_mask=torch.zeros(1, 12, 8),
        correction=torch.zeros(1, 12, 8),
        correction_mask=torch.ones(1, 12, 8),
        brake=torch.zeros(1, 12),
    )
    a, _ = driving_loss(p, t)
    changed = copy.deepcopy(t)
    changed["speeds"] -= 100
    b, _ = driving_loss(p, changed)
    assert torch.equal(a, b)
    a.backward()
    assert torch.isfinite(model.stop.weight.grad).all()
    assert "light" not in p and "distance" not in p


def test_weights_reject_bad_hash_before_unpickling_and_policy_forbids_training_data(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not weights")
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenGridEncoder(dict(kind="imagenet_resnet18", checkpoint=str(path), sha256="wrong"))
    for module in (
        "src.navigation.spatial_data",
        "src.navigation.label_budget",
        "src.navigation.jepa_pretraining",
        "src.navigation.rule_prefix",
        "src.navigation.rule_suite",
        "src.navigation.trace_storage",
    ):
        with pytest.raises(ImportError, match="Forbidden"):
            PrivilegeGuard().find_spec(module)


def test_actual_study_labels_and_validation_positions_preserve_registered_split():
    m = study_manifest()
    assert len([e for e in m["episodes"] if e["role"] == "train"]) == 59
    assert (
        sum(
            validation_positions(e["end"] - e["start"], "cpu").sum().item()
            for e in m["episodes"]
            if e["role"] == "validation"
        )
        == 1312
    )
    selected = set()
    for a in m["annotation_budget"]["allocations"]:
        assert selected <= set(a["groups"]) and a["actual_fraction"] <= a["nominal_fraction"]
        selected = set(a["groups"])
    assert m["annotation_budget"]["total_labels"] == 16639


@pytest.mark.parametrize("architecture", ["spatial-stop-go-v1", "spatial-visual-decision-v2"])
def test_spatial_export_runs_in_actual_privilege_guarded_worker(tmp_path, architecture):
    import json
    import struct
    import subprocess
    import sys

    import numpy as np

    from src.navigation.contracts import VIEWS, CameraCalibration, SensorPacket
    from src.navigation.policy_ipc import encode_packet

    model = SpatialCameraStudent(hidden=32, scratch=True)
    path = tmp_path / "spatial.pt"
    torch.save(
        dict(
            architecture=architecture,
            encoder_kind="cnn_scratch",
            encoder_dim=384,
            hidden=32,
            color_enabled=True,
            model=model.state_dict(),
            step=0,
        ),
        path,
    )
    calibration = CameraCalibration(512, 288, 100, (1.3, 0, 2.3), (0, 0, 0))
    packet = SensorPacket(
        0,
        0.0,
        {v: np.zeros((288, 512, 3), np.uint8) for v in VIEWS},
        dict.fromkeys(VIEWS, calibration),
        np.zeros((16, 2), np.float32),
        np.zeros(3, np.float32),
    )
    payload = encode_packet(packet)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.navigation.policy_worker",
            "--checkpoint",
            str(path),
            "--device",
            "cpu",
        ],
        input=struct.pack("<Q", len(payload)) + payload + struct.pack("<Q", 0),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    ready, prediction = map(json.loads, result.stdout.splitlines())
    assert ready["status"] == "ready" and prediction["interventions"] == []
    assert (
        prediction["auxiliary_perception_available"] is False
        and prediction["light_probabilities"] is None
    )
    assert len(prediction["control"]) == 3 and not prediction["rule_overrides_enabled"]


def test_shared_head_initialization_and_label_pool_are_identical_across_encoder_arms():
    from scripts.train_navigation_spatial_student import initialize, selected_pool

    a = initialize("imagenet_resnet18", 512, 42)
    b = initialize("dinov2_small", 384, 42)
    for key, value in a.state_dict().items():
        if not key.startswith("low."):
            assert torch.equal(value, b.state_dict()[key]), key
    m = study_manifest()
    pool, allocation = selected_pool(m, 0.05)
    assert all(e["role"] == "train" for e in pool)
    assert {e["id"] for e in pool} == set(allocation["groups"])
