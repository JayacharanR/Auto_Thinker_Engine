import torch

from src.navigation.fact_student import FactStudent, fact_loss
from src.navigation.spatial_student import driving_loss


def test_fact_targets_train_camera_attention_without_entering_forward_inputs():
    torch.set_num_threads(2)
    model = FactStudent(hidden=32)
    inputs = dict(
        features=torch.randn(1, 12, 3, 16, 384),
        optical=torch.randn(1, 12, 3, 144, 11),
        navigation=torch.randn(1, 12, 16, 2),
        ego=torch.randn(1, 12, 3),
    )
    prediction, hidden = model(**inputs)
    targets = dict(
        correction=torch.zeros(1, 12, 8),
        correction_mask=torch.ones(1, 12, 8),
        current_speed=torch.ones(1, 12),
        speeds=torch.ones(1, 12, 8),
        speed_mask=torch.ones(1, 12, 8),
        brake=torch.zeros(1, 12),
        light=torch.ones(1, 12, dtype=torch.long),
        presence=torch.ones(1, 12, 2),
        distance=torch.ones(1, 12, 2) * 10,
        hazards=torch.zeros(1, 12, 4),
    )
    loss, parts = fact_loss(prediction, targets)
    loss.backward()
    assert torch.isfinite(loss) and model._recurrent_sequence is None
    assert model.attention.in_proj_weight.grad.abs().sum() > 0
    assert model.facts.light.weight.grad.abs().sum() > 0
    assert prediction["light"].shape == (1, 12, 5) and hidden.shape == (1, 1, 32)
    assert parts["fact_distance"] >= 0


def test_paired_candidate_initialization_preserves_every_fresh_driving_parameter():
    from scripts.train_navigation_fact_student import initialize_candidate

    for encoder, dimension in (("cnn_scratch", 384), ("imagenet_resnet18", 512)):
        plain = initialize_candidate(encoder, dimension, 42, False)
        facts = initialize_candidate(encoder, dimension, 42, True)
        assert all(
            torch.equal(value, facts.state_dict()[name])
            for name, value in plain.state_dict().items()
        )
        assert set(facts.state_dict()) - set(plain.state_dict()) == {
            f"facts.{head}.{part}"
            for head in ("light", "presence", "distance", "hazards")
            for part in ("weight", "bias")
        }


def test_cold_start_supervision_reaches_initial_stationary_go_observation():
    # Only the first observation has a speed error: burn-in8 masks exactly the
    # deployment-start mistake that the paired prospective study is targeting.
    targets = dict(
        correction=torch.zeros(1, 12, 8),
        correction_mask=torch.ones(1, 12, 8),
        current_speed=torch.ones(1, 12),
        speeds=torch.ones(1, 12, 8),
        speed_mask=torch.ones(1, 12, 8),
        brake=torch.zeros(1, 12),
    )
    prediction = {
        key: value.clone().requires_grad_()
        for key, value in targets.items()
        if key not in ("correction_mask", "speed_mask")
    }
    prediction["cruise_speeds"] = torch.ones(1, 12, 9, requires_grad=True)
    prediction["stop_logits"] = torch.zeros(1, 12, 9, requires_grad=True)
    prediction["comfort_brake"] = torch.zeros(1, 12, requires_grad=True)
    with torch.no_grad():
        prediction["cruise_speeds"][0, 0, 0] = 0
    loss, _ = driving_loss(prediction, targets, burn_in=0)
    loss.backward()
    assert prediction["cruise_speeds"].grad[0, 0, 0] < 0
    prediction["cruise_speeds"].grad.zero_()
    old_loss, _ = driving_loss(prediction, targets, burn_in=8)
    old_loss.backward()
    assert prediction["cruise_speeds"].grad[0, 0, 0] == 0


def test_fact_checkpoint_runs_in_privilege_guarded_worker_with_overrides_off(tmp_path):
    import json
    import struct
    import subprocess
    import sys

    import numpy as np

    from scripts.run_navigation import checkpoint_metadata
    from src.navigation.contracts import VIEWS, CameraCalibration, SensorPacket
    from src.navigation.policy_ipc import encode_packet

    model = FactStudent(hidden=32, scratch=True)
    checkpoint = tmp_path / "facts.pt"
    torch.save(
        dict(
            architecture="spatial-visual-facts-v3",
            encoder_kind="cnn_scratch",
            encoder_dim=384,
            hidden=32,
            color_enabled=True,
            model=model.state_dict(),
            fact_supervision=True,
            burn_in=0,
            step=0,
        ),
        checkpoint,
    )
    metadata = checkpoint_metadata(checkpoint)
    assert metadata["fact_supervision"] and not metadata["neural_ego_speed_enabled"]
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
            str(checkpoint),
            "--device",
            "cpu",
        ],
        input=struct.pack("<Q", len(payload)) + payload + struct.pack("<Q", 0),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    ready, prediction = map(json.loads, result.stdout.splitlines())
    assert ready["status"] == "ready"
    assert prediction["auxiliary_perception_available"]
    assert prediction["auxiliary_predictions_used_for_control"] is False
    assert prediction["rule_overrides_enabled"] is False and prediction["interventions"] == []
    assert len(prediction["light_probabilities"]) == 5 and len(prediction["control"]) == 3
