"""Causal inference, location separation, target masks and complete recovery."""

import json

import numpy as np
import pytest
import torch

from scripts.train_navigation_student import atomic_checkpoint, restore_rng, rng_state
from src.navigation.learning_data import (
    correction_targets,
    future_speed_targets,
    prepare_cache,
    split_episode_locations,
)
from src.navigation.student import CameraStopGoStudent, CameraStudent, make_student, student_loss


def test_shared_junctions_never_cross_internal_split():
    episodes = [
        dict(town="TownA", partition="train", junction_ids=[j], family=str(f))
        for j in range(12)
        for f in range(6)
    ]
    episodes.append(dict(town="TownA", partition="train", junction_ids=[0, 1], family="0"))
    roles, metadata = split_episode_locations(episodes)
    locations = {
        r: {
            (e["town"], j)
            for e, role in zip(episodes, roles)
            if role == r
            for j in e["junction_ids"]
        }
        for r in ("train", "validation")
    }
    assert not locations["train"] & locations["validation"]
    assert split_episode_locations([{**e, "loss": 999} for e in episodes]) == (roles, metadata)
    with pytest.raises(ValueError, match="training locations"):
        split_episode_locations([{**episodes[0], "town": "Town05_Opt"}])


def test_future_speed_labels_mask_end_without_wrapping_or_crossing_episodes():
    speeds, mask = future_speed_targets(np.arange(12))
    assert speeds[0, :2].tolist() == [5, 10]
    assert mask[0].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert not mask[-1].any()


def test_correction_is_relative_to_route_and_masks_unseen_teacher_horizon():
    nav = np.column_stack([np.arange(2, 33, 2), np.zeros(16)])
    teacher = [[2, 0], [4, 1], [8, 1], [12, 1]]
    correction, mask = correction_targets(nav, teacher)
    assert 0.8 < correction[0] < 1.1 and mask[:3].all()
    assert not mask[3:].any()


@pytest.mark.parametrize("model_type", [CameraStudent, CameraStopGoStudent])
def test_cnn_memory_cannot_use_future_frames(model_type):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = model_type(hidden=32).eval()
    inputs = dict(
        rgb=torch.randint(256, (1, 4, 3, 3, 48, 64), dtype=torch.uint8),
        front=torch.randint(256, (1, 4, 3, 96, 128), dtype=torch.uint8),
        navigation=torch.randn(1, 4, 16, 2),
        ego=torch.randn(1, 4, 3),
    )
    with torch.no_grad():
        original, _ = model(**inputs)
        altered = {k: v.clone() for k, v in inputs.items()}
        for value in altered.values():
            value[:, 2:] = 0
        changed, _ = model(**altered)
        for key in original:
            assert torch.allclose(original[key][:, :2], changed[key][:, :2], atol=1e-6)
        # Streaming and whole-clip inference agree, including the recurrent state.
        first, hidden = model(**{k: v[:, :2] for k, v in inputs.items()})
        second, _ = model(**{k: v[:, 2:] for k, v in inputs.items()}, hidden=hidden)
        assert torch.allclose(
            torch.cat([first["current_speed"], second["current_speed"]], 1),
            original["current_speed"],
            atol=1e-6,
        )


def test_checkpoint_restores_next_sampler_and_optimizer_update(tmp_path):
    torch.manual_seed(1)
    rng = np.random.default_rng(3)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters())

    def update():
        draw = rng.integers(100)
        optimizer.zero_grad()
        loss = model(torch.randn(2, 3)).square().mean()
        loss.backward()
        optimizer.step()
        return draw, {k: v.clone() for k, v in model.state_dict().items()}

    update()
    checkpoint = tmp_path / "latest.pt"
    atomic_checkpoint(
        checkpoint,
        dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(rng)),
    )
    expected_draw, expected = update()
    state = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"], rng)
    actual_draw, actual = update()
    assert actual_draw == expected_draw
    assert all(torch.equal(actual[k], expected[k]) for k in actual)
    assert not checkpoint.with_suffix(".part").exists()


def test_learning_cache_refuses_superseded_collection(tmp_path):
    root = tmp_path / "collection"
    root.mkdir()
    (root / "data_audit.json").write_text(json.dumps({"audit_passed": True, "status": "complete"}))
    (root / "collection_review.json").write_text(json.dumps({"status": "superseded_do_not_train"}))
    with pytest.raises(ValueError, match="excluded"):
        prepare_cache(root, tmp_path / "cache", tmp_path / "STOP", 0, lambda *args: None)


def test_policy_wire_rejects_extra_privileged_fields():
    from src.navigation.policy_ipc import decode_packet, encode_packet
    from tests.test_navigation_collection import packet

    original = packet()
    decoded = decode_packet(encode_packet(original))
    assert np.array_equal(decoded.rgb["left"], original.rgb["left"])
    object.__setattr__(original, "teacher_hazards", {"red": True})
    with pytest.raises(ValueError, match="exact SensorPacket"):
        encode_packet(original)


def test_route_controller_turns_right_and_holds_stop_without_creeping():
    from src.navigation.controller import track

    route = np.column_stack([np.arange(2, 33, 2), np.arange(2, 33, 2) * 0.2])
    control = track(route, np.zeros(8), 0, 4, 0)
    assert control.steer > 0 and control.throttle > 0 and control.brake == 0
    stopped = track(route, np.zeros(8), 0, 0, 0)
    assert stopped.throttle == 0 and stopped.brake == 1
    with pytest.raises(ValueError):
        track(route, np.zeros(8), 0, float("nan"), 0)


@pytest.mark.parametrize("architecture", ["camera-student-v1", "camera-stop-go-v2"])
def test_fresh_policy_process_runs_without_simulator_or_labels(tmp_path, architecture):
    from dataclasses import replace

    from src.navigation.policy_ipc import PolicyClient
    from src.navigation.policy_worker import PrivilegeGuard
    from tests.test_navigation_collection import packet

    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("carla")
    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("src.navigation.learning_data")
    model = make_student(architecture, hidden=32)
    checkpoint = tmp_path / "policy.pt"
    atomic_checkpoint(
        checkpoint, {"model": model.state_dict(), "hidden": 32, "architecture": architecture}
    )
    client = PolicyClient(checkpoint, tmp_path / "stderr.log", device="cpu")
    try:
        example = replace(
            packet(), navigation_waypoints=np.column_stack([np.arange(2, 33, 2), np.zeros(16)])
        )
        response = client.step(example)
        assert response["frame"] == 10 and not response["rule_overrides_enabled"]
        assert len(response["control"]) == 3 and response["interventions"] == []
    finally:
        client.close()
    assert client.process.returncode == 0


def test_semantic_stop_controls_speed_and_brake_with_trainable_release():
    model = CameraStopGoStudent(hidden=32)
    with torch.no_grad():
        model.stop.weight.zero_()
        model.stop.bias.fill_(1)
    features = torch.randn(1, 4, 32)
    stopped = model.decode(features)
    assert not stopped["current_speed"].any() and not stopped["speeds"].any()
    assert (stopped["brake"] >= 0).all()
    with torch.no_grad():
        model.stop.bias.fill_(-1)
    released = model.decode(features)
    assert (released["current_speed"] > 0).all() and (released["brake"] < 0).all()
    targets = {
        "current_speed": torch.tensor([[0.0, 0.0, 0.0, 5.0]]),
        "speeds": torch.zeros(1, 4, 8),
        "speed_mask": torch.zeros(1, 4, 8),
        "correction": torch.zeros(1, 4, 8),
        "correction_mask": torch.zeros(1, 4, 8),
        "brake": torch.ones(1, 4),
        "light": torch.ones(1, 4, dtype=torch.long),
        "hazards": torch.zeros(1, 4, 4),
        "distance": torch.zeros(1, 4, 2),
        "presence": torch.zeros(1, 4, 2),
    }
    loss, parts = student_loss(released, targets)
    assert torch.isfinite(loss) and parts["future_speed"] == 0 and parts["future_stop"] == 0
    assert parts["light"] == 0  # known lamp outside the audited distance range
    released["cruise_speeds"].retain_grad()
    released["stop_logits"].retain_grad()
    loss.backward()
    # Hard inference gating must not cut off learning of either decision.
    assert released["stop_logits"].grad[0, 2, 0] < 0  # stopped label learns to hold
    assert released["stop_logits"].grad[0, 3, 0] > 0  # go label learns to release
    assert released["cruise_speeds"].grad[0, 2, 0] == 0
    assert released["cruise_speeds"].grad[0, 3, 0] != 0
    assert not released["cruise_speeds"].grad[..., 1:].any()


def test_evaluation_metadata_uses_exported_model_variant(tmp_path):
    from scripts.run_navigation import checkpoint_metadata

    path = tmp_path / "policy.pt"
    for architecture in ("camera-student-v1", "camera-stop-go-v2"):
        model = make_student(architecture, hidden=32)
        atomic_checkpoint(
            path, dict(model=model.state_dict(), architecture=architecture, hidden=32, step=123)
        )
        metadata = checkpoint_metadata(path)
        assert metadata["architecture"] == architecture
        assert metadata["selected_training_step"] == 123 and metadata["hidden"] == 32
    atomic_checkpoint(path, dict(architecture="unknown", hidden=32))
    with pytest.raises(ValueError, match="Unsupported"):
        checkpoint_metadata(path)
