"""Causal RGB packing, matched null ablation and protected parent behavior."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.cache_navigation_signal_policy import selected_episodes
from scripts.train_navigation_signal_policy import passes, stop_loss, validation_mask
from src.navigation.contracts import VIEWS, CameraCalibration, SensorPacket
from src.navigation.signal_policy import (
    MAX_OBJECTS,
    TOKEN_DIM,
    CausalSignalFacts,
    SignalResidualStudent,
    StopResidual,
    pack_observations,
)
from src.navigation.student import CameraStopGoStudent, make_student


def packet(i):
    return SensorPacket(
        i * 2,
        i / 10,
        {v: np.zeros((288, 512, 3), np.uint8) for v in VIEWS},
        {
            v: CameraCalibration(512, 288, 100, (1.3, 0, 2.3), (0, yaw, 0))
            for v, yaw in zip(VIEWS, (0, -60, 60))
        },
        np.zeros((16, 2), np.float32),
        np.zeros(3, np.float32),
    )


def row(view="front", p=(1, 0, 0)):
    return dict(view=view, box=[254, 20, 258, 30], score=0.9, probabilities=list(p))


def test_native_packer_keeps_all_detections_and_has_no_oracle_fields():
    tokens, valid = pack_observations(packet(0), [row(v) for v in VIEWS for _ in range(100)])
    assert tokens.shape == (301, 17) and valid.all()
    assert np.array_equal(tokens[0], np.zeros(17))
    assert tokens[101, 8] < 0 and tokens[201, 8] > 0  # Physical camera rotation.
    changed = replace(
        packet(0),
        navigation_waypoints=np.ones((16, 2), np.float32),
        ego_motion=np.ones(3, np.float32),
    )
    assert np.array_equal(
        pack_observations(changed, [row()])[0], pack_observations(packet(0), [row()])[0]
    )
    with pytest.raises(ValueError, match="capacity"):
        pack_observations(packet(0), [row()] * 301)
    with pytest.raises(ValueError, match="distribution"):
        pack_observations(packet(0), [row(p=(-1, 1, 1))])


def test_cache_runtime_share_past_only_cadence_and_age():
    class Perception:
        def __init__(self):
            self.frames = []

        def infer(self, p):
            self.frames.append(p.frame)
            return [row(p=(1, 0, 0) if p.frame < 8 else (0, 0, 1))]

    perception = Perception()
    facts = CausalSignalFacts(perception)
    old = []
    for i in range(5):
        t, m = facts.step(packet(i))
        old.append(t)
        assert m.sum() == 2
        assert t[1, -1] == pytest.approx((i % 4) / 10)
    assert perception.frames == [0, 8]
    assert all(t[1, 5] == 1 for t in old[:4]) and old[4][1, 7] == 1
    with pytest.raises(ValueError, match="contiguous"):
        facts.step(packet(6))


def test_zero_residual_exact_parent_and_updates_cannot_change_path_or_cruise():
    torch.manual_seed(2)
    torch.set_num_threads(2)
    parent = CameraStopGoStudent(32)
    model = SignalResidualStudent(32)
    model.load_state_dict(parent.state_dict(), strict=False)
    original = {k: v.clone() for k, v in parent.state_dict().items()}
    x = dict(
        rgb=torch.randint(256, (1, 4, 3, 3, 48, 64), dtype=torch.uint8),
        front=torch.randint(256, (1, 4, 3, 96, 128), dtype=torch.uint8),
        navigation=torch.randn(1, 4, 16, 2),
        ego=torch.randn(1, 4, 3),
    )
    facts = torch.zeros(1, 4, MAX_OBJECTS + 1, TOKEN_DIM)
    valid = torch.zeros(1, 4, MAX_OBJECTS + 1, dtype=torch.bool)
    valid[..., 0] = True
    expected, _ = parent(**x)
    before, _ = model(**x, signal_facts=facts, signal_valid=valid)
    for k in expected:
        assert torch.equal(expected[k], before[k]), k
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    before["stop_logits"].sum().backward()
    opt.step()
    after, _ = model(**x, signal_facts=facts, signal_valid=valid)
    assert not torch.equal(after["stop_logits"], before["stop_logits"])
    for k in ("correction", "cruise_speeds", "comfort_brake", "light", "hazards"):
        assert torch.equal(after[k], expected[k]), k
    assert all(torch.equal(model.state_dict()[k], v) for k, v in original.items())
    assert isinstance(make_student("camera-signal-residual-v3", 32), SignalResidualStudent)


def test_attention_masks_padding_and_carries_causal_memory():
    torch.manual_seed(3)
    model = StopResidual(32)
    torch.nn.init.normal_(model.output.weight)
    r, nav, ego = torch.randn(1, 3, 32), torch.randn(1, 3, 16, 2), torch.randn(1, 3, 3)
    f = torch.randn(1, 3, 301, 17)
    v = torch.zeros(1, 3, 301, dtype=torch.bool)
    v[..., :2] = True
    p, _ = model(r, nav, ego, f, v)
    f2 = f.clone()
    f2[..., 2:, :] = 1000
    assert torch.equal(p, model(r, nav, ego, f2, v)[0])
    hidden, pieces = None, []
    for i in range(3):
        pred, hidden = model(
            r[:, i : i + 1],
            nav[:, i : i + 1],
            ego[:, i : i + 1],
            f[:, i : i + 1],
            v[:, i : i + 1],
            hidden,
        )
        pieces.append(pred)
    assert torch.allclose(p, torch.cat(pieces, 1), atol=1e-6)
    f3 = f.clone()
    f3[:, 2] += 5
    assert torch.equal(p[:, :2], model(r, nav, ego, f3, v)[0][:, :2])


def test_cache_excludes_new_component_train_locations_preserves_original_validation():
    episodes = [
        dict(role="train", family="signal_stop", town="Town01", junction_ids=[1]),
        dict(role="train", family="signal_stop", town="Town01", junction_ids=[2]),
        dict(role="validation", family="signal_stop", town="Town01", junction_ids=[3]),
    ]
    selected = selected_episodes(dict(episodes=episodes), {"Town01": [1]})
    assert selected == [episodes[0], episodes[2]]


def test_selector_catches_family_regression_hidden_by_aggregate_improvement():
    import copy

    parent = dict(
        counts=dict(speed=1312),
        families=dict(
            light=dict(stopped=100, stop_hits=50, controller_go=80),
            sign=dict(stopped=50, stop_hits=30, controller_go=40),
        ),
        controller_stop_recall=0.53,
        controller_false_hold_fraction=0.05,
        go_speed_mae_mps=2.0,
        selection_score=0.3,
    )
    improved = copy.deepcopy(parent)
    improved.update(controller_stop_recall=0.7, selection_score=0.2)
    assert passes(improved, parent)
    improved["families"]["sign"]["stop_hits"] = 28
    assert not passes(improved, parent)
    improved = copy.deepcopy(parent)
    improved["counts"]["speed"] -= 1
    assert not passes(improved, parent)
    assert validation_mask(70, "cpu").nonzero().flatten().tolist() == list(range(16, 32)) + list(
        range(48, 64)
    )


def test_residual_loss_uses_only_past_burnin_and_valid_future_labels():
    r = torch.randn(1, 12, 9, requires_grad=True)
    a = dict(
        stop_logits=torch.zeros_like(r),
        current_speed=torch.ones(1, 12),
        speeds=torch.ones(1, 12, 8),
        speed_mask=torch.zeros(1, 12, 8),
    )
    stop_loss(r, a, True).backward()
    assert (
        not r.grad[:, :8].any() and not r.grad[..., 1:].any() and r.grad[:, 8:, 0].abs().sum() > 0
    )


def test_exported_residual_runtime_is_usable_in_privilege_guarded_child(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys

    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    model = SignalResidualStudent(32)
    export = tmp_path / "policy.pt"
    torch.save(
        dict(
            model=model.state_dict(),
            hidden=32,
            architecture="camera-signal-residual-v3",
            fact_arm="null",
            signal_bundle=str(bundle),
            signal_bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        ),
        export,
    )
    import struct

    from src.navigation.policy_ipc import encode_packet

    payload = encode_packet(packet(0))
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.navigation.policy_worker",
            "--checkpoint",
            str(export),
            "--device",
            "cpu",
        ],
        input=struct.pack("<Q", len(payload)) + payload + struct.pack("<Q", 0),
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stderr.decode()
    ready, result = [json.loads(s) for s in child.stdout.splitlines()]
    assert (
        ready["status"] == "ready"
        and result["rule_overrides_enabled"] is False
        and result["frame"] == 0
    )
