"""Version negotiation, own-actuator isolation, real IPC and camera-control parity."""

import io
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
import yaml

from src.navigation.ego_action_feedback import (
    AppliedActionHistory,
    EgoActionFeedback,
    decode_step,
    encode_step,
)
from src.navigation.feedback_policy import FeedbackPolicySession
from src.navigation.policy_input_contract import (
    ACTUATOR_SCHEMA,
    LEGACY_SCHEMA,
    policy_input_schema,
    validate_config_schema,
)
from src.navigation.policy_ipc import PolicyClient, decode_packet, encode_packet
from src.navigation.student import make_student
from tests.test_navigation_collection import packet


def observation(frame=10):
    return replace(
        packet(frame),
        navigation_waypoints=np.column_stack([np.arange(2, 33, 2), np.zeros(16)]).astype(
            np.float32
        ),
        ego_motion=packet(frame).ego_motion.astype(np.float32),
    )


def ack(frame=10, steer=0.1, throttle=0.4, brake=0.0):
    return EgoActionFeedback(frame, frame - 1, steer, throttle, brake)


def test_versioned_roundtrip_preserves_legacy_packet_bytes():
    old = encode_packet(observation())
    restored, actual = decode_step(encode_step(observation(), ack()))
    assert restored.frame == 10 and actual == ack()
    assert encode_packet(restored) == old
    assert decode_packet(old).frame == 10
    assert set(vars(restored)) == set(vars(observation()))


def test_history_uses_actual_scripted_controls_and_requires_two_real_consecutive_acks():
    history = AppliedActionHistory()
    history.observe(10, ack(10, 0.1, 0.0, 1.0))
    with pytest.raises(ValueError, match="Warm up"):
        history.at_frame(10)
    history.observe(11, ack(11, 0.2, 0.3, 0.0))
    assert np.array_equal(history.at_frame(11), np.array([[0.1, 0, 1], [0.2, 0.3, 0]], np.float32))
    assert not history.at_frame(11).flags.writeable
    with pytest.raises(ValueError, match="Stale"):
        history.observe(11, ack(11))
    with pytest.raises(ValueError, match="missing"):
        history.observe(13, ack(13))
    with pytest.raises(ValueError):
        history.at_frame(12)


@pytest.mark.parametrize(
    "bad",
    [
        EgoActionFeedback(10, 8, 0.0, 0.0, 1.0),
        ack(10, 2.0, 0.0, 0.0),
        ack(10, 0.0, float("nan"), 0.0),
        ack(10, 0.0, True, 0.0),
        ack(10, 0.0, 0.2, 0.5),
        EgoActionFeedback(True, 0, 0.0, 0.0, 0.0),
    ],
)
def test_bad_frames_actuators_and_non_numeric_values_fail_before_transport(bad):
    with pytest.raises(ValueError):
        encode_step(observation(), bad)


def test_hidden_oracle_fields_and_object_envelopes_are_rejected():
    hidden = ack()
    object.__setattr__(hidden, "control_source", "teacher")
    with pytest.raises(ValueError, match="exact"):
        encode_step(observation(), hidden)
    with np.load(io.BytesIO(encode_step(observation(), ack())), allow_pickle=False) as payload:
        fields = {k: payload[k].copy() for k in payload.files}
    oracle = json.loads(fields["feedback"].tobytes())
    oracle["traffic_light"] = "Red"
    fields["feedback"] = np.frombuffer(json.dumps(oracle).encode(), np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, **fields)
    with pytest.raises(ValueError, match="forbidden"):
        decode_step(buffer.getvalue())
    fields["feedback"] = np.array([object()], object)
    buffer = io.BytesIO()
    np.savez(buffer, **fields)
    with pytest.raises(ValueError):
        decode_step(buffer.getvalue())


def test_contract_is_explicit_and_does_not_extend_legacy_by_default():
    assert policy_input_schema({}) == LEGACY_SCHEMA
    validate_config_schema({}, LEGACY_SCHEMA)
    with pytest.raises(ValueError, match="approved"):
        validate_config_schema({}, ACTUATOR_SCHEMA)
    for invalid in (None, [], "unknown"):
        with pytest.raises(ValueError):
            policy_input_schema({"policy_input_schema": invalid})
    config = yaml.safe_load(open("configs/navigation.yaml"))
    validate_config_schema(config, ACTUATOR_SCHEMA)
    config["policy_input_extensions"][ACTUATOR_SCHEMA]["fields"].append("teacher_control")
    with pytest.raises(ValueError, match="approved"):
        validate_config_schema(config, ACTUATOR_SCHEMA)


def test_session_keeps_parent_visual_memory_through_scripted_handoff():
    class Parent:
        def __init__(self):
            self.frames = []

        def step(self, current):
            self.frames.append(current.frame)
            return dict(frame=current.frame, control=[0.0, 0.7, 0.0], interventions=[])

    parent = Parent()
    session = FeedbackPolicySession(parent)
    first = session.step(observation(10), ack(10, 0.0, 0.0, 1.0))
    second = session.step(observation(11), ack(11, 0.2, 0.0, 1.0))
    assert parent.frames == [10, 11]
    assert not first["actuator_history_ready"] and second["actuator_history_ready"]
    assert second["control"] == [0.0, 0.7, 0.0]
    assert np.allclose(session.history.at_frame(11), [[0, 0, 1], [0.2, 0, 1]])


@pytest.mark.parametrize("architecture", ["camera-stop-go-v2", "spatial-visual-facts-v3"])
def test_real_fresh_workers_have_exact_legacy_control_parity_and_explicit_feedback(
    tmp_path, architecture
):
    torch.set_num_threads(2)
    torch.manual_seed(112)
    if architecture == "spatial-visual-facts-v3":
        from src.navigation.fact_student import FactStudent

        model = FactStudent(384, 32, scratch=True, color_enabled=True)
        state = dict(
            architecture=architecture,
            hidden=32,
            model=model.state_dict(),
            encoder_kind="cnn_scratch",
            encoder_dim=384,
            color_enabled=True,
        )
    else:
        model = make_student(architecture, hidden=32)
        state = dict(architecture=architecture, hidden=32, model=model.state_dict())
    legacy, extended = tmp_path / "legacy.pt", tmp_path / "extended.pt"
    torch.save(state, legacy)
    torch.save(dict(state, policy_input_schema=ACTUATOR_SCHEMA), extended)
    old = PolicyClient(legacy, tmp_path / "legacy.log", device="cpu")
    new = PolicyClient(
        extended, tmp_path / "extended.log", device="cpu", input_schema=ACTUATOR_SCHEMA
    )
    try:
        with pytest.raises(ValueError, match="Legacy"):
            old.step(observation(), ack())
        with pytest.raises(ValueError, match="exact"):
            new.step(observation())
        for frame in range(10, 16):
            current = observation(frame)
            if architecture == "spatial-visual-facts-v3":
                current = replace(
                    current,
                    rgb={k: np.tile(v, (36, 32, 1)) for k, v in current.rgb.items()},
                    calibration={
                        k: replace(v, width=512, height=288) for k, v in current.calibration.items()
                    },
                )
            actual = ack(frame, steer=0.1 * (frame % 3), throttle=0.0, brake=1.0)
            before, after = old.step(current), new.step(current, actual)
            extra = {
                k: after.pop(k)
                for k in ("policy_input_schema", "ego_action_feedback", "actuator_history_ready")
            }
            assert after == before  # All parent predictions and controls, including GRU memory.
            assert extra["ego_action_feedback"] == vars(actual)
            assert extra["actuator_history_ready"] == (frame > 10)
    finally:
        old.close()
        new.close()
    assert old.process.returncode == new.process.returncode == 0


def test_worker_rejects_checkpoint_transport_disagreement(tmp_path):
    model = make_student("camera-stop-go-v2", hidden=32)
    path = tmp_path / "extended.pt"
    torch.save(
        dict(
            architecture="camera-stop-go-v2",
            hidden=32,
            model=model.state_dict(),
            policy_input_schema=ACTUATOR_SCHEMA,
        ),
        path,
    )
    with pytest.raises(RuntimeError, match="exited"):
        PolicyClient(path, tmp_path / "rejected.log", device="cpu")
    assert "Checkpoint and transport input schema differ" in (tmp_path / "rejected.log").read_text()


def test_native_audit_rejects_fabricated_proposal_history_and_missing_acknowledgments():
    from scripts.audit_navigation_ego_feedback import verify_feedback_rows

    rows = [
        dict(
            frame=10,
            policy_input_schema=ACTUATOR_SCHEMA,
            ego_action_feedback=vars(ack(10, 0.0, 0.0, 1.0)),
            actuator_history_ready=False,
            control=[0.1, 0.5, 0.0],
        ),
        dict(
            frame=11,
            policy_input_schema=ACTUATOR_SCHEMA,
            ego_action_feedback=vars(ack(11, 0.1, 0.5, 0.0)),
            actuator_history_ready=True,
            control=[0.0, 0.0, 1.0],
        ),
    ]
    aligned = {10: dict(control=[0.0, 0.0, 1.0]), 11: dict(control=[0.1, 0.5, 0.0])}
    assert verify_feedback_rows(rows, aligned)["consecutive_prior_commands"] == 1
    with pytest.raises(ValueError, match="unavailable"):
        verify_feedback_rows(rows, {10: aligned[10]})
    wrong = [dict(rows[0], control=[0.0, 0.0, 1.0]), rows[1]]
    with pytest.raises(ValueError, match="preceding"):
        verify_feedback_rows(wrong, aligned)
    wrong = [dict(rows[0], actuator_history_ready=True), rows[1]]
    with pytest.raises(ValueError, match="readiness"):
        verify_feedback_rows(wrong, aligned)


def test_final_candidate_binds_declared_checkpoint_and_qualification_input_schema(tmp_path):
    from src.navigation.final_evaluation import (
        SUITE,
        candidate_request,
        digest,
        validate_final_request,
    )

    candidate = tmp_path / "fixture_only.pt"
    report = tmp_path / "fixture_report.json"
    qualification = tmp_path / "fixture_qualification.json"
    report.write_text("{}")
    torch.save(dict(policy_input_schema=ACTUATOR_SCHEMA), candidate)
    evidence = dict(
        candidate_sha256=digest(candidate),
        town05_used_for_training_or_selection=False,
        milestones={
            stage: dict(
                evidence_complete=True,
                gate_passed=True,
                reports=[dict(path=str(report), sha256=digest(report))],
            )
            for stage in ("M2", "M3", "M4", "M5")
        },
    )
    qualification.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="Qualification input schema"):
        candidate_request(candidate, qualification, tmp_path / "fixture_results")
    qualification.write_text(json.dumps(dict(evidence, policy_input_schema=ACTUATOR_SCHEMA)))
    request = candidate_request(candidate, qualification, tmp_path / "fixture_results")
    assert request["policy_input_schema"] == ACTUATOR_SCHEMA
    registration = tmp_path / "fixture_registration.json"
    registration.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="identity"):
        validate_final_request(
            registration,
            SUITE / "Town05_Opt.xml",
            tmp_path / "fixture_results/seed3031",
            3031,
            "0",
            dict(sha256=digest(candidate), rule_overrides=False, policy_input_schema=LEGACY_SCHEMA),
            None,
        )
