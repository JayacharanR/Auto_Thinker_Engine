"""Versioned ego-only actuator feedback; approved 2026-10-20.

No simulator, actor, teacher or world facts cross this boundary.
"""

import io
import json
from dataclasses import dataclass

import numpy as np

from src.navigation.contracts import ControlCommand
from src.navigation.policy_ipc import MAX_PAYLOAD, decode_packet, encode_packet

SCHEMA = "navigation-ego-action-step-v2"
FEEDBACK_FIELDS = {"observation_frame", "action_frame", "steer", "throttle", "brake"}
ENVELOPE_FIELDS = {"schema", "packet", "feedback"}


@dataclass(frozen=True)
class EgoActionFeedback:
    observation_frame: int  # Frame f, observing the motor command applied during f-1.
    action_frame: int
    steer: float
    throttle: float
    brake: float

    def validate(self, packet_frame):
        if type(self) is not EgoActionFeedback or set(vars(self)) != FEEDBACK_FIELDS:
            raise ValueError("Only exact ego-control feedback fields are allowed")
        if (
            type(self.observation_frame) is not int
            or type(self.action_frame) is not int
            or type(packet_frame) is not int
            or self.observation_frame != packet_frame
            or self.action_frame != packet_frame - 1
            or self.action_frame < 0
        ):
            raise ValueError("Feedback must acknowledge action f-1 at the current packet frame f")
        values = (self.steer, self.throttle, self.brake)
        if any(type(value) not in (float, int) for value in values):
            raise ValueError("Plain numeric ego actuator values only")
        ControlCommand(*values).validate()


def encode_step(packet, feedback):
    if type(feedback) is not EgoActionFeedback:
        raise ValueError("An exact ego-control acknowledgment is required")
    feedback.validate(packet.frame)
    raw = encode_packet(packet)  # Existing strict SensorPacket/array allowlist is unchanged.
    buffer = io.BytesIO()
    np.savez(
        buffer,
        schema=np.frombuffer(SCHEMA.encode(), np.uint8),
        packet=np.frombuffer(raw, np.uint8),
        feedback=np.frombuffer(json.dumps(vars(feedback), allow_nan=False).encode(), np.uint8),
    )
    result = buffer.getvalue()
    if len(result) > MAX_PAYLOAD:
        raise ValueError("Versioned step payload too large")
    return result


def decode_step(payload):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("Versioned step payload too large")
    with np.load(io.BytesIO(payload), allow_pickle=False) as values:
        if set(values.files) != ENVELOPE_FIELDS or any(
            values[name].dtype != np.uint8 or values[name].ndim != 1 for name in values.files
        ):
            raise ValueError("Only the exact versioned byte-array envelope is allowed")
        if values["schema"].tobytes().decode() != SCHEMA:
            raise ValueError("Unsupported ego-control step schema")
        packet = decode_packet(values["packet"].tobytes())
        feedback = json.loads(values["feedback"].tobytes())
        if not isinstance(feedback, dict) or set(feedback) != FEEDBACK_FIELDS:
            raise ValueError("Unexpected ego feedback fields; actor/teacher facts are forbidden")
        feedback = EgoActionFeedback(**feedback)
        feedback.validate(packet.frame)
    return packet, feedback


class AppliedActionHistory:
    def __init__(self):
        self.rows = []

    def observe(self, packet_frame, feedback):
        if type(feedback) is not EgoActionFeedback:
            raise ValueError("Exact acknowledged ego-control history required")
        feedback.validate(packet_frame)
        if self.rows and feedback.observation_frame != self.rows[-1].observation_frame + 1:
            raise ValueError("Stale, duplicate or missing native ego-control acknowledgment")
        self.rows = [*self.rows[-1:], feedback]

    def at_frame(self, frame):
        if len(self.rows) != 2 or self.rows[-1].observation_frame != frame:
            raise ValueError("Warm up with two actual acknowledgments; never duplicate a guess")
        if [r.action_frame for r in self.rows] != [frame - 2, frame - 1]:
            raise ValueError("Two chronological prior applied commands required")
        controls = np.array([[r.steer, r.throttle, r.brake] for r in self.rows], dtype=np.float32)
        controls.setflags(write=False)
        return controls
