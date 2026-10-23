"""Own-actuator session: camera memory runs on every tick, including scripted setup.

This first integration is a parity reference, not an active trajectory planner.
The parent receives the unchanged camera packet. Future registered CV/RSSM/JEPA
planners can consume the same two measured commands through ``history.at_frame``.
No guessed parent output is substituted for the acknowledged applied command.
"""

from src.navigation.ego_action_feedback import AppliedActionHistory
from src.navigation.policy_input_contract import ACTUATOR_SCHEMA


class FeedbackPolicySession:
    def __init__(self, parent):
        self.parent = parent
        self.history = AppliedActionHistory()

    def step(self, packet, feedback):
        self.history.observe(packet.frame, feedback)
        result = self.parent.step(packet)
        if result["frame"] != packet.frame:
            raise ValueError("Camera parent returned a stale control")
        reserved = {"policy_input_schema", "ego_action_feedback", "actuator_history_ready"}
        if reserved.intersection(result):
            raise ValueError("Camera parent supplied reserved actuator acknowledgment fields")
        return dict(
            result,
            policy_input_schema=ACTUATOR_SCHEMA,
            ego_action_feedback=dict(vars(feedback)),
            actuator_history_ready=len(self.history.rows) == 2,
        )
