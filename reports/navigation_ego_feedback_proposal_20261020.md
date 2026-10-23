# Ego actuator feedback proposal — approval required, not applied

Current policy_inputs in configs/navigation.yaml are RGB, camera calibration,
navigation waypoints, ego motion and timestamps. SensorPacket is an exact six-field
allowlist; the inference guard excludes simulator/training modules. Do not quietly
add fields or infer that scripted-prefix proposals were the applied motor commands.

The M5shared command generator requires the last two actually applied ego commands
to initialize the measured two-tick delay. During ordinary unassisted driving the
agent can remember its outputs. During M3scripted setup those outputs are ignored;
substituting them creates a false dynamics initial condition. Changing this public
input contract needs user approval under the benchmark-change rule.

Proposed extension, identically available to CV/RSSM/JEPA: an explicitly versioned
step envelope containing the unchanged SensorPacket and EgoActionFeedback with
ONLY observation frame f, acknowledged action frame f-1, steering, throttle and
brake measured from the ego vehicle's own get_control(). No actor IDs, other-car
states, traffic-light/sign state, route-line distance, teacher proposals, control
source or intervention reason may enter the feedback. It is own actuator status,
not an oracle hazard observation. Verify against next-tick native controls.

Legacy policies retain byte-identical v1payloads; only a declared v2candidate
negotiates feedback. The initial planner waits until it has two real consecutive
acknowledgments, using its camera driving parent during warm-up. Duplicate/missing/
stale or malformed acknowledgments fail closed; never fabricate a second command.
Do not erase parent visual memory at the scripted handoff. Every new model/arm
declares the input version and records the acknowledgment and actual control audits.

Concrete staged codec/history implementation and adversarial tests:
outputs/navigation_actuator_feedback_staging_20261020. This directory is outside
the pinned inference package. No running/queued policy, configuration, final suite,
route, scoring or dataset is modified. A future approved source generation must
wire camera extraction/IPC/inference and verify old-v1control parity before use.
Final candidate registration must bind the declared input version and actual entry
points. Approval would not waive M2–M6safety, fairness, coverage or evidence gates.
