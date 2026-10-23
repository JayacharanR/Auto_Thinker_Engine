"""Training-only factual windows; oracle outcomes never cross the policy boundary."""

import numpy as np

from src.navigation.trajectory_candidates import ordered_projection


def factual_window(
    index,
    frames,
    navigation,
    ego,
    positions,
    aligned,
    contact_world_frames,
    first_safety_frame=None,
    native_red_zero_observed=False,
):
    """Return current inputs, ordered acknowledged actions and separately masked targets.

    Unknown action tails are zero padded ONLY beyond every scored horizon. The
    ordered dynamics are causal at each horizon; no unknown tail earns a label.
    Observed collisions remain positive even in an assisted/censored future.
    """
    frame = int(frames[index])
    actions = np.zeros((80, 3), np.float32)
    known = 0
    for tick in range(80):
        record = aligned.get(frame + tick + 1)
        if record is None:
            break
        if record.get("handbrake", False) or record.get("gear", 1) < 0:
            break  # Three-action models cannot silently omit an active actuator.
        actions[tick] = record["control"]
        known += 1
    targets = index + np.arange(1, 9) * 5
    latent_valid = (targets < len(frames)) & (np.arange(1, 9) * 10 <= known)
    target_indices = np.minimum(targets, len(frames) - 1)
    outcomes = np.full((8, 3), np.nan, np.float32)
    cost_valid = np.zeros((8, 3), np.float32)
    yaw = float(ego[index, 1])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    future_xy = (positions[target_indices, :2] - positions[index, :2]) @ rotation
    progress = ordered_projection(future_xy, navigation[index])
    for horizon in range(8):
        if not latent_valid[horizon]:
            continue
        end = frame + (horizon + 1) * 10
        if frames[target_indices[horizon]] != end:
            raise ValueError("Factual source is not contiguous10Hz")
        collision = any(frame < contact <= end for contact in contact_world_frames)
        uncensored = first_safety_frame is None or end < first_safety_frame
        # A recorded harmful outcome is evidence; an assisted zero is not a
        # counterfactual safe label. Keep progress as actual executed-action data.
        if collision or uncensored:
            outcomes[horizon, 0] = float(collision)
            cost_valid[horizon, 0] = 1
        if native_red_zero_observed and uncensored:
            outcomes[horizon, 1] = 0
            cost_valid[horizon, 1] = 1
        outcomes[horizon, 2] = progress[horizon]
        cost_valid[horizon, 2] = 1
    condition = np.r_[navigation[index].ravel() / 32, ego[index] / [10, np.pi, 1]].astype(
        np.float32
    )
    if condition.shape != (35,) or not np.isfinite(condition).all():
        raise ValueError("Only finite declared navigation and ego-motion conditioning allowed")
    return dict(
        input_index=index,
        future_indices=target_indices,
        conditioning=condition,
        controls=actions,
        known_control_ticks=known,
        latent_valid=latent_valid.astype(np.float32),
        outcomes=outcomes,
        cost_valid=cost_valid,
    )
