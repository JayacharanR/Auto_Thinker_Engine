"""Scoring/partition helpers ONLY. Never import this module in camera inference."""

from collections import Counter

import numpy as np

from src.navigation.signal_perception import STATES


def episode_scope(meta, guard, roles):
    town = meta["town"]
    junctions = set(meta["junction_ids"])
    if (
        town.removesuffix("_Opt") == "Town05"
        or meta["partition"] != "train"
        or not junctions
        or not junctions <= set(guard["allowed_junctions"].get(town, []))
    ):
        raise ValueError("Episode is outside approved internal driving TRAIN")
    overlap = sorted(j for j in junctions if roles.get((town, j)) == "validation")
    return dict(eligible=not overlap, component_validation_overlap=overlap)


def iou(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.0


def score_frame(row, predicted):
    """GT matching only after RGB inference; not visibility-certified state labels."""
    signal = row["signal"]
    truth = row["next_light_state"]
    if signal is None or truth not in STATES:
        return dict(known=False, projected_housings=0, matches=[])
    targets = []
    for view, geometry in signal["views"].items():
        for head, box in enumerate(geometry["boxes"]):
            if box["in_image"] is True:
                b = np.clip(np.asarray(box["bbox_xyxy"]), [0, 0, 0, 0], [512, 288, 512, 288])
                if b[2] > b[0] and b[3] > b[1]:
                    targets.append((view, head, b))
    pairs = []
    for index, observation in enumerate(predicted):
        for target, (view, _, box) in enumerate(targets):
            overlap = iou(observation["box"], box) if observation["view"] == view else 0
            if overlap >= 0.3:
                pairs.append((-overlap, index, target))
    used_predictions, used_targets, matches = set(), set(), []
    for negative_overlap, index, target in sorted(pairs):
        if index in used_predictions or target in used_targets:
            continue
        used_predictions.add(index)
        used_targets.add(target)
        observation, (view, head, _) = predicted[index], targets[target]
        matches.append(
            dict(
                view=view,
                head=head,
                actor=signal["recorded_actor_id"],
                prediction=index,
                track_id=observation["track_id"],
                iou=-negative_overlap,
                truth=truth,
                raw_state=observation["raw_state"],
                tracked_argmax=observation["tracked_argmax"],
                tracked_state=observation["tracked_state"],
            )
        )
    return dict(known=True, projected_housings=len(targets), matches=matches)


def summarize_frames(rows):
    counts, previous, confusion = (
        Counter(),
        {},
        {k: np.zeros((3, 3), int) for k in ("raw_state", "tracked_argmax")},
    )
    gap, maximum_gap = 0, 0
    for row in rows:
        scored, frame = row["scored"], row["frame"]
        counts["timestamps"] += 1
        counts["detections_all_views"] += len(row["observations"])
        counts["known_timestamps"] += int(scored["known"])
        counts["projected_housings"] += scored["projected_housings"]
        counts["matched_housings"] += len(scored["matches"])
        counts["known_timestamps_with_match"] += int(scored["known"] and bool(scored["matches"]))
        if scored["known"] and not scored["matches"]:
            gap += 1
            maximum_gap = max(gap, maximum_gap)
        else:
            gap = 0
        for match in scored["matches"]:
            truth = STATES.index(match["truth"])
            counts[f"matched_truth_{match['truth']}"] += 1
            for name in confusion:
                confusion[name][truth, STATES.index(match[name])] += 1
                counts[f"{name}_correct"] += int(match[name] == match["truth"])
            confident = match["tracked_state"] != "Unknown"
            counts["tracked_confident"] += int(confident)
            counts["tracked_confident_correct"] += int(match["tracked_state"] == match["truth"])
            counts["tracked_unknown"] += int(not confident)
            if match["truth"] == "Red":
                counts["raw_red_called_green"] += int(match["raw_state"] == "Green")
                counts["tracked_red_called_green"] += int(match["tracked_state"] == "Green")
            key = match["view"], match["actor"], match["head"]
            old = previous.get(key)
            if old is not None and frame == old[0] + 2:
                counts["consecutive_matched_housing_pairs"] += 1
                counts["consecutive_track_id_changes"] += int(old[1] != match["track_id"])
            previous[key] = frame, match["track_id"]
    return dict(
        counts=dict(counts),
        maximum_known_no_detection_run_s=maximum_gap / 10,
        confusion_order=list(STATES),
        confusion={k: v.tolist() for k, v in confusion.items()},
    )


def transition_delays(rows):
    """Oracle-associated diagnostic; no selected applicable track is fed to runtime."""
    result = []
    for index in range(1, len(rows)):
        old, current = rows[index - 1], rows[index]
        if (
            old["signal"] is None
            or current["signal"] is None
            or old["signal"]["recorded_actor_id"] != current["signal"]["recorded_actor_id"]
            or old["next_light_state"] not in STATES
            or current["next_light_state"] not in STATES
            or old["next_light_state"] == current["next_light_state"]
        ):
            continue
        target, actor = current["next_light_state"], current["signal"]["recorded_actor_id"]
        hits = dict.fromkeys(("raw_state", "tracked_argmax", "tracked_state"))
        observed_until = current["timestamp"]
        for later in rows[index:]:
            if (
                later["timestamp"] - current["timestamp"] > 3.000001
                or later["signal"] is None
                or later["next_light_state"] != target
                or later["signal"]["recorded_actor_id"] != actor
            ):
                break
            observed_until = later["timestamp"]
            for name in hits:
                if hits[name] is None and any(
                    m[name] == target for m in later["scored"]["matches"]
                ):
                    hits[name] = later["timestamp"] - current["timestamp"]
        result.append(
            dict(
                frame=current["frame"],
                timestamp=current["timestamp"],
                previous_state=old["next_light_state"],
                next_state=target,
                first_correct_associated_evidence_delay_s=hits,
                observed_window_s=observed_until - current["timestamp"],
            )
        )
    return result
