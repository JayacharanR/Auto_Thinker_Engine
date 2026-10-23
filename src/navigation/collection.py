"""Map-only navigation and collection-side control acknowledgement."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.navigation.contracts import TrainingTargets
from src.navigation.metrics import summarize_records
from src.navigation.runtime import atomic_json


def empty_training_background(spec):
    collection = spec.get("collection") or {}
    if collection.get("background", "canonical") != "empty":
        return False
    if (
        spec.get("student")
        or collection.get("mode") != "expert"
        or not collection.get("learning_guard")
        or collection["learning_guard"].get("kind") != "continuous-approach-train-v1"
    ):
        raise ValueError("Empty background is restricted to guarded expert TRAIN collection")
    return True


def training_signal_wait(spec):
    collection = spec.get("collection") or {}
    value = collection.get("signal_stationary_seconds")
    if value is None:
        return None
    if not empty_training_background(spec) or not np.isfinite(value) or not 0 < value <= 15:
        raise ValueError("Signal wait override requires guarded no-lead TRAIN collection,0–15s")
    return float(value)


def training_light_gap(spec):
    """A declared expert-training parameter; never camera/benchmark behavior."""
    collection = spec.get("collection") or {}
    value = collection.get("training_light_gap_m")
    if value is None:
        return None
    if not empty_training_background(spec):
        raise ValueError("Light-gap override requires guarded empty expert TRAIN collection")
    value = float(value)
    if not np.isfinite(value) or not 6 <= value <= 16:
        raise ValueError("TRAIN light gap must be finite and between6 and16m")
    return value


def training_position(world_map, pose, allowed, frame, violation_path=None):
    waypoint = world_map.get_waypoint(pose.location)
    if waypoint.is_junction and waypoint.junction_id not in allowed:
        if violation_path:
            atomic_json(
                violation_path,
                dict(
                    frame=frame,
                    junction_id=waypoint.junction_id,
                    road_id=waypoint.road_id,
                    lane_id=waypoint.lane_id,
                    ego_xyz=[pose.location.x, pose.location.y, pose.location.z],
                ),
            )
        raise RuntimeError("Actual collection trajectory entered a held-out junction")
    return pose


def verify_collection_catalog(catalog_dir, route):
    directory, route = Path(catalog_dir).resolve(), Path(route).resolve()
    data = (directory / "catalog.json").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != (directory / "catalog.sha256").read_text().split()[0]:
        raise ValueError("Collection catalog hash mismatch")
    catalog = json.loads(data)
    if catalog.get("role") != "train":
        raise ValueError("Collection requires the frozen training partition")
    split_data = (directory.parent / "locations.json").read_bytes()
    if (
        hashlib.sha256(split_data).hexdigest()
        != (directory.parent / "locations.sha256").read_text().strip()
    ):
        raise ValueError("Location split hash mismatch")
    split = json.loads(split_data)
    if split["catalogs"]["train"]["sha256"] != digest:
        raise ValueError("Training catalog differs from the approved split")
    town = next((t for t in catalog["towns"] if route == directory / f"{t}.xml"), None)
    if town is None or town.removesuffix("_Opt") == "Town05":
        raise ValueError("Route is not a training-partition route")
    for entry in catalog["encounters"]:
        allowed = split["assignments"][entry["town"]]["train"]
        if not set(entry["crossed_junction_ids"]) <= set(allowed):
            raise ValueError("Collection route crosses a held-out junction")
    return {"catalog": str(directory), "sha256": digest, "town": town}


def finalize_collection(output):
    """Publish conservative episode eligibility after durable evaluator results exist."""
    output = Path(output)
    summary_path, raw_path = output / "summary.json", output / "leaderboard.json"
    if summary_path.exists():
        episodes = json.loads(summary_path.read_text())["episodes"]
    elif raw_path.exists():
        episodes = summarize_records(json.loads(raw_path.read_text())["_checkpoint"]["records"])[
            "episodes"
        ]
    else:
        episodes = []
    records = {e["route_id"]: e for e in episodes}
    for path in output.glob("episodes/*/data/episode.json"):
        metadata = json.loads(path.read_text())
        profile_path = path.parent.parent / "collection_profile.json"
        profile = json.loads(profile_path.read_text()) if profile_path.exists() else None
        record = records.get(f"RouteScenario_{metadata['route_id']}_rep0")
        scenario = metadata.get("scenario")
        setup_path = output / "scenario_setup" / f"{metadata['route_id']}.json"
        setup_started = not scenario or (
            setup_path.exists()
            and any(
                s["type"] == scenario for s in json.loads(setup_path.read_text()).get("started", [])
            )
        )
        outcome = {
            "evaluator_record": record,
            "declared_scenario_started": bool(setup_started),
            "imitation_eligible": bool(
                record
                and record["rule_safe_completed"]
                and profile
                and profile["samples"] > 0
                and not profile["control_differences"]
                and not profile.get("safety_stop_ticks")
                and setup_started
            ),
            "prediction_eligible": True,
            "episode_censored": (
                not bool(record and record["completed"] and profile)
                or bool(profile and profile.get("safety_stop_ticks"))
            ),
            "counterfactual_safe_label": False,
            "note": "Unknown/truncated/unsafe episodes are never clean imitation targets; "
            "committed observations remain available for prediction with explicit censoring.",
        }
        if metadata.get("mode") == "dagger":
            outcome.update(
                imitation_eligible=False,
                correction_labels_eligible=bool(
                    record
                    and record["rule_safe_completed"]
                    and profile
                    and profile["samples"] > 0
                    and not profile["control_differences"]
                    and setup_started
                    and not (output / "collection_partition_violation.json").exists()
                ),
                episode_censored=bool(
                    outcome["episode_censored"]
                    or (profile and profile.get("privileged_intervention_ticks"))
                ),
                note="DAgger training mixture: only instantaneous expert corrections may be used. "
                "Never imitate student/executed actions or infer an unassisted safe outcome. "
                "Assisted/censored future trajectories are not counterfactual labels.",
            )
        destination = path.with_name("outcome.json")
        if destination.exists():
            if json.loads(destination.read_text()) != outcome:
                raise ValueError("Refusing to change finalized episode eligibility")
        else:
            atomic_json(destination, outcome)


class RouteProjector:
    """Project ego onto a fixed navigation polyline, independent of expert hazards."""

    def __init__(self, points, distances=None):
        self.points = np.asarray(points, dtype=np.float64)[:, :2]
        if len(self.points) < 2 or not np.isfinite(self.points).all():
            raise ValueError("Need a finite navigation polyline")
        lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        keep = np.r_[True, lengths > 1e-6]
        self.points = self.points[keep]
        if len(self.points) < 2:
            raise ValueError("Navigation has zero length")
        self.arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        self.distances = np.asarray(distances if distances is not None else np.arange(2, 34, 2))
        self.index = 0

    def project(self, xy, yaw):
        # A bounded forward search avoids jumping to a later visit at a route crossing.
        stop = min(len(self.points) - 1, self.index + 80)
        start = max(0, self.index - 5)
        a, b = self.points[start:stop], self.points[start + 1 : stop + 1]
        delta = b - a
        ratio = np.clip(np.sum((xy - a) * delta, axis=1) / np.sum(delta * delta, axis=1), 0, 1)
        nearest = a + ratio[:, None] * delta
        local = int(np.argmin(np.linalg.norm(nearest - xy, axis=1)))
        index = start + local
        self.index = max(self.index, index)
        progress = self.arc[index] + ratio[local] * (self.arc[index + 1] - self.arc[index])
        ahead = np.minimum(progress + self.distances, self.arc[-1])
        world = np.column_stack([np.interp(ahead, self.arc, self.points[:, i]) for i in (0, 1)])
        c, s = np.cos(yaw), np.sin(yaw)
        return ((world - xy) @ np.array([[c, -s], [s, c]])).astype(np.float32)


class CollectionRecorder:
    """Commit an observation only after observing its command at the next physics tick."""

    def __init__(self, writer):
        self.writer, self.pending = writer, None
        self.control_differences = 0

    def capture(self, packet, teacher_control, requested_control, facts, intervention=None):
        if self.pending is not None:
            raise RuntimeError("Previous control has not been acknowledged")
        self.pending = packet, tuple(teacher_control), tuple(requested_control), facts, intervention

    def acknowledge(self, frame, observed_control):
        if self.pending is None:
            return
        packet, teacher, requested, facts, intervention = self.pending
        if frame == packet.frame:
            return  # no physics tick has happened; never pretend it executed
        if frame != packet.frame + 1:
            raise ValueError("Actuator acknowledgement must be at the next physics frame")
        changed = not np.allclose(requested, observed_control, atol=1e-5, rtol=0)
        self.control_differences += int(changed)
        self.writer.append(
            packet,
            TrainingTargets(
                frame=packet.frame,
                expert_control=teacher,
                executed_control=tuple(observed_control),
                intervention="actuator_command_difference" if changed else intervention,
                facts={
                    **facts,
                    "requested_control": list(requested),
                    "actuator_feedback_frame": frame,
                    "action_execution_observed": True,
                },
            ),
        )
        self.pending = None

    def close(self):
        self.writer.close()
        return {
            "samples": self.writer.samples,
            "shards": self.writer.shards,
            "control_differences": self.control_differences,
            "unacknowledged_samples": int(self.pending is not None),
        }
