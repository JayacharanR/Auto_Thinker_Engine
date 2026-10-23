"""Causal per-camera bearing tracks. Identity is provisional; applicability unknown."""

import math

import numpy as np

from src.navigation.contracts import VIEWS
from src.navigation.signal_perception import STATES


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def angular_box(box, calibration, heading):
    if abs(calibration.pitch_yaw_roll[0]) > 1e-6 or abs(calibration.pitch_yaw_roll[2]) > 1e-6:
        raise ValueError("Angular tracker currently requires zero camera pitch/roll")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= calibration.width and 0 <= y0 < y1 <= calibration.height):
        raise ValueError("Expected finite in-image detection")
    focal = float(calibration.intrinsic[0, 0])
    ax = np.arctan((np.array([x0, x1]) - calibration.width / 2) / focal)
    ay = np.arctan((calibration.height / 2 - np.array([y1, y0])) / focal)
    return np.array(
        [
            wrap(float(ax.mean()) + math.radians(calibration.pitch_yaw_roll[1]) + heading),
            float(ay.mean()),
            float(ax[1] - ax[0]),
            float(ay[1] - ay[0]),
        ]
    )


def association_cost(first, second):
    ratios = second[2:] / first[2:]
    if np.any(ratios < 0.4) or np.any(ratios > 2.5):
        return None
    gates = np.clip(2 * np.maximum(first[2:], second[2:]), math.radians(1), math.radians(4))
    distance = np.array([abs(wrap(first[0] - second[0])), abs(first[1] - second[1])]) / gates
    return float(np.linalg.norm(distance)) if np.all(distance <= 1) else None


class SignalTracker:
    """State smoothing only; never releases a vehicle or asserts route applicability."""

    def __init__(self):
        self.tracks, self.next_id, self.previous = {}, 1, None

    def step(self, packet, observations):
        packet.validate()
        if self.previous is not None and (
            packet.frame != self.previous[0] + 2
            or abs(packet.timestamp - self.previous[1] - 0.1) > 1e-5
        ):
            raise ValueError("Tracker requires contiguous10Hz packets; reset per episode")
        # Validate the entire observation list before changing recurrent state.
        bearings = []
        for row in observations:
            p = np.asarray(row["probabilities"], dtype=float)
            if (
                row["view"] not in VIEWS
                or p.shape != (3,)
                or not np.isfinite(p).all()
                or np.any(p < 0)
                or not np.isclose(p.sum(), 1)
                or not np.isfinite(row["score"])
                or not 0.5 <= row["score"] <= 1
            ):
                raise ValueError("Invalid RGB signal observation")
            bearings.append(
                angular_box(
                    row["box"], packet.calibration[row["view"]], float(packet.ego_motion[1])
                )
            )
        now = packet.timestamp
        self.previous = packet.frame, now
        self.tracks = {k: t for k, t in self.tracks.items() if now - t["seen"] <= 0.500001}
        pairs = []
        for track_id, track in self.tracks.items():
            for i, (row, bearing) in enumerate(zip(observations, bearings)):
                if row["view"] == track["view"]:
                    cost = association_cost(track["bearing"], bearing)
                    if cost is not None:
                        pairs.append((cost, track_id, i))
        assigned, used = {}, set()
        for _, track_id, index in sorted(pairs):
            if index not in assigned and track_id not in used:
                assigned[index] = track_id
                used.add(track_id)
        result = []
        for index, (row, bearing) in enumerate(zip(observations, bearings)):
            p = np.asarray(row["probabilities"], dtype=float)
            track_id = assigned.get(index)
            if track_id is None:
                track_id, self.next_id = self.next_id, self.next_id + 1
                track = dict(view=row["view"], probabilities=p.copy(), hits=1)
                self.tracks[track_id] = track
            else:
                track = self.tracks[track_id]
                elapsed = now - track["seen"]
                alpha = 1 - math.exp(-elapsed / 0.3)
                track["probabilities"] += alpha * (p - track["probabilities"])
                track["hits"] = track["hits"] + 1 if elapsed <= 0.10001 else 1
            track.update(bearing=bearing, seen=now, box=row["box"], score=row["score"])
            result.append(
                dict(
                    **row,
                    track_id=track_id,
                    raw_state=STATES[int(p.argmax())],
                    tracked_probabilities=track["probabilities"].tolist(),
                    tracked_argmax=STATES[int(track["probabilities"].argmax())],
                    tracked_state=self._state(track, now),
                    applicability="unresolved",
                )
            )
        return dict(observations=result, tracks=self.snapshot(now))

    @staticmethod
    def _state(track, now):
        if (
            now - track["seen"] > 0.200001
            or track["hits"] < 2
            or track["probabilities"].max() < 0.6
        ):
            return "Unknown"
        return STATES[int(track["probabilities"].argmax())]

    def snapshot(self, now):
        if self.previous is None or now != self.previous[1]:
            raise ValueError("Snapshot only at the latest causal packet timestamp")
        return [
            dict(
                track_id=k,
                view=t["view"],
                state=self._state(t, now),
                age_s=now - t["seen"],
                confirmations=t["hits"],
                probabilities=t["probabilities"].tolist(),
                applicability="unresolved",
            )
            for k, t in sorted(self.tracks.items())
        ]
