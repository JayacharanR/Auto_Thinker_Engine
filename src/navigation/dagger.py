"""Collection-only expert corrections; never imported by camera inference."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.navigation.contracts import ControlCommand


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def internal_location_guard(manifest, assignments):
    """Preserve the already frozen internal validation junctions during aggregation."""
    reserved = {town: set() for town in assignments}
    observed_train = {town: set() for town in assignments}
    for episode in manifest["episodes"]:
        town = episode["town"]
        if (
            town.removesuffix("_Opt") == "Town05"
            or episode["partition"] != "train"
            or episode["role"] not in ("train", "validation")
            or not set(episode["junction_ids"]) <= set(assignments[town]["train"])
        ):
            raise ValueError("Correction source must use approved training geography")
        target = reserved if episode["role"] == "validation" else observed_train
        target[town].update(episode["junction_ids"])
    if any(reserved[t] & observed_train[t] for t in assignments):
        raise ValueError("Internal training/validation junction overlap")
    return {
        "reserved_internal_junctions": {t: sorted(v) for t, v in reserved.items()},
        "allowed_junctions": {
            t: sorted(set(a["train"]) - reserved[t]) for t, a in assignments.items()
        },
    }


def validate_guard(path, catalog_dir, town, subset):
    guard = json.loads(Path(path).read_text())
    catalog_dir = Path(catalog_dir)
    manifest = Path(guard["internal_manifest"])
    if digest(manifest) != guard["internal_manifest_sha256"]:
        raise ValueError("Internal split manifest changed")
    if digest(catalog_dir / "catalog.json") != guard["catalog_sha256"]:
        raise ValueError("DAgger catalog changed")
    assignments = json.loads((catalog_dir.parent / "locations.json").read_text())["assignments"]
    expected = internal_location_guard(json.loads(manifest.read_text()), assignments)
    if any(guard[key] != value for key, value in expected.items()):
        raise ValueError("DAgger guard changes the frozen internal split")
    if not subset:
        raise ValueError("DAgger requires explicit eligible route IDs")
    ids = [int(r) for r in subset.split(",")]
    entries = {
        e["route_id"]: e
        for e in json.loads((catalog_dir / "catalog.json").read_text())["encounters"]
    }
    for rid in ids:
        entry = entries[rid]
        if (
            entry["town"] != town
            or rid not in guard["geometry_eligibility"]["kept_route_ids"]
            or not set(entry["crossed_junction_ids"]) <= set(guard["allowed_junctions"][town])
        ):
            raise ValueError("DAgger route touches reserved geography")
    return {**guard, "guard_sha256": digest(path)}


class CorrectionMixer:
    """Training behavior only: student control, privileged braking and labeled recovery.

    A recovery is permitted only while the expert would go. It is not an
    evaluation controller or a safety guarantee. Every teacher-selected tick is
    marked as an intervention even when its command matches the student's.
    """

    def __init__(self, stall_seconds=4.0, recovery_seconds=3.0):
        self.stall_seconds, self.recovery_seconds = stall_seconds, recovery_seconds
        self.stalled_since = self.recover_until = self.previous_time = None
        self.counts = Counter()

    def select(self, timestamp, student, teacher, speed, teacher_speed, hazard):
        for control in (student, teacher):
            ControlCommand(*control).validate()
        if not np.isfinite([timestamp, speed, teacher_speed]).all():
            raise ValueError("Nonfinite correction state")
        if self.previous_time is not None and abs(timestamp - self.previous_time - 0.05) > 1e-5:
            raise ValueError("Correction mixer requires contiguous 20Hz ticks")
        self.previous_time = timestamp
        if hazard or teacher[2] > 0 or teacher_speed < 0.1:
            self.stalled_since = self.recover_until = None
            source = "teacher_hazard" if hazard else "teacher_braking"
            chosen = teacher
        elif self.recover_until is not None and timestamp < self.recover_until:
            source, chosen = "teacher_recovery", teacher
        else:
            self.recover_until = None
            would_go = teacher_speed > 1 and teacher[1] > 0 and teacher[2] == 0
            if speed < 0.2 and would_go:
                if self.stalled_since is None:
                    self.stalled_since = timestamp
            else:
                self.stalled_since = None
            if (
                self.stalled_since is not None
                and timestamp - self.stalled_since >= self.stall_seconds
            ):
                self.recover_until = timestamp + self.recovery_seconds
                self.stalled_since = None
                self.counts["recovery_events"] += 1
                source, chosen = "teacher_recovery", teacher
            else:
                source, chosen = "student", student
        self.counts[source] += 1
        return tuple(chosen), None if source == "student" else f"privileged_{source}"


def correction_records(root):
    """Instantaneous expert targets only; never turn an assisted rollout into a demo."""
    import zipfile

    from src.navigation.dataset import shard_records

    root = Path(root)
    outcome = root / "outcome.json"
    if not outcome.exists() or not json.loads(outcome.read_text()).get(
        "correction_labels_eligible"
    ):
        return
    for shard in sorted(root.glob("shard_*.zip")):
        with zipfile.ZipFile(shard) as archive:
            for record in shard_records(shard):
                target = json.loads(archive.read(f"{record['prefix']}/targets.json"))
                if not target["facts"].get("expert_label_valid"):
                    continue
                yield shard, record, target
