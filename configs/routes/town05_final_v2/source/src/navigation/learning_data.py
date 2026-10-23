"""Training-only preparation: whole-location split, raw image cache, future labels."""

import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.navigation.contracts import VIEWS
from src.navigation.dataset import load_policy_sample, shard_records
from src.navigation.runtime import atomic_json

LIGHTS = {"None": 0, "Red": 1, "Yellow": 2, "Green": 3}
HAZARDS = ("vehicle_hazard", "walker_hazard", "light_hazard", "stop_sign_hazard")


def split_episode_locations(episodes, seed=42):
    """Connected junction groups stay together; never split overlapping clips."""
    parent = list(range(len(episodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, episode in enumerate(episodes):
        if episode["town"].removesuffix("_Opt") == "Town05" or episode["partition"] != "train":
            raise ValueError("Only approved training locations may enter the learning cache")
        for junction in episode["junction_ids"]:
            key = episode["town"], junction
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    groups = defaultdict(list)
    for i in range(len(episodes)):
        groups[find(i)].append(i)
    components = sorted(groups.values(), key=lambda ids: min(ids))
    if len(components) < 3:
        raise ValueError("Too few independent training-location groups for internal validation")
    rng = np.random.default_rng(seed)
    families = {e["family"] for e in episodes}
    for attempt in range(1000):
        order = rng.permutation(len(components))
        selected = set(int(i) for i in order[: max(1, math.ceil(len(components) * 0.2))])
        val = {i for g in selected for i in components[g]}
        if {episodes[i]["family"] for i in val} == families and {
            e["family"] for i, e in enumerate(episodes) if i not in val
        } == families:
            return ["validation" if i in val else "train" for i in range(len(episodes))], {
                "seed": seed,
                "selection_attempt": attempt,
                "rule": "First seeded 20% junction-component split covering all families "
                "on both sides",
                "components": components,
                "validation_components": sorted(selected),
                "scope": "Internal validation uses only collection training geography; "
                "not heldout_dev",
            }
    raise ValueError("No internal location split with all six families")


def correction_targets(navigation, teacher_path):
    """Lateral route correction at 4,8,...32m; mask beyond the teacher's path."""
    nav = np.asarray(navigation, dtype=np.float32)
    teacher = np.vstack([np.zeros((1, 2)), np.asarray(teacher_path, dtype=np.float32)])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(teacher, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-5]
    teacher, arc = teacher[keep], arc[keep]
    distances = np.arange(4, 33, 4)
    desired = np.column_stack([np.interp(distances, arc, teacher[:, i]) for i in (0, 1)])
    points = nav[1::2]
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-5)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    correction = np.sum((desired - points) * normal, axis=1)
    return np.clip(correction, -2, 2), (distances <= arc[-1]).astype(np.float32)


def future_speed_targets(speeds):
    """Targets may look forward; the cached policy inputs never do."""
    speeds = np.asarray(speeds, dtype=np.float32)
    indices = np.arange(len(speeds))[:, None] + np.arange(5, 41, 5)[None]
    valid = indices < len(speeds)
    return speeds[np.minimum(indices, len(speeds) - 1)], valid.astype(np.float32)


def prepare_cache(collection, destination, stop, deadline, progress):
    collection, destination = Path(collection).resolve(), Path(destination).resolve()
    audit_path = collection / "data_audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit["audit_passed"] or audit["status"] != "complete":
        raise ValueError("Learning requires the completed collection integrity audit")
    review = collection / "collection_review.json"
    if review.exists() and json.loads(review.read_text())["status"] == "superseded_do_not_train":
        raise ValueError("This collection was explicitly excluded from training")
    provenance = {
        "collection": str(collection),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "request_sha256": hashlib.sha256((collection / "request.json").read_bytes()).hexdigest(),
        "image_cache": "Three 256x144 RGB views and original front 512x288, uint8; no features",
        "schema": "camera-student-cache-v1",
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["provenance"] != provenance:
            raise ValueError("Learning cache provenance changed")
        return manifest
    if destination.exists():
        raise FileExistsError("Incomplete cache preserved; choose a new cache directory")
    episodes = []
    request = json.loads((collection / "request.json").read_text())
    for entry in audit["episodes"]:
        data = Path(entry["path"])
        outcome = json.loads((data / "outcome.json").read_text())
        if not outcome["imitation_eligible"]:
            continue
        meta = json.loads((data / "episode.json").read_text())
        if (
            meta["collection_catalog_sha256"] != request["catalog_sha256"]
            or meta["route_id"] not in request["geometry_eligibility"]["kept_route_ids"]
        ):
            raise ValueError("Episode is outside the frozen collection eligibility")
        episodes.append(
            {
                **meta,
                "data": str(data),
                "samples": entry["samples"],
                "outcome_sha256": hashlib.sha256((data / "outcome.json").read_bytes()).hexdigest(),
            }
        )
    roles, split = split_episode_locations(episodes)
    count = sum(e["samples"] for e in episodes)
    required = count * (3 * 144 * 256 * 3 + 288 * 512 * 3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required + 25e9:
        raise RuntimeError("Raw image cache would violate the 25GB disk reserve")
    destination.mkdir()
    rgb = np.lib.format.open_memmap(
        destination / "rgb.npy", mode="w+", dtype=np.uint8, shape=(count, 3, 3, 144, 256)
    )
    front = np.lib.format.open_memmap(
        destination / "front.npy", mode="w+", dtype=np.uint8, shape=(count, 3, 288, 512)
    )
    arrays = {
        "navigation": np.zeros((count, 16, 2), np.float32),
        "ego": np.zeros((count, 3), np.float32),
        "correction": np.zeros((count, 8), np.float32),
        "correction_mask": np.zeros((count, 8), np.float32),
        "speeds": np.zeros((count, 8), np.float32),
        "speed_mask": np.zeros((count, 8), np.float32),
        "current_speed": np.zeros(count, np.float32),
        "brake": np.zeros(count, np.float32),
        "light": np.zeros(count, np.int64),
        "hazards": np.zeros((count, 4), np.float32),
        "distance": np.zeros((count, 2), np.float32),
        "presence": np.zeros((count, 2), np.float32),
    }
    index = 0
    for episode, role in zip(episodes, roles):
        episode.update(role=role, start=index)
        start = index
        for shard in sorted(Path(episode["data"]).glob("shard_*.zip")):
            if Path(stop).exists() or time.monotonic() >= deadline:
                raise InterruptedError(
                    "Stopped during cache preparation; incomplete cache retained"
                )
            with zipfile.ZipFile(shard) as archive:
                for record in shard_records(shard):
                    packet = load_policy_sample(shard, record)
                    if packet.navigation_waypoints.shape != (16, 2):
                        raise ValueError("Expected the declared 16-point navigation input")
                    for v, view in enumerate(VIEWS):
                        raw = packet.rgb[view]
                        if raw.shape != (288, 512, 3):
                            raise ValueError("Expected original 512x288 camera images")
                        image = Image.fromarray(raw).resize((256, 144), Image.Resampling.BILINEAR)
                        rgb[index, v] = np.asarray(image).transpose(2, 0, 1)
                    front[index] = packet.rgb["front"].transpose(2, 0, 1)
                    label = json.loads(archive.read(f"{record['prefix']}/targets.json"))
                    facts = label["facts"]
                    arrays["navigation"][index] = packet.navigation_waypoints
                    arrays["ego"][index] = packet.ego_motion
                    arrays["correction"][index], arrays["correction_mask"][index] = (
                        correction_targets(
                            packet.navigation_waypoints, facts["teacher_planned_path_ego"]
                        )
                    )
                    arrays["current_speed"][index] = facts["teacher_target_speed_mps"]
                    arrays["brake"][index] = label["expert_control"][2] > 0.5
                    arrays["light"][index] = LIGHTS.get(str(facts["next_light_state"]), 4)
                    arrays["hazards"][index] = [facts["teacher_hazards"][k] for k in HAZARDS]
                    for j, key in enumerate(("next_light_distance_m", "next_stop_distance_m")):
                        distance = facts[key]
                        arrays["presence"][index, j] = distance is not None and distance <= 80
                        arrays["distance"][index, j] = min(80, max(0, distance or 0))
                    index += 1
            progress(index, count)
        if index - start != episode["samples"]:
            raise ValueError("Cache and audited episode lengths differ")
        arrays["speeds"][start:index], arrays["speed_mask"][start:index] = future_speed_targets(
            arrays["current_speed"][start:index]
        )
        episode["end"] = index
    rgb.flush()
    front.flush()
    del rgb, front
    for name, values in arrays.items():
        np.save(destination / f"{name}.npy", values, allow_pickle=False)
    for path in destination.glob("*.npy"):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    manifest = {
        "provenance": provenance,
        "samples": count,
        "episodes": episodes,
        "split": split,
        "required_image_bytes": required,
        "target_horizons_seconds": [0.5 * i for i in range(1, 9)],
        "path_distances_m": list(range(4, 33, 4)),
    }
    atomic_json(manifest_path, manifest)
    return manifest


class ClipCache:
    def __init__(self, root, length=16):
        self.root, self.length = Path(root), length
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.arrays = {
            p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in self.root.glob("*.npy")
        }
        self.pools = defaultdict(list)
        self.validation = []
        for episode in self.manifest["episodes"]:
            if episode["end"] - episode["start"] < length:
                continue
            if episode["role"] == "train":
                self.pools[episode["family"]].append(episode)
            else:
                self.validation.extend(range(episode["start"], episode["end"] - length + 1, length))
        self.families = sorted(self.pools)
        if not self.validation or len(self.families) != 6:
            raise ValueError("Need all six training families and nonempty internal validation")

    def sample(self, rng, size):
        starts = []
        for _ in range(size):
            family = self.families[int(rng.integers(len(self.families)))]
            episodes = self.pools[family]
            episode = episodes[int(rng.integers(len(episodes)))]
            # Equal go/stop sampling when both exist, to expose permanent-stop collapse.
            ends = np.arange(episode["start"] + self.length - 1, episode["end"])
            stopped = self.arrays["current_speed"][ends] < 0.1
            chosen = ends[stopped == bool(rng.integers(2))]
            end = int(rng.choice(chosen if len(chosen) else ends))
            starts.append(end - self.length + 1)
        return starts

    def batch(self, starts, device):
        indices = np.asarray(starts)[:, None] + np.arange(self.length)[None]
        result = {
            k: torch.from_numpy(np.asarray(a[indices]).copy()).to(device)
            for k, a in self.arrays.items()
        }
        inputs = {k: result[k] for k in ("rgb", "front", "navigation", "ego")}
        targets = {k: v for k, v in result.items() if k not in inputs}
        return inputs, targets
