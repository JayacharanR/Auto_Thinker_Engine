"""Train-only DAgger cache: expert instant targets, no mixture future supervision."""

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.navigation.contracts import VIEWS
from src.navigation.dagger import correction_records, digest
from src.navigation.dataset import load_policy_sample
from src.navigation.learning_data import HAZARDS, LIGHTS, correction_targets
from src.navigation.runtime import atomic_json


def prepare_corrections(report, destination, base_cache, stop, deadline, progress):
    report, destination, base_cache = Path(report), Path(destination), Path(base_cache)
    assessment = json.loads(report.read_text())
    if not assessment["audit_passed"] or not assessment["correction_data_gate_passed"]:
        raise ValueError("Corrections need the explicit passed integrity/exposure gate")
    if digest(base_cache / "manifest.json") != assessment["internal_manifest_sha256"]:
        raise ValueError("Corrections would change the original internal validation split")
    data_paths = [Path(p) for p in assessment["eligible_correction_paths"]]
    provenance = {
        "assessment": str(report.resolve()),
        "assessment_sha256": digest(report),
        "base_manifest_sha256": digest(base_cache / "manifest.json"),
        "builder_sha256": digest(__file__),
        "episode_outcomes": {str(p): digest(p / "outcome.json") for p in data_paths},
        "schema": "instant-expert-correction-cache-v1",
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["provenance"] != provenance:
            raise ValueError("Correction cache provenance changed")
        return manifest
    if destination.exists():
        raise FileExistsError("Incomplete correction cache retained; choose a fresh directory")
    rows = {e["data"]: e for e in assessment["episodes"] if e["correction_eligible"]}
    count = sum(rows[str(p)]["counts"]["samples"] for p in data_paths)
    required = count * (3 * 3 * 144 * 256 + 3 * 288 * 512)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required + 25e9:
        raise RuntimeError("Correction cache would violate25GB reserve")
    destination.mkdir()
    images = {
        "rgb": np.lib.format.open_memmap(
            destination / "rgb.npy", mode="w+", dtype=np.uint8, shape=(count, 3, 3, 144, 256)
        ),
        "front": np.lib.format.open_memmap(
            destination / "front.npy", mode="w+", dtype=np.uint8, shape=(count, 3, 288, 512)
        ),
    }
    arrays = {
        k: np.zeros((count, *shape), np.float32)
        for k, shape in {
            "navigation": (16, 2),
            "ego": (3,),
            "correction": (8,),
            "correction_mask": (8,),
            "speeds": (8,),
            "speed_mask": (8,),
            "current_speed": (),
            "brake": (),
            "hazards": (4,),
            "distance": (2,),
            "presence": (2,),
            "priority_go": (),
        }.items()
    }
    arrays["light"] = np.zeros(count, np.int64)
    index, episodes = 0, []
    base_manifest = json.loads((base_cache / "manifest.json").read_text())
    reserved = {
        (e["town"], j)
        for e in base_manifest["episodes"]
        if e["role"] == "validation"
        for j in e["junction_ids"]
    }
    for data in data_paths:
        meta = json.loads((data / "episode.json").read_text())
        if (
            meta["partition"] != "train"
            or meta["town"].removesuffix("_Opt") == "Town05"
            or any((meta["town"], j) in reserved for j in meta["junction_ids"])
        ):
            raise ValueError("Correction inputs touch reserved geography")
        start, previous = index, None
        for shard, record, label in correction_records(data):
            if Path(stop).exists() or time.monotonic() >= deadline:
                raise InterruptedError("Correction cache preparation stopped; prefix retained")
            packet = load_policy_sample(shard, record)
            if previous is not None and packet.frame != previous + 2:
                raise ValueError("Correction clips cannot bridge missing labels")
            previous = packet.frame
            if packet.navigation_waypoints.shape != (16, 2):
                raise ValueError("Wrong correction navigation shape")
            for view, name in enumerate(VIEWS):
                raw = packet.rgb[name]
                if raw.shape != (288, 512, 3):
                    raise ValueError("Wrong correction camera shape")
                images["rgb"][index, view] = np.asarray(
                    Image.fromarray(raw).resize((256, 144), Image.Resampling.BILINEAR)
                ).transpose(2, 0, 1)
            images["front"][index] = packet.rgb["front"].transpose(2, 0, 1)
            facts = label["facts"]
            arrays["navigation"][index], arrays["ego"][index] = (
                packet.navigation_waypoints,
                packet.ego_motion,
            )
            arrays["correction"][index], arrays["correction_mask"][index] = correction_targets(
                packet.navigation_waypoints, facts["teacher_planned_path_ego"]
            )
            arrays["current_speed"][index] = facts["teacher_target_speed_mps"]
            arrays["brake"][index] = label["expert_control"][2] > 0.5
            arrays["light"][index] = LIGHTS.get(str(facts["next_light_state"]), 4)
            arrays["hazards"][index] = [facts["teacher_hazards"][k] for k in HAZARDS]
            arrays["priority_go"][index] = (
                facts["teacher_target_speed_mps"] > 1
                and label["expert_control"][2] < 0.5
                and packet.ego_motion[0] < 0.2
                and packet.timestamp > 5
                and (
                    facts["student_requested_control"][2] >= 0.5
                    or facts["student_target_speed_mps"] < 0.15
                )
            )
            for j, key in enumerate(("next_light_distance_m", "next_stop_distance_m")):
                distance = facts[key]
                arrays["presence"][index, j] = distance is not None and distance <= 80
                arrays["distance"][index, j] = min(80, max(0, distance or 0))
            index += 1
            if index % 32 == 0:
                progress(index, count)
        if index - start != rows[str(data)]["counts"]["samples"]:
            raise ValueError("Correction cache length differs from audited eligible labels")
        episodes.append({**meta, "data": str(data), "role": "train", "start": start, "end": index})
    if index != count:
        raise ValueError("Correction cache count mismatch")
    for array in images.values():
        array.flush()
    for name, array in arrays.items():
        np.save(destination / f"{name}.npy", array, allow_pickle=False)
    for path in destination.glob("*.npy"):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    manifest = dict(
        provenance=provenance,
        samples=count,
        episodes=episodes,
        future_speed_supervision="all masked; observed mixture is not an expert future",
        required_image_bytes=required,
    )
    atomic_json(manifest_path, manifest)
    return manifest


class CorrectionPool:
    def __init__(self, root, length):
        self.root, self.length = Path(root), length
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.arrays = {
            p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in self.root.glob("*.npy")
        }
        if self.arrays["speed_mask"].any():
            raise ValueError("Assisted future labels must remain masked")
        self.pools = {"go_correction": [], "stop": []}
        for episode in self.manifest["episodes"]:
            ends = np.arange(episode["start"] + length - 1, episode["end"])
            for name, mask in (
                ("go_correction", self.arrays["priority_go"][ends] > 0.5),
                ("stop", self.arrays["current_speed"][ends] < 0.1),
            ):
                if mask.any():
                    self.pools[name].append(ends[mask])
        if any(not pool for pool in self.pools.values()):
            raise ValueError("Need post-burn-in go corrections and required stops")

    def sample(self, rng, size):
        starts = []
        for _ in range(size):
            pools = self.pools["go_correction" if rng.integers(2) else "stop"]
            ends = pools[int(rng.integers(len(pools)))]
            starts.append(int(rng.choice(ends)) - self.length + 1)
        return starts

    def batch(self, starts, device):
        indices = np.asarray(starts)[:, None] + np.arange(self.length)[None]
        values = {
            k: torch.from_numpy(np.asarray(a[indices]).copy()).to(device)
            for k, a in self.arrays.items()
            if k != "priority_go"
        }
        inputs = {k: values[k] for k in ("rgb", "front", "navigation", "ego")}
        return inputs, {k: v for k, v in values.items() if k not in inputs}
