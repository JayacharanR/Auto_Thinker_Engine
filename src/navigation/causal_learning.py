"""TRAIN-only continuous expert cache and complete-episode causal forwarding."""

import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint

from src.navigation.contracts import VIEWS
from src.navigation.dagger import digest
from src.navigation.dataset import load_policy_sample, shard_records
from src.navigation.learning_data import correction_targets, future_speed_targets
from src.navigation.runtime import atomic_json

INPUTS = ("rgb", "front", "navigation", "ego")


def stream_forward(model, inputs, chunk=8, gradients=False):
    """Retain GRU history/gradients from the episode start; visual chunks recompute."""
    outputs, hidden = {}, None
    for start in range(0, inputs["rgb"].shape[1], chunk):
        values = [inputs[k][:, start : start + chunk] for k in INPUTS]
        if gradients:
            predicted, hidden = checkpoint(model, *values, hidden, use_reentrant=False)
        else:
            predicted, hidden = model(*values, hidden)
        for key, value in predicted.items():
            outputs.setdefault(key, []).append(value)
    return {key: torch.cat(values, dim=1) for key, values in outputs.items()}


def original_training_episodes(manifest, allowed):
    selected, excluded = [], []
    for e in manifest["episodes"]:
        if e["role"] != "train":
            continue
        target = selected if set(e["junction_ids"]) <= set(allowed[e["town"]]) else excluded
        target.append(e)
    if {e["family"] for e in selected} != {e["family"] for e in manifest["episodes"]}:
        raise ValueError("Strict TRAIN filter leaves a missing original family")
    return selected, excluded


def prepare_approach_cache(collection, destination, base, stop, deadline, progress):
    collection, destination, base = map(Path, (collection, destination, base))
    audit = json.loads((collection / "data_audit.json").read_text())
    request = json.loads((collection / "request.json").read_text())
    plan = json.loads((collection / "plan.json").read_text())
    if (
        not audit["audit_passed"]
        or not plan["varied_waits"]
        or json.loads((collection / "progress.json").read_text())["status"] != "complete"
        or digest(collection / "plan.json") != request["plan_sha256"]
    ):
        raise ValueError("Completed source-frozen varied-wait TRAIN collection required")
    if any(digest(collection / "source" / p) != h for p, h in request["source_hashes"].items()):
        raise ValueError("Collection source archive changed")
    eligible = {
        e["path"] for e in audit["continuous_approach_episodes"] if e["approach_imitation_eligible"]
    }
    source_episodes = [e for e in audit["episodes"] if e["path"] in eligible]
    if not source_episodes:
        raise ValueError("No accepted continuous expert episodes")
    provenance = dict(
        collection=str(collection.resolve()),
        collection_audit_sha256=digest(collection / "data_audit.json"),
        base_manifest_sha256=digest(base / "manifest.json"),
        builder_sha256=digest(__file__),
        schema="continuous-expert-TRAIN-cache-v1",
        perception_labels_approved=False,
    )
    if (destination / "manifest.json").exists():
        if (destination / "evicted_image_arrays.json").exists():
            raise FileNotFoundError(
                "Regenerable image cache evicted; rebuild from preserved shards at a fresh path"
            )
        m = json.loads((destination / "manifest.json").read_text())
        if m["provenance"] != provenance:
            raise ValueError("Supplement cache provenance changed")
        return m
    if destination.exists():
        raise FileExistsError("Incomplete cache preserved; choose fresh output")
    original = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in base.glob("*.npy")}
    count = sum(e["samples"] for e in source_episodes)
    required = sum(count * np.prod(a.shape[1:]) * a.dtype.itemsize for a in original.values())
    if shutil.disk_usage(destination.parent).free < required + 25e9:
        raise RuntimeError("Supplement cache would violate25GB disk reserve")
    destination.mkdir(parents=True)
    arrays = {
        name: np.lib.format.open_memmap(
            destination / f"{name}.npy", mode="w+", dtype=a.dtype, shape=(count, *a.shape[1:])
        )
        for name, a in original.items()
    }
    episodes = []
    index = 0
    for e in source_episodes:
        data = Path(e["path"])
        meta = json.loads((data / "episode.json").read_text())
        if meta["partition"] != "train" or meta["town"].removesuffix("_Opt") == "Town05":
            raise ValueError("Supplement touches held-out data")
        start = index
        for shard in sorted(data.glob("shard_*.zip")):
            if Path(stop).exists() or time.monotonic() >= deadline:
                raise InterruptedError("Cache STOP/deadline")
            with zipfile.ZipFile(shard) as archive:
                for record in shard_records(shard):
                    packet = load_policy_sample(shard, record)
                    label = json.loads(archive.read(f"{record['prefix']}/targets.json"))
                    f = label["facts"]
                    for j, view in enumerate(VIEWS):
                        arrays["rgb"][index, j] = np.asarray(
                            Image.fromarray(packet.rgb[view]).resize(
                                (256, 144), Image.Resampling.BILINEAR
                            )
                        ).transpose(2, 0, 1)
                    arrays["front"][index] = packet.rgb["front"].transpose(2, 0, 1)
                    arrays["navigation"][index] = packet.navigation_waypoints
                    arrays["ego"][index] = packet.ego_motion
                    arrays["current_speed"][index] = f["teacher_target_speed_mps"]
                    arrays["brake"][index] = label["expert_control"][2] > 0.5
                    arrays["correction"][index], arrays["correction_mask"][index] = (
                        correction_targets(
                            packet.navigation_waypoints, f["teacher_planned_path_ego"]
                        )
                    )
                    # Unapproved perception arrays deliberately have no supervised loss.
                    for name in ("light", "hazards", "distance", "presence"):
                        arrays[name][index] = 0
                    index += 1
            progress(index, count)
        arrays["speeds"][start:index], arrays["speed_mask"][start:index] = future_speed_targets(
            arrays["current_speed"][start:index]
        )
        episodes.append(
            dict(
                meta,
                role="train",
                start=start,
                end=index,
                data=str(data),
                perception_supervision=False,
            )
        )
    if index != count:
        raise ValueError("Supplement count changed")
    for a in arrays.values():
        a.flush()
    manifest = dict(
        provenance=provenance, samples=count, episodes=episodes, required_bytes=int(required)
    )
    atomic_json(destination / "manifest.json", manifest)
    return manifest


class EpisodeCache:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.arrays = {
            p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in self.root.glob("*.npy")
        }

    def episode(self, e, device):
        if any(k not in self.arrays for k in INPUTS):
            raise FileNotFoundError(
                "Raw image arrays evicted; rebuild from preserved lossless shards "
                "at a fresh cache path"
            )
        values = {
            k: torch.from_numpy(np.asarray(a[e["start"] : e["end"]]).copy()).to(device)[None]
            for k, a in self.arrays.items()
        }
        return {k: values[k] for k in INPUTS}, {k: v for k, v in values.items() if k not in INPUTS}
