"""Training-only fixed study split, annotation budgets and lossless RGB access."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.navigation.contracts import VIEWS
from src.navigation.dagger import digest
from src.navigation.dataset import load_policy_sample, shard_records
from src.navigation.label_budget import nested_budgets

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/navigation_m2_cache_20k_20261015"
EXTRA = ROOT / "outputs/navigation_m2_causal_approach_cache_20261018"
DOMAINS = {"original": BASE, "approach": EXTRA}
INPUT_ARRAYS = ("navigation", "ego")
TARGET_ARRAYS = ("current_speed", "speeds", "speed_mask", "brake", "correction", "correction_mask")


def study_manifest():
    pinned = json.loads(
        (ROOT / "outputs/navigation_m2_signal_policy_cache_20261018/request.json").read_text()
    )
    if (
        digest(BASE / "manifest.json") != pinned["base_manifest_sha256"]
        or digest(EXTRA / "manifest.json") != pinned["extra_manifest_sha256"]
    ):
        raise ValueError("Protected original/supplement manifests changed")
    old = json.loads((BASE / "manifest.json").read_text())
    extra = json.loads((EXTRA / "manifest.json").read_text())
    guarded = json.loads(
        (ROOT / "outputs/navigation_m2_causal_history_20261018/training_domains.json").read_text()
    )
    allowed = set(guarded["original"])
    episodes = []
    offset = 0
    for domain, manifest in (("original", old), ("approach", extra)):
        for source in manifest["episodes"]:
            if domain == "original" and source["role"] == "train" and source["data"] not in allowed:
                continue
            if source["partition"] != "train" or source["town"].removesuffix("_Opt") == "Town05":
                raise ValueError("Foreign geography entered research corpus")
            count = source["end"] - source["start"]
            identity = (
                f"{domain}:{source['town']}:{source['route_id']}:{source['seed']}:{source['start']}"
            )
            episodes.append(
                dict(
                    source,
                    id=identity,
                    domain=domain,
                    source_start=source["start"],
                    source_end=source["end"],
                    start=offset,
                    end=offset + count,
                    labels=count,
                )
            )
            offset += count
    train = [e for e in episodes if e["role"] == "train"]
    validation = [e for e in episodes if e["role"] == "validation"]
    if (
        len(train) != 59
        or sum(e["labels"] for e in train) != 16639
        or len(validation) != 13
        or offset != 19461
    ):
        raise ValueError("Protected original/continuous corpus counts changed")
    train_locations = {(e["town"], j) for e in train for j in e["junction_ids"]}
    if train_locations & {(e["town"], j) for e in validation for j in e["junction_ids"]}:
        raise ValueError("Training/validation locations overlap")
    budget = nested_budgets(
        [
            dict(
                id=e["id"],
                role="train",
                partition="train",
                town=e["town"],
                labels=e["labels"],
                family=e["family"],
            )
            for e in train
        ]
    )
    return dict(
        schema="navigation-spatial-study-data-v1",
        episodes=episodes,
        samples=offset,
        source_manifest_sha256={
            name: digest(root / "manifest.json") for name, root in DOMAINS.items()
        },
        base_guard_sha256=digest(ROOT / "reports/navigation_m2_dagger_guard_20261016.json"),
        annotation_budget=budget,
        original_validation_unchanged=True,
        forbidden_sources=[
            "Town05",
            "heldout_dev",
            "diagnostic light-gap pilot",
            "full-label inherited driving/perception heads",
        ],
    )


class RawContext:
    """Original RGB mmap plus a RAM-only reconstruction of evicted supplement RGB."""

    def __init__(self, manifest):
        self.manifest = manifest
        self.arrays = {}
        self.extra_ram = None
        for domain, root in DOMAINS.items():
            self.arrays[domain] = {
                k: np.load(root / f"{k}.npy", mmap_mode="r", allow_pickle=False)
                for k in (*INPUT_ARRAYS, *TARGET_ARRAYS)
            }
            if (root / "rgb.npy").exists():
                self.arrays[domain]["rgb"] = np.load(
                    root / "rgb.npy", mmap_mode="r", allow_pickle=False
                )

    def preload_missing(self, check=lambda: None, progress=lambda n, total: None):
        if "rgb" in self.arrays["approach"]:
            return
        total = json.loads((EXTRA / "manifest.json").read_text())["samples"]
        rgb = np.empty((total, 3, 3, 144, 256), np.uint8)
        done = 0
        for e in self.manifest["episodes"]:
            if e["domain"] != "approach":
                continue
            offset = e["source_start"]
            for shard in sorted(Path(e["data"]).glob("shard_*.zip")):
                check()
                for record in shard_records(shard):
                    packet = load_policy_sample(shard, record)
                    for v, view in enumerate(VIEWS):
                        rgb[offset, v] = np.asarray(
                            Image.fromarray(packet.rgb[view]).resize(
                                (256, 144), Image.Resampling.BILINEAR
                            )
                        ).transpose(2, 0, 1)
                    offset += 1
                    done += 1
                progress(done, total)
            if offset != e["source_end"]:
                raise ValueError("Evicted RGB reconstruction length changed")
        if done != total:
            raise ValueError("Incomplete supplement RAM reconstruction")
        self.extra_ram = rgb
        self.arrays["approach"]["rgb"] = rgb

    def rgb(self, e):
        return self.arrays[e["domain"]]["rgb"][e["source_start"] : e["source_end"]]

    def targets(self, e, device):
        a = self.arrays[e["domain"]]
        return {
            k: torch.from_numpy(np.asarray(a[k][e["source_start"] : e["source_end"]]).copy()).to(
                device
            )[None]
            for k in TARGET_ARRAYS
        }


class SpatialCorpus:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.arrays = {
            k: np.load(self.root / f"{k}.npy", mmap_mode="r", allow_pickle=False)
            for k in (*INPUT_ARRAYS, *TARGET_ARRAYS, "optical")
        }
        self.raw = RawContext(self.manifest)

    def episode(self, e, device, features=None):
        values = {
            k: torch.from_numpy(np.asarray(a[e["start"] : e["end"]]).copy()).to(device)[None]
            for k, a in self.arrays.items()
        }
        inputs = {k: values[k] for k in (*INPUT_ARRAYS, "optical")}
        if features is not None:
            inputs["features"] = (
                torch.from_numpy(np.asarray(features[e["start"] : e["end"]]).copy())
                .to(device)[None]
                .float()
            )
        return inputs, {k: values[k] for k in TARGET_ARRAYS}


def validation_positions(length, device):
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    for start in range(0, length - 31, 32):
        mask[start + 16 : start + 32] = True
    return mask
