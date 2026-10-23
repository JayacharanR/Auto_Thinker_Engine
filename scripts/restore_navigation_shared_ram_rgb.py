#!/usr/bin/env python3
"""Restore the exact missing shared raw-RGB cache without changing a running fit."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402
from src.navigation.spatial_data import EXTRA, RawContext  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.out / "restoration_source.py")
    original_receipt = ROOT / "outputs/navigation_ram_rgb_20261019/receipt.json"
    receipt = json.loads(original_receipt.read_text())
    eviction_path = EXTRA / "evicted_image_arrays.json"
    eviction = json.loads(eviction_path.read_text())
    expected = eviction["arrays"]["rgb.npy"]
    link, target = EXTRA / "rgb.npy", Path(receipt["ram_file"])
    if (
        not link.is_symlink() or link.exists() or Path(os.readlink(link)) != target
        or target.parent.parent != Path("/tmp")
        or not target.parent.name.startswith("auto_thinker_rgb_")
        or target.name != "rgb.npy"
        or receipt["sha256"] != expected["sha256"] or receipt["bytes"] != expected["bytes"]
    ):
        raise ValueError("Only the registered broken owned RAM-cache link can be restored")
    if (
        psutil.virtual_memory().available < 3 * expected["bytes"] + 12e9
        or shutil.disk_usage("/tmp").free < expected["bytes"] + 1e9
    ):
        raise RuntimeError("Insufficient bounded RAM/tmpfs for exact raw-pixel cache restoration")
    manifest = ROOT / "outputs/navigation_spatial_data_repaired_20261019/manifest.json"
    atomic_json(args.out / "request.json", dict(
        original_receipt_sha256=digest(original_receipt), eviction_sha256=digest(eviction_path),
        expected=expected, target=str(target), logical_path=str(link),
        source_sha256=digest(__file__), corrected_manifest_sha256=digest(manifest),
        exact_raw_pixels_only=True, no_model_feature_or_label_cache_change=True,
    ))
    started = time.monotonic()

    def check():
        if (args.out / "STOP").exists():
            raise InterruptedError("Shared raw RGB restoration STOP")
        if psutil.virtual_memory().available < 12e9:
            raise InterruptedError("Maintain12GBavailable RAM during reconstruction")

    def progress(done, total):
        atomic_json(args.out / "progress.json", dict(
            status="running", stage="Verified raw RGB in RAM", frames=done,
            target_frames=total, updated_at=time.time(),
        ))

    with RunLock(ROOT / "outputs/.navigation_ram_rgb.lock"):
        for path, expected_sha in eviction["preserved_source_shards"].items():
            check()
            if digest(path) != expected_sha:
                raise ValueError("Preserved raw RGB source shard changed")
        raw = RawContext(json.loads(manifest.read_text()))
        raw.preload_missing(check, progress)
        target.parent.mkdir(exist_ok=True)
        temporary = target.parent / f"rgb.restore.{os.getpid()}.npy"
        if temporary.exists() or target.exists():
            raise FileExistsError("Do not replace an existing RAM cache or restoration temporary")
        np.save(temporary, raw.extra_ram, allow_pickle=False)
        if temporary.stat().st_size != expected["bytes"] or digest(temporary) != expected["sha256"]:
            raise ValueError("Reconstructed pixels differ from the original; not published")
        temporary.chmod(0o400)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, target)  # Atomic publication, refusing any concurrent existing target.
        temporary.unlink()
        if digest(link) != expected["sha256"]:
            raise ValueError("Published logical RGB cache does not match original")
        atomic_json(args.out / "receipt.json", dict(
            target=str(target), logical_path=str(link), sha256=expected["sha256"],
            bytes=expected["bytes"], exact_original_restored=True,
            existing_link_unchanged=True, existing_trainer_sources_unchanged=True,
            preexisting_worker_private_reconstruction_unchanged=True,
            source_shards_preserved=True, backbone_features_cached=False,
            elapsed_s=time.monotonic() - started,
            cleanup="Keep through corrected67study and all4fact-study fits; owned RAM cache only",
        ))
        atomic_json(args.out / "progress.json", dict(
            status="complete", stage="Verified raw RGB in RAM", frames=len(raw.extra_ram),
            target_frames=len(raw.extra_ram), updated_at=time.time(),
        ))
    print(json.dumps(dict(bytes=expected["bytes"], exact_original_restored=True,
                          seconds=time.monotonic() - started)), flush=True)


if __name__ == "__main__":
    main()
