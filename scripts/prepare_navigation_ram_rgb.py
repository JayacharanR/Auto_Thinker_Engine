#!/usr/bin/env python3
"""Share verified supplementary raw RGB in tmpfs, without caching CNN features."""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402
from src.navigation.spatial_data import EXTRA, RawContext, study_manifest  # noqa:E402


def main(args):
    link = EXTRA / "rgb.npy"
    if args.cleanup:
        study = json.loads((args.study / "progress.json").read_text())
        if study["status"] == "running":
            raise RuntimeError("Keep the shared RGB while the study runs")
        receipt = json.loads((args.out / "receipt.json").read_text())
        target = Path(receipt["ram_file"])
        with RunLock(ROOT / "outputs/.navigation.lock"):
            if link.is_symlink() and link.resolve() == target:
                link.unlink()
            if target.exists():
                if digest(target) != receipt["sha256"]:
                    raise ValueError("Changed RAM file retained")
                target.unlink()
                target.parent.rmdir()
            atomic_json(args.out / "cleanup.json", dict(cleaned=True, time=time.time()))
        return
    if args.out.exists():
        raise FileExistsError("Keep the prior RAM cache recipe")
    if link.exists() or link.is_symlink():
        raise FileExistsError("RGB path already exists; do not overwrite")
    expected = json.loads((EXTRA / "evicted_image_arrays.json").read_text())["arrays"]["rgb.npy"]
    if (
        shutil.disk_usage("/tmp").free < expected["bytes"] + 1e8
        or psutil.virtual_memory().available < 3 * expected["bytes"] + 12e9
    ):
        raise RuntimeError("Insufficient tmpfs/RAM for bounded shared RGB reconstruction")
    args.out.mkdir(parents=True)
    atomic_json(
        args.out / "request.json",
        dict(
            expected=expected,
            eviction_receipt_sha256=digest(EXTRA / "evicted_image_arrays.json"),
            script_sha256=digest(__file__),
            raw_only=True,
            backbone_features_cached=False,
            source_sha256=digest(ROOT / "src/navigation/spatial_data.py"),
        ),
    )
    started = time.monotonic()

    def check():
        if (args.out / "STOP").exists():
            raise InterruptedError("RAM RGB STOP")

    def progress(done, total):
        atomic_json(
            args.out / "progress.json",
            dict(
                status="running",
                stage="Verified raw RGB in RAM",
                frames=done,
                target_frames=total,
                updated_at=time.time(),
            ),
        )

    raw = RawContext(study_manifest())
    raw.preload_missing(check, progress)
    check()
    directory = Path(tempfile.mkdtemp(prefix="auto_thinker_rgb_", dir="/tmp"))
    target = directory / "rgb.npy"
    np.save(target, raw.extra_ram, allow_pickle=False)
    if target.stat().st_size != expected["bytes"] or digest(target) != expected["sha256"]:
        raise ValueError("Rebuilt raw RGB differs from the preserved original hash; not published")
    target.chmod(0o400)
    link.symlink_to(target)
    atomic_json(
        args.out / "receipt.json",
        dict(
            ram_file=str(target),
            logical_path=str(link),
            sha256=expected["sha256"],
            bytes=expected["bytes"],
            exact_evicted_original_reconstructed=True,
            source_shards_preserved=True,
            model_features_cached=False,
            elapsed_s=time.monotonic() - started,
            lifecycle=(
                'Temporary raw-image cache; remove owned link/tmpfs file after '
                'the study finishes'
            ),
        ),
    )
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            stage="Verified raw RGB in RAM",
            frames=len(raw.extra_ram),
            target_frames=len(raw.extra_ram),
            updated_at=time.time(),
        ),
    )
    print(
        json.dumps(
            dict(
                raw_frames=len(raw.extra_ram),
                exact_original_hash=True,
                seconds=time.monotonic() - started,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/navigation_ram_rgb_20261019")
    parser.add_argument(
        "--study", type=Path, default=ROOT / "outputs/navigation_visual_study_20261019"
    )
    parser.add_argument("--cleanup", action="store_true")
    with RunLock(ROOT / "outputs/.navigation_ram_rgb.lock"):
        main(parser.parse_args())
