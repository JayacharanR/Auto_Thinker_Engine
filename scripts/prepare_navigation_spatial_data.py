#!/usr/bin/env python3
"""Freeze label budgets and cache annotation-free native RGB fields, not a fit."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa: E402
from src.navigation.optical_fields import optical_fields  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.spatial_data import (  # noqa: E402
    DOMAINS,
    INPUT_ARRAYS,
    TARGET_ARRAYS,
    study_manifest,
)


def prepare(args):
    if args.out.exists():
        raise FileExistsError("Preserve prior data preparation; choose a fresh output")
    manifest = study_manifest()
    if (
        shutil.disk_usage(args.out.parent).free
        < 25e9 + manifest["samples"] * 3 * 144 * 11 * 2 + 20e6
    ):
        raise RuntimeError("Optical fields would violate25GB reserve")
    check_resources(args.out, 25)
    sources = [
        Path(__file__),
        ROOT / "src/navigation/spatial_data.py",
        ROOT / "src/navigation/optical_fields.py",
        ROOT / "src/navigation/label_budget.py",
        ROOT / "src/navigation/dataset.py",
        ROOT / "reports/navigation_m4_spatial_protocol_20261019.md",
    ]
    request = dict(
        samples=manifest["samples"],
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        maximum_wall_seconds=7200,
        carla_labels_used_for_optical_fields=0,
    )
    for p in sources:
        dest = args.out / "source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    atomic_json(args.out / "request.json", request)
    original = {
        domain: {
            k: np.load(root / f"{k}.npy", mmap_mode="r", allow_pickle=False)
            for k in (*INPUT_ARRAYS, *TARGET_ARRAYS)
        }
        for domain, root in DOMAINS.items()
    }
    arrays = {
        k: np.lib.format.open_memmap(
            args.out / f"{k}.npy",
            mode="w+",
            dtype=a.dtype,
            shape=(manifest["samples"], *a.shape[1:]),
        )
        for k, a in original["original"].items()
    }
    fields = np.lib.format.open_memmap(
        args.out / "optical.npy",
        mode="w+",
        dtype=np.float16,
        shape=(manifest["samples"], 3, 144, 11),
    )
    started = time.monotonic()
    done = 0
    raw_hashes = {}
    torch.set_num_threads(2)

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 7200:
            raise InterruptedError("Optical cache STOP/wall cap")

    try:
        for e in manifest["episodes"]:
            check()
            for key, a in original[e["domain"]].items():
                arrays[key][e["start"] : e["end"]] = a[e["source_start"] : e["source_end"]]
            i = e["start"]
            for shard in sorted(Path(e["data"]).glob("shard_*.zip")):
                check()
                raw_hashes[str(shard)] = digest(shard)
                for record in shard_records(shard):
                    packet = load_policy_sample(shard, record)
                    if not np.array_equal(
                        packet.navigation_waypoints, arrays["navigation"][i]
                    ) or not np.array_equal(packet.ego_motion, arrays["ego"][i]):
                        raise ValueError("Native field/cache source frame mismatch")
                    rgb = torch.from_numpy(
                        np.stack(
                            [packet.rgb[v].transpose(2, 0, 1) for v in ("front", "left", "right")]
                        )
                    ).to(args.device)
                    with torch.inference_mode():
                        fields[i] = optical_fields(rgb).half().cpu().numpy()
                    i += 1
                    done += 1
                fields.flush()
                elapsed = time.monotonic() - started
                atomic_json(
                    args.out / "progress.json",
                    dict(
                        status="running",
                        stage="M4 optical data cache",
                        frames=done,
                        target_frames=manifest["samples"],
                        frames_per_second=done / elapsed,
                        eta_seconds=(manifest["samples"] - done) * elapsed / done,
                        updated_at=time.time(),
                    ),
                )
            if i != e["end"]:
                raise ValueError("Optical episode length mismatch")
        for a in arrays.values():
            a.flush()
        fields.flush()
        manifest.update(
            request_sha256=digest(args.out / "request.json"),
            source_shard_sha256=raw_hashes,
            array_sha256={p.name: digest(p) for p in args.out.glob("*.npy")},
            elapsed_s=time.monotonic() - started,
        )
        atomic_json(args.out / "manifest.json", manifest)
        atomic_json(
            args.out / "progress.json",
            dict(
                status="complete",
                stage="M4 optical data cache",
                frames=done,
                target_frames=manifest["samples"],
                updated_at=time.time(),
            ),
        )
        print(
            json.dumps(
                dict(
                    samples=done,
                    elapsed_s=manifest["elapsed_s"],
                    labels=[
                        (a["nominal_fraction"], a["actual_labels"])
                        for a in manifest["annotation_budget"]["allocations"]
                    ],
                )
            ),
            flush=True,
        )
    except Exception as error:
        atomic_json(
            args.out / "progress.json",
            dict(
                status="stopped" if isinstance(error, InterruptedError) else "failed",
                error=str(error),
                frames=done,
                target_frames=manifest["samples"],
                updated_at=time.time(),
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/navigation_spatial_data_20261019"
    )
    parser.add_argument("--device", default="cuda")
    with RunLock(ROOT / "outputs/.navigation.lock"):
        prepare(parser.parse_args())
