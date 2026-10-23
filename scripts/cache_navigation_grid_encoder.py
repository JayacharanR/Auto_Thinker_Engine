#!/usr/bin/env python3
"""One regenerable frozen-encoder spatial cache, with episode-bounded history."""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.grid_encoders import FrozenGridEncoder  # noqa: E402
from src.navigation.optical_fields import past_average  # noqa: E402
from src.navigation.rectangular_jepa import causal_clip_indices  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.spatial_data import RawContext  # noqa: E402


def cache(args):
    if args.out.exists():
        raise FileExistsError("Encoder cache attempt already exists; no blind restart")
    data = json.loads((args.data / "manifest.json").read_text())
    if json.loads((args.data / "progress.json").read_text())["status"] != "complete":
        raise ValueError("Complete shared optical/annotation corpus first")
    descriptor = json.loads(args.registry.read_text())[args.encoder]
    dimension = 512 if args.encoder == "imagenet_resnet18" else 384
    required = data["samples"] * 3 * 16 * dimension * 2 + 128
    ram_features = getattr(args, "features_ram", False)
    if shutil.disk_usage(args.out.parent).free < 25e9 + (0 if ram_features else required):
        raise RuntimeError("Frozen encoder cache would breach25GB reserve")
    if ram_features and (
        shutil.disk_usage("/tmp").free < required + 1e9
        or psutil.virtual_memory().available < required * 2 + 12e9
    ):
        raise RuntimeError("Insufficient tmpfs/RAM for bounded frozen features")
    check_resources(args.out, 25)
    sources = [
        Path(__file__),
        ROOT / "src/navigation/grid_encoders.py",
        ROOT / "src/navigation/rectangular_jepa.py",
        ROOT / "src/navigation/optical_fields.py",
        ROOT / "src/navigation/spatial_data.py",
        ROOT / "src/jepa/encoder.py",
        ROOT / "reports/navigation_m4_spatial_protocol_20261019.md",
    ]
    request = dict(
        encoder=args.encoder,
        encoder_descriptor=descriptor,
        data_manifest_sha256=digest(args.data / "manifest.json"),
        precision="BF16 frozen forward, FP32 temporal average, FP16 exported tokens",
        maximum_wall_seconds=7200,
        feature_storage="temporary_RAM" if ram_features else "project_filesystem",
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
    )
    for p in sources:
        target = args.out / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    atomic_json(args.out / "request.json", request)
    started = time.monotonic()
    done = 0

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 7200:
            raise InterruptedError("Encoder cache STOP/wall cap")

    def progress(phase="extracting", status="running"):
        elapsed = time.monotonic() - started
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M4 frozen encoder cache",
                encoder=args.encoder,
                frames=done,
                target_frames=data["samples"],
                phase=phase,
                elapsed_s=elapsed,
                frames_per_second=done / max(0.001, elapsed),
                eta_seconds=(data["samples"] - done) * elapsed / max(1, done),
                updated_at=time.time(),
            ),
        )

    progress("reconstructing evicted supplement in RAM")
    raw = RawContext(data)
    raw.preload_missing(check)
    torch.set_num_threads(2)
    model = FrozenGridEncoder(descriptor).cuda()
    feature_path = args.out / "features.npy"
    if ram_features:
        temporary = Path(tempfile.mkdtemp(prefix="auto_thinker_frozen_features_", dir="/tmp"))
        temporary_feature = temporary / "features.npy"
        feature_path.symlink_to(temporary_feature)
        atomic_json(
            args.out / "ram_feature_storage.json",
            dict(
                target=str(temporary_feature),
                expected_bytes=required,
                lifecycle=(
                    "Regenerable RAM cache; retain requests/manifests/weights and exact SHA"
                ),
            ),
        )
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(data["samples"], 3, 16, dimension),
    )
    try:
        for e in data["episodes"]:
            check()
            frames = raw.rgb(e)
            length = len(frames)
            if args.encoder in ("jepa_comma2k19", "jepa_carla", "jepa_random"):
                for start in range(0, length, 8):
                    check()
                    ids = np.array(
                        [causal_clip_indices(i, 0) for i in range(start, min(length, start + 8))]
                    )
                    video = (
                        torch.from_numpy(np.asarray(frames[ids]).copy())
                        .cuda()
                        .permute(0, 2, 1, 3, 4, 5)
                    )
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        value = model.video_features(video)
                    features[e["start"] + start : e["start"] + start + len(ids)] = (
                        value.half().cpu().numpy()
                    )
                    done += len(ids)
                    progress()
            else:
                per_frame = []
                for start in range(0, length, 16):
                    check()
                    rgb = torch.from_numpy(np.asarray(frames[start : start + 16]).copy()).cuda()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        value = (
                            model.frame_features(rgb.flatten(0, 1))
                            .float()
                            .reshape(-1, 3, 16, dimension)
                        )
                    per_frame.append(value.cpu())
                    done += len(rgb)
                    progress()
                averaged = past_average(torch.cat(per_frame, 0))
                features[e["start"] : e["end"]] = averaged.half().numpy()
            features.flush()
        if done != data["samples"]:
            raise ValueError("Encoder cache count changed")
        atomic_json(
            args.out / "manifest.json",
            dict(
                samples=done,
                dimension=dimension,
                encoder=args.encoder,
                features_sha256=digest(args.out / "features.npy"),
                request_sha256=digest(args.out / "request.json"),
                data_manifest_sha256=digest(args.data / "manifest.json"),
                elapsed_s=time.monotonic() - started,
            ),
        )
        progress("complete", status="complete")
        print(
            json.dumps(
                dict(encoder=args.encoder, samples=done, elapsed_s=time.monotonic() - started)
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
                target_frames=data["samples"],
                updated_at=time.time(),
            ),
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--encoder",
        choices=(
            "imagenet_resnet18",
            "dinov2_small",
            "jepa_comma2k19",
            "jepa_carla",
            "jepa_random",
        ),
        required=True,
    )
    p.add_argument("--data", type=Path, default=ROOT / "outputs/navigation_spatial_data_20261019")
    p.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "outputs/navigation_models/navigation_backbones_20261019.json",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--features-ram", action="store_true")
    with RunLock(ROOT / "outputs/.navigation.lock"):
        cache(p.parse_args())
