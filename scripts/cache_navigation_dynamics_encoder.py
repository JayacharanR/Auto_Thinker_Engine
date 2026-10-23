#!/usr/bin/env python3
"""Cache causal RGB tokens for the repaired factual corpus after clean JEPA fitting."""

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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.contracts import VIEWS  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa:E402
from src.navigation.grid_encoders import FrozenGridEncoder  # noqa:E402
from src.navigation.rectangular_jepa import causal_clip_indices  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--after-study", type=Path, required=True)
    parser.add_argument(
        "--after-stage",
        type=Path,
        help="Optional additional serial dependency, preserving the actual encoder-study identity",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("Preserve prior causal-encoder cache attempts")
    data = json.loads((args.data / "manifest.json").read_text())
    args.out.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "src/navigation/grid_encoders.py",
        ROOT / "src/navigation/rectangular_jepa.py",
        ROOT / "src/jepa/encoder.py",
        ROOT / "src/navigation/dataset.py",
        ROOT / "src/navigation/contracts.py",
        ROOT / "reports/navigation_dynamics_data_protocol_20261019.md",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "queued_request.json",
        dict(
            data_manifest_sha256=digest(args.data / "manifest.json"),
            source_hashes=hashes,
            after_study=str(args.after_study.resolve()),
            after_stage=str(args.after_stage.resolve()) if args.after_stage else None,
            storage="regenerable temporary_RAM",
            no_training_updates=True,
        ),
    )
    started, done, status = time.monotonic(), 0, "running"

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M5 past-only factual encoder cache",
                frames=done,
                target_frames=data["samples"],
                phase=phase,
                updated_at=time.time(),
                elapsed_s=time.monotonic() - started,
            ),
        )

    def check():
        if (args.out / "STOP").exists():
            raise InterruptedError("Dynamics cache STOP")
        if time.monotonic() - started > 7200:
            raise InterruptedError("Dynamics cache2h wall cap")

    try:
        progress("Waiting for full corrected study; no concurrent simulator or fit")
        for dependency in [args.after_study] + ([args.after_stage] if args.after_stage else []):
            while True:
                if (args.out / "STOP").exists():
                    raise InterruptedError("Dynamics cache STOP while queued")
                path = dependency / "progress.json"
                if path.exists():
                    prior = json.loads(path.read_text())["status"]
                    if prior != "running":
                        if prior != "complete":
                            raise RuntimeError(
                                f"Dependency {dependency} ended {prior}; inspect before encoding"
                            )
                        break
                time.sleep(3)
        started = time.monotonic()
        if any(digest(ROOT / p) != h for p, h in hashes.items()):
            raise ValueError("Queued causal encoder source changed")
        descriptor_path = args.after_study / "jepa_pretraining/encoder_descriptor.json"
        descriptor = json.loads(descriptor_path.read_text())
        required = data["samples"] * 48 * 384 * 2 + 128
        if (
            shutil.disk_usage("/tmp").free < required + 1e9
            or psutil.virtual_memory().available < required * 2 + 12e9
        ):
            raise RuntimeError("Insufficient bounded tmpfs/RAM for factual features")
        with RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 25)
            atomic_json(
                args.out / "request.json",
                dict(
                    descriptor=descriptor,
                    descriptor_sha256=digest(descriptor_path),
                    data_manifest_sha256=digest(args.data / "manifest.json"),
                    source_hashes=hashes,
                    history_frames=8,
                    three_views=True,
                    spatial_tokens=48,
                    dimension=384,
                    precision="BF16 frozen forward,FP16 exports; same causal clip as deployment",
                    future_rgb_target_only=True,
                    action_or_outcome_inputs_to_encoder=False,
                    maximum_wall_seconds=7200,
                ),
            )
            temporary = Path(tempfile.mkdtemp(prefix="auto_thinker_factual_features_", dir="/tmp"))
            target = temporary / "features.npy"
            (args.out / "features.npy").symlink_to(target)
            atomic_json(
                args.out / "ram_storage.json",
                dict(
                    target=str(target),
                    logical_path=str(args.out / "features.npy"),
                    reproducible=True,
                ),
            )
            features = np.lib.format.open_memmap(
                target, mode="w+", dtype=np.float16, shape=(data["samples"], 48, 384)
            )
            torch.set_num_threads(2)
            encoder = FrozenGridEncoder(descriptor).cuda()
            shard_hashes = {}
            for episode in data["episodes"]:
                check()
                rgb = []
                for shard in episode["source_shards"]:
                    check()
                    if digest(shard["path"]) != shard["sha256"]:
                        raise ValueError("Factual source shard changed")
                    shard_hashes[shard["path"]] = shard["sha256"]
                    for record in shard_records(shard["path"]):
                        packet = load_policy_sample(shard["path"], record)
                        rgb.append(
                            np.stack(
                                [
                                    np.asarray(
                                        Image.fromarray(packet.rgb[view]).resize(
                                            (256, 144), Image.Resampling.BILINEAR
                                        )
                                    ).transpose(2, 0, 1)
                                    for view in VIEWS
                                ]
                            )
                        )
                rgb = np.stack(rgb)
                if len(rgb) != episode["samples"]:
                    raise ValueError("Factual image/label sample count mismatch")
                for start in range(0, len(rgb), 8):
                    check()
                    indices = np.array(
                        [causal_clip_indices(i, 0) for i in range(start, min(start + 8, len(rgb)))]
                    )
                    video = torch.from_numpy(rgb[indices].copy()).cuda().permute(0, 2, 1, 3, 4, 5)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        value = encoder.video_features(video).reshape(-1, 48, 384)
                    features[episode["start"] + start : episode["start"] + start + len(indices)] = (
                        value.half().cpu().numpy()
                    )
                    done += len(indices)
                    progress("Past-only RGB encoding; future embeddings are training targets only")
                features.flush()
            if done != data["samples"]:
                raise ValueError("Factual encoder coverage incomplete")
            atomic_json(
                args.out / "manifest.json",
                dict(
                    samples=done,
                    features_sha256=digest(target),
                    request_sha256=digest(args.out / "request.json"),
                    data_manifest_sha256=digest(args.data / "manifest.json"),
                    source_shard_sha256=shard_hashes,
                    elapsed_s=time.monotonic() - started,
                ),
            )
            status = "complete"
    except InterruptedError:
        status = "stopped"
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error)))
        raise
    finally:
        progress("finished")
    print(json.dumps(dict(status=status, frames=done, total=data["samples"])), flush=True)


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation_factual_encoder.lock"):
        main()
