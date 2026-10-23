#!/usr/bin/env python3
"""Measure preview-only recording options on identical clean TRAIN RGB frames."""

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dataset import load_policy_sample, shard_records  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.out / "profile_source.py")
    manifest_path = ROOT / "outputs/navigation_spatial_data_repaired_20261019/manifest.json"
    episodes = json.loads(manifest_path.read_text())["episodes"]
    # Fixed TRAIN-only source selection; never inspect driving outcomes or VAL.
    selected = {}
    for episode in episodes:
        if episode["role"] == "train" and episode["end"] - episode["start"] >= 64:
            selected.setdefault(episode["town"], episode)
    frames, sources = [], []
    for town, episode in sorted(selected.items()):
        count = 0
        for shard in sorted(Path(episode["data"]).glob("shard_*.zip")):
            used = []
            for record in shard_records(shard):
                packet = load_policy_sample(shard, record)
                frames.append(
                    np.concatenate([packet.rgb[v] for v in ("left", "front", "right")], axis=1)
                )
                used.append(dict(prefix=record["prefix"], frame=record["frame"]))
                count += 1
                if count == 64:
                    break
            sources.append(dict(town=town, shard=str(shard), sha256=sha(shard), samples=used))
            if count == 64:
                break
        if count != 64:
            raise ValueError("Selected TRAIN episode lacks its declared64 frames")
    if len(selected) != 5 or len(frames) != 320:
        raise ValueError("Expected fixed64 TRAIN frames from each of five towns")
    atomic_json(
        args.out / "request.json",
        dict(
            manifest_sha256=sha(manifest_path),
            source_sha256=sha(__file__),
            sources=sources,
            policy_input_frames_modified=False,
            original_artifacts_modified=False,
            preview_options_only=True,
        ),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="auto_thinker_compact_video_", dir="/tmp") as folder:
        for crf, preset, fps in (
            (23, "veryfast", 10),
            (26, "medium", 10),
            (28, "medium", 10),
            (26, "medium", 5),
            (28, "medium", 5),
        ):
            selected_frames = frames[::10 // fps]
            path = Path(folder) / f"crf{crf}_{preset}_{fps}fps.mp4"
            before = time.monotonic()
            latencies = []
            with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
                stream = container.add_stream("libx264", rate=fps)
                stream.width, stream.height = 1536, 288
                stream.pix_fmt = "yuv420p"
                stream.thread_count = 2
                stream.options = dict(crf=str(crf), preset=preset, profile="baseline")
                for rgb in selected_frames:
                    tick = time.monotonic()
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                        container.mux(packet)
                    latencies.append(time.monotonic() - tick)
                for packet in stream.encode():
                    container.mux(packet)
            elapsed = time.monotonic() - before
            errors, squared, decoded = [], [], 0
            with av.open(str(path)) as check:
                for index, frame in enumerate(check.decode(video=0)):
                    if index >= len(selected_frames):
                        raise ValueError("Encoder emitted unexpected extra preview frames")
                    delta = frame.to_ndarray(format="rgb24").astype(np.float32)
                    delta -= selected_frames[index]
                    errors.append(float(np.abs(delta).mean()))
                    squared.append(float(np.square(delta).mean()))
                    decoded += 1
            if decoded != len(selected_frames):
                raise ValueError("Encoder lost intended preview frames")
            row = dict(
                crf=crf,
                preset=preset,
                fps=fps,
                width=1536,
                height=288,
                duration_s=32,
                frames=decoded,
                bytes=path.stat().st_size,
                bits_per_second=path.stat().st_size * 8 / 32,
                frames_per_second=decoded / elapsed,
                encode_call_p95_ms=float(np.quantile(latencies, .95) * 1000),
                source_pixel_mae_8bit=float(np.mean(errors)),
                source_pixel_psnr_db=10 * math.log10(255**2 / float(np.mean(squared))),
                all_intended_frames_decoded=True,
                preview_sha256=sha(path),
            )
            results.append(row)
            atomic_json(args.out / "profile.json", dict(results=results, complete=False))
    atomic_json(
        args.out / "profile.json",
        dict(
            results=results,
            complete=True,
            source_manifest_sha256=sha(manifest_path),
            source_request_sha256=sha(args.out / "request.json"),
            note=(
                "Clean TRAIN pixel comparison only. PSNR/MAE do not certify semantic "
                "traffic-light visibility. Preview quality never changes policy inputs, "
                "lossless training shards, native recorder events or benchmark scoring."
            ),
        ),
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
