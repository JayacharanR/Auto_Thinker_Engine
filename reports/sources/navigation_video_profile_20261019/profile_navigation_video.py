#!/usr/bin/env python3
"""Bounded CPU recording-codec feasibility; preserve every original video."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Retain prior codec profile")
    frames = []
    with av.open(str(args.video)) as source:
        for frame in source.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == 320:
                break
    results = []
    with tempfile.TemporaryDirectory(prefix="auto_thinker_codec_", dir="/tmp") as folder:
        for preset in ("ultrafast", "veryfast"):
            path = Path(folder) / f"{preset}.mp4"
            started = time.monotonic()
            latencies = []
            with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
                stream = container.add_stream("libx264", rate=10)
                stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
                stream.pix_fmt = "yuv420p"
                stream.thread_count = 2
                stream.options = dict(crf="23", preset=preset, profile="baseline")
                for rgb in frames:
                    before = time.monotonic()
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                        container.mux(packet)
                    latencies.append(time.monotonic() - before)
                for packet in stream.encode():
                    container.mux(packet)
            elapsed = time.monotonic() - started
            decoded = []
            with av.open(str(path)) as check:
                for frame in check.decode(video=0):
                    decoded.append(frame.to_ndarray(format="rgb24"))
            if len(decoded) != len(frames):
                raise ValueError("Codec profile dropped frames")
            mae = float(
                np.mean(
                    [
                        np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()
                        for a, b in zip(decoded, frames)
                    ]
                )
            )
            results.append(
                dict(
                    preset=preset,
                    frames=len(frames),
                    bytes=path.stat().st_size,
                    total_s=elapsed,
                    frames_per_second=len(frames) / elapsed,
                    encode_call_p95_ms=float(np.quantile(latencies, 0.95) * 1000),
                    source_pixel_mae_8bit=mae,
                    all_frames_decoded=True,
                )
            )
    atomic_json(
        args.out,
        dict(
            source_video=str(args.video),
            source_sha256=digest(args.video),
            script_sha256=digest(__file__),
            results=results,
            original_videos_modified=False,
            note="Codec/storage measurement only; no policy/reward/scoring result. Decode source is already compressed; pixel MAE is not semantic visibility certification.",
        ),
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
