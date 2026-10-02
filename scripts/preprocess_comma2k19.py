#!/usr/bin/env python3
"""
Decode comma2k19 once into small memory-mappable arrays for Phase 2.

Reading raw HEVC per sample (decode from frame 0 up to the clip, at
1164x874) makes training CPU-bound at seconds per clip. This script decodes
every segment exactly once, keeps every ``stride``-th frame (20 Hz -> 10 Hz by
default), resizes to ``size`` x ``size`` and writes per segment:

    <out>/<segment_id>/frames.npy      uint8 (N, size, size, 3)
    <out>/<segment_id>/telemetry.npy   float32 (N, 2): steering angle (deg), speed (m/s)
    <out>/<segment_id>/meta.json

plus ``<out>/manifest.json`` listing all segments. Segments already written
are skipped, so the script can be re-run after an interruption or after more
chunks are downloaded.

Usage:
    python scripts/preprocess_comma2k19.py --src data/comma2k19 --out data/comma2k19_128
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

VIDEO_FPS = 20.0
TELEMETRY = ("steering_angle", "speed")


def discover_segments(root: Path) -> list[Path]:
    """Segment directories (``Chunk_*/<route>/<segment>/video.hevc``) below ``root``."""
    return sorted(p.parent for p in root.glob("*/*/*/video.hevc"))


def segment_id(segment: Path) -> str:
    """``<route>/<n>`` with filesystem-safe characters (routes contain ``|``)."""
    return f"{segment.parent.name}--{segment.name}".replace("|", "_")


def frame_times(segment: Path, num_frames: int) -> tuple[np.ndarray, str]:
    """Frame timestamps on the CAN clock, and where they came from.

    ``global_pose/frame_times`` holds per-frame times on the log clock. Without
    it, frames are assumed to start with the CAN log at 20 Hz.
    """
    path = segment / "global_pose" / "frame_times"
    if path.exists():
        times = np.load(path).reshape(-1).astype(np.float64)
        if len(times) >= num_frames:
            return times[:num_frames], "global_pose/frame_times"
    can_t = segment / "processed_log" / "CAN" / TELEMETRY[0] / "t"
    start = float(np.load(can_t).reshape(-1)[0]) if can_t.exists() else 0.0
    return start + np.arange(num_frames) / VIDEO_FPS, "can_start+20Hz"


def aligned_telemetry(segment: Path, times: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """(N, 2) CAN signals interpolated to ``times``; missing signals are NaN."""
    columns, missing = [], []
    for name in TELEMETRY:
        base = segment / "processed_log" / "CAN" / name
        try:
            t = np.load(base / "t").reshape(-1)
            v = np.load(base / "value").reshape(len(t), -1)[:, 0]
            columns.append(np.interp(times, t, v).astype(np.float32))
        except (FileNotFoundError, ValueError):
            columns.append(np.full(len(times), np.nan, np.float32))
            missing.append(name)
    return np.stack(columns, axis=1), missing


def decode_frames(video: Path, stride: int, size: int) -> tuple[np.ndarray, list[int]]:
    """Every ``stride``-th frame resized to ``size`` x ``size`` RGB, and their source indices."""
    import av
    import cv2

    frames, indices = [], []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i % stride:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            frames.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA))
            indices.append(i)
    if not frames:
        raise ValueError("no frames decoded")
    return np.stack(frames), indices


def process_segment(args: tuple) -> dict:
    segment, out_root, stride, size = args
    seg_id = segment_id(segment)
    out = out_root / seg_id
    meta_path = out / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    started = time.time()
    try:
        frames, indices = decode_frames(segment / "video.hevc", stride, size)
        times, time_source = frame_times(segment, indices[-1] + 1)
        telemetry, missing = aligned_telemetry(segment, times[indices])
    except Exception as error:  # report and continue with other segments
        return {"id": seg_id, "source": str(segment), "error": repr(error)}

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "frames.npy", frames)
    np.save(out / "telemetry.npy", telemetry)
    meta = {
        "id": seg_id,
        "source": str(segment),
        "num_frames": int(len(frames)),
        "fps": VIDEO_FPS / stride,
        "size": size,
        "time_source": time_source,
        "missing_telemetry": missing,
        "seconds": round(time.time() - started, 2),
    }
    meta_path.write_text(json.dumps(meta))  # written last: marks the segment complete
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, default=Path("data/comma2k19"))
    parser.add_argument("--out", type=Path, default=Path("data/comma2k19_128"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=2, help="Keep every Nth 20 Hz frame")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--limit", type=int, default=None, help="Process at most N segments")
    args = parser.parse_args()

    segments = discover_segments(args.src)[: args.limit]
    if not segments:
        print(f"No segments (*/<route>/<segment>/video.hevc) found below {args.src}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Preprocessing {len(segments)} segments with {args.workers} workers -> {args.out}")

    jobs = [(s, args.out, args.stride, args.size) for s in segments]
    results, started = [], time.time()
    with mp.Pool(args.workers) as pool:
        for i, meta in enumerate(pool.imap_unordered(process_segment, jobs), 1):
            results.append(meta)
            status = meta.get("error") or f"{meta['num_frames']} frames"
            print(f"[{i}/{len(jobs)}] {meta['id']}: {status}", flush=True)

    done = sorted((m for m in results if "error" not in m), key=lambda m: m["id"])
    failed = [m for m in results if "error" in m]
    manifest = {
        "size": args.size,
        "fps": VIDEO_FPS / args.stride,
        "telemetry_columns": ["steering_angle_deg", "speed_mps"],
        "segments": [{"id": m["id"], "num_frames": m["num_frames"]} for m in done],
        "failed": failed,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    frames = sum(m["num_frames"] for m in done)
    print(f"Done in {time.time() - started:.0f}s: {len(done)} segments, {frames} frames, "
          f"{len(failed)} failed. Manifest: {args.out / 'manifest.json'}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
