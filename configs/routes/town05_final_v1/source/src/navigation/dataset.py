"""Lossless, atomic episode shards with separate policy inputs and training labels.

Only renamed *.zip files are committed. Partial archives survive as *.part files,
and a reader never guesses that an unfinalized episode is a clean demonstration.
"""

import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
from PIL import Image

from src.navigation.contracts import VIEWS, CameraCalibration, ControlCommand, SensorPacket
from src.navigation.runtime import atomic_json

SCHEMA = "navigation-shards-v2"
SUPPORTED_SCHEMAS = {"navigation-shards-v1", SCHEMA}
POLICY_KEYS = {"frame", "timestamp", "calibration", "navigation_waypoints", "ego_motion"}


def json_bytes(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class EpisodeWriter:
    def __init__(self, root, metadata, shard_samples=32, minimum_free_gb=25):
        if shard_samples < 1:
            raise ValueError("shard_samples must be positive")
        self.root = Path(root)
        # Every simulator attempt/episode owns a new directory, never overwrites data.
        self.root.mkdir(parents=True, exist_ok=False)
        atomic_json(
            self.root / "episode.json",
            {**metadata, "schema": SCHEMA, "rgb_encoding": "lossless_h264_rgb"},
        )
        self.shard_samples, self.minimum_free_gb = shard_samples, minimum_free_gb
        self.archive = None
        self.pending = []
        self.shards = 0
        self.samples = 0
        self.previous = None
        self.closed = False
        self.encoders = {}

    def _member(self, name, payload):
        self.archive.writestr(name, payload)
        return hashlib.sha256(payload).hexdigest()

    def append(self, packet, targets):
        if self.closed:
            raise RuntimeError("Episode writer is closed")
        packet.validate()
        if targets.frame != packet.frame:
            raise ValueError("Training labels and camera packet frame differ")
        for control in (targets.expert_control, targets.executed_control):
            ControlCommand(*control).validate()
        if self.previous is not None:
            frame, timestamp = self.previous
            if packet.frame != frame + 2 or abs(packet.timestamp - timestamp - 0.1) > 1e-5:
                raise ValueError("Collection requires contiguous 10Hz samples from 20Hz physics")
        raw_size = sum(a.nbytes for a in packet.rgb.values())
        if shutil.disk_usage(self.root).free - raw_size < self.minimum_free_gb * 1e9:
            raise RuntimeError("Collection disk reserve reached")
        # Validate all JSON before writing any member of this sample.
        policy = {
            "frame": packet.frame,
            "timestamp": packet.timestamp,
            "calibration": {k: asdict(c) for k, c in packet.calibration.items()},
            "navigation_waypoints": packet.navigation_waypoints.tolist(),
            "ego_motion": packet.ego_motion.tolist(),
        }
        policy_bytes, target_bytes = json_bytes(policy), json_bytes(asdict(targets))
        if self.archive is None:
            fd, name = tempfile.mkstemp(prefix="shard_", suffix=".part", dir=self.root)
            os.close(fd)
            self.part = Path(name)
            self.archive = zipfile.ZipFile(self.part, "w", compression=zipfile.ZIP_STORED)
            for view in VIEWS:
                buffer = io.BytesIO()
                container = av.open(buffer, "w", format="matroska")
                stream = container.add_stream("libx264rgb", rate=10)
                stream.width, stream.height = packet.rgb[view].shape[1], packet.rgb[view].shape[0]
                stream.pix_fmt = "rgb24"
                stream.options = {"crf": "0", "preset": "veryfast", "x264-params": "bframes=0"}
                self.encoders[view] = buffer, container, stream
        prefix = f"{self.samples:08d}"
        hashes = {
            "policy.json": self._member(f"{prefix}/policy.json", policy_bytes),
            "targets.json": self._member(f"{prefix}/targets.json", target_bytes),
        }
        rgb_hashes = {}
        for view in VIEWS:
            _, container, stream = self.encoders[view]
            for encoded in stream.encode(
                av.VideoFrame.from_ndarray(packet.rgb[view], format="rgb24")
            ):
                container.mux(encoded)
            rgb_hashes[view] = hashlib.sha256(packet.rgb[view].tobytes()).hexdigest()
        self.pending.append(
            {
                "prefix": prefix,
                "frame": packet.frame,
                "timestamp": packet.timestamp,
                "sha256": hashes,
                "rgb_sha256": rgb_hashes,
            }
        )
        self.samples += 1
        self.previous = packet.frame, packet.timestamp
        if len(self.pending) >= self.shard_samples:
            self.commit()

    def commit(self):
        if self.archive is None:
            return
        media = {}
        for view, (buffer, container, stream) in self.encoders.items():
            for encoded in stream.encode():
                container.mux(encoded)
            container.close()
            media[f"{view}.mkv"] = self._member(f"{view}.mkv", buffer.getvalue())
        self.encoders = {}
        self.archive.writestr(
            "manifest.json",
            json_bytes(
                {
                    "schema": SCHEMA,
                    "encoding": "lossless_h264_rgb",
                    "media_sha256": media,
                    "samples": self.pending,
                }
            ),
        )
        self.archive.close()
        self.archive = None
        with self.part.open("rb") as f:
            os.fsync(f.fileno())
        destination = self.root / f"shard_{self.shards:06d}.zip"
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(self.part, destination)
        sync_directory(self.root)
        self.shards += 1
        self.pending = []

    def close(self):
        if not self.closed:
            self.commit()
            self.closed = True


def shard_records(path, verify=True):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["schema"] not in SUPPORTED_SCHEMAS:
            raise ValueError("Unknown dataset schema")
        if verify:
            for name, digest in manifest.get("media_sha256", {}).items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError(f"Shard media changed: {name}")
        for record in manifest["samples"]:
            if verify:
                for name, digest in record["sha256"].items():
                    data = archive.read(f"{record['prefix']}/{name}")
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise ValueError(f"Shard member changed: {name}")
            yield record


@lru_cache(maxsize=1)
def decode_rgb_shard(path, mtime_ns, size):
    """One 32-sample RGB cache (~43MB); no privileged labels are opened."""
    views = {}
    with zipfile.ZipFile(path) as archive:
        for view in VIEWS:
            with av.open(io.BytesIO(archive.read(f"{view}.mkv"))) as container:
                views[view] = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    return views


def load_policy_sample(path, record):
    """Read only the allowlisted policy members; never deserialize targets.json."""
    prefix = record["prefix"]
    with zipfile.ZipFile(path) as archive:
        policy = json.loads(archive.read(f"{prefix}/policy.json"))
        if set(policy) != POLICY_KEYS:
            raise ValueError("Unexpected policy fields; possible privileged input leakage")
        rgb = {}
        if "rgb_sha256" in record:
            stat = Path(path).stat()
            views = decode_rgb_shard(str(path), stat.st_mtime_ns, stat.st_size)
            manifest = json.loads(archive.read("manifest.json"))
            index = next(i for i, r in enumerate(manifest["samples"]) if r["prefix"] == prefix)
            for view in VIEWS:
                if len(views[view]) != len(manifest["samples"]):
                    raise ValueError("RGB video and label frame counts differ")
                rgb[view] = views[view][index].copy()
                if hashlib.sha256(rgb[view].tobytes()).hexdigest() != record["rgb_sha256"][view]:
                    raise ValueError("Lossless RGB pixel hash mismatch")
        else:
            for view in VIEWS:
                with Image.open(io.BytesIO(archive.read(f"{prefix}/{view}.png"))) as img:
                    rgb[view] = np.asarray(img.convert("RGB")).copy()
    packet = SensorPacket(
        frame=policy["frame"],
        timestamp=policy["timestamp"],
        rgb=rgb,
        calibration={k: CameraCalibration(**v) for k, v in policy["calibration"].items()},
        navigation_waypoints=np.asarray(policy["navigation_waypoints"], dtype=np.float32),
        ego_motion=np.asarray(policy["ego_motion"], dtype=np.float32),
    )
    packet.validate()
    if packet.frame != record["frame"]:
        raise ValueError("Policy and manifest frame differ")
    return packet


def training_records(root, purpose):
    """Yield committed records, requiring explicit episode eligibility for imitation."""
    root = Path(root)
    if purpose not in ("imitation", "prediction"):
        raise ValueError("Choose imitation or prediction explicitly")
    if purpose == "imitation":
        outcome = root / "outcome.json"
        if not outcome.exists() or not json.loads(outcome.read_text())["imitation_eligible"]:
            return
    for shard in sorted(root.glob("shard_*.zip")):
        for record in shard_records(shard):
            yield shard, record
