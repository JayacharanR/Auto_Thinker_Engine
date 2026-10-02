"""
comma2k19 Dataset Loader.

Reads clips from the arrays written by ``scripts/preprocess_comma2k19.py``
(one decode per segment, frames resized and memory-mapped), so a sample is a
slice of a uint8 array instead of an HEVC decode.

Processed layout:
    <root>/manifest.json
    <root>/<segment_id>/frames.npy       uint8 (N, S, S, 3), 10 Hz by default
    <root>/<segment_id>/telemetry.npy    float32 (N, 2): steering angle, speed

Train/val splitting is by segment, so clips from one drive never appear in
both splits.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def split_segments(segment_ids: list[str], split: str, split_ratio: float) -> list[str]:
    """Deterministic segment-level split (seed 42 permutation of the sorted ids)."""
    ids = sorted(segment_ids)
    order = np.random.RandomState(42).permutation(len(ids))
    cut = int(len(ids) * split_ratio)
    if split == "train":
        chosen = order[:cut]
    elif split == "val":
        chosen = order[cut:]
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")
    return [ids[i] for i in sorted(chosen)]


class Comma2k19Dataset(Dataset):
    """
    Video clips with aligned steering/speed telemetry.

    Args:
        dataset_root: Directory written by scripts/preprocess_comma2k19.py.
        num_frames: Frames per clip.
        frame_stride: Step between clip frames, in stored frames (1 = 10 Hz).
        split: 'train' or 'val'.
        split_ratio: Fraction of segments used for training.
        transform: Optional transform on the (C, T, H, W) clip in [0, 1].
        sample_step: Spacing of clip start positions. Training adds a random
            offset in [0, sample_step) so every start is reachable.
        use_steering / use_speed: Telemetry channels returned.
        normalize_telemetry: Z-score telemetry with training-split statistics.
        random_offset: Jitter clip starts (default: on for the train split).
    """

    def __init__(
        self,
        dataset_root: str,
        num_frames: int = 8,
        frame_stride: int = 1,
        split: str = "train",
        split_ratio: float = 0.9,
        transform: Optional[object] = None,
        sample_step: Optional[int] = None,
        use_steering: bool = True,
        use_speed: bool = True,
        normalize_telemetry: bool = True,
        random_offset: Optional[bool] = None,
    ):
        super().__init__()
        self.root = Path(dataset_root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{manifest_path} not found. Preprocess the raw dataset first: "
                f"python scripts/preprocess_comma2k19.py --src <raw> --out {self.root}"
            )
        self.manifest = json.loads(manifest_path.read_text())
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.span = (num_frames - 1) * frame_stride + 1
        self.split = split
        self.transform = transform
        self.sample_step = sample_step or max(1, num_frames // 2)
        self.random_offset = split == "train" if random_offset is None else random_offset
        self.channels = [i for i, use in enumerate((use_steering, use_speed)) if use]

        lengths = {s["id"]: s["num_frames"] for s in self.manifest["segments"]}
        self.segment_ids = [
            s for s in split_segments(list(lengths), split, split_ratio)
            if lengths[s] >= self.span
        ]
        self._frames: dict[int, np.ndarray] = {}  # opened lazily (per worker)
        self._telemetry: dict[int, np.ndarray] = {}

        # (segment index, first start) for every clip; starts every sample_step.
        self.samples = [
            (i, start)
            for i, seg in enumerate(self.segment_ids)
            for start in range(0, lengths[seg] - self.span + 1, self.sample_step)
        ]
        self._lengths = [lengths[s] for s in self.segment_ids]

        self.telemetry_mean = np.zeros(2, np.float32)
        self.telemetry_std = np.ones(2, np.float32)
        if normalize_telemetry:
            self.telemetry_mean, self.telemetry_std = self._train_telemetry_stats(
                list(lengths), split_ratio
            )

    def _train_telemetry_stats(self, all_ids: list[str], split_ratio: float):
        values = [
            np.load(self.root / seg / "telemetry.npy", mmap_mode="r")
            for seg in split_segments(all_ids, "train", split_ratio)
        ]
        if not values:
            return np.zeros(2, np.float32), np.ones(2, np.float32)
        stacked = np.concatenate(values)
        mean = np.nan_to_num(np.nanmean(stacked, axis=0)).astype(np.float32)
        std = np.nan_to_num(np.nanstd(stacked, axis=0), nan=1.0).astype(np.float32)
        return mean, np.maximum(std, 1e-6)

    def _arrays(self, seg_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if seg_idx not in self._frames:
            seg_dir = self.root / self.segment_ids[seg_idx]
            self._frames[seg_idx] = np.load(seg_dir / "frames.npy", mmap_mode="r")
            self._telemetry[seg_idx] = np.load(seg_dir / "telemetry.npy", mmap_mode="r")
        return self._frames[seg_idx], self._telemetry[seg_idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dict with:
            - 'video': (C, T, H, W) float clip (transformed if a transform is set)
            - 'telemetry': (T, A) normalised steering/speed (NaN -> 0)
            - 'segment_path': segment id
            - 'start_frame': first stored-frame index of the clip
        """
        seg_idx, start = self.samples[idx]
        if self.random_offset:
            last_start = self._lengths[seg_idx] - self.span
            start = min(start + int(np.random.randint(self.sample_step)), last_start)
        frames, telemetry = self._arrays(seg_idx)
        clip = slice(start, start + self.span, self.frame_stride)

        video = torch.from_numpy(np.array(frames[clip]))  # (T, H, W, C) uint8, writable copy
        video = video.permute(3, 0, 1, 2).float().div_(255.0)  # (C, T, H, W)
        if self.transform is not None:
            video = self.transform(video)

        tel = (np.asarray(telemetry[clip], np.float32) - self.telemetry_mean) / self.telemetry_std
        tel = torch.from_numpy(np.nan_to_num(tel[:, self.channels]))
        return {
            "video": video,
            "telemetry": tel,
            "segment_path": self.segment_ids[seg_idx],
            "start_frame": start,
        }


def create_comma2k19_dataloaders(
    config: dict,
    seed: int = 42,
    deterministic: bool = False,
) -> tuple:
    """
    Create train and validation DataLoaders from config.

    Args:
        config: Full Phase 2 YAML config.
        seed: Random seed for reproducibility.
        deterministic: Train split without augmentation, clip jitter, shuffling
            or dropped batches (feature extraction for the linear probe).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    from src.data.transforms import create_video_transform
    from src.utils.seeding import get_generator, worker_init_fn

    data_cfg = config["data"]
    tubelet_cfg = data_cfg.get("tubelet", {})
    telemetry_cfg = data_cfg.get("telemetry", {})
    common = dict(
        dataset_root=data_cfg["dataset_root"],
        num_frames=tubelet_cfg.get("num_frames", 8),
        frame_stride=tubelet_cfg.get("frame_stride", 1),
        split_ratio=data_cfg.get("split_ratio", 0.9),
        sample_step=tubelet_cfg.get("sample_step"),
        use_steering=telemetry_cfg.get("use_steering", True),
        use_speed=telemetry_cfg.get("use_speed", True),
        normalize_telemetry=telemetry_cfg.get("normalize", True),
    )
    train_dataset = Comma2k19Dataset(
        split="train",
        transform=create_video_transform(config, is_train=not deterministic),
        random_offset=not deterministic,
        **common,
    )
    val_dataset = Comma2k19Dataset(
        split="val", transform=create_video_transform(config, is_train=False), **common
    )

    num_workers = data_cfg.get("num_workers", 4)
    loader_args = dict(
        batch_size=config["training"]["batch_size"],
        num_workers=num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_args["prefetch_factor"] = data_cfg.get("prefetch_factor", 2)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=not deterministic,
        drop_last=not deterministic,
        generator=get_generator(seed),
        worker_init_fn=worker_init_fn,
        **loader_args,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_args
    )
    return train_loader, val_loader
