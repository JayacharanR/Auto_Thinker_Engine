"""
comma2k19 Dataset Loader.

Loads video segments from the comma2k19 dataset as 3D tubelets
and aligns frames with synchronized steering/speed telemetry.

Dataset structure (per segment):
    route_id/segment_number/
        video.hevc          # 20 Hz road-facing camera
        processed_log/      # numpy arrays: CAN data, IMU, etc.
        global_pos/          # numpy arrays: GPS positions

This loader handles:
- HEVC video decoding via PyAV
- Telemetry alignment by timestamp interpolation
- 3D tubelet sampling (consecutive frames with stride)
- Train/val splitting by segment
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class Comma2k19Dataset(Dataset):
    """
    PyTorch Dataset for comma2k19 video + telemetry data.

    Loads video segments as tubelets (sequences of frames) and aligns
    each frame with the corresponding steering angle and speed from
    the CAN bus telemetry.

    Args:
        dataset_root: Path to the extracted comma2k19 dataset.
        num_frames: Number of frames per tubelet.
        frame_stride: Temporal stride between frames (e.g., 2 = every other frame).
        spatial_size: Target spatial resolution (frames resized to this).
        split: 'train' or 'val'.
        split_ratio: Fraction of segments used for training.
        transform: Optional torchvision transform for frames.
        min_segment_length: Skip segments with fewer frames than this.
        use_steering: Whether to load steering telemetry.
        use_speed: Whether to load speed telemetry.
        normalize_telemetry: Whether to z-score normalize telemetry.
    """

    def __init__(
        self,
        dataset_root: str,
        num_frames: int = 16,
        frame_stride: int = 2,
        spatial_size: int = 224,
        split: str = "train",
        split_ratio: float = 0.9,
        transform: Optional[object] = None,
        min_segment_length: int = 32,
        use_steering: bool = True,
        use_speed: bool = True,
        normalize_telemetry: bool = True,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.spatial_size = spatial_size
        self.split = split
        self.transform = transform
        self.min_segment_length = min_segment_length
        self.use_steering = use_steering
        self.use_speed = use_speed
        self.normalize_telemetry = normalize_telemetry

        # Total frames needed per tubelet
        self.total_frames_needed = (num_frames - 1) * frame_stride + 1

        # Discover all segments
        self.segments = self._discover_segments()

        # Split into train/val
        np.random.seed(42)  # Deterministic split
        indices = np.random.permutation(len(self.segments))
        split_idx = int(len(indices) * split_ratio)

        if split == "train":
            selected = indices[:split_idx]
        elif split == "val":
            selected = indices[split_idx:]
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")

        self.segments = [self.segments[i] for i in selected]

        # Build index: (segment_idx, start_frame) pairs for all valid tubelets
        self.samples = self._build_sample_index()

        # Compute telemetry normalization statistics (from training set only)
        self._steering_mean = 0.0
        self._steering_std = 1.0
        self._speed_mean = 0.0
        self._speed_std = 1.0
        if normalize_telemetry and split == "train":
            self._compute_telemetry_stats()

    def _discover_segments(self) -> list[Path]:
        """Find all valid segment directories in the dataset."""
        segments = []

        if not self.dataset_root.exists():
            print(
                f"WARNING: Dataset root {self.dataset_root} does not exist. "
                f"Run the download script first: scripts/download_comma2k19.py"
            )
            return segments

        # comma2k19 structure: Chunk_N/route_id/segment_number/
        for chunk_dir in sorted(self.dataset_root.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for route_dir in sorted(chunk_dir.iterdir()):
                if not route_dir.is_dir():
                    continue
                for segment_dir in sorted(route_dir.iterdir()):
                    if not segment_dir.is_dir():
                        continue
                    # Check for required files
                    video_path = segment_dir / "video.hevc"
                    if video_path.exists():
                        segments.append(segment_dir)

        return segments

    def _build_sample_index(self) -> list[tuple[int, int]]:
        """
        Build an index of (segment_idx, start_frame) pairs.

        Each sample is a valid starting position for a tubelet within
        a segment. Segments shorter than total_frames_needed are skipped.
        """
        samples = []
        for seg_idx, segment_path in enumerate(self.segments):
            # Estimate segment length from video metadata
            video_path = segment_path / "video.hevc"
            try:
                num_total_frames = self._get_video_length(video_path)
            except Exception:
                continue

            if num_total_frames < self.total_frames_needed:
                continue

            if num_total_frames < self.min_segment_length:
                continue

            # Create samples at every possible starting position
            max_start = num_total_frames - self.total_frames_needed
            for start in range(0, max_start + 1, self.frame_stride):
                samples.append((seg_idx, start))

        return samples

    @staticmethod
    def _get_video_length(video_path: Path) -> int:
        """Get the number of frames in an HEVC video file."""
        import av

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            # Use stream.frames if available, otherwise count
            if stream.frames > 0:
                return stream.frames
            # Fallback: count frames (slower)
            count = 0
            for _ in container.decode(stream):
                count += 1
            return count

    def _load_video_frames(
        self, video_path: Path, start_frame: int
    ) -> torch.Tensor:
        """
        Load a tubelet of frames from an HEVC video.

        Args:
            video_path: Path to the .hevc file.
            start_frame: Starting frame index.

        Returns:
            (C, T, H, W) tensor of frames, normalized to [0, 1].
        """
        import av
        from torchvision.transforms.functional import resize

        frame_indices = [
            start_frame + i * self.frame_stride for i in range(self.num_frames)
        ]

        frames = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            frame_count = 0

            for frame in container.decode(stream):
                if frame_count in frame_indices:
                    # Convert to RGB numpy array
                    img = frame.to_ndarray(format="rgb24")
                    # Convert to tensor (C, H, W) and normalize to [0, 1]
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    # Resize to target spatial size
                    img_tensor = resize(
                        img_tensor,
                        [self.spatial_size, self.spatial_size],
                        antialias=True,
                    )
                    frames.append(img_tensor)

                    if len(frames) == self.num_frames:
                        break

                frame_count += 1

                # Early exit if we've passed all needed frames
                if frame_count > frame_indices[-1]:
                    break

        if len(frames) < self.num_frames:
            # Pad with last frame if video ended early
            while len(frames) < self.num_frames:
                frames.append(frames[-1].clone())

        # Stack: (T, C, H, W) → rearrange to (C, T, H, W)
        video = torch.stack(frames, dim=0)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)

        return video

    def _load_telemetry(
        self, segment_path: Path, start_frame: int
    ) -> torch.Tensor:
        """
        Load steering and speed telemetry aligned with video frames.

        Uses timestamp-based interpolation against the official comma2k19
        CAN field layout:
          - CAN/steering_angle/t (timestamps) + CAN/steering_angle/value
          - CAN/car_speed/t (timestamps) + CAN/car_speed/value

        Args:
            segment_path: Path to the segment directory.
            start_frame: Starting frame index.

        Returns:
            (T, A) tensor where A = number of telemetry channels
            (2 if both steering and speed are used).
        """
        processed_log = segment_path / "processed_log"
        telemetry_channels = []

        frame_indices = [
            start_frame + i * self.frame_stride for i in range(self.num_frames)
        ]

        # Video timestamps at 20 Hz
        video_fps = 20.0
        frame_timestamps = np.array([fi / video_fps for fi in frame_indices])

        if self.use_steering:
            aligned = self._interpolate_can_signal(
                processed_log, "steering_angle", frame_timestamps
            )
            telemetry_channels.append(torch.from_numpy(aligned))

        if self.use_speed:
            # Official comma2k19 field is "car_speed", not "speed"
            aligned = self._interpolate_can_signal(
                processed_log, "car_speed", frame_timestamps
            )
            telemetry_channels.append(torch.from_numpy(aligned))

        if not telemetry_channels:
            return torch.zeros(self.num_frames, 0)

        # Stack: (T, A)
        telemetry = torch.stack(telemetry_channels, dim=-1)

        # Normalize
        if self.normalize_telemetry:
            if self.use_steering and telemetry.shape[-1] >= 1:
                telemetry[:, 0] = (
                    telemetry[:, 0] - self._steering_mean
                ) / max(self._steering_std, 1e-6)
            if self.use_speed and telemetry.shape[-1] >= 2:
                telemetry[:, 1] = (
                    telemetry[:, 1] - self._speed_mean
                ) / max(self._speed_std, 1e-6)

        return telemetry

    @staticmethod
    def _interpolate_can_signal(
        processed_log: Path,
        signal_name: str,
        target_timestamps: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate a CAN signal to target timestamps.

        The official comma2k19 layout stores each CAN signal as:
          processed_log/CAN/{signal_name}/t      — timestamps (seconds)
          processed_log/CAN/{signal_name}/value   — signal values

        We use np.interp to align CAN values (sampled at ~100 Hz)
        to video frame timestamps (at 20 Hz).

        Args:
            processed_log: Path to the segment's processed_log directory.
            signal_name: CAN signal name (e.g., 'steering_angle', 'car_speed').
            target_timestamps: Timestamps (seconds) to interpolate to.

        Returns:
            np.ndarray of interpolated values at target_timestamps.
        """
        t_path = processed_log / "CAN" / signal_name / "t"
        v_path = processed_log / "CAN" / signal_name / "value"

        if t_path.exists() and v_path.exists():
            try:
                t = np.load(t_path).flatten()
                v = np.load(v_path).flatten()

                if len(t) > 0 and len(v) > 0 and len(t) == len(v):
                    # Normalize timestamps relative to segment start
                    t_rel = t - t[0]
                    return np.interp(
                        target_timestamps, t_rel, v
                    ).astype(np.float32)
            except Exception:
                pass

        # Fallback: zeros if signal unavailable
        return np.zeros(len(target_timestamps), dtype=np.float32)

    def _compute_telemetry_stats(self) -> None:
        """
        Compute mean/std of telemetry for z-score normalization.

        Uses official comma2k19 CAN field names:
          - steering_angle (not steering)
          - car_speed (not speed)
        """
        steering_values = []
        speed_values = []

        # Sample up to 100 segments for statistics
        sample_segments = self.segments[:min(100, len(self.segments))]

        for segment_path in sample_segments:
            processed_log = segment_path / "processed_log"

            if self.use_steering:
                path = processed_log / "CAN" / "steering_angle" / "value"
                if path.exists():
                    try:
                        data = np.load(path).flatten()
                        steering_values.extend(data.tolist())
                    except Exception:
                        pass

            if self.use_speed:
                # Official field: car_speed, NOT speed
                path = processed_log / "CAN" / "car_speed" / "value"
                if path.exists():
                    try:
                        data = np.load(path).flatten()
                        speed_values.extend(data.tolist())
                    except Exception:
                        pass

        if steering_values:
            self._steering_mean = float(np.mean(steering_values))
            self._steering_std = float(np.std(steering_values))
        if speed_values:
            self._speed_mean = float(np.mean(speed_values))
            self._speed_std = float(np.std(speed_values))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Load a single tubelet sample.

        Returns:
            Dict with:
            - 'video': (C, T, H, W) video tubelet tensor
            - 'telemetry': (T, A) aligned telemetry tensor
            - 'segment_path': str path to source segment
            - 'start_frame': int starting frame index
        """
        seg_idx, start_frame = self.samples[idx]
        segment_path = self.segments[seg_idx]
        video_path = segment_path / "video.hevc"

        # Load video frames
        video = self._load_video_frames(video_path, start_frame)

        # Apply transforms
        if self.transform is not None:
            video = self.transform(video)

        # Load telemetry
        telemetry = self._load_telemetry(segment_path, start_frame)

        return {
            "video": video,
            "telemetry": telemetry,
            "segment_path": str(segment_path),
            "start_frame": start_frame,
        }


def create_comma2k19_dataloaders(
    config: dict,
    seed: int = 42,
) -> tuple:
    """
    Create train and validation DataLoaders from config.

    Args:
        config: Data section of the YAML config.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    from src.data.transforms import create_video_transform
    from src.utils.seeding import get_generator, worker_init_fn

    data_cfg = config["data"]
    tubelet_cfg = data_cfg.get("tubelet", {})

    train_transform = create_video_transform(config, is_train=True)
    val_transform = create_video_transform(config, is_train=False)

    train_dataset = Comma2k19Dataset(
        dataset_root=data_cfg["dataset_root"],
        num_frames=tubelet_cfg.get("num_frames", 16),
        frame_stride=tubelet_cfg.get("frame_stride", 2),
        spatial_size=tubelet_cfg.get("spatial_size", 224),
        split="train",
        split_ratio=data_cfg.get("split_ratio", 0.9),
        transform=train_transform,
        min_segment_length=tubelet_cfg.get("min_segment_length", 32),
        use_steering=data_cfg.get("telemetry", {}).get("use_steering", True),
        use_speed=data_cfg.get("telemetry", {}).get("use_speed", True),
        normalize_telemetry=data_cfg.get("telemetry", {}).get("normalize", True),
    )

    val_dataset = Comma2k19Dataset(
        dataset_root=data_cfg["dataset_root"],
        num_frames=tubelet_cfg.get("num_frames", 16),
        frame_stride=tubelet_cfg.get("frame_stride", 2),
        spatial_size=tubelet_cfg.get("spatial_size", 224),
        split="val",
        split_ratio=data_cfg.get("split_ratio", 0.9),
        transform=val_transform,
        min_segment_length=tubelet_cfg.get("min_segment_length", 32),
        use_steering=data_cfg.get("telemetry", {}).get("use_steering", True),
        use_speed=data_cfg.get("telemetry", {}).get("use_speed", True),
        normalize_telemetry=data_cfg.get("telemetry", {}).get("normalize", True),
    )

    # Copy normalization stats from train to val
    val_dataset._steering_mean = train_dataset._steering_mean
    val_dataset._steering_std = train_dataset._steering_std
    val_dataset._speed_mean = train_dataset._speed_mean
    val_dataset._speed_std = train_dataset._speed_std

    g = get_generator(seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        drop_last=True,
        generator=g,
        worker_init_fn=worker_init_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        drop_last=False,
    )

    return train_loader, val_loader
