"""
Frame transforms for video preprocessing.

Handles resizing, normalization, and augmentation for both
training and evaluation. Horizontal flip is deliberately
disabled — driving scenes are laterally asymmetric (road side).
"""

from typing import Optional

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class VideoTransform:
    """
    Transform pipeline for video tubelets.

    Applies spatial augmentations consistently across all frames
    in a tubelet (same crop/flip for temporal consistency).

    Args:
        spatial_size: Target spatial resolution.
        is_train: If True, applies augmentation. If False, only
                  resize and normalize.
        random_crop: Whether to apply random resized crop.
        crop_scale: Scale range for random crop.
        color_jitter: Color jitter parameters dict.
        normalize_mean: Channel-wise mean for normalization.
        normalize_std: Channel-wise std for normalization.
    """

    def __init__(
        self,
        spatial_size: int = 224,
        is_train: bool = True,
        random_crop: bool = True,
        crop_scale: tuple[float, float] = (0.8, 1.0),
        color_jitter: Optional[dict] = None,
        normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.spatial_size = spatial_size
        self.is_train = is_train
        self.random_crop = random_crop
        self.crop_scale = crop_scale
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        # Color jitter (applied independently per frame is fine)
        self.color_jitter_transform = None
        if is_train and color_jitter:
            self.color_jitter_transform = T.ColorJitter(
                brightness=color_jitter.get("brightness", 0.2),
                contrast=color_jitter.get("contrast", 0.2),
                saturation=color_jitter.get("saturation", 0.1),
                hue=color_jitter.get("hue", 0.05),
            )

        # Normalization
        self.normalize = T.Normalize(
            mean=list(normalize_mean),
            std=list(normalize_std),
        )

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply transforms to a video tubelet.

        Args:
            video: (C, T, H, W) tensor in [0, 1].

        Returns:
            (C, T, H, W) transformed tensor.
        """
        # NOTE: Do NOT use 'T' as variable name — it shadows the
        # torchvision.transforms import alias 'T'.
        n_channels, n_frames, height, width = video.shape

        if self.is_train and self.random_crop:
            # Get crop parameters (same for all frames)
            ratio = (0.9, 1.1)  # Slight aspect ratio variation
            i, j, h, w = T.RandomResizedCrop.get_params(
                video[:, 0],  # Use first frame for param computation
                scale=(self.crop_scale[0], self.crop_scale[1]),
                ratio=ratio,
            )

            # Apply same crop to all frames
            frames = []
            for t in range(n_frames):
                frame = TF.resized_crop(
                    video[:, t], i, j, h, w,
                    [self.spatial_size, self.spatial_size],
                    antialias=True,
                )
                frames.append(frame)
            video = torch.stack(frames, dim=1)  # (C, T, H, W)
        else:
            # Evaluation: center resize
            frames = []
            for t in range(n_frames):
                frame = TF.resize(
                    video[:, t],
                    [self.spatial_size, self.spatial_size],
                    antialias=True,
                )
                frames.append(frame)
            video = torch.stack(frames, dim=1)

        # Color jitter (per-frame, but with temporal coherence from
        # applying the same jitter instance)
        if self.color_jitter_transform is not None:
            frames = []
            for t in range(n_frames):
                frames.append(self.color_jitter_transform(video[:, t]))
            video = torch.stack(frames, dim=1)

        # Normalize each frame
        frames = []
        for t in range(n_frames):
            frames.append(self.normalize(video[:, t]))
        video = torch.stack(frames, dim=1)

        return video


def create_video_transform(config: dict, is_train: bool = True) -> VideoTransform:
    """
    Create a VideoTransform from config.

    Args:
        config: Full YAML config dict.
        is_train: Whether this is for training or evaluation.

    Returns:
        Configured VideoTransform instance.
    """
    aug_cfg = config.get("augmentation", {})
    tubelet_cfg = config.get("data", {}).get("tubelet", {})

    normalize_cfg = aug_cfg.get("normalize", {})
    color_cfg = aug_cfg.get("color_jitter", None) if is_train else None

    return VideoTransform(
        spatial_size=tubelet_cfg.get("spatial_size", 224),
        is_train=is_train,
        random_crop=aug_cfg.get("random_crop", True) if is_train else False,
        crop_scale=tuple(aug_cfg.get("crop_scale", [0.8, 1.0])),
        color_jitter=color_cfg,
        normalize_mean=tuple(normalize_cfg.get("mean", [0.485, 0.456, 0.406])),
        normalize_std=tuple(normalize_cfg.get("std", [0.229, 0.224, 0.225])),
    )
