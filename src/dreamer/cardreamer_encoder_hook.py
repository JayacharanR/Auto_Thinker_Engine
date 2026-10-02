"""
Encoder hook for dreamerv3-torch (PyTorch DreamerV3 backbone).

This module provides:
1. FrameStacker — rolling buffer for temporal encoders
2. DreamerV3EncoderHook — replaces dreamerv3-torch's MultiEncoder
   with our pretrained encoder (JEPA/V-JEPA2) or project CNN

Unlike the previous CarDreamer hook (which couldn't work due to JAX/PyTorch
mismatch), this targets a PyTorch nn.Module so the swap is direct attribute
assignment on agent._wm.encoder.

Interface contract with dreamerv3-torch:
    - forward(obs: dict) → Tensor of shape (B, T, embed_dim)
    - self.outdim: int — used by RSSM for input sizing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.dreamer.encoder_adapter import EncoderAdapter, create_encoder


class FrameStacker:
    """
    Rolling frame buffer for temporal encoders.

    Accumulates single frames into (C, T, H, W) clips.
    Pads with first-frame copies until buffer is full.

    Args:
        num_frames: Number of frames per clip.
        device: Target torch device.
    """

    def __init__(self, num_frames: int = 4, device: str = "cpu"):
        self.num_frames = num_frames
        self.device = device
        self._buffer = []

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Push a single (C, H, W) frame and return (C, T, H, W) clip.

        On first call, pads buffer with copies of the first frame.
        After buffer fills, oldest frame drops (sliding window).
        """
        frame = frame.to(self.device)

        if len(self._buffer) == 0:
            # Pad with copies of first frame
            self._buffer = [frame.clone() for _ in range(self.num_frames)]
        else:
            self._buffer.append(frame)
            if len(self._buffer) > self.num_frames:
                self._buffer.pop(0)

        # Stack: list of (C, H, W) → (C, T, H, W)
        return torch.stack(self._buffer, dim=1)

    def reset(self):
        """Clear the buffer for a new episode."""
        self._buffer = []


class DreamerV3EncoderHook(nn.Module):
    """
    Drop-in replacement for dreamerv3-torch's MultiEncoder.

    Interface contract:
        forward(obs: dict) → Tensor(B, T, embed_dim)
        self.outdim: int

    This processes the 'image' key from obs dict through our encoder
    (CNN/JEPA/V-JEPA2) + adapter, producing the embedding the RSSM expects.

    For temporal encoders (JEPA, V-JEPA2), FrameStacker accumulates frames
    into clips of T >= tubelet_size before encoding. For CNN, single frames
    are processed directly.

    Args:
        arm: One of 'cnn', 'custom_jepa', 'vjepa2'.
        config: Phase 3 config dict with 'encoders' and 'adapter' keys.
        device: Target device string.
        obs_key: Key in obs dict for image data (default: 'image').
        num_temporal_frames: Frames to stack for temporal encoders.
    """

    def __init__(
        self,
        arm: str,
        config: dict,
        device: str = "cuda",
        obs_key: str = "image",
        num_temporal_frames: int = 4,
    ):
        super().__init__()
        self.arm = arm
        self.device = device
        self.obs_key = obs_key

        # Per-arm resolution from encoder config
        enc_cfg = config.get("encoders", {}).get(arm, {})
        self.target_resolution = enc_cfg.get("input_resolution", 224)

        # Create encoder + adapter
        self.encoder, self.adapter = create_encoder(arm, config, device)
        self.encoder = self.encoder.to(device)
        self.adapter = self.adapter.to(device)

        # Output dim — this is what RSSM reads as embed_size
        self.outdim = config["adapter"]["target_dim"]

        # Temporal handling
        self._needs_temporal = arm in ("custom_jepa", "vjepa2")
        self._tubelet_size = enc_cfg.get("tubelet_size", 2)
        self._num_temporal_frames = num_temporal_frames

        # Ensure num_temporal_frames >= tubelet_size (blocker #4 fix)
        if self._needs_temporal:
            assert num_temporal_frames >= self._tubelet_size, (
                f"num_temporal_frames ({num_temporal_frames}) must be >= "
                f"tubelet_size ({self._tubelet_size})"
            )

    def forward(self, obs: dict) -> torch.Tensor:
        """
        Process observation dict → (B, T, embed_dim).

        dreamerv3-torch's WorldModel._train() passes data dict with:
            obs['image']: (B, T, H, W, C) — note channel-last!
            obs['action']: (B, T, act_dim)
            obs['reward']: (B, T)
            obs['is_first']: (B, T)

        This method:
        1. Extracts image from obs dict
        2. Converts channel-last → channel-first
        3. Resizes to target resolution
        4. For temporal encoders: groups frames into clips
        5. Encodes through our encoder + adapter
        6. Returns (B, T, embed_dim)
        """
        x = obs[self.obs_key]
        online_step = x.dim() == 4
        if online_step:
            # Dreamer policy calls the encoder with one observation per env:
            # (B, H, W, C). Replay training calls it with (B, T, H, W, C).
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(
                f"Expected {self.obs_key!r} with 4 or 5 dimensions, got {tuple(x.shape)}"
            )

        B, T = x.shape[:2]

        # Channel-last → channel-first: (B, T, H, W, C) → (B, T, C, H, W)
        if x.shape[-1] in (1, 3):  # channel-last detection
            x = x.permute(0, 1, 4, 2, 3)

        # Normalize [0, 255] → [0, 1] if needed
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 2.0:
            x = x / 255.0

        C, H, W = x.shape[2], x.shape[3], x.shape[4]

        # Resize if needed: flatten (B*T, C, H, W), resize, reshape back
        if H != self.target_resolution or W != self.target_resolution:
            x_flat = x.reshape(B * T, C, H, W)
            x_flat = F.interpolate(
                x_flat,
                size=(self.target_resolution, self.target_resolution),
                mode="bilinear",
                align_corners=False,
            )
            x = x_flat.reshape(B, T, C, self.target_resolution, self.target_resolution)

        if self._needs_temporal:
            output = self._encode_temporal(x, B, T)
        else:
            output = self._encode_single_frame(x, B, T)
        return output[:, 0] if online_step else output

    def _encode_single_frame(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        CNN arm: encode each frame independently.

        Input: (B, T, C, H, W)
        Output: (B, T, embed_dim)
        """
        C = x.shape[2]
        x_flat = x.reshape(B * T, C, x.shape[3], x.shape[4])  # (B*T, C, H, W)
        features = self.encoder(x_flat)  # (B*T, encoder_dim)
        adapted = self.adapter(features)  # (B*T, embed_dim)
        return adapted.reshape(B, T, -1)  # (B, T, embed_dim)

    def _encode_temporal(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        JEPA/V-JEPA2 arm: encode overlapping temporal clips.

        For each timestep t, create a clip of the last `num_temporal_frames`
        frames (with padding for early timesteps). This ensures T >= tubelet_size
        for every clip, fixing blocker #4.

        Input: (B, T, C, H, W)
        Output: (B, T, embed_dim)
        """
        outputs = []
        C = x.shape[2]
        clip_len = self._num_temporal_frames

        for t in range(T):
            # Build clip ending at frame t
            start = max(0, t - clip_len + 1)
            clip = x[:, start:t + 1]  # (B, actual_len, C, H, W)

            # Pad if needed (early timesteps)
            actual_len = clip.shape[1]
            if actual_len < clip_len:
                pad_frame = clip[:, :1].expand(-1, clip_len - actual_len, -1, -1, -1)
                clip = torch.cat([pad_frame, clip], dim=1)  # (B, clip_len, C, H, W)

            # Convert to (B, C, T, H, W) for our encoder
            clip = clip.permute(0, 2, 1, 3, 4)  # (B, C, clip_len, H, W)

            # Encode
            features = self.encoder(clip)  # (B, encoder_dim) or (B, N, D)

            # Pool if needed (ViT returns (B, N, D))
            if features.dim() == 3:
                features = features.mean(dim=1)  # (B, D)

            adapted = self.adapter(features)  # (B, embed_dim)
            outputs.append(adapted)

        return torch.stack(outputs, dim=1)  # (B, T, embed_dim)


def patch_dreamerv3_encoder(agent, hook: DreamerV3EncoderHook):
    """
    Replace dreamerv3-torch agent's encoder with our hook.

    This is a direct PyTorch nn.Module attribute swap — both sides
    are PyTorch, so this just works (unlike the JAX CarDreamer case).

    Also updates embed_size so the RSSM input dimension matches.

    Args:
        agent: dreamerv3-torch Dreamer agent
        hook: Our DreamerV3EncoderHook instance
    """
    # Swap the encoder module
    agent._wm.encoder = hook

    # Update embed_size so RSSM knows the new input dimension
    agent._wm.embed_size = hook.outdim

    # Verify the swap
    assert agent._wm.encoder is hook, "Encoder swap failed"
    assert agent._wm.embed_size == hook.outdim, "embed_size mismatch"

    print(f"[hook] Replaced MultiEncoder with {hook.arm} hook (outdim={hook.outdim})")


class FrozenFeatureExtractor:
    """
    Run a frozen encoder once per environment step, at collection time.

    Replay then stores the embedding (``feat``) instead of re-encoding clips
    for every training batch, which removes all ViT forwards from Dreamer
    updates. Frames are stacked per episode (``reset`` at episode start), so
    clips never span two episodes.

    Args:
        encoder: Frozen module mapping (B, C, T, H, W) clips to (B, D) or (B, N, D).
        num_frames: Clip length.
        resolution: Input size the encoder expects (frames are resized).
        mean / std: Per-channel normalisation applied to [0, 1] frames, or None.
        device: Device for the encoder.
    """

    def __init__(self, encoder: nn.Module, num_frames: int, resolution: Optional[int],
                 mean=None, std=None, device: str = "cuda"):
        self.encoder = encoder.to(device).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.device = device
        self.resolution = resolution
        self.stacker = FrameStacker(num_frames, device)
        self._norm = None
        if mean is not None:
            self._norm = (torch.tensor(mean, device=device).view(3, 1, 1),
                          torch.tensor(std, device=device).view(3, 1, 1))
        self.dim = int(self._encode(torch.zeros(1, 3, num_frames, resolution or 64,
                                                resolution or 64, device=device)).shape[-1])

    @torch.no_grad()
    def _encode(self, clip: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=clip.is_cuda):
            features = self.encoder(clip)
        if features.dim() == 3:
            features = features.mean(dim=1)
        return features.float()

    def reset(self) -> None:
        self.stacker.reset()

    @torch.no_grad()
    def __call__(self, frame) -> np.ndarray:
        # CarDreamer's camera is a channel-reversed view (negative strides): copy first.
        frame = np.ascontiguousarray(frame)
        x = torch.as_tensor(frame, device=self.device).permute(2, 0, 1).float() / 255.0
        if self.resolution and tuple(x.shape[-2:]) != (self.resolution, self.resolution):
            x = F.interpolate(x[None], size=(self.resolution, self.resolution),
                              mode="bilinear", align_corners=False)[0]
        if self._norm is not None:
            x = (x - self._norm[0]) / self._norm[1]
        clip = self.stacker.push(x)  # (C, T, H, W)
        return self._encode(clip[None])[0].cpu().numpy()


def build_feature_extractor(arm: str, phase3_config: dict, device: str):
    """
    Frozen-encoder extractor for an arm, or None when the arm trains its encoder.

    ``custom_jepa`` is rebuilt from the configuration stored in its Phase 2
    checkpoint, so resolution, clip length and input normalisation match
    pretraining, and its weights are loaded strictly.
    """
    enc_cfg = phase3_config.get("encoders", {}).get(arm, {})
    if arm == "cnn" or not enc_cfg.get("freeze", True) or enc_cfg.get("use_lora", False):
        return None
    if arm == "custom_jepa":
        from src.jepa.encoder import ViTEncoder

        path = enc_cfg.get("checkpoint")
        if not path:
            raise FileNotFoundError("encoders.custom_jepa.checkpoint is not set")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        ctx, tubelet = cfg["model"]["context_encoder"], cfg["data"]["tubelet"]
        encoder = ViTEncoder(
            img_size=tubelet["spatial_size"], patch_size=ctx["patch_size"],
            tubelet_size=ctx["tubelet_size"], embed_dim=ctx["embed_dim"], depth=ctx["depth"],
            num_heads=ctx["num_heads"], mlp_ratio=ctx["mlp_ratio"],
            num_frames=tubelet["num_frames"],
        )
        encoder.load_state_dict(ckpt["context_encoder_state_dict"], strict=True)
        norm = cfg.get("augmentation", {}).get("normalize", {})
        print(f"[encoder] custom_jepa from {path} ({ckpt.get('regularizer', '?')}, "
              f"{tubelet['spatial_size']} px x {tubelet['num_frames']} frames)")
        return FrozenFeatureExtractor(
            encoder, tubelet["num_frames"], tubelet["spatial_size"],
            norm.get("mean"), norm.get("std"), device,
        )
    if arm == "vjepa2":
        from src.dreamer.encoder_adapter import VJEPAEncoder

        encoder = VJEPAEncoder(
            model_name=enc_cfg.get("model_name", "facebook/vjepa2-vitl-fpc64-256"), freeze=True
        )
        # The V-JEPA2 wrapper normalises and resizes internally.
        frames = phase3_config.get("frame_stacking", {}).get("num_frames", 4)
        return FrozenFeatureExtractor(encoder, frames, None, device=device)
    raise ValueError(f"Unknown arm: {arm}")


class PrecomputedFeatureEncoder(nn.Module):
    """
    Dreamer encoder for frozen arms: a trainable adapter over the stored
    ``feat`` embedding, concatenated with ``vector_encoder``'s output (the
    same dreamerv3-torch MLP the CNN arm uses for the ``route`` vector).
    """

    def __init__(self, feat_dim: int, adapter_cfg: dict, vector_encoder: Optional[nn.Module]):
        super().__init__()
        self.adapter = EncoderAdapter(
            input_dim=feat_dim,
            target_dim=adapter_cfg["target_dim"],
            use_layer_norm=adapter_cfg.get("use_layer_norm", True),
            activation=adapter_cfg.get("activation", "silu"),
        )
        self.vector_encoder = vector_encoder
        self.outdim = adapter_cfg["target_dim"] + (vector_encoder.outdim if vector_encoder else 0)

    def forward(self, obs: dict) -> torch.Tensor:
        feat = obs["feat"]  # (B, D) when acting, (B, T, D) in replay batches
        # Flatten: EncoderAdapter would treat a 3-D input as ViT tokens and pool them.
        adapted = self.adapter(feat.reshape(-1, feat.shape[-1]))
        parts = [adapted.reshape(*feat.shape[:-1], -1)]
        if self.vector_encoder is not None:
            parts.append(self.vector_encoder(obs))
        return torch.cat(parts, dim=-1)
