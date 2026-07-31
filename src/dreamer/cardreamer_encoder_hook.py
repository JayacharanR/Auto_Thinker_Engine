"""
CarDreamer Encoder Hook — integrate our encoder adapters into CarDreamer's DreamerV3.

This module provides the bridge between our encoder research (JEPA pretraining,
V-JEPA2 transfer, CNN baseline) and CarDreamer's tested DreamerV3 training loop.

Instead of reimplementing RSSM/actor-critic/replay (which introduced bugs #2, #7, #8),
we hook into CarDreamer's pipeline at the encoder level — which is the actual
research contribution of this project.

Architecture:
    CarDreamer Env (Gym) → image obs
        → OUR encoder (CNN / custom_jepa / vjepa2)
        → OUR adapter (projects to fixed dim)
        → CarDreamer's RSSM (replace their encoder output)
        → CarDreamer's actor-critic
        → action → CarDreamer Env

Design Decision: Frame Stacking (Option A)
    For temporal encoders (custom_jepa, vjepa2), we maintain a rolling buffer
    of the last N CARLA frames. This preserves the scientific claim that
    temporal video pretraining transfers to control, rather than degenerating
    to 2D patches (Option B).

Resolution: 224×224 everywhere
    Phase 2 JEPA pretrains at 224×224. Phase 3 resizes CARLA frames to 224×224
    before encoding. This ensures positional embeddings match and checkpoint
    loading works without interpolation hacks.
"""

import collections
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from src.dreamer.encoder_adapter import create_encoder


class FrameStacker:
    """
    Rolling buffer of the last N frames for temporal encoders.

    CNN arm: bypasses this (single frame).
    custom_jepa / vjepa2: accumulates frames into (C, T, H, W) clips.

    Pads with repeated first frame if buffer isn't full yet
    (avoids zeros which would confuse pretrained encoders).
    """

    def __init__(self, num_frames: int = 4, device: str = "cuda"):
        self.num_frames = num_frames
        self.device = device
        self.buffer: collections.deque = collections.deque(maxlen=num_frames)

    def reset(self):
        self.buffer.clear()

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Add a frame and return the full temporal clip.

        Args:
            frame: (C, H, W) single frame tensor.

        Returns:
            (C, T, H, W) temporal clip tensor.
        """
        self.buffer.append(frame.detach())

        # Pad with first frame if not full
        while len(self.buffer) < self.num_frames:
            self.buffer.appendleft(self.buffer[0].clone())

        stacked = torch.stack(list(self.buffer), dim=0)  # (T, C, H, W)
        return stacked.permute(1, 0, 2, 3)  # (C, T, H, W)


class CarDreamerEncoderHook(nn.Module):
    """
    Drop-in encoder replacement for CarDreamer's DreamerV3.

    CarDreamer's DreamerV3 expects an encoder that takes image observations
    and returns a flat feature vector. This module wraps our three encoder
    arms to conform to that interface.

    Usage in CarDreamer integration:
        # In the modified dreamerv3/nets.py or via monkey-patching:
        hook = CarDreamerEncoderHook(arm='custom_jepa', config=config)
        # Replace DreamerV3's encoder.forward() output with hook(obs)

    Args:
        arm: One of 'cnn', 'custom_jepa', 'vjepa2'.
        config: Full experiment config dict.
        device: Target device.
        target_resolution: Resize CARLA frames to this before encoding.
            Must match Phase 2 training resolution for JEPA arms.
        num_temporal_frames: Number of frames to stack for temporal encoders.
    """

    def __init__(
        self,
        arm: str,
        config: dict,
        device: str = "cuda",
        target_resolution: int = 224,
        num_temporal_frames: int = 4,
    ):
        super().__init__()
        self.arm = arm
        self.device = device
        self.target_resolution = target_resolution

        # Create encoder + adapter from our adapter factory
        self.encoder, self.adapter = create_encoder(arm, config, device)

        # Move to device
        self.encoder = self.encoder.to(device)
        self.adapter = self.adapter.to(device)

        # Temporal encoders need frame stacking
        self._needs_temporal = arm in ("custom_jepa", "vjepa2")
        if self._needs_temporal:
            self.frame_stacker = FrameStacker(
                num_frames=num_temporal_frames,
                device=device,
            )
        else:
            self.frame_stacker = None

        # Output dim for CarDreamer's RSSM
        self.output_dim = config["adapter"]["target_dim"]

        # Log encoder info
        total_params = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"[EncoderHook] arm='{arm}', total={total_params/1e6:.1f}M, "
              f"trainable={trainable/1e6:.1f}M, output_dim={self.output_dim}")

    def forward(self, obs_image: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of CARLA observations.

        Args:
            obs_image: (B, C, H, W) image tensor from CarDreamer env,
                       values in [0, 1] or [0, 255] depending on env config.

        Returns:
            (B, D) feature tensor for CarDreamer's RSSM.
        """
        B = obs_image.shape[0]

        # Normalize to [0, 1] if needed
        if obs_image.max() > 1.0:
            obs_image = obs_image.float() / 255.0

        # Resize to target resolution (224×224)
        if obs_image.shape[-1] != self.target_resolution:
            obs_image = TF.resize(
                obs_image,
                [self.target_resolution, self.target_resolution],
                antialias=True,
            )

        if self._needs_temporal and self.frame_stacker is not None:
            # Process each sample in the batch through the frame stacker
            # NOTE: This assumes sequential single-step calls during rollout.
            # For batch training from replay, CarDreamer handles temporality
            # through the RSSM, so we add a trivial temporal dim.
            if B == 1:
                # Online rollout: use frame stacker
                frame = obs_image.squeeze(0)  # (C, H, W)
                clip = self.frame_stacker.push(frame)  # (C, T, H, W)
                clip = clip.unsqueeze(0).to(self.device)  # (1, C, T, H, W)
                features = self.encoder(clip)
            else:
                # Batch from replay: add trivial temporal dim
                # The RSSM provides real temporal modeling
                obs_5d = obs_image.unsqueeze(2)  # (B, C, 1, H, W)
                features = self.encoder(obs_5d)
        else:
            # CNN arm: single frame
            features = self.encoder(obs_image)

        return self.adapter(features)  # (B, D)

    def reset(self):
        """Reset frame stacker at episode boundaries."""
        if self.frame_stacker is not None:
            self.frame_stacker.reset()

    @property
    def frozen(self) -> bool:
        """Whether the encoder weights are frozen (JEPA/V-JEPA2 arms)."""
        return self.arm in ("custom_jepa", "vjepa2")


def patch_dreamerv3_encoder(agent, hook: CarDreamerEncoderHook):
    """
    Monkey-patch CarDreamer's DreamerV3 agent to use our encoder hook.

    This replaces the agent's encoder forward pass with our hook,
    while keeping all other DreamerV3 components (RSSM, actor-critic,
    replay buffer, training loop) intact.

    Args:
        agent: CarDreamer's DreamerV3 agent instance.
        hook: Our encoder hook with the desired arm.

    Note:
        The exact attribute path depends on CarDreamer's DreamerV3
        implementation structure. This may need adjustment based on
        the actual CarDreamer version. The two most likely patterns:

        Pattern A (attribute replacement):
            agent.wm.encoder = hook

        Pattern B (forward hook):
            agent.wm.encoder.register_forward_hook(...)

        We implement Pattern A as the default, with Pattern B as fallback.
    """
    # Pattern A: direct replacement
    # CarDreamer's DreamerV3 typically has: agent.wm.encoder
    if hasattr(agent, 'wm') and hasattr(agent.wm, 'encoder'):
        # Store original for potential restoration
        agent._original_encoder = agent.wm.encoder
        agent.wm.encoder = hook
        print(f"[patch] Replaced agent.wm.encoder with {hook.arm} hook")
        return True

    # Pattern B: if agent structure differs, try _nets or _model
    for attr in ('_nets', '_model', 'model'):
        obj = getattr(agent, attr, None)
        if obj is not None and hasattr(obj, 'encoder'):
            obj._original_encoder = obj.encoder
            obj.encoder = hook
            print(f"[patch] Replaced agent.{attr}.encoder with {hook.arm} hook")
            return True

    raise RuntimeError(
        "Could not find encoder in CarDreamer's DreamerV3 agent. "
        "Check the agent's attribute structure and update patch_dreamerv3_encoder()."
    )
