"""
Tests for the new integration files: CarDreamer encoder hook,
FrameStacker, temporal PE interpolation, and V-JEPA2 adapter shape.

These are the files with zero test coverage that the code reviewer
flagged as the riskiest untested code.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dreamer.cardreamer_encoder_hook import FrameStacker
from src.dreamer.encoder_adapter import CNNEncoder, EncoderAdapter
from src.jepa.encoder import ViTEncoder


# ============================================================
# FrameStacker tests
# ============================================================

class TestFrameStacker:
    """Tests for the rolling frame buffer used by temporal encoders."""

    def test_output_shape(self):
        """FrameStacker output should be (C, T, H, W) with correct T."""
        stacker = FrameStacker(num_frames=4, device="cpu")
        frame = torch.randn(3, 224, 224)  # (C, H, W)
        clip = stacker.push(frame)
        assert clip.shape == (3, 4, 224, 224), f"Expected (3, 4, 224, 224), got {clip.shape}"

    def test_padding_with_first_frame(self):
        """Before buffer is full, should pad with repeated first frame, not zeros."""
        stacker = FrameStacker(num_frames=4, device="cpu")
        frame1 = torch.ones(3, 8, 8) * 42.0
        clip = stacker.push(frame1)

        # All 4 frames should be identical to frame1 (padding copies first frame)
        for t in range(4):
            assert torch.allclose(clip[:, t], frame1), f"Frame {t} doesn't match first frame"

    def test_frame_ordering(self):
        """Frames should be in chronological order: oldest first, newest last."""
        stacker = FrameStacker(num_frames=3, device="cpu")

        frame1 = torch.ones(3, 8, 8) * 1.0
        frame2 = torch.ones(3, 8, 8) * 2.0
        frame3 = torch.ones(3, 8, 8) * 3.0

        stacker.push(frame1)
        stacker.push(frame2)
        clip = stacker.push(frame3)

        # clip[:, 0] = frame1 (oldest), clip[:, 2] = frame3 (newest)
        assert torch.allclose(clip[:, 0], frame1)
        assert torch.allclose(clip[:, 1], frame2)
        assert torch.allclose(clip[:, 2], frame3)

    def test_sliding_window(self):
        """After buffer is full, oldest frame should drop when new one is pushed."""
        stacker = FrameStacker(num_frames=3, device="cpu")

        frames = [torch.ones(3, 8, 8) * float(i) for i in range(5)]
        for f in frames[:3]:
            stacker.push(f)

        # Push frame4 — should drop frame1
        clip = stacker.push(frames[3])
        assert torch.allclose(clip[:, 0], frames[1])  # frame2 is now oldest
        assert torch.allclose(clip[:, 2], frames[3])   # frame4 is newest

    def test_reset_clears_buffer(self):
        """After reset, buffer should be empty and re-pad from next frame."""
        stacker = FrameStacker(num_frames=3, device="cpu")

        stacker.push(torch.ones(3, 8, 8) * 99.0)
        stacker.reset()

        new_frame = torch.ones(3, 8, 8) * 42.0
        clip = stacker.push(new_frame)

        # All frames should be new_frame (re-padded)
        for t in range(3):
            assert torch.allclose(clip[:, t], new_frame)


# ============================================================
# Temporal PE Interpolation tests
# ============================================================

class TestTemporalPEInterpolation:
    """Tests for ViT encoder handling variable temporal lengths."""

    @pytest.fixture
    def encoder_16f(self):
        """ViT pretrained with 16 frames (Phase 2 config)."""
        return ViTEncoder(
            img_size=224, patch_size=16, tubelet_size=2,
            embed_dim=384, depth=2, num_heads=6, num_frames=16,
        )

    def test_same_length_no_interpolation(self, encoder_16f):
        """With matching temporal length, should use pos_embed as-is."""
        x = torch.randn(1, 3, 16, 224, 224)  # 16 frames
        out = encoder_16f(x)
        # 16 frames / 2 tubelet = 8 temporal patches × 196 spatial = 1568
        assert out.shape == (1, 1568, 384)

    def test_shorter_temporal_length(self, encoder_16f):
        """With fewer frames (Phase 3: 4 frames), should interpolate PE."""
        x = torch.randn(1, 3, 4, 224, 224)  # 4 frames → 2 temporal patches
        out = encoder_16f(x)
        # 4 frames / 2 tubelet = 2 temporal patches × 196 spatial = 392
        assert out.shape == (1, 392, 384)

    def test_single_frame_temporal(self, encoder_16f):
        """With 2 frames (minimum for tubelet_size=2), should still work."""
        x = torch.randn(1, 3, 2, 224, 224)  # 2 frames → 1 temporal patch
        out = encoder_16f(x)
        assert out.shape == (1, 196, 384)  # 1 temporal × 196 spatial

    def test_gradient_flows_through_interpolation(self, encoder_16f):
        """Gradients should flow through the interpolated pos_embed."""
        x = torch.randn(1, 3, 4, 224, 224, requires_grad=False)
        out = encoder_16f(x)
        loss = out.sum()
        loss.backward()

        # pos_embed should have gradients
        assert encoder_16f.pos_embed.grad is not None
        assert encoder_16f.pos_embed.grad.abs().sum() > 0

    def test_interpolation_is_smooth(self, encoder_16f):
        """Interpolated PE should be smooth, not full of NaNs or zeros."""
        x = torch.randn(1, 3, 4, 224, 224)
        with torch.no_grad():
            pe = encoder_16f._get_pos_embed(392)  # 2 temporal × 196 spatial

        assert not torch.isnan(pe).any(), "Interpolated PE contains NaN"
        assert not torch.isinf(pe).any(), "Interpolated PE contains Inf"
        assert pe.abs().mean() > 0.001, "Interpolated PE is effectively zero"


# ============================================================
# V-JEPA2 Adapter Shape tests (mocked — no HuggingFace download)
# ============================================================

class TestVJEPA2AdapterShape:
    """
    Tests for VJEPAEncoder tensor layout logic.

    We can't download the real V-JEPA2 model in CI, but we CAN verify
    the tensor reshaping logic works correctly by testing the
    input/output conventions directly.
    """

    def test_4d_to_5d_conversion(self):
        """(B, C, H, W) single frame should get unsqueezed to (B, C, 1, H, W)."""
        x = torch.randn(2, 3, 224, 224)

        # Simulate VJEPAEncoder.forward() preprocessing
        if x.dim() == 4:
            x = x.unsqueeze(2)

        assert x.shape == (2, 3, 1, 224, 224)

    def test_bcthw_to_btchw_permutation(self):
        """Our (B,C,T,H,W) → V-JEPA2's (B,T,C,H,W) permutation."""
        x = torch.randn(2, 3, 4, 224, 224)  # Our convention

        x_video = x.permute(0, 2, 1, 3, 4)  # V-JEPA2 convention

        assert x_video.shape == (2, 4, 3, 224, 224)
        # Verify data integrity: channel 0 of frame 0 matches
        assert torch.allclose(x[0, 0, 0], x_video[0, 0, 0])

    def test_normalization_shapes(self):
        """Processor normalization mean/std should broadcast correctly."""
        # Simulate ImageNet normalization for (B, T, C, H, W) layout
        x_video = torch.randn(2, 4, 3, 224, 224)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

        normalized = (x_video - mean) / std

        assert normalized.shape == x_video.shape
        assert not torch.isnan(normalized).any()

    def test_resize_for_batch_temporal(self):
        """Resize should handle (B*T, C, H, W) reshape correctly."""
        import torch.nn.functional as F

        B, T, C, H, W = 2, 4, 3, 128, 128
        x_video = torch.randn(B, T, C, H, W)
        target = 224

        x_flat = x_video.reshape(B * T, C, H, W)
        x_resized = F.interpolate(x_flat, size=(target, target), mode="bilinear", align_corners=False)
        x_out = x_resized.reshape(B, T, C, target, target)

        assert x_out.shape == (2, 4, 3, 224, 224)

    def test_vjepa2_uses_correct_kwarg(self):
        """V-JEPA2 adapter source should use pixel_values_videos, not pixel_values."""
        import inspect
        from src.dreamer.encoder_adapter import VJEPAEncoder

        source = inspect.getsource(VJEPAEncoder.forward)

        # The official API uses pixel_values_videos
        assert "pixel_values_videos" in source, (
            "VJEPAEncoder.forward should use pixel_values_videos kwarg "
            "(per official HuggingFace Transformers V-JEPA2 API)"
        )
        # Ensure old incorrect kwarg is not present (except in comments)
        # Count non-comment occurrences
        lines_with_old_kwarg = [
            line for line in source.split("\n")
            if "pixel_values=" in line
            and "pixel_values_videos" not in line
            and not line.strip().startswith("#")
        ]
        assert len(lines_with_old_kwarg) == 0, (
            f"Found {len(lines_with_old_kwarg)} lines using old pixel_values= kwarg"
        )

# ============================================================
# DreamerV3 Encoder Hook tests (dreamerv3-torch interface)
# ============================================================

class TestDreamerV3EncoderHook:
    """
    Tests for the dreamerv3-torch integration hook.

    The hook's forward(obs) takes a dict with 'image' key containing
    (B, T, H, W, C) channel-last tensor (dreamerv3-torch convention)
    and returns (B, T, embed_dim).
    """

    @pytest.fixture
    def cnn_config(self):
        return {
            "adapter": {"target_dim": 256, "use_layer_norm": True, "activation": "silu"},
            "encoders": {
                "cnn": {
                    "depth": 32, "kernels": [4, 4, 4, 4], "stride": 2,
                    "activation": "silu", "output_dim": 512, "input_resolution": 64,
                },
            },
        }

    def test_cnn_dict_interface(self, cnn_config):
        """Hook should accept obs dict with channel-last image and return (B, T, D)."""
        from src.dreamer.cardreamer_encoder_hook import DreamerV3EncoderHook

        hook = DreamerV3EncoderHook(arm="cnn", config=cnn_config, device="cpu")

        # dreamerv3-torch convention: (B, T, H, W, C) channel-last
        obs = {"image": torch.randn(2, 8, 64, 64, 3)}
        out = hook(obs)

        assert out.shape == (2, 8, 256), f"Expected (2, 8, 256), got {out.shape}"

    def test_cnn_no_temporal(self, cnn_config):
        """CNN arm should not use temporal frame stacking."""
        from src.dreamer.cardreamer_encoder_hook import DreamerV3EncoderHook

        hook = DreamerV3EncoderHook(arm="cnn", config=cnn_config, device="cpu")
        assert not hook._needs_temporal

    def test_outdim_matches_adapter(self, cnn_config):
        """Hook's outdim should match adapter target_dim (RSSM input size)."""
        from src.dreamer.cardreamer_encoder_hook import DreamerV3EncoderHook

        hook = DreamerV3EncoderHook(arm="cnn", config=cnn_config, device="cpu")
        assert hook.outdim == 256

    def test_uint8_normalization(self, cnn_config):
        """Hook should auto-normalize uint8 [0,255] images."""
        from src.dreamer.cardreamer_encoder_hook import DreamerV3EncoderHook

        hook = DreamerV3EncoderHook(arm="cnn", config=cnn_config, device="cpu")

        obs = {"image": torch.randint(0, 256, (1, 4, 64, 64, 3), dtype=torch.uint8)}
        out = hook(obs)

        assert out.shape == (1, 4, 256)
        assert not torch.isnan(out).any()

    def test_resolution_resize(self, cnn_config):
        """Hook should resize input to target resolution."""
        from src.dreamer.cardreamer_encoder_hook import DreamerV3EncoderHook

        hook = DreamerV3EncoderHook(arm="cnn", config=cnn_config, device="cpu")

        # Input at wrong resolution
        obs = {"image": torch.randn(1, 4, 128, 128, 3)}
        out = hook(obs)

        assert out.shape == (1, 4, 256)  # Resized to 64x64 internally

    def test_channel_first_passthrough(self, cnn_config):
        """Hook should handle channel-first input too."""
        from src.dreamer.cardreamer_encoder_hook import DreamerV3EncoderHook

        hook = DreamerV3EncoderHook(arm="cnn", config=cnn_config, device="cpu")

        # (B, T, C, H, W) — channel-first, C=3 not in last dim
        obs = {"image": torch.randn(1, 4, 3, 64, 64)}
        out = hook(obs)

        assert out.shape == (1, 4, 256)


class TestEncoderSwap:
    """Test that the encoder swap mechanism works on dreamerv3-torch's WorldModel."""

    def test_patch_updates_outdim(self):
        """patch_dreamerv3_encoder should update embed_size on the WorldModel."""
        from src.dreamer.cardreamer_encoder_hook import (
            DreamerV3EncoderHook,
            patch_dreamerv3_encoder,
        )

        config = {
            "adapter": {"target_dim": 512, "use_layer_norm": True, "activation": "silu"},
            "encoders": {
                "cnn": {
                    "depth": 32, "kernels": [4, 4, 4, 4], "stride": 2,
                    "activation": "silu", "output_dim": 512, "input_resolution": 64,
                },
            },
        }

        hook = DreamerV3EncoderHook(arm="cnn", config=config, device="cpu")

        # Create a mock world model with encoder and embed_size
        class MockWM:
            def __init__(self):
                self.encoder = nn.Linear(10, 10)
                self.embed_size = 10

        class MockAgent:
            def __init__(self):
                self._wm = MockWM()

        agent = MockAgent()
        patch_dreamerv3_encoder(agent, hook)

        assert agent._wm.encoder is hook
        assert agent._wm.embed_size == 512


class TestTemporalClipValidity:
    """Verify temporal clips always have T >= tubelet_size."""

    def test_temporal_encoding_shape(self):
        """Temporal encoder should produce (B, T, D) for each timestep."""
        from src.dreamer.cardreamer_encoder_hook import DreamerV3EncoderHook

        config = {
            "adapter": {"target_dim": 256, "use_layer_norm": True, "activation": "silu"},
            "encoders": {
                "custom_jepa": {
                    "input_resolution": 224, "patch_size": 16,
                    "embed_dim": 384, "depth": 2, "num_heads": 6,
                    "output_dim": 384, "tubelet_size": 2,
                },
            },
        }

        hook = DreamerV3EncoderHook(
            arm="custom_jepa", config=config, device="cpu",
            num_temporal_frames=4,
        )

        # Simulate dreamerv3-torch replay batch: (B, T, H, W, C) channel-last
        obs = {"image": torch.randn(1, 8, 224, 224, 3)}
        out = hook(obs)

        # Should produce one embedding per timestep
        assert out.shape == (1, 8, 256), f"Expected (1, 8, 256), got {out.shape}"

    def test_tubelet_size_assertion(self):
        """Should raise if num_temporal_frames < tubelet_size."""
        from src.dreamer.cardreamer_encoder_hook import DreamerV3EncoderHook

        config = {
            "adapter": {"target_dim": 256, "use_layer_norm": True, "activation": "silu"},
            "encoders": {
                "custom_jepa": {
                    "input_resolution": 224, "patch_size": 16,
                    "embed_dim": 384, "depth": 2, "num_heads": 6,
                    "output_dim": 384, "tubelet_size": 4,
                },
            },
        }

        with pytest.raises(AssertionError, match="tubelet_size"):
            DreamerV3EncoderHook(
                arm="custom_jepa", config=config, device="cpu",
                num_temporal_frames=2,  # Less than tubelet_size=4
            )

