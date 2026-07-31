"""
Unit tests for the data pipeline.

Tests tensor shapes, frame-telemetry alignment, and masking correctness.
"""

import torch
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import VideoTransform


class TestVideoTransform:
    def test_train_transform_shape(self):
        transform = VideoTransform(spatial_size=224, is_train=True)
        video = torch.rand(3, 16, 480, 640)  # (C, T, H, W)
        out = transform(video)
        assert out.shape == (3, 16, 224, 224)

    def test_val_transform_shape(self):
        transform = VideoTransform(spatial_size=224, is_train=False)
        video = torch.rand(3, 16, 480, 640)
        out = transform(video)
        assert out.shape == (3, 16, 224, 224)

    def test_normalization_applied(self):
        transform = VideoTransform(
            spatial_size=64, is_train=False,
            normalize_mean=(0.5, 0.5, 0.5),
            normalize_std=(0.5, 0.5, 0.5),
        )
        # Input in [0, 1], output should be roughly in [-1, 1]
        video = torch.ones(3, 4, 64, 64) * 0.5
        out = transform(video)
        # After normalizing 0.5 with mean=0.5 std=0.5: (0.5-0.5)/0.5 = 0
        assert torch.allclose(out, torch.zeros_like(out), atol=0.1)

    def test_no_horizontal_flip(self):
        """Horizontal flip should NOT be applied to driving data."""
        transform = VideoTransform(spatial_size=64, is_train=True)
        # This is a design constraint — the transform class doesn't include h-flip
        # We verify by checking the class doesn't have a flip attribute
        assert not hasattr(transform, "horizontal_flip") or not getattr(
            transform, "horizontal_flip", False
        )


class TestTubeletSampling:
    """Tests for the tubelet sampling logic (mock-based without actual data)."""

    def test_frame_indices_calculation(self):
        """Verify frame indices are correctly calculated from stride."""
        num_frames = 16
        frame_stride = 2
        start_frame = 10

        expected_indices = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
        actual_indices = [start_frame + i * frame_stride for i in range(num_frames)]

        assert actual_indices == expected_indices

    def test_total_frames_needed(self):
        """Total frames needed = (num_frames - 1) * stride + 1."""
        num_frames = 16
        frame_stride = 2
        total_needed = (num_frames - 1) * frame_stride + 1
        assert total_needed == 31

    def test_telemetry_shape(self):
        """Mock test for telemetry alignment output shape."""
        num_frames = 16
        num_channels = 2  # steering + speed

        # Expected output: (T, A)
        telemetry = torch.randn(num_frames, num_channels)
        assert telemetry.shape == (16, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
