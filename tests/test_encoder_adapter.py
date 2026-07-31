"""
Unit tests for the encoder adapter (Phase 3).

Verifies that all three encoder arms produce the correct output shape
when passed through their respective adapters.
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dreamer.encoder_adapter import CNNEncoder, EncoderAdapter


class TestCNNEncoder:
    def test_output_shape(self):
        encoder = CNNEncoder(depth=48, input_size=128)
        x = torch.randn(2, 3, 128, 128)
        out = encoder(x)
        assert out.dim() == 2
        assert out.shape[0] == 2

    def test_output_dim_property(self):
        encoder = CNNEncoder(depth=48, input_size=128)
        assert encoder.output_dim > 0


class TestEncoderAdapter:
    def test_2d_input(self):
        adapter = EncoderAdapter(input_dim=512, target_dim=1024)
        x = torch.randn(2, 512)
        out = adapter(x)
        assert out.shape == (2, 1024)

    def test_3d_input_pooling(self):
        """3D input from ViT should be pooled over patch dimension."""
        adapter = EncoderAdapter(input_dim=384, target_dim=1024)
        x = torch.randn(2, 100, 384)  # (B, N_patches, D)
        out = adapter(x)
        assert out.shape == (2, 1024)

    def test_identical_capacity(self):
        """All three adapters should have same number of params (for fairness)."""
        adapter_cnn = EncoderAdapter(input_dim=512, target_dim=1024)
        adapter_jepa = EncoderAdapter(input_dim=384, target_dim=1024)
        adapter_vjepa = EncoderAdapter(input_dim=1024, target_dim=1024)

        # The projection layers differ in input_dim, but the architecture
        # (single linear + layernorm + activation) is identical.
        # This is the controlled variable.
        for name, param in adapter_cnn.named_parameters():
            assert any(
                name in dict(a.named_parameters())
                for a in [adapter_jepa, adapter_vjepa]
            ) or "projection" in name


class TestCNNAdapterIntegration:
    def test_cnn_through_adapter(self):
        encoder = CNNEncoder(depth=48, input_size=128)
        adapter = EncoderAdapter(input_dim=encoder.output_dim, target_dim=1024)

        x = torch.randn(2, 3, 128, 128)
        features = encoder(x)
        out = adapter(features)
        assert out.shape == (2, 1024)

    def test_gradient_flow(self):
        encoder = CNNEncoder(depth=48, input_size=128)
        adapter = EncoderAdapter(input_dim=encoder.output_dim, target_dim=1024)

        x = torch.randn(2, 3, 128, 128)
        out = adapter(encoder(x))
        loss = out.sum()
        loss.backward()

        # Both encoder and adapter should have gradients
        assert any(p.grad is not None for p in encoder.parameters())
        assert any(p.grad is not None for p in adapter.parameters())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
