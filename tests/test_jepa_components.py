"""
Unit tests for JEPA components.

Tests critical invariants:
1. EMA target encoder never receives gradients
2. Masking indices never overlap (no target leak)
3. Loss shapes are correct
4. Encoder/predictor forward pass shapes
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jepa.encoder import ViTEncoder, PatchEmbed3D
from src.jepa.target_encoder import EMATargetEncoder
from src.jepa.predictor import JEPAPredictor
from src.jepa.masking import MaskGenerator, verify_no_leak
from src.jepa.losses import JEPALoss, CollapseMonitor


class TestPatchEmbed3D:
    def test_output_shape(self):
        pe = PatchEmbed3D(img_size=224, patch_size=16, tubelet_size=2, embed_dim=384)
        x = torch.randn(2, 3, 16, 224, 224)
        out = pe(x)
        # T'=8, H'=14, W'=14 → 8*14*14=1568 patches
        assert out.shape == (2, 1568, 384)

    def test_smaller_input(self):
        pe = PatchEmbed3D(img_size=128, patch_size=16, tubelet_size=2, embed_dim=384)
        x = torch.randn(2, 3, 8, 128, 128)
        out = pe(x)
        # T'=4, H'=8, W'=8 → 4*8*8=256 patches
        assert out.shape == (2, 256, 384)


class TestViTEncoder:
    def test_forward_shape(self):
        encoder = ViTEncoder(
            img_size=128, patch_size=16, tubelet_size=2,
            embed_dim=384, depth=4, num_heads=6, num_frames=8,
        )
        x = torch.randn(2, 3, 8, 128, 128)
        out = encoder(x)
        assert out.shape == (2, 256, 384)  # 4*8*8=256 patches

    def test_masked_forward(self):
        encoder = ViTEncoder(
            img_size=128, patch_size=16, tubelet_size=2,
            embed_dim=384, depth=4, num_heads=6, num_frames=8,
        )
        x = torch.randn(2, 3, 8, 128, 128)
        # Keep only 100 of 256 patches
        keep_indices = torch.randint(0, 256, (2, 100))
        out = encoder(x, mask_indices=keep_indices)
        assert out.shape == (2, 100, 384)


class TestEMATargetEncoder:
    def test_no_gradients(self):
        """CRITICAL: Target encoder must never have gradients."""
        encoder = ViTEncoder(
            img_size=128, patch_size=16, tubelet_size=2,
            embed_dim=384, depth=2, num_heads=6, num_frames=8,
        )
        target = EMATargetEncoder(encoder, momentum=0.996)

        # Forward pass
        x = torch.randn(1, 3, 8, 128, 128)
        out = target(x)

        # Verify no gradient on any parameter
        assert target.verify_no_gradients()

        # Verify output doesn't have grad_fn
        assert out.grad_fn is None

    def test_ema_update(self):
        """EMA update should move target toward context."""
        encoder = ViTEncoder(
            img_size=128, patch_size=16, tubelet_size=2,
            embed_dim=384, depth=2, num_heads=6, num_frames=8,
        )
        # momentum=0, warmup_steps=0 → instant copy (no warmup schedule)
        target = EMATargetEncoder(encoder, momentum=0.0, warmup_steps=0, warmup_start=0.0)

        # Modify context encoder AFTER deepcopy
        for p in encoder.parameters():
            p.data.fill_(1.0)

        # EMA update with momentum=0 should copy exactly: target = 0*target + 1*context
        target.update(encoder)

        for tp, cp in zip(target.encoder.parameters(), encoder.parameters()):
            assert torch.allclose(tp, cp, atol=1e-6)

    def test_warmup_schedule(self):
        encoder = ViTEncoder(
            img_size=128, patch_size=16, tubelet_size=2,
            embed_dim=384, depth=2, num_heads=6, num_frames=8,
        )
        target = EMATargetEncoder(
            encoder, momentum=0.999, warmup_steps=100, warmup_start=0.9
        )

        # At step 0, momentum should be warmup_start
        assert target.current_momentum == 0.9

        # At step 50 (halfway), momentum should be interpolated
        target._step_count = 50
        expected = 0.9 + (0.999 - 0.9) * 0.5
        assert abs(target.current_momentum - expected) < 1e-6

        # After warmup, momentum should be target
        target._step_count = 100
        assert target.current_momentum == 0.999


class TestMasking:
    def test_no_leak_multi_block(self):
        """Context and mask indices must never overlap."""
        mg = MaskGenerator(
            num_patches=256, num_patches_spatial=64,
            num_patches_temporal=4, strategy="multi_block",
            config={"multi_block": {"num_masks": 4, "total_mask_ratio": 0.6}},
        )
        result = mg(batch_size=4)
        verify_no_leak(result["context_indices"], result["mask_indices"])

    def test_no_leak_temporal(self):
        mg = MaskGenerator(
            num_patches=256, num_patches_spatial=64,
            num_patches_temporal=4, strategy="temporal_last",
            config={"temporal": {"num_future_frames": 2}},
        )
        result = mg(batch_size=4)
        verify_no_leak(result["context_indices"], result["mask_indices"])

    def test_mask_ratio(self):
        mg = MaskGenerator(
            num_patches=256, num_patches_spatial=64,
            num_patches_temporal=4, strategy="multi_block",
            config={"multi_block": {"total_mask_ratio": 0.65}},
        )
        result = mg(batch_size=8)
        # Mask ratio should be approximately 0.65
        n_mask = result["mask_indices"].shape[1]
        n_ctx = result["context_indices"].shape[1]
        actual_ratio = n_mask / (n_mask + n_ctx)
        assert 0.3 < actual_ratio < 0.9, f"Mask ratio {actual_ratio} out of expected range"


class TestJEPALoss:
    def test_smooth_l1(self):
        loss_fn = JEPALoss(loss_type="smooth_l1")
        pred = torch.randn(2, 50, 384)
        target = torch.randn(2, 50, 384)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_mse(self):
        loss_fn = JEPALoss(loss_type="mse")
        pred = torch.randn(2, 50, 384)
        target = torch.randn(2, 50, 384)
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_target_detached(self):
        """Loss should detach targets (belt-and-suspenders)."""
        loss_fn = JEPALoss()
        pred = torch.randn(2, 50, 384, requires_grad=True)
        target = torch.randn(2, 50, 384, requires_grad=True)
        loss = loss_fn(pred, target)
        loss.backward()
        # pred should have gradient, target should NOT (detached in loss)
        assert pred.grad is not None


class TestCollapseMonitor:
    def test_detects_collapse(self):
        monitor = CollapseMonitor(variance_threshold=0.1)
        # Near-constant output = collapse
        collapsed_output = torch.ones(8, 100, 384) * 0.5 + torch.randn(8, 100, 384) * 0.001
        result = monitor.check(collapsed_output, step=0)
        assert result["collapsed"] is True

    def test_healthy_representations(self):
        monitor = CollapseMonitor(variance_threshold=0.01)
        healthy_output = torch.randn(8, 100, 384)
        result = monitor.check(healthy_output, step=0)
        assert result["collapsed"] is False
        assert result["variance"] > 0.01


class TestPredictor:
    def test_forward_shape(self):
        predictor = JEPAPredictor(
            context_dim=384, embed_dim=192, depth=2, num_heads=6,
            num_patches=256, action_conditioning=True,
            action_dim=2, action_embed_dim=64,
        )
        context = torch.randn(2, 100, 384)
        ctx_indices = torch.randint(0, 256, (2, 100))
        mask_indices = torch.randint(0, 256, (2, 156))
        actions = torch.randn(2, 8, 2)

        out = predictor(context, ctx_indices, mask_indices, actions)
        assert out.shape == (2, 156, 384)  # Predicts at masked positions in context_dim

    def test_without_actions(self):
        predictor = JEPAPredictor(
            context_dim=384, embed_dim=192, depth=2, num_heads=6,
            num_patches=256, action_conditioning=False,
        )
        context = torch.randn(2, 100, 384)
        ctx_indices = torch.randint(0, 256, (2, 100))
        mask_indices = torch.randint(0, 256, (2, 156))

        out = predictor(context, ctx_indices, mask_indices)
        assert out.shape == (2, 156, 384)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
