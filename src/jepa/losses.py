"""
JEPA Loss Functions.

Computes loss between predictor output and target encoder output
at masked positions ONLY. No decoder, no reconstruction loss —
the loss lives entirely in the latent space.

Supports Smooth L1 (Huber) and L2 (MSE) losses as configured.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    """
    JEPA latent prediction loss.

    Computes the distance between predicted representations and target
    encoder representations, evaluated ONLY at masked positions.

    This is the sole training signal for the JEPA framework. There is
    deliberately no pixel-level reconstruction loss anywhere.

    Args:
        loss_type: 'smooth_l1' (Huber) or 'mse' (L2).
        beta: Smooth L1 transition point (only for smooth_l1).
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        beta: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute JEPA loss at masked positions.

        Args:
            predictions: (B, N_mask, D) predictor output at masked positions.
            targets: (B, N_mask, D) target encoder output at masked positions.
                     Must be detached (no gradient through target encoder).

        Returns:
            Scalar loss value.
        """
        assert predictions.shape == targets.shape, (
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

        # Ensure targets are detached (belt-and-suspenders — the target encoder
        # forward should already be @torch.no_grad(), but we verify here)
        targets = targets.detach()

        if self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(predictions, targets, beta=self.beta)
        elif self.loss_type == "mse":
            loss = F.mse_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class CollapseMonitor:
    """
    Monitor for representation collapse during JEPA training.

    Representation collapse occurs when the encoder outputs near-constant
    vectors regardless of input — the loss trivially goes to ~0 but the
    representations are useless. This is the HIGHEST RISK in Phase 2.

    Detection: track the variance of encoder outputs across a batch.
    Near-zero variance = collapse. Catch this early rather than after
    a full training run.

    If variance drops below threshold:
    1. First check EMA momentum (too low → target tracks context too closely)
    2. Then check predictor capacity (too large → predictor memorizes)
    3. Only then investigate architecture changes

    Args:
        variance_threshold: Minimum acceptable variance. Below this
            triggers a warning.
        log_every: Compute variance every N steps.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        log_every: int = 100,
    ):
        self.variance_threshold = variance_threshold
        self.log_every = log_every
        self._step = 0
        self._collapse_warnings = 0
        self._history: list[float] = []

    @torch.no_grad()
    def check(
        self,
        encoder_output: torch.Tensor,
        step: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Check encoder output for signs of collapse.

        Args:
            encoder_output: (B, N, D) encoder representations.
            step: Current training step (for logging frequency).

        Returns:
            Dict with:
            - 'variance': batch variance of encoder output
            - 'mean_norm': mean L2 norm of representations
            - 'collapsed': bool flag if variance below threshold
            - 'std': standard deviation of representations
        """
        if step is not None:
            self._step = step

        # Compute per-dimension variance across batch and patches
        flat = encoder_output.reshape(-1, encoder_output.shape[-1])
        variance = flat.var(dim=0).mean().item()
        std = flat.std(dim=0).mean().item()
        mean_norm = flat.norm(dim=-1).mean().item()

        collapsed = variance < self.variance_threshold
        if collapsed:
            self._collapse_warnings += 1

        self._history.append(variance)

        result = {
            "variance": variance,
            "std": std,
            "mean_norm": mean_norm,
            "collapsed": collapsed,
            "total_collapse_warnings": self._collapse_warnings,
        }

        if collapsed:
            print(
                f"\n⚠️  COLLAPSE WARNING (step {self._step}): "
                f"Encoder output variance = {variance:.6f} "
                f"(threshold: {self.variance_threshold})\n"
                f"  Action items:\n"
                f"  1. Check EMA momentum (current target too close to context?)\n"
                f"  2. Check predictor capacity (too powerful?)\n"
                f"  3. Total warnings so far: {self._collapse_warnings}\n"
            )

        return result

    def should_check(self, step: int) -> bool:
        """Whether to run collapse check at this step."""
        return step % self.log_every == 0

    @property
    def variance_history(self) -> list[float]:
        """Full history of variance measurements."""
        return self._history
