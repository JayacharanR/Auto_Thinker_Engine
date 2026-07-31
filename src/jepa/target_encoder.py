"""
EMA Target Encoder for JEPA pretraining.

The target encoder is architecturally identical to the context encoder
but its weights are an exponential moving average (EMA) of the context
encoder's weights. NO gradient flows through it — this is critical
and is enforced by the @torch.no_grad() decorator on the update method.

Classic bug source: target encoder accidentally receiving gradients.
This module includes assertions to verify no .grad is set.
"""

import copy

import torch
import torch.nn as nn


class EMATargetEncoder(nn.Module):
    """
    EMA (Exponential Moving Average) target encoder.

    Wraps a context encoder and maintains an EMA copy of its weights.
    The target encoder produces the prediction targets for the JEPA
    predictor — it is never directly optimized.

    The EMA momentum can optionally warm up from a lower value to the
    target value over a specified number of steps. This prevents the
    target encoder from being too close to the context encoder early
    in training (which can contribute to collapse).

    Args:
        context_encoder: The context encoder whose weights are EMA-tracked.
        momentum: Target EMA momentum (τ). Higher = slower update.
            Typical: 0.996.
        warmup_steps: Number of steps to linearly ramp momentum from
            warmup_start to momentum.
        warmup_start: Starting momentum for warmup. Typical: 0.99.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        momentum: float = 0.996,
        warmup_steps: int = 5000,
        warmup_start: float = 0.99,
    ):
        super().__init__()

        # Deep copy the encoder — independent parameters
        self.encoder = copy.deepcopy(context_encoder)

        # Freeze all parameters — no gradient should ever flow through
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.target_momentum = momentum
        self.warmup_steps = warmup_steps
        self.warmup_start = warmup_start
        self._step_count = 0

    @property
    def current_momentum(self) -> float:
        """Get current EMA momentum (accounts for warmup schedule)."""
        if self._step_count >= self.warmup_steps:
            return self.target_momentum

        # Linear ramp from warmup_start to target_momentum
        progress = self._step_count / max(self.warmup_steps, 1)
        return self.warmup_start + (self.target_momentum - self.warmup_start) * progress

    @torch.no_grad()
    def update(self, context_encoder: nn.Module) -> float:
        """
        Update target encoder weights as EMA of context encoder.

        This method MUST be called with @torch.no_grad() (enforced by
        decorator). It performs:
            target_params = τ * target_params + (1 - τ) * context_params

        Args:
            context_encoder: The context encoder with updated weights.

        Returns:
            The momentum value used for this update (for logging).
        """
        momentum = self.current_momentum
        self._step_count += 1

        for target_param, context_param in zip(
            self.encoder.parameters(), context_encoder.parameters()
        ):
            target_param.data.mul_(momentum).add_(
                context_param.data, alpha=1.0 - momentum
            )

        return momentum

    @torch.no_grad()
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through target encoder. No gradients.

        Args:
            x: Input tensor (same format as context encoder).
            **kwargs: Additional arguments passed to encoder.forward().

        Returns:
            Target encoder representations.
        """
        return self.encoder(x, **kwargs)

    def verify_no_gradients(self) -> bool:
        """
        Verify that no parameter in the target encoder has a gradient.

        This is a safety check — call it periodically during training
        to catch gradient leaks early. A gradient on any target encoder
        parameter indicates a bug in the training loop.

        Returns:
            True if no gradients found (correct behavior).

        Raises:
            AssertionError if any parameter has a gradient.
        """
        for name, param in self.encoder.named_parameters():
            assert param.grad is None, (
                f"TARGET ENCODER BUG: Parameter '{name}' has a gradient! "
                f"The target encoder must never receive gradients. "
                f"Check that all forward passes through the target encoder "
                f"are wrapped in torch.no_grad()."
            )
            assert not param.requires_grad, (
                f"TARGET ENCODER BUG: Parameter '{name}' has requires_grad=True! "
                f"All target encoder parameters must be frozen."
            )
        return True

    def state_dict_with_metadata(self) -> dict:
        """
        Get state dict with EMA metadata for checkpointing.

        Returns:
            Dict containing encoder state, step count, and momentum.
        """
        return {
            "encoder_state_dict": self.encoder.state_dict(),
            "step_count": self._step_count,
            "target_momentum": self.target_momentum,
            "warmup_steps": self.warmup_steps,
            "warmup_start": self.warmup_start,
        }

    def load_state_dict_with_metadata(self, state: dict) -> None:
        """Load state dict with EMA metadata."""
        self.encoder.load_state_dict(state["encoder_state_dict"])
        self._step_count = state["step_count"]
        self.target_momentum = state["target_momentum"]
        self.warmup_steps = state["warmup_steps"]
        self.warmup_start = state["warmup_start"]
