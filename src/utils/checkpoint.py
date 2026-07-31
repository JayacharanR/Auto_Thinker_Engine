"""
Checkpoint management with rotation policy.

Saves and loads model/optimizer state with a disk-discipline policy:
keep the last N checkpoints + the best-by-metric checkpoint.
This prevents disk fill from frequent checkpointing + video logging.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Optional

import torch


class CheckpointManager:
    """
    Manages model checkpoints with automatic rotation.

    Policy: keep the last `max_keep` checkpoints plus the single
    best checkpoint by a tracked metric. Old checkpoints beyond
    `max_keep` are deleted automatically.

    Args:
        checkpoint_dir: Directory to store checkpoints.
        max_keep: Maximum number of recent checkpoints to retain.
        metric_name: Name of the metric used to determine "best" checkpoint.
        metric_mode: 'max' (higher is better, e.g. success rate) or
                     'min' (lower is better, e.g. loss).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_keep: int = 3,
        metric_name: str = "eval/success_rate",
        metric_mode: str = "max",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        self.best_metric: Optional[float] = None
        self.recent_checkpoints: list[Path] = []

    def save(
        self,
        state: dict[str, Any],
        step: int,
        metric_value: Optional[float] = None,
    ) -> Path:
        """
        Save a checkpoint and apply rotation policy.

        Args:
            state: Dict containing model_state_dict, optimizer_state_dict,
                   step, config, and any other state to persist.
            step: Global training step (used in filename).
            metric_value: Current value of the tracked metric. If this is
                         the best seen so far, it's also saved as 'best.pt'.

        Returns:
            Path to the saved checkpoint file.
        """
        # Save timestamped checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_step{step:08d}.pt"
        state["step"] = step
        if metric_value is not None:
            state["metric_value"] = metric_value
            state["metric_name"] = self.metric_name

        torch.save(state, ckpt_path)
        self.recent_checkpoints.append(ckpt_path)

        # Check if this is the best checkpoint
        if metric_value is not None:
            is_best = False
            if self.best_metric is None:
                is_best = True
            elif self.metric_mode == "max" and metric_value > self.best_metric:
                is_best = True
            elif self.metric_mode == "min" and metric_value < self.best_metric:
                is_best = True

            if is_best:
                self.best_metric = metric_value
                best_path = self.checkpoint_dir / "best.pt"
                shutil.copy2(ckpt_path, best_path)

        # Rotate: delete old checkpoints beyond max_keep
        while len(self.recent_checkpoints) > self.max_keep:
            old_ckpt = self.recent_checkpoints.pop(0)
            if old_ckpt.exists() and old_ckpt.name != "best.pt":
                old_ckpt.unlink()

        return ckpt_path

    @staticmethod
    def load(
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda",
    ) -> dict[str, Any]:
        """
        Load a checkpoint into model and optimizer.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            model: Model to load state into.
            optimizer: Optional optimizer to load state into.
            device: Device to map tensors to.

        Returns:
            The full checkpoint dict (for extracting step, config, etc.).
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        return ckpt

    @staticmethod
    def load_best(
        checkpoint_dir: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Load the best checkpoint from a directory."""
        best_path = Path(checkpoint_dir) / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(
                f"No best checkpoint found at {best_path}. "
                "Has training produced a checkpoint with metric tracking?"
            )
        return CheckpointManager.load(str(best_path), model, optimizer, device)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint path, or None if no checkpoints exist."""
        ckpts = sorted(self.checkpoint_dir.glob("checkpoint_step*.pt"))
        return ckpts[-1] if ckpts else None


def build_checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Optional[dict] = None,
    **extra_state: Any,
) -> dict[str, Any]:
    """
    Build a standardized checkpoint state dict.

    Args:
        model: The model (or dict of models).
        optimizer: The optimizer (or dict of optimizers).
        step: Current training step.
        config: Run configuration dict.
        **extra_state: Any additional state to save (e.g., EMA weights).

    Returns:
        Checkpoint state dict ready for torch.save().
    """
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }
    if config is not None:
        state["config"] = config
    state.update(extra_state)
    return state
