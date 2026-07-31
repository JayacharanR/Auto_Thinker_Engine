"""
Experiment logging utilities.

Provides a unified ExperimentLogger that writes to TensorBoard (always)
and optionally to Weights & Biases. All training scripts use this
logger to ensure consistent metric naming and tracking.

Run naming convention: phase{N}_{arm}_{seed}_{date}
"""

import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    Unified experiment logger for TensorBoard + optional W&B.

    Every training script should instantiate one ExperimentLogger and use it
    for all metric logging. This ensures consistent run naming, VRAM tracking,
    and video rollout logging across all phases.

    Args:
        log_dir: Base directory for TensorBoard logs (under outputs/).
        run_name: Run name following convention: phase{N}_{arm}_{seed}_{date}.
        use_wandb: Whether to also log to Weights & Biases.
        wandb_project: W&B project name (only used if use_wandb=True).
        config: Full run config dict to log as hyperparameters.
    """

    def __init__(
        self,
        log_dir: str,
        run_name: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "driving-world-model",
        config: Optional[dict] = None,
    ):
        if run_name is None:
            run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

        self.run_name = run_name
        self.log_path = Path(log_dir) / run_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        # TensorBoard (always active)
        self.tb_writer = SummaryWriter(log_dir=str(self.log_path))

        # W&B (optional)
        self.use_wandb = use_wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config=config or {},
                    dir=str(self.log_path),
                )
            except ImportError:
                print("[WARNING] wandb not installed. Falling back to TensorBoard only.")
                self.use_wandb = False
            except Exception as e:
                print(f"[WARNING] W&B init failed: {e}. Falling back to TensorBoard only.")
                self.use_wandb = False

        # Log config as hyperparameters
        if config:
            self.tb_writer.add_hparams(
                hparam_dict=_flatten_dict(config),
                metric_dict={},
                run_name=run_name,
            )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric."""
        self.tb_writer.add_scalar(tag, value, step)
        if self.use_wandb and self.wandb_run:
            import wandb

            wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        """Log multiple related scalars (e.g., per-term reward breakdown)."""
        self.tb_writer.add_scalars(main_tag, tag_scalar_dict, step)
        if self.use_wandb and self.wandb_run:
            import wandb

            wandb.log(
                {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()},
                step=step,
            )

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram (e.g., encoder output distribution for collapse monitoring)."""
        self.tb_writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        """
        Log an image tensor.

        Args:
            tag: Image tag.
            image: Tensor of shape (C, H, W) in [0, 1].
            step: Global step.
        """
        self.tb_writer.add_image(tag, image, step)
        if self.use_wandb and self.wandb_run:
            import wandb

            wandb.log({tag: wandb.Image(image.permute(1, 2, 0).cpu().numpy())}, step=step)

    def log_video(self, tag: str, video: torch.Tensor, step: int, fps: int = 10) -> None:
        """
        Log a video rollout.

        Args:
            tag: Video tag.
            video: Tensor of shape (T, C, H, W) in [0, 1].
            step: Global step.
            fps: Frames per second for playback.
        """
        # TensorBoard expects (N, T, C, H, W) — add batch dim
        self.tb_writer.add_video(tag, video.unsqueeze(0), step, fps=fps)
        if self.use_wandb and self.wandb_run:
            import wandb

            # Convert to numpy (T, H, W, C) for W&B
            video_np = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
            wandb.log({tag: wandb.Video(video_np, fps=fps)}, step=step)

    def log_vram(self, step: int) -> None:
        """
        Log current and peak GPU VRAM usage.

        This should be called periodically (every N steps) to track
        memory pressure — critical for the shared-GPU setup.
        """
        if torch.cuda.is_available():
            current_mb = torch.cuda.memory_allocated() / 1e6
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            reserved_mb = torch.cuda.memory_reserved() / 1e6

            self.log_scalar("system/vram_current_mb", current_mb, step)
            self.log_scalar("system/vram_peak_mb", peak_mb, step)
            self.log_scalar("system/vram_reserved_mb", reserved_mb, step)

    def log_throughput(self, steps_per_sec: float, step: int) -> None:
        """Log training throughput in steps/sec."""
        self.log_scalar("system/steps_per_sec", steps_per_sec, step)

    def close(self) -> None:
        """Flush and close all writers."""
        self.tb_writer.close()
        if self.use_wandb and self.wandb_run:
            import wandb

            wandb.finish()


def make_run_name(phase: int, arm: str = "default", seed: int = 0) -> str:
    """
    Generate a standardized run name.

    Convention: phase{N}_{arm}_{seed}_{date}

    Args:
        phase: Phase number (0-3).
        arm: Arm identifier (e.g., 'cnn', 'custom_jepa', 'vjepa2').
        seed: Random seed.

    Returns:
        Formatted run name string.
    """
    date_str = time.strftime("%Y%m%d")
    return f"phase{phase}_{arm}_{seed}_{date_str}"


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dict for TensorBoard hparams logging."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (int, float, str, bool)):
            items.append((new_key, v))
        else:
            items.append((new_key, str(v)))
    return dict(items)
