"""
Linear Probe for evaluating JEPA encoder quality.

Freezes the pretrained encoder and trains a single linear layer
on top to regress steering angle from held-out comma2k19 clips.

CRITICAL: Always run the identical probe on an UNTRAINED (randomly
initialized) encoder of the same architecture as a control.
Report BOTH numbers — the delta is the evidence, not the trained
number alone.
"""

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class LinearProbe(nn.Module):
    """
    Linear probe for evaluating encoder representations.

    Takes the frozen encoder's output, applies global average pooling,
    then a single linear layer to predict the target (e.g., steering angle).

    Args:
        encoder: Pretrained encoder (will be frozen).
        encoder_dim: Dimension of encoder output.
        output_dim: Prediction dimension (1 for steering regression).
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 384,
        output_dim: int = 1,
    ):
        super().__init__()

        # Freeze the encoder — no gradient flows through it
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Single linear probe layer
        self.probe = nn.Linear(encoder_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) video input.

        Returns:
            (B, output_dim) predictions.
        """
        with torch.no_grad():
            features = self.encoder(x)  # (B, N, D)

        # Global average pooling over patches
        features = features.mean(dim=1)  # (B, D)

        return self.probe(features)


def train_linear_probe(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str = "cuda",
    logger: Optional[object] = None,
) -> dict[str, float]:
    """
    Train and evaluate a linear probe on an encoder.

    Args:
        encoder: The encoder to probe (will be frozen internally).
        train_loader: Training data (video + steering labels).
        val_loader: Validation data.
        config: Linear probe config section.
        device: Torch device.
        logger: Optional ExperimentLogger for metric tracking.

    Returns:
        Dict with 'mae', 'r2', and 'final_loss' metrics on validation set.
    """
    encoder_dim = config.get("probe_hidden_dim") or _infer_encoder_dim(encoder, device)
    epochs = config.get("probe_epochs", 50)
    lr = config.get("probe_lr", 1e-3)
    batch_size = config.get("probe_batch_size", 64)

    probe = LinearProbe(
        encoder=encoder,
        encoder_dim=encoder_dim,
        output_dim=1,
    ).to(device)

    optimizer = torch.optim.Adam(probe.probe.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training
        probe.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Probe epoch {epoch+1}/{epochs}", leave=False):
            video = batch["video"].to(device)
            telemetry = batch["telemetry"].to(device)

            # Target: mean steering angle over the tubelet
            if telemetry.shape[-1] >= 1:
                target = telemetry[:, :, 0].mean(dim=1, keepdim=True)  # (B, 1)
            else:
                continue

            prediction = probe(video)
            loss = F.mse_loss(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        val_metrics = evaluate_linear_probe(probe, val_loader, device)

        if logger is not None:
            logger.log_scalar("probe/train_loss", np.mean(train_losses), epoch)
            logger.log_scalar("probe/val_mae", val_metrics["mae"], epoch)
            logger.log_scalar("probe/val_r2", val_metrics["r2"], epoch)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

    return val_metrics


def evaluate_linear_probe(
    probe: LinearProbe,
    data_loader: DataLoader,
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate a trained linear probe on a dataset."""
    probe.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in data_loader:
            video = batch["video"].to(device)
            telemetry = batch["telemetry"].to(device)

            if telemetry.shape[-1] >= 1:
                target = telemetry[:, :, 0].mean(dim=1, keepdim=True)
            else:
                continue

            prediction = probe(video)
            loss = F.mse_loss(prediction, target)

            total_loss += loss.item() * video.shape[0]
            count += video.shape[0]

            all_predictions.append(prediction.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    if not all_predictions:
        return {"mae": float("inf"), "r2": 0.0, "loss": float("inf")}

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions) if len(targets) > 1 else 0.0

    return {
        "mae": float(mae),
        "r2": float(r2),
        "loss": total_loss / max(count, 1),
    }


def run_probe_comparison(
    trained_encoder: nn.Module,
    encoder_class: type,
    encoder_kwargs: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str = "cuda",
    logger: Optional[object] = None,
) -> dict[str, dict[str, float]]:
    """
    Run the critical trained vs random-init probe comparison.

    This is the actual proof of JEPA pretraining quality. The trained
    encoder should meaningfully beat the random-init control on steering
    prediction. If it doesn't, pretraining didn't learn useful representations.

    Args:
        trained_encoder: The pretrained JEPA encoder.
        encoder_class: Class to instantiate for random baseline.
        encoder_kwargs: Kwargs for random encoder instantiation.
        train_loader: Training data.
        val_loader: Validation data.
        config: Probe config.
        device: Torch device.
        logger: Optional logger.

    Returns:
        Dict with 'trained' and 'random' result dicts, each containing
        'mae' and 'r2' metrics.
    """
    print("=" * 60)
    print("LINEAR PROBE COMPARISON: Trained vs Random-Init Encoder")
    print("=" * 60)

    # Probe on trained encoder
    print("\n--- Probing TRAINED encoder ---")
    trained_results = train_linear_probe(
        encoder=trained_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger,
    )

    # Probe on random-init encoder (identical architecture, untrained)
    print("\n--- Probing RANDOM-INIT encoder (control) ---")
    random_encoder = encoder_class(**encoder_kwargs).to(device)
    random_results = train_linear_probe(
        encoder=random_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger,
    )

    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"  Trained encoder MAE:   {trained_results['mae']:.4f}")
    print(f"  Random  encoder MAE:   {random_results['mae']:.4f}")
    print(f"  Delta MAE:             {random_results['mae'] - trained_results['mae']:.4f}")
    print(f"  Trained encoder R²:    {trained_results['r2']:.4f}")
    print(f"  Random  encoder R²:    {random_results['r2']:.4f}")
    print(f"  Delta R²:              {trained_results['r2'] - random_results['r2']:.4f}")
    print("=" * 60)

    if trained_results["mae"] >= random_results["mae"]:
        print("⚠️  WARNING: Trained encoder did NOT beat random baseline!")
        print("    Possible causes:")
        print("    1. Representation collapse during pretraining")
        print("    2. Insufficient pretraining epochs")
        print("    3. Wrong EMA momentum or predictor capacity")

    return {
        "trained": trained_results,
        "random": random_results,
    }


def _infer_encoder_dim(encoder: nn.Module, device: str) -> int:
    """Infer encoder output dimension by running a dummy forward pass."""
    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 16, 224, 224).to(device)
        try:
            out = encoder(dummy)
            return out.shape[-1]
        except Exception:
            return 384  # ViT-S default
