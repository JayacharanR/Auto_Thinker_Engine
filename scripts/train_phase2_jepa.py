"""
Phase 2 Training Script: JEPA Pretraining on comma2k19.

Self-supervised pretraining of the ViT-Small encoder using the
JEPA framework on driving video data. No labels, no pixel decoding.

Key training loop components:
1. Sample video tubelets from comma2k19
2. Generate multi-block masks
3. Context encoder processes unmasked patches
4. Target encoder (EMA) processes all patches
5. Predictor predicts target representations at masked positions
6. Loss = smooth L1 between predictions and targets at masked positions
7. EMA update of target encoder
8. Collapse monitoring throughout

Usage:
    python scripts/train_phase2_jepa.py --config configs/phase2_jepa_pretrain.yaml
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.comma2k19_dataset import create_comma2k19_dataloaders
from src.jepa.encoder import ViTEncoder
from src.jepa.losses import CollapseMonitor, JEPALoss
from src.jepa.masking import MaskGenerator, verify_no_leak
from src.jepa.predictor import JEPAPredictor
from src.jepa.target_encoder import EMATargetEncoder
from src.utils.checkpoint import CheckpointManager, build_checkpoint_state
from src.utils.logging_utils import ExperimentLogger, make_run_name
from src.utils.seeding import seed_everything


def train(config: dict):
    """Main Phase 2 JEPA training loop."""
    # Setup
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    mask_cfg = config["masking"]
    loss_cfg = config["loss"]

    seed = exp_cfg["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(seed)

    run_name = exp_cfg.get("run_name") or make_run_name(
        phase=2, arm="jepa_pretrain", seed=seed
    )
    logger = ExperimentLogger(
        log_dir=exp_cfg["log_dir"],
        run_name=run_name,
        use_wandb=exp_cfg.get("use_wandb", False),
        config=config,
    )

    ckpt_manager = CheckpointManager(
        checkpoint_dir=exp_cfg["checkpoint_dir"],
        max_keep=train_cfg.get("max_keep_checkpoints", 3),
        metric_name="val/loss",
        metric_mode="min",
    )

    # Data
    print("Loading comma2k19 data...")
    train_loader, val_loader = create_comma2k19_dataloaders(config, seed=seed)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples:   {len(val_loader.dataset)}")

    # Model
    ctx_cfg = model_cfg["context_encoder"]
    pred_cfg = model_cfg["predictor"]
    target_cfg = model_cfg["target_encoder"]
    tubelet_cfg = config["data"]["tubelet"]

    num_frames = tubelet_cfg["num_frames"]
    spatial_size = tubelet_cfg["spatial_size"]
    patch_size = ctx_cfg["patch_size"]
    tubelet_size = ctx_cfg["tubelet_size"]

    num_patches_spatial = (spatial_size // patch_size) ** 2
    num_patches_temporal = num_frames // tubelet_size
    num_patches = num_patches_spatial * num_patches_temporal

    # Context encoder
    context_encoder = ViTEncoder(
        img_size=spatial_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        embed_dim=ctx_cfg["embed_dim"],
        depth=ctx_cfg["depth"],
        num_heads=ctx_cfg["num_heads"],
        mlp_ratio=ctx_cfg["mlp_ratio"],
        drop_rate=ctx_cfg.get("drop_rate", 0.0),
        attn_drop_rate=ctx_cfg.get("attn_drop_rate", 0.0),
        num_frames=num_frames,
    ).to(device)

    # Target encoder (EMA copy)
    target_encoder = EMATargetEncoder(
        context_encoder=context_encoder,
        momentum=target_cfg["ema_momentum"],
        warmup_steps=target_cfg.get("ema_warmup_steps", 5000),
        warmup_start=target_cfg.get("ema_warmup_start", 0.99),
    ).to(device)

    # Predictor
    predictor = JEPAPredictor(
        context_dim=ctx_cfg["embed_dim"],
        embed_dim=pred_cfg["embed_dim"],
        depth=pred_cfg["depth"],
        num_heads=pred_cfg["num_heads"],
        mlp_ratio=pred_cfg["mlp_ratio"],
        num_patches=num_patches,
        action_conditioning=pred_cfg.get("action_conditioning", True),
        action_dim=pred_cfg.get("action_dim", 2),
        action_embed_dim=pred_cfg.get("action_embed_dim", 64),
    ).to(device)

    # Mask generator
    mask_generator = MaskGenerator(
        num_patches=num_patches,
        num_patches_spatial=num_patches_spatial,
        num_patches_temporal=num_patches_temporal,
        strategy=mask_cfg["strategy"],
        config=mask_cfg,
    )

    # Loss
    criterion = JEPALoss(
        loss_type=loss_cfg["type"],
        beta=loss_cfg.get("beta", 1.0),
    )

    # Collapse monitor
    collapse_cfg = train_cfg.get("collapse_monitoring", {})
    collapse_monitor = CollapseMonitor(
        variance_threshold=collapse_cfg.get("variance_threshold", 0.01),
        log_every=collapse_cfg.get("log_variance_every", 100),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg.get("adam_betas", [0.9, 0.95])),
    )

    # Cosine LR schedule with warmup
    total_epochs = train_cfg["total_epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    min_lr = train_cfg.get("min_lr", 1e-6)

    def lr_schedule(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(min_lr / train_cfg["learning_rate"], 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if train_cfg.get("mixed_precision", True) and device == "cuda" else None

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    print(f"\nStarting Phase 2 JEPA training: {run_name}")
    print(f"  Encoder: ViT-S ({sum(p.numel() for p in context_encoder.parameters())/1e6:.1f}M params)")
    print(f"  Predictor: ({sum(p.numel() for p in predictor.parameters())/1e6:.1f}M params)")
    print(f"  Patches: {num_patches} ({num_patches_spatial} spatial × {num_patches_temporal} temporal)")
    print(f"  Masking: {mask_cfg['strategy']}")
    print(f"  Device: {device}")

    for epoch in range(total_epochs):
        context_encoder.train()
        predictor.train()
        epoch_losses = []
        epoch_start = time.time()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")):
            video = batch["video"].to(device)  # (B, C, T, H, W)
            telemetry = batch["telemetry"].to(device)  # (B, T, A)
            batch_size = video.shape[0]

            # Generate masks
            masks = mask_generator(batch_size)
            context_indices = masks["context_indices"].to(device)
            mask_indices = masks["mask_indices"].to(device)

            # Verify no information leak (periodically)
            if global_step % 1000 == 0:
                verify_no_leak(context_indices, mask_indices)

            # Forward pass
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                # Context encoder: process unmasked patches
                context_output = context_encoder(video, mask_indices=context_indices)

                # Target encoder: process ALL patches (no masking)
                with torch.no_grad():
                    target_output = target_encoder(video)

                    # Extract target at masked positions
                    target_at_mask = torch.gather(
                        target_output,
                        dim=1,
                        index=mask_indices.unsqueeze(-1).expand(
                            -1, -1, target_output.shape[-1]
                        ),
                    )

                # Predictor: predict target at masked positions
                predictions = predictor(
                    context_tokens=context_output,
                    context_indices=context_indices,
                    mask_indices=mask_indices,
                    actions=telemetry if predictor.action_conditioning else None,
                )

                # Loss
                loss = criterion(predictions, target_at_mask)

            # Backward
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(context_encoder.parameters()) + list(predictor.parameters()),
                    train_cfg.get("grad_clip", 1.0),
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(context_encoder.parameters()) + list(predictor.parameters()),
                    train_cfg.get("grad_clip", 1.0),
                )
                optimizer.step()

            # EMA update (no gradient!)
            momentum = target_encoder.update(context_encoder)

            epoch_losses.append(loss.item())
            global_step += 1

            # Logging
            if global_step % train_cfg.get("log_every", 50) == 0:
                logger.log_scalar("train/loss", loss.item(), global_step)
                logger.log_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                logger.log_scalar("train/ema_momentum", momentum, global_step)
                logger.log_scalar("train/mask_ratio", masks["mask_ratio"], global_step)
                logger.log_vram(global_step)

            # Collapse monitoring
            if collapse_monitor.should_check(global_step):
                collapse_info = collapse_monitor.check(context_output.detach(), global_step)
                logger.log_scalar("collapse/variance", collapse_info["variance"], global_step)
                logger.log_scalar("collapse/std", collapse_info["std"], global_step)
                logger.log_scalar("collapse/mean_norm", collapse_info["mean_norm"], global_step)

                # Also verify target encoder has no gradients
                target_encoder.verify_no_gradients()

        scheduler.step()

        # Epoch summary
        epoch_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1}: loss={epoch_loss:.4f}, time={epoch_time:.1f}s")

        # Validation
        val_loss = validate(context_encoder, target_encoder, predictor, mask_generator,
                           criterion, val_loader, device, scaler)
        logger.log_scalar("val/loss", val_loss, global_step)
        print(f"  Val loss: {val_loss:.4f}")

        # Checkpoint
        if (epoch + 1) % train_cfg.get("checkpoint_every_epochs", 5) == 0:
            ckpt_state = {
                "context_encoder_state_dict": context_encoder.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "target_encoder": target_encoder.state_dict_with_metadata(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
            }
            ckpt_manager.save(ckpt_state, global_step, val_loss)

    logger.close()
    print(f"\nPhase 2 training complete. Best val loss: {best_val_loss:.4f}")


def validate(context_encoder, target_encoder, predictor, mask_generator,
             criterion, val_loader, device, scaler) -> float:
    """Run validation and return mean loss."""
    context_encoder.eval()
    losses = []

    with torch.no_grad():
        for batch in val_loader:
            video = batch["video"].to(device)
            telemetry = batch["telemetry"].to(device)
            batch_size = video.shape[0]

            masks = mask_generator(batch_size)
            context_indices = masks["context_indices"].to(device)
            mask_indices = masks["mask_indices"].to(device)

            context_output = context_encoder(video, mask_indices=context_indices)
            target_output = target_encoder(video)
            target_at_mask = torch.gather(
                target_output, dim=1,
                index=mask_indices.unsqueeze(-1).expand(-1, -1, target_output.shape[-1]),
            )

            predictions = predictor(
                context_tokens=context_output,
                context_indices=context_indices,
                mask_indices=mask_indices,
                actions=telemetry if predictor.action_conditioning else None,
            )

            loss = criterion(predictions, target_at_mask)
            losses.append(loss.item())

    context_encoder.train()
    return np.mean(losses) if losses else float("inf")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: JEPA Pretraining on comma2k19")
    parser.add_argument("--config", default="configs/phase2_jepa_pretrain.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
