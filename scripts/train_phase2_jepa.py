"""
Phase 2 Training Script: JEPA Pretraining on comma2k19.

Self-supervised pretraining of the ViT-Small encoder using the
JEPA framework on driving video data. No labels, no pixel decoding.

Key training loop components:
1. Sample video clips from the preprocessed comma2k19 arrays
2. Generate tube masks (spatial blocks masked in every frame)
3. Context encoder processes unmasked patches
4. Targets for the masked patches come from the full clip
5. Predictor predicts target representations at masked positions
6. Loss = smooth L1 between predictions and targets at masked positions
7. Collapse prevention, chosen by ``model.regularizer``:
   - ``ema``: targets from an EMA teacher with stop-gradient (V-JEPA)
   - ``sigreg``: targets from the online encoder; SIGReg keeps the
     embeddings near an isotropic Gaussian (LeJEPA)
8. Collapse monitoring throughout

Writes ``latest.pt`` (for --resume) and ``best.pt`` (lowest validation
prediction loss) to the checkpoint directory.

Usage:
    python scripts/train_phase2_jepa.py --config configs/phase2_jepa_laptop.yaml
    python scripts/train_phase2_jepa.py --config configs/phase2_jepa_laptop.yaml \\
        --regularizer sigreg --checkpoint-dir outputs/checkpoints/phase2_sigreg
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.comma2k19_dataset import create_comma2k19_dataloaders  # noqa: E402
from src.jepa.encoder import ViTEncoder  # noqa: E402
from src.jepa.losses import CollapseMonitor, JEPALoss  # noqa: E402
from src.jepa.masking import MaskGenerator, verify_no_leak  # noqa: E402
from src.jepa.predictor import JEPAPredictor  # noqa: E402
from src.jepa.sigreg import SIGReg  # noqa: E402
from src.jepa.target_encoder import EMATargetEncoder  # noqa: E402
from src.utils.logging_utils import ExperimentLogger, make_run_name  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402

REGULARIZERS = ("ema", "sigreg")


def patch_geometry(config: dict) -> dict:
    """Patch-grid sizes implied by the clip and encoder settings."""
    tubelet_cfg = config["data"]["tubelet"]
    ctx_cfg = config["model"]["context_encoder"]
    spatial = (tubelet_cfg["spatial_size"] // ctx_cfg["patch_size"]) ** 2
    temporal = tubelet_cfg["num_frames"] // ctx_cfg["tubelet_size"]
    return {"spatial": spatial, "temporal": temporal, "total": spatial * temporal}


def build_models(config: dict, device: str):
    """Context encoder, predictor and (for ``ema``) the EMA target encoder."""
    model_cfg = config["model"]
    ctx_cfg = model_cfg["context_encoder"]
    pred_cfg = model_cfg["predictor"]
    tubelet_cfg = config["data"]["tubelet"]
    regularizer = model_cfg.get("regularizer", "ema")
    if regularizer not in REGULARIZERS:
        raise ValueError(f"model.regularizer must be one of {REGULARIZERS}, got {regularizer!r}")

    context_encoder = ViTEncoder(
        img_size=tubelet_cfg["spatial_size"],
        patch_size=ctx_cfg["patch_size"],
        tubelet_size=ctx_cfg["tubelet_size"],
        embed_dim=ctx_cfg["embed_dim"],
        depth=ctx_cfg["depth"],
        num_heads=ctx_cfg["num_heads"],
        mlp_ratio=ctx_cfg["mlp_ratio"],
        drop_rate=ctx_cfg.get("drop_rate", 0.0),
        attn_drop_rate=ctx_cfg.get("attn_drop_rate", 0.0),
        num_frames=tubelet_cfg["num_frames"],
    ).to(device)
    predictor = JEPAPredictor(
        context_dim=ctx_cfg["embed_dim"],
        embed_dim=pred_cfg["embed_dim"],
        depth=pred_cfg["depth"],
        num_heads=pred_cfg["num_heads"],
        mlp_ratio=pred_cfg["mlp_ratio"],
        num_patches=patch_geometry(config)["total"],
        action_conditioning=pred_cfg.get("action_conditioning", False),
        action_dim=pred_cfg.get("action_dim", 2),
        action_embed_dim=pred_cfg.get("action_embed_dim", 64),
    ).to(device)

    target_encoder = None
    if regularizer == "ema":
        target_cfg = model_cfg["target_encoder"]
        target_encoder = EMATargetEncoder(
            context_encoder=context_encoder,
            momentum=target_cfg["ema_momentum"],
            warmup_steps=target_cfg.get("ema_warmup_steps", 5000),
            warmup_start=target_cfg.get("ema_warmup_start", 0.99),
        ).to(device)
    return context_encoder, predictor, target_encoder


def compute_losses(
    context_encoder: nn.Module,
    predictor: JEPAPredictor,
    target_encoder,
    criterion: JEPALoss,
    sigreg,
    sigreg_lambda: float,
    video: torch.Tensor,
    telemetry: torch.Tensor,
    context_indices: torch.Tensor,
    mask_indices: torch.Tensor,
    step: int,
) -> dict:
    """
    One JEPA objective evaluation.

    With an EMA ``target_encoder`` the targets carry no gradient. Without it
    (SIGReg), the online encoder encodes the full clip for the targets and
    ``loss = (1 - lambda) * prediction + lambda * SIGReg(full-clip tokens)``.
    """
    context_output = context_encoder(video, mask_indices=context_indices)
    if target_encoder is not None:
        with torch.no_grad():
            full_output = target_encoder(video)
    else:
        full_output = context_encoder(video)
    target_at_mask = torch.gather(
        full_output, dim=1,
        index=mask_indices.unsqueeze(-1).expand(-1, -1, full_output.shape[-1]),
    )
    predictions = predictor(
        context_tokens=context_output,
        context_indices=context_indices,
        mask_indices=mask_indices,
        actions=telemetry if predictor.action_conditioning else None,
    )
    pred_loss = criterion(predictions.float(), target_at_mask.float())
    out = {"pred_loss": pred_loss, "loss": pred_loss, "context_output": context_output}
    if sigreg is not None:
        out["sigreg"] = sigreg(full_output, step)
        out["loss"] = (1.0 - sigreg_lambda) * pred_loss + sigreg_lambda * out["sigreg"]
    return out


def _autocast_dtype(train_cfg: dict, device: str):
    """bf16 / fp16 autocast dtype, or None for fp32."""
    if device != "cuda":
        return None
    precision = str(train_cfg.get(
        "precision", "fp16" if train_cfg.get("mixed_precision", True) else "fp32"
    ))
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)


def train(config: dict, resume: bool = False):
    """Main Phase 2 JEPA training loop."""
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    mask_cfg = config["masking"]
    loss_cfg = config["loss"]
    regularizer = model_cfg.get("regularizer", "ema")

    seed = exp_cfg["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)

    ckpt_dir = Path(exp_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = ckpt_dir / "latest.pt", ckpt_dir / "best.pt"
    ckpt = None
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"--resume: no checkpoint at {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
    elif latest_path.is_file():
        raise FileExistsError(f"{ckpt_dir} holds an earlier run; pass --resume or another dir")

    run_name = exp_cfg.get("run_name") or make_run_name(
        phase=2, arm=f"jepa_{regularizer}", seed=seed
    )
    logger = ExperimentLogger(
        log_dir=exp_cfg["log_dir"],
        run_name=run_name,
        use_wandb=exp_cfg.get("use_wandb", False),
        config=config,
    )

    # Data
    print("Loading comma2k19 data...")
    train_loader, val_loader = create_comma2k19_dataloaders(config, seed=seed)
    print(f"  Train clips: {len(train_loader.dataset)}  Val clips: {len(val_loader.dataset)}")

    # Model
    context_encoder, predictor, target_encoder = build_models(config, device)
    geometry = patch_geometry(config)
    mask_generator = MaskGenerator(
        num_patches=geometry["total"],
        num_patches_spatial=geometry["spatial"],
        num_patches_temporal=geometry["temporal"],
        strategy=mask_cfg["strategy"],
        config=mask_cfg,
    )
    criterion = JEPALoss(
        loss_type=loss_cfg["type"],
        beta=loss_cfg.get("beta", 1.0),
        detach_targets=regularizer == "ema",
    )
    sigreg, sigreg_lambda = None, 0.0
    if regularizer == "sigreg":
        sigreg_cfg = model_cfg.get("sigreg", {})
        sigreg = SIGReg(
            num_slices=sigreg_cfg.get("num_slices", 256),
            max_samples=sigreg_cfg.get("max_samples", 4096),
        ).to(device)
        sigreg_lambda = float(sigreg_cfg.get("lambda", 0.05))

    collapse_cfg = train_cfg.get("collapse_monitoring", {})
    collapse_monitor = CollapseMonitor(
        variance_threshold=collapse_cfg.get("variance_threshold", 0.01),
        log_every=collapse_cfg.get("log_variance_every", 100),
    )

    params = list(context_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg.get("adam_betas", [0.9, 0.95])),
    )

    # Per-step cosine schedule with linear warmup. max_steps (optional) caps
    # the run so that differently regularised runs get the same budget.
    steps_per_epoch = len(train_loader)
    total_steps = train_cfg["total_epochs"] * steps_per_epoch
    if train_cfg.get("max_steps"):
        total_steps = min(total_steps, int(train_cfg["max_steps"]))
    warmup_steps = int(train_cfg.get("warmup_epochs", 1) * steps_per_epoch)
    if train_cfg.get("warmup_steps") is not None:
        warmup_steps = int(train_cfg["warmup_steps"])
    min_ratio = train_cfg.get("min_lr", 1e-6) / train_cfg["learning_rate"]

    def lr_schedule(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_ratio, 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    amp_dtype = _autocast_dtype(train_cfg, device)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    global_step, start_epoch, best_val = 0, 0, float("inf")
    if ckpt is not None:
        context_encoder.load_state_dict(ckpt["context_encoder_state_dict"])
        predictor.load_state_dict(ckpt["predictor_state_dict"])
        if target_encoder is not None:
            target_encoder.load_state_dict_with_metadata(ckpt["target_encoder"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step, start_epoch = ckpt["global_step"], ckpt["epoch"] + 1
        best_val = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {latest_path} at step {global_step} (epoch {start_epoch})")

    def checkpoint_state(epoch: int, val_loss: float) -> dict:
        state = {
            "context_encoder_state_dict": context_encoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "regularizer": regularizer,
            "config": config,
        }
        if target_encoder is not None:
            state["target_encoder"] = target_encoder.state_dict_with_metadata()
        return state

    print(f"\nStarting Phase 2 JEPA training: {run_name}")
    print(f"  Regularizer: {regularizer}" + (f" (lambda={sigreg_lambda})" if sigreg else ""))
    print(f"  Encoder: ViT ({sum(p.numel() for p in context_encoder.parameters())/1e6:.1f}M)"
          f", predictor ({sum(p.numel() for p in predictor.parameters())/1e6:.1f}M)")
    print(f"  Patches: {geometry['total']} ({geometry['spatial']} spatial x "
          f"{geometry['temporal']} temporal); masking: {mask_cfg['strategy']}")
    print(f"  Steps: {total_steps} ({steps_per_epoch}/epoch); device {device}, amp {amp_dtype}")

    epoch = start_epoch
    while global_step < total_steps:
        context_encoder.train()
        predictor.train()
        epoch_losses, epoch_start, seen = [], time.time(), 0

        # No per-step bar in log files (nohup / detached runs).
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=not sys.stderr.isatty()):
            if global_step >= total_steps:
                break
            video = batch["video"].to(device, non_blocking=True)  # (B, C, T, H, W)
            telemetry = batch["telemetry"].to(device, non_blocking=True)  # (B, T, A)
            masks = mask_generator(video.shape[0])
            context_indices = masks["context_indices"].to(device)
            mask_indices = masks["mask_indices"].to(device)
            if global_step % 1000 == 0:
                verify_no_leak(context_indices, mask_indices)

            with torch.autocast("cuda", dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                out = compute_losses(
                    context_encoder, predictor, target_encoder, criterion, sigreg,
                    sigreg_lambda, video, telemetry, context_indices, mask_indices, global_step,
                )
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(params, train_cfg.get("grad_clip", 1.0))
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            momentum = target_encoder.update(context_encoder) if target_encoder else None
            epoch_losses.append(out["pred_loss"].item())
            seen += video.shape[0]
            global_step += 1

            if global_step % train_cfg.get("log_every", 50) == 0:
                logger.log_scalar("train/loss", loss.item(), global_step)
                logger.log_scalar("train/pred_loss", out["pred_loss"].item(), global_step)
                if "sigreg" in out:
                    logger.log_scalar("train/sigreg", out["sigreg"].item(), global_step)
                if momentum is not None:
                    logger.log_scalar("train/ema_momentum", momentum, global_step)
                logger.log_scalar("train/grad_norm", float(grad_norm), global_step)
                logger.log_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                logger.log_scalar("train/mask_ratio", masks["mask_ratio"], global_step)
                logger.log_scalar(
                    "train/clips_per_sec", seen / max(time.time() - epoch_start, 1e-9), global_step
                )
                logger.log_vram(global_step)

            if collapse_monitor.should_check(global_step):
                info = collapse_monitor.check(out["context_output"].detach().float(), global_step)
                logger.log_scalar("collapse/variance", info["variance"], global_step)
                logger.log_scalar("collapse/std", info["std"], global_step)
                logger.log_scalar("collapse/mean_norm", info["mean_norm"], global_step)
                if target_encoder is not None:
                    target_encoder.verify_no_gradients()

        epoch_time = time.time() - epoch_start
        val_loss = validate(context_encoder, target_encoder, predictor, mask_generator,
                            criterion, val_loader, device, amp_dtype)
        logger.log_scalar("val/loss", val_loss, global_step)
        print(f"  Epoch {epoch + 1}: train pred loss {np.mean(epoch_losses):.4f}, "
              f"val loss {val_loss:.4f}, {seen / max(epoch_time, 1e-9):.1f} clips/s, "
              f"step {global_step}/{total_steps}")

        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        state = checkpoint_state(epoch, val_loss)
        torch.save(state, latest_path)
        if is_best:
            torch.save(state, best_path)
            print(f"  New best -> {best_path}")
        epoch += 1

    logger.close()
    print(f"\nPhase 2 training complete. Best val loss: {best_val:.4f} ({best_path})")


@torch.no_grad()
def validate(context_encoder, target_encoder, predictor, mask_generator,
             criterion, val_loader, device, amp_dtype=None) -> float:
    """Mean validation prediction loss.

    Masks are drawn from a fixed seed so that epochs are compared on the same
    masks. Targets come from the EMA teacher if present, else the online encoder.
    """
    context_encoder.eval()
    predictor.eval()
    losses = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for batch in val_loader:
            video = batch["video"].to(device)
            telemetry = batch["telemetry"].to(device)
            masks = mask_generator(video.shape[0])
            with torch.autocast("cuda", dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                out = compute_losses(
                    context_encoder, predictor, target_encoder, criterion, None, 0.0,
                    video, telemetry, masks["context_indices"].to(device),
                    masks["mask_indices"].to(device), 0,
                )
            losses.append(out["pred_loss"].item())
    context_encoder.train()
    predictor.train()
    return float(np.mean(losses)) if losses else float("inf")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: JEPA Pretraining on comma2k19")
    parser.add_argument("--config", default="configs/phase2_jepa_laptop.yaml")
    parser.add_argument("--regularizer", choices=REGULARIZERS, default=None,
                        help="Override model.regularizer")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override experiment.checkpoint_dir")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override training.max_steps")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <checkpoint-dir>/latest.pt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.regularizer:
        config["model"]["regularizer"] = args.regularizer
    if args.checkpoint_dir:
        config["experiment"]["checkpoint_dir"] = args.checkpoint_dir
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps

    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
