"""
Phase 2 Linear Probe Evaluation Script.

Evaluates JEPA encoder quality by training a linear probe to
predict steering angle, and comparing against a random-init control.

Usage:
    python scripts/probe_phase2.py --config configs/phase2_jepa_pretrain.yaml \
                                   --checkpoint outputs/checkpoints/phase2/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.comma2k19_dataset import create_comma2k19_dataloaders
from src.eval.linear_probe import run_probe_comparison
from src.jepa.encoder import ViTEncoder
from src.utils.logging_utils import ExperimentLogger, make_run_name
from src.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Linear Probe Evaluation")
    parser.add_argument("--config", default="configs/phase2_jepa_pretrain.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to Phase 2 encoder checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    # Load data
    print("Loading comma2k19 data for probing...")
    train_loader, val_loader = create_comma2k19_dataloaders(config, seed=args.seed)

    # Build encoder architecture
    ctx_cfg = config["model"]["context_encoder"]
    tubelet_cfg = config["data"]["tubelet"]

    encoder_kwargs = {
        "img_size": tubelet_cfg["spatial_size"],
        "patch_size": ctx_cfg["patch_size"],
        "tubelet_size": ctx_cfg["tubelet_size"],
        "embed_dim": ctx_cfg["embed_dim"],
        "depth": ctx_cfg["depth"],
        "num_heads": ctx_cfg["num_heads"],
        "mlp_ratio": ctx_cfg["mlp_ratio"],
        "num_frames": tubelet_cfg["num_frames"],
    }

    # Load trained encoder
    print(f"Loading checkpoint: {args.checkpoint}")
    trained_encoder = ViTEncoder(**encoder_kwargs).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "context_encoder_state_dict" in ckpt:
        trained_encoder.load_state_dict(ckpt["context_encoder_state_dict"])
    elif "model_state_dict" in ckpt:
        trained_encoder.load_state_dict(ckpt["model_state_dict"])

    # Logger
    run_name = make_run_name(phase=2, arm="probe", seed=args.seed)
    logger = ExperimentLogger(
        log_dir=config["experiment"]["log_dir"],
        run_name=run_name,
        config=config,
    )

    # Run comparison
    probe_cfg = config.get("linear_probe", {})
    results = run_probe_comparison(
        trained_encoder=trained_encoder,
        encoder_class=ViTEncoder,
        encoder_kwargs=encoder_kwargs,
        train_loader=train_loader,
        val_loader=val_loader,
        config=probe_cfg,
        device=device,
        logger=logger,
    )

    # Save results
    output_dir = Path("outputs/probe_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"probe_results_seed{args.seed}.txt"
    with open(results_path, "w") as f:
        f.write("Phase 2 Linear Probe Results\n")
        f.write("=" * 40 + "\n")
        for key, metrics in results.items():
            f.write(f"\n{key.upper()} Encoder:\n")
            for m, v in metrics.items():
                f.write(f"  {m}: {v:.4f}\n")

    print(f"\nResults saved to {results_path}")
    logger.close()


if __name__ == "__main__":
    main()
