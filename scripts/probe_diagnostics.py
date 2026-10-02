#!/usr/bin/env python3
"""
Probe diagnostics for Phase 2 encoders: what information do they carry?

Each encoder (the given checkpoints plus a random-init control of the same
architecture) encodes the deterministic train/val clips once. Ridge probes
are then fitted for every combination of
  - pooling: global mean, 2x2 grid, 4x4 grid (spatial layout kept)
  - target:  clip-mean steering angle, clip-mean speed
and scored on held-out validation drives (R2; ridge strength chosen on
held-out training drives).

Usage:
    python scripts/probe_diagnostics.py outputs/checkpoints/phase2_ema/final.pt \\
        outputs/checkpoints/phase2_sigreg/final.pt
Writes outputs/probe_results/diagnostics.json and prints a table.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.comma2k19_dataset import create_comma2k19_dataloaders  # noqa: E402
from src.eval.linear_probe import extract_token_grid, fit_linear_probe, pool_grid  # noqa: E402
from src.jepa.encoder import ViTEncoder  # noqa: E402

TARGETS = {"steering": 0, "speed": 1}
GRIDS = (1, 2, 4)


def encoder_from_config(config: dict) -> ViTEncoder:
    ctx, tubelet = config["model"]["context_encoder"], config["data"]["tubelet"]
    return ViTEncoder(
        img_size=tubelet["spatial_size"], patch_size=ctx["patch_size"],
        tubelet_size=ctx["tubelet_size"], embed_dim=ctx["embed_dim"], depth=ctx["depth"],
        num_heads=ctx["num_heads"], mlp_ratio=ctx["mlp_ratio"], num_frames=tubelet["num_frames"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=0, help="Random-init control seed")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/probe_results/diagnostics.json"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = {p: torch.load(p, map_location="cpu", weights_only=False) for p in args.checkpoints}
    config = next(iter(ckpts.values()))["config"]
    train_loader, val_loader = create_comma2k19_dataloaders(config, deterministic=True)

    encoders = {}
    for path, ckpt in ckpts.items():
        encoder = encoder_from_config(ckpt["config"])
        encoder.load_state_dict(ckpt["context_encoder_state_dict"])
        encoders[f"{path.parent.name}/{path.stem}"] = encoder
    torch.manual_seed(args.seed)
    encoders["random_init"] = encoder_from_config(config)

    results = {}
    for name, encoder in encoders.items():
        started = time.time()
        encoder.to(device)
        train_f, train_y, train_g = extract_token_grid(encoder, train_loader, device, grid=4)
        val_f, val_y, _ = extract_token_grid(encoder, val_loader, device, grid=4)
        encoder.cpu()
        for target, channel in TARGETS.items():
            for grid in GRIDS:
                r = fit_linear_probe(pool_grid(train_f, grid), train_y[:, channel],
                                     pool_grid(val_f, grid), val_y[:, channel], train_g)
                results[f"{name}|{target}|{grid}x{grid}"] = r
        print(f"  {name}: done in {time.time() - started:.0f}s", flush=True)

    names = list(encoders)
    print(f"\nValidation R2 (higher is better; ~0 = no linear signal)\n{'probe':22s}"
          + "".join(f"{n.split('/')[0]:>18s}" for n in names))
    for target in TARGETS:
        for grid in GRIDS:
            row = f"{target + ' ' + str(grid) + 'x' + str(grid):22s}"
            row += "".join(f"{results[f'{n}|{target}|{grid}x{grid}']['r2']:18.3f}" for n in names)
            print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"checkpoints": [str(p) for p in args.checkpoints], "results": results}, indent=2))
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
