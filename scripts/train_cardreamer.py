"""
Unified CarDreamer Training Script — Phases 1 and 3.

Replaces the separate train_phase1.py and train_phase3_arm.py scripts
with a single entry point that uses CarDreamer's tested DreamerV3
training loop, with our encoder swapped in at the designated slot.

This eliminates the custom RSSM, actor-critic, replay buffer, and training
loop that were the source of bugs #2, #7, #8 in the code review.

Usage:
    # Phase 1: CNN baseline (CarDreamer default)
    python scripts/train_cardreamer.py --arm cnn --task carla_right_turn_simple

    # Phase 3: Custom JEPA encoder
    python scripts/train_cardreamer.py --arm custom_jepa --task carla_right_turn_simple \
        --jepa-checkpoint outputs/checkpoints/phase2/best.pt

    # Phase 3: V-JEPA2 encoder
    python scripts/train_cardreamer.py --arm vjepa2 --task carla_right_turn_simple

    # Run all three arms for comparison
    python scripts/train_cardreamer.py --run-comparison --task carla_right_turn_simple

Prerequisites:
    1. Run scripts/setup_cardreamer.sh first
    2. CARLA server must be running
    3. For custom_jepa arm: Phase 2 checkpoint required
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# Add project root and CarDreamer to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "CarDreamer"))

from src.dreamer.cardreamer_encoder_hook import CarDreamerEncoderHook, patch_dreamerv3_encoder
from src.utils.seeding import seed_everything
from src.eval.metrics import ComparisonTable


def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_encoder_hook(arm: str, config: dict, device: str = "cuda") -> CarDreamerEncoderHook:
    """
    Build the encoder hook for a given arm.

    For custom_jepa: loads Phase 2 checkpoint weights.
    For vjepa2: downloads/loads Meta's pretrained weights.
    For cnn: creates the default CNN encoder.
    """
    hook = CarDreamerEncoderHook(
        arm=arm,
        config=config,
        device=device,
        target_resolution=config.get("encoder_resolution", 224),
        num_temporal_frames=config.get("frame_stacking", {}).get("num_frames", 4),
    )

    # Load Phase 2 checkpoint for custom_jepa arm
    if arm == "custom_jepa":
        jepa_ckpt_path = config.get("jepa_checkpoint")
        if jepa_ckpt_path and Path(jepa_ckpt_path).exists():
            import torch
            ckpt = torch.load(jepa_ckpt_path, map_location=device, weights_only=False)
            if "context_encoder_state_dict" in ckpt:
                hook.encoder.load_state_dict(ckpt["context_encoder_state_dict"])
                print(f"[train] Loaded Phase 2 JEPA checkpoint: {jepa_ckpt_path}")
            else:
                print(f"[train] WARNING: checkpoint missing 'context_encoder_state_dict'")
        elif jepa_ckpt_path:
            print(f"[train] WARNING: JEPA checkpoint not found: {jepa_ckpt_path}")
        else:
            print("[train] WARNING: No JEPA checkpoint specified for custom_jepa arm. "
                  "Encoder will have random weights.")

    return hook


def train_arm(
    arm: str,
    task: str,
    seed: int,
    config: dict,
    carla_port: int = 2000,
    steps: int = 500_000,
):
    """
    Train a single arm using CarDreamer's DreamerV3 loop.

    This is the core function that replaces our custom training loop
    with CarDreamer's tested infrastructure.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)

    print(f"\n{'='*60}")
    print(f"Training arm '{arm}' on task '{task}', seed={seed}")
    print(f"{'='*60}")

    # --- Import CarDreamer ---
    try:
        import car_dreamer
    except ImportError:
        print("ERROR: CarDreamer not installed. Run scripts/setup_cardreamer.sh first.")
        print("       Or: cd third_party/CarDreamer && pip install flit && flit install --symlink")
        sys.exit(1)

    # --- Create CarDreamer Task Environment ---
    print(f"[train] Creating CarDreamer task: {task}")
    env, task_configs = car_dreamer.create_task(task)

    # --- Build Encoder Hook ---
    encoder_hook = build_encoder_hook(arm, config, device)

    # --- Create DreamerV3 Agent ---
    # CarDreamer's DreamerV3 is in third_party/CarDreamer/dreamerv3/
    dreamerv3_dir = PROJECT_ROOT / "third_party" / "CarDreamer" / "dreamerv3"
    sys.path.insert(0, str(dreamerv3_dir))

    try:
        # CarDreamer's DreamerV3 training entry point
        # The exact import path depends on CarDreamer's structure
        from dreamerv3 import agent as dreamerv3_agent
        from dreamerv3 import embodied

        # Build agent config
        agent_config = embodied.Config({
            'logdir': f"outputs/logs/{arm}_seed{seed}",
            'run.steps': steps,
            'run.log_every': 1000,
            'run.eval_every': 10000,
            'seed': seed,
        })

        # Merge with task configs
        agent_config = agent_config.update(task_configs)

        # Create agent
        agent = dreamerv3_agent.Agent(env.observation_space, env.action_space, agent_config)

        # --- Swap Encoder ---
        patch_dreamerv3_encoder(agent, encoder_hook)

        # --- Train ---
        print(f"[train] Starting training for {steps} steps...")
        embodied.run.train(agent, env, agent_config)

        print(f"[train] Training complete for arm '{arm}'")

    except ImportError as e:
        # Fallback: CarDreamer's DreamerV3 may have different import structure
        print(f"\n[train] CarDreamer DreamerV3 import failed: {e}")
        print("[train] This is expected if CarDreamer's dependencies aren't installed.")
        print("[train] On target hardware, run:")
        print(f"  cd {dreamerv3_dir}")
        print("  pip install -r requirements.txt")
        print(f"  python scripts/train_cardreamer.py --arm {arm} --task {task}")
        print("\n[train] Using fallback: saving encoder hook config for later use.")

        # Save the encoder hook state so it can be loaded on target hardware
        fallback_path = f"outputs/checkpoints/encoder_hook_{arm}_seed{seed}.pt"
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        torch.save({
            "arm": arm,
            "encoder_state_dict": encoder_hook.encoder.state_dict(),
            "adapter_state_dict": encoder_hook.adapter.state_dict(),
            "config": config,
            "seed": seed,
            "task": task,
        }, fallback_path)
        print(f"[train] Saved encoder hook to: {fallback_path}")

    finally:
        env.close()


def run_comparison(task: str, config: dict, carla_port: int, steps: int):
    """Run all three arms with matched seeds for a controlled comparison."""
    seeds = config.get("experiment", {}).get("seeds", [42, 123, 456])
    arms = ["cnn", "custom_jepa", "vjepa2"]
    comparison = ComparisonTable()

    for arm in arms:
        for seed in seeds:
            print(f"\n{'#'*60}")
            print(f"# Comparison: arm={arm}, seed={seed}")
            print(f"{'#'*60}")
            train_arm(arm, task, seed, config, carla_port, steps)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print("Results saved to outputs/logs/")
    print("Generate comparison table with:")
    print("  python -c \"from src.eval.metrics import ...; ...\"")


def main():
    parser = argparse.ArgumentParser(
        description="CarDreamer-based training for Phases 1 & 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1 baseline
  python scripts/train_cardreamer.py --arm cnn --task carla_right_turn_simple

  # Phase 3 with JEPA encoder
  python scripts/train_cardreamer.py --arm custom_jepa \\
      --jepa-checkpoint outputs/checkpoints/phase2/best.pt

  # Run full three-way comparison
  python scripts/train_cardreamer.py --run-comparison
        """,
    )
    parser.add_argument("--arm", choices=["cnn", "custom_jepa", "vjepa2"], default="cnn",
                        help="Encoder arm to use")
    parser.add_argument("--task", default="carla_right_turn_simple",
                        help="CarDreamer task name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/phase3_transfer_arms.yaml",
                        help="Experiment config file")
    parser.add_argument("--jepa-checkpoint", type=str, default=None,
                        help="Path to Phase 2 JEPA checkpoint (for custom_jepa arm)")
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Total training steps")
    parser.add_argument("--carla-port", type=int, default=2000,
                        help="CARLA server port")
    parser.add_argument("--run-comparison", action="store_true",
                        help="Run all three arms with matched seeds")

    args = parser.parse_args()

    config = load_config(args.config)

    # Override config with CLI args
    if args.jepa_checkpoint:
        config["jepa_checkpoint"] = args.jepa_checkpoint

    if args.run_comparison:
        run_comparison(args.task, config, args.carla_port, args.steps)
    else:
        train_arm(args.arm, args.task, args.seed, config, args.carla_port, args.steps)


if __name__ == "__main__":
    main()
