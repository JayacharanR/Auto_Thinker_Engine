#!/usr/bin/env python3
"""
Unified training script for Phase 1 (baseline CNN) and Phase 3 (encoder comparison).

Uses dreamerv3-torch (PyTorch) as the DreamerV3 backbone, with CarDreamer's
CARLA task definitions for the environment.

This replaces the previous train_cardreamer.py that tried to use CarDreamer's
JAX-based DreamerV3 (which was a framework mismatch with our PyTorch encoders).

Usage:
    # Single arm
    python scripts/train_cardreamer.py --arm cnn --task carla_right_turn_simple

    # Full 3-arm comparison (Phase 3)
    python scripts/train_cardreamer.py --comparison --task carla_right_turn_simple

    # Phase 1 baseline
    python scripts/train_cardreamer.py --arm cnn --task carla_right_turn_simple --steps 500000
"""

import argparse
import functools
import os
import pathlib
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# dreamerv3-torch
DREAMER_DIR = PROJECT_ROOT / "third_party" / "dreamerv3_torch"
sys.path.insert(0, str(DREAMER_DIR))

# CarDreamer (for CARLA task definitions only — NOT for DreamerV3)
CARDREAMER_DIR = PROJECT_ROOT / "third_party" / "CarDreamer"
sys.path.insert(0, str(CARDREAMER_DIR))

from src.dreamer.cardreamer_encoder_hook import (
    DreamerV3EncoderHook,
    patch_dreamerv3_encoder,
)


def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_phase3_config(config_path: Optional[str] = None) -> dict:
    """Load Phase 3 encoder comparison config."""
    from ruamel.yaml import YAML

    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "phase3_transfer_arms.yaml")
    ryaml = YAML(typ="safe", pure=True)
    with open(config_path) as f:
        return ryaml.load(f)


def build_encoder_hook(
    arm: str,
    phase3_config: dict,
    device: str,
) -> DreamerV3EncoderHook:
    """
    Build encoder hook for a specific arm.

    For custom_jepa, loads Phase 2 checkpoint if available.
    """
    hook = DreamerV3EncoderHook(
        arm=arm,
        config=phase3_config,
        device=device,
        obs_key="image",
        num_temporal_frames=phase3_config.get("frame_stacking", {}).get("num_frames", 4),
    )

    # Load Phase 2 JEPA checkpoint for custom_jepa arm
    if arm == "custom_jepa":
        jepa_ckpt_path = phase3_config.get("encoders", {}).get("custom_jepa", {}).get("checkpoint")
        if jepa_ckpt_path and os.path.exists(jepa_ckpt_path):
            ckpt = torch.load(jepa_ckpt_path, map_location=device, weights_only=False)
            if "context_encoder_state_dict" in ckpt:
                hook.encoder.load_state_dict(ckpt["context_encoder_state_dict"], strict=False)
                print(f"[train] Loaded Phase 2 JEPA checkpoint: {jepa_ckpt_path}")
            elif "model_state_dict" in ckpt:
                hook.encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
                print(f"[train] Loaded Phase 2 checkpoint: {jepa_ckpt_path}")
        elif jepa_ckpt_path:
            print(f"[train] WARNING: JEPA checkpoint not found: {jepa_ckpt_path}")
        else:
            print("[train] WARNING: No JEPA checkpoint specified for custom_jepa arm.")

    return hook


def make_carla_env(task_name: str, seed: int = 0, image_size: tuple = (64, 64)):
    """
    Create a CARLA environment using CarDreamer's task definitions.

    CarDreamer's task creation is framework-agnostic (plain Python + CARLA client).
    We wrap the resulting Gym env to match dreamerv3-torch's expected interface.

    Returns a gym-compatible environment.
    """
    import envs.wrappers as wrappers

    try:
        import car_dreamer
        env, task_config = car_dreamer.create_task(task_name)
    except ImportError:
        print("[train] CarDreamer not installed. Creating a placeholder env.")
        print("[train] On target hardware: bash scripts/setup_cardreamer.sh /path/to/carla")
        raise

    # Wrap for dreamerv3-torch compatibility
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, 1000)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)

    return env


def load_dreamer_config(
    arm: str,
    task: str,
    seed: int,
    steps: int,
    image_size: tuple,
    device: str,
) -> argparse.Namespace:
    """
    Load dreamerv3-torch's config with our overrides.

    Loads defaults from dreamerv3-torch's configs.yaml, then applies
    our Phase 3 settings (resolution, batch size, etc.).
    """
    from ruamel.yaml import YAML

    # Load dreamerv3-torch defaults
    configs_path = DREAMER_DIR / "configs.yaml"
    ryaml = YAML(typ="safe", pure=True)
    all_configs = ryaml.load(configs_path.read_text())

    # Start with defaults
    config = all_configs["defaults"].copy()

    # Our overrides
    logdir = str(PROJECT_ROOT / "outputs" / "logs" / f"{arm}_seed{seed}")
    config.update({
        "logdir": logdir,
        "seed": seed,
        "steps": steps,
        "task": f"carla_{task}" if not task.startswith("carla_") else task,
        "device": device,
        "size": list(image_size),
        "action_repeat": 1,  # CARLA already runs at real time
        "time_limit": 1000,
        "prefill": 5000,
        "eval_every": 10000,
        "log_every": 1000,
        "eval_episode_num": 5,
        "compile": False,  # Disable torch.compile for encoder swap compatibility
        "precision": 32,
        "video_pred_log": True,
    })

    # Convert nested dicts to proper format
    for key in ["encoder", "decoder", "actor", "critic", "reward_head", "cont_head"]:
        if isinstance(config.get(key), dict):
            pass  # Already a dict, fine

    # Convert to namespace (dreamerv3-torch uses argparse.Namespace)
    return argparse.Namespace(**config)


def train_arm(
    arm: str,
    task: str,
    seed: int,
    phase3_config: dict,
    steps: int = 500_000,
    image_size: tuple = (64, 64),
) -> Dict[str, float]:
    """
    Train a single arm using dreamerv3-torch's training loop.

    Returns dict of final metrics for comparison table.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)

    print(f"\n{'='*60}")
    print(f"Training arm '{arm}' on task '{task}', seed={seed}")
    print(f"Device: {device}, Steps: {steps}")
    print(f"{'='*60}")

    # --- Load dreamerv3-torch config ---
    config = load_dreamer_config(arm, task, seed, steps, image_size, device)
    logdir = pathlib.Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # --- Import dreamerv3-torch modules ---
    try:
        import models
        import tools
        import envs.wrappers as wrappers
        from parallel import Damy
    except ImportError as e:
        print(f"[train] dreamerv3-torch import failed: {e}")
        print("[train] Ensure third_party/dreamerv3_torch is set up.")
        return {}

    # --- Create environment ---
    try:
        env = make_carla_env(task, seed, image_size)
        train_envs = [Damy(env)]
        eval_envs = [Damy(env)]
    except ImportError:
        # Fallback: save encoder hook for target hardware
        hook = build_encoder_hook(arm, phase3_config, device)
        fallback_path = f"outputs/checkpoints/encoder_hook_{arm}_seed{seed}.pt"
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        torch.save({
            "arm": arm,
            "encoder_state_dict": hook.encoder.state_dict(),
            "adapter_state_dict": hook.adapter.state_dict(),
            "config": phase3_config,
            "seed": seed,
            "task": task,
        }, fallback_path)
        print(f"[train] Saved encoder hook to: {fallback_path}")
        return {}

    # --- Set up training ---
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    step = 0
    logger = tools.Logger(logdir, 0)
    train_eps = tools.load_episodes(logdir / "train_eps", limit=config.dataset_size)
    eval_eps = tools.load_episodes(logdir / "eval_eps", limit=1)

    # --- Prefill with random actions ---
    from dreamer import make_dataset
    from torch import distributions as torchd

    if hasattr(acts, "discrete"):
        random_actor = tools.OneHotDist(
            torch.zeros(config.num_actions).repeat(config.envs, 1)
        )
    else:
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(
                torch.tensor(acts.low).repeat(config.envs, 1),
                torch.tensor(acts.high).repeat(config.envs, 1),
            ),
            1,
        )

    def random_agent(o, d, s):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {"action": action, "logprob": logprob}, None

    print(f"[train] Prefilling dataset ({config.prefill} steps)...")
    state = tools.simulate(
        random_agent, train_envs, train_eps,
        logdir / "train_eps", logger,
        limit=config.dataset_size, steps=config.prefill,
    )
    logger.step += config.prefill * config.action_repeat

    # --- Create agent ---
    from dreamer import Dreamer

    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)

    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(device)
    agent.requires_grad_(requires_grad=False)

    # --- Swap encoder (this is the key operation) ---
    hook = build_encoder_hook(arm, phase3_config, device)
    patch_dreamerv3_encoder(agent, hook)

    # Re-enable gradients for the hook
    for param in hook.parameters():
        param.requires_grad = True

    # --- Training loop ---
    print(f"[train] Starting training for {steps} steps...")
    metrics_history = []

    while agent._step < config.steps + config.eval_every:
        logger.write()

        if config.eval_episode_num > 0:
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy, eval_envs, eval_eps,
                logdir / "eval_eps", logger,
                is_eval=True, episodes=config.eval_episode_num,
            )

        state = tools.simulate(
            agent, train_envs, train_eps,
            logdir / "train_eps", logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )

        # Save checkpoint
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "arm": arm,
            "seed": seed,
            "step": agent._step,
        }
        torch.save(items_to_save, logdir / "latest.pt")

    # --- Collect final metrics ---
    final_metrics = {
        "arm": arm,
        "seed": seed,
        "total_steps": int(agent._step),
    }

    # Extract from logger
    for key in ["eval_reward", "eval_length", "eval_success"]:
        if key in agent._metrics and agent._metrics[key]:
            final_metrics[key] = float(np.mean(agent._metrics[key]))

    print(f"[train] Training complete for arm '{arm}', seed={seed}")
    print(f"[train] Final metrics: {final_metrics}")

    # Clean up
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

    return final_metrics


def run_comparison(
    task: str,
    phase3_config: dict,
    steps: int = 500_000,
    image_size: tuple = (64, 64),
):
    """
    Run all three arms with matched seeds for a controlled comparison.

    Collects metrics from each arm/seed run and outputs a comparison table.
    """
    seeds = phase3_config.get("experiment", {}).get("seeds", [42, 123, 456])
    arms = ["cnn", "custom_jepa", "vjepa2"]

    all_results = []

    for arm in arms:
        for seed in seeds:
            print(f"\n{'#'*60}")
            print(f"# Comparison: {arm} / seed {seed}")
            print(f"{'#'*60}")

            metrics = train_arm(
                arm=arm,
                task=task,
                seed=seed,
                phase3_config=phase3_config,
                steps=steps,
                image_size=image_size,
            )
            all_results.append(metrics)

    # --- Build and print comparison table ---
    print(f"\n{'='*80}")
    print("PHASE 3 COMPARISON RESULTS")
    print(f"{'='*80}")

    header = f"{'Arm':<15} {'Seed':<8} {'Steps':<10} {'Eval Reward':<15} {'Eval Length':<15}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        arm_name = r.get("arm", "?")
        seed = r.get("seed", "?")
        total_steps = r.get("total_steps", 0)
        eval_reward = r.get("eval_reward", float("nan"))
        eval_length = r.get("eval_length", float("nan"))
        print(f"{arm_name:<15} {seed:<8} {total_steps:<10} {eval_reward:<15.2f} {eval_length:<15.1f}")

    # Aggregate by arm
    print(f"\n{'='*80}")
    print("AGGREGATE (mean ± std across seeds)")
    print(f"{'='*80}")

    for arm in arms:
        arm_results = [r for r in all_results if r.get("arm") == arm]
        rewards = [r.get("eval_reward", float("nan")) for r in arm_results]
        rewards = [r for r in rewards if not np.isnan(r)]
        if rewards:
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            print(f"{arm:<15} reward: {mean_r:.2f} ± {std_r:.2f} (n={len(rewards)})")
        else:
            print(f"{arm:<15} reward: no data")

    # Save results
    import json
    results_path = PROJECT_ROOT / "outputs" / "phase3_comparison.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[comparison] Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="DreamerV3 training with encoder comparison")
    parser.add_argument("--arm", type=str, default="cnn", choices=["cnn", "custom_jepa", "vjepa2"])
    parser.add_argument("--task", type=str, default="carla_right_turn_simple")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--config", type=str, default=None, help="Phase 3 config path")
    parser.add_argument("--comparison", action="store_true", help="Run 3-arm comparison")
    parser.add_argument("--image-size", type=int, nargs=2, default=[64, 64])
    args = parser.parse_args()

    phase3_config = load_phase3_config(args.config)

    if args.comparison:
        run_comparison(
            task=args.task,
            phase3_config=phase3_config,
            steps=args.steps,
            image_size=tuple(args.image_size),
        )
    else:
        train_arm(
            arm=args.arm,
            task=args.task,
            seed=args.seed,
            phase3_config=phase3_config,
            steps=args.steps,
            image_size=tuple(args.image_size),
        )


if __name__ == "__main__":
    main()
