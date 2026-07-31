"""
Phase 3 Training Script: Three-way Encoder Comparison.

The project's centerpiece: a CONTROLLED experiment comparing three
encoder arms under identical conditions.

  --arm cnn          : CarDreamer's default CNN (baseline, Phase 1)
  --arm custom_jepa  : Phase 2 JEPA encoder (frozen)
  --arm vjepa2       : Meta's V-JEPA2 (frozen transfer learning)

All three feed into the SAME RSSM and actor-critic. The encoder swap
is the ONLY variable. Same reward function, same task, same seeds,
same training budget.

Usage:
    python scripts/train_phase3_arm.py --arm cnn --seed 42
    python scripts/train_phase3_arm.py --arm custom_jepa --seed 42
    python scripts/train_phase3_arm.py --arm vjepa2 --seed 42
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dreamer.encoder_adapter import EncoderAdapter, create_encoder
from src.dreamer.rssm_wrapper import RSSM, RSSMState
from src.envs.carla_wrapper import CarlaEnvWrapper
from src.envs.reward import RewardFunction, create_reward_function
from src.eval.metrics import ComparisonTable, MetricsTracker
from src.utils.checkpoint import CheckpointManager
from src.utils.logging_utils import ExperimentLogger, make_run_name
from src.utils.seeding import seed_everything

# Import actor/critic from Phase 1 (identical architecture)
from scripts.train_phase1 import (
    Actor,
    Critic,
    ContinuePredictor,
    ImageDecoder,
    RewardPredictor,
)


class Phase3Agent:
    """
    DreamerV3 agent with swappable encoder for Phase 3 comparison.

    The ONLY difference between arms is the encoder + adapter.
    Everything else (RSSM, actor, critic, decoder, reward predictor,
    continue predictor) is identical.
    """

    def __init__(
        self,
        arm: str,
        config: dict,
        device: str = "cuda",
    ):
        self.arm = arm
        self.config = config
        self.device = device

        dreamer_cfg = config["dreamer"]
        rssm_cfg = dreamer_cfg["rssm"]
        train_cfg = config["training"]
        env_cfg = config["environment"]

        img_size = env_cfg["observation"]["camera"]["width"]
        action_dim = 2
        adapter_dim = config["adapter"]["target_dim"]

        # Encoder + adapter (THIS is what changes per arm)
        self.encoder, self.adapter = create_encoder(arm, config, device)

        # Count trainable params for this arm
        enc_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        adapt_params = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        print(f"  Arm '{arm}': encoder trainable params = {enc_params/1e6:.2f}M, "
              f"adapter params = {adapt_params/1e3:.1f}K")

        # RSSM (identical across arms)
        self.rssm = RSSM(
            stochastic_size=rssm_cfg["stochastic_size"],
            deterministic_size=rssm_cfg["deterministic_size"],
            hidden_size=rssm_cfg["hidden_size"],
            num_categories=rssm_cfg["num_categories"],
            num_classes=rssm_cfg["num_classes"],
            action_dim=action_dim,
            embed_dim=adapter_dim,
        ).to(device)

        state_dim = rssm_cfg["num_categories"] * rssm_cfg["num_classes"] + rssm_cfg["deterministic_size"]

        # Decoder (identical)
        self.decoder = ImageDecoder(
            state_dim=state_dim,
            channels=3,
            depth=dreamer_cfg["decoder"]["depth"],
            output_size=img_size,
        ).to(device)

        # Reward & continue predictors (identical)
        self.reward_pred = RewardPredictor(state_dim).to(device)
        self.continue_pred = ContinuePredictor(state_dim).to(device)

        # Actor-critic (identical)
        ac_cfg = dreamer_cfg["actor_critic"]
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=ac_cfg["actor_units"],
            num_layers=ac_cfg["actor_layers"],
        ).to(device)
        self.critic = Critic(
            state_dim=state_dim,
            hidden_dim=ac_cfg["critic_units"],
            num_layers=ac_cfg["critic_layers"],
        ).to(device)

        # Optimizers — only optimize trainable params of encoder
        lr = train_cfg["learning_rate"]
        wm_params = (
            [p for p in self.encoder.parameters() if p.requires_grad]
            + list(self.adapter.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_pred.parameters())
            + list(self.continue_pred.parameters())
        )
        self.world_model_opt = torch.optim.Adam(wm_params, lr=lr, eps=train_cfg["adam_eps"])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=train_cfg["adam_eps"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=train_cfg["adam_eps"])

        self.grad_clip = train_cfg["grad_clip"]
        self.imagination_horizon = train_cfg["imagination_horizon"]
        self.discount = ac_cfg["discount"]
        self.lambda_gae = ac_cfg["lambda_gae"]
        self._current_state = None

    def encode_observation(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image through arm-specific encoder + adapter."""
        features = self.encoder(image)
        return self.adapter(features)

    def act(self, obs: dict, explore: bool = True) -> np.ndarray:
        """Select action from observation."""
        with torch.no_grad():
            image = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(self.device) / 255.0
            embed = self.encode_observation(image)

            if self._current_state is None:
                self._current_state = self.rssm.initial_state(1, self.device)

            action = torch.zeros(1, 2, device=self.device)
            self._current_state, _ = self.rssm.observe_step(
                self._current_state, action, embed
            )
            action = self.actor(self._current_state.combined)

            if explore:
                action = action + torch.randn_like(action) * 0.3
                action = torch.clamp(action, -1.0, 1.0)

        return action.cpu().numpy().squeeze()

    def reset(self):
        self._current_state = None


def train_arm(arm: str, seed: int, config: dict):
    """Train a single Phase 3 arm."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = config["training"]

    seed_everything(seed)

    run_name = make_run_name(phase=3, arm=arm, seed=seed)
    print(f"\n{'='*60}")
    print(f"Phase 3: Training arm '{arm}', seed={seed}")
    print(f"Run name: {run_name}")
    print(f"{'='*60}")

    logger = ExperimentLogger(
        log_dir=config["experiment"]["log_dir"],
        run_name=run_name,
        use_wandb=config["experiment"].get("use_wandb", False),
        config=config,
    )

    ckpt_dir = f"{config['experiment']['checkpoint_dir']}/{arm}_seed{seed}"
    ckpt_manager = CheckpointManager(
        checkpoint_dir=ckpt_dir,
        max_keep=train_cfg["max_keep_checkpoints"],
        metric_name="eval/success_rate",
        metric_mode="max",
    )

    # Load reward config from Phase 1 (MUST NOT be modified)
    reward_config_path = config.get("reward_config", "configs/phase1_dreamer_baseline.yaml")
    with open(reward_config_path, "r") as f:
        reward_config = yaml.safe_load(f)
    reward_fn = RewardFunction(reward_config["reward"])

    # Create environment
    env = CarlaEnvWrapper(config=config["environment"], reward_fn=reward_fn)
    env.connect()

    # Create agent
    agent = Phase3Agent(arm=arm, config=config, device=device)

    # Log VRAM baseline
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Training loop (same structure as Phase 1)
    replay_buffer = []
    metrics_tracker = MetricsTracker(name=f"phase3_{arm}")
    total_steps = 0
    episode = 0
    start_time = time.time()

    while total_steps < train_cfg["total_steps"]:
        obs, info = env.reset()
        agent.reset()
        metrics_tracker.start_episode()
        done = False

        while not done and total_steps < train_cfg["total_steps"]:
            action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            replay_buffer.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "done": done,
            })

            if len(replay_buffer) > 100_000:
                replay_buffer = replay_buffer[-100_000:]

            obs = next_obs
            total_steps += 1
            metrics_tracker.step(step_info)

            if total_steps % train_cfg["log_every"] == 0:
                logger.log_vram(total_steps)

        episode_metrics = metrics_tracker.end_episode(
            success=not terminated and truncated,
        )
        episode += 1

        # Evaluation
        if total_steps % train_cfg["eval_every"] == 0:
            eval_tracker = MetricsTracker(name=f"eval_{arm}")
            for _ in range(config["environment"]["num_eval_episodes"]):
                obs, _ = env.reset()
                agent.reset()
                eval_tracker.start_episode()
                d = False
                while not d:
                    action = agent.act(obs, explore=False)
                    obs, _, terminated, truncated, step_info = env.step(action)
                    d = terminated or truncated
                    eval_tracker.step(step_info)
                eval_tracker.end_episode(success=not terminated and truncated)

            eval_summary = eval_tracker.summary()
            for k, v in eval_summary.items():
                logger.log_scalar(f"eval/{k}", v, total_steps)

    # Final metrics
    wall_clock = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    final_metrics = metrics_tracker.summary()
    final_metrics["wall_clock_seconds"] = wall_clock
    final_metrics["peak_vram_mb"] = peak_vram

    env.close()
    logger.close()

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Three-way Encoder Comparison")
    parser.add_argument("--arm", required=True, choices=["cnn", "custom_jepa", "vjepa2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/phase3_transfer_arms.yaml")
    parser.add_argument("--run-all-seeds", action="store_true",
                        help="Run with all seeds from config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.run_all_seeds:
        seeds = config["experiment"]["seeds"]
        comparison = ComparisonTable()

        for seed in seeds:
            metrics = train_arm(args.arm, seed, config)
            comparison.add_result(args.arm, seed, metrics)

        # Print comparison
        print("\n" + comparison.generate_table())
        comparison.save_table(f"outputs/comparison_{args.arm}.md")
    else:
        train_arm(args.arm, args.seed, config)


if __name__ == "__main__":
    main()
