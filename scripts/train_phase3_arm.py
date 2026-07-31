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
import collections
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dreamer.encoder_adapter import EncoderAdapter, create_encoder
from src._deprecated.rssm_wrapper import RSSM, RSSMState
from src._deprecated.carla_wrapper import CarlaEnvWrapper
from src._deprecated.reward import RewardFunction
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


class FrameStacker:
    """
    Accumulates consecutive frames into a temporal clip for video encoders.

    The CNN arm uses a single frame, but custom_jepa and vjepa2 expect
    video input (B, C, T, H, W). This stacker buffers the last N frames
    and returns them as a temporal clip.

    Args:
        num_frames: Number of frames to stack.
        frame_shape: (C, H, W) shape of each frame.
    """

    def __init__(self, num_frames: int = 4, frame_shape: tuple = (3, 128, 128)):
        self.num_frames = num_frames
        self.frame_shape = frame_shape
        self.buffer: collections.deque = collections.deque(maxlen=num_frames)

    def reset(self):
        """Clear the buffer."""
        self.buffer.clear()

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Add a frame and return the stacked clip.

        If fewer than num_frames are available, repeats the first frame
        to fill the buffer (avoids zeros which would confuse the encoder).

        Args:
            frame: (C, H, W) single frame tensor.

        Returns:
            (C, T, H, W) temporal clip tensor.
        """
        self.buffer.append(frame)

        # Pad with first frame if buffer isn't full yet
        while len(self.buffer) < self.num_frames:
            self.buffer.appendleft(self.buffer[0].clone())

        # Stack: list of (C, H, W) → (T, C, H, W) → (C, T, H, W)
        stacked = torch.stack(list(self.buffer), dim=0)  # (T, C, H, W)
        return stacked.permute(1, 0, 2, 3)  # (C, T, H, W)


class Phase3Agent:
    """
    DreamerV3 agent with swappable encoder for Phase 3 comparison.

    The ONLY difference between arms is the encoder + adapter.
    Everything else (RSSM, actor, critic, decoder, reward predictor,
    continue predictor) is identical.

    Fixes applied from code review:
    - train_step() is now implemented (Blocker 1)
    - Previous action is tracked for RSSM conditioning (Blocker 4)
    - Frame stacking for temporal encoders (Blocker 2)
    - Checkpointing in training loop (Blocker 1)
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

        # Frame stacking for temporal encoders (custom_jepa, vjepa2)
        self._needs_temporal = arm in ("custom_jepa", "vjepa2")
        if self._needs_temporal:
            temporal_frames = config.get("frame_stacking", {}).get("num_frames", 4)
            self.frame_stacker = FrameStacker(
                num_frames=temporal_frames,
                frame_shape=(3, img_size, img_size),
            )
        else:
            self.frame_stacker = None

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

        # State tracking — includes previous action (Blocker 4 fix)
        self._current_state: Optional[RSSMState] = None
        self._prev_action: Optional[torch.Tensor] = None

    def encode_observation(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode observation through arm-specific encoder + adapter.

        For temporal encoders (custom_jepa, vjepa2), the frame stacker
        accumulates frames into a temporal clip before encoding.
        """
        if self._needs_temporal and self.frame_stacker is not None:
            # image: (B, C, H, W) single frame → stacker returns (C, T, H, W)
            frame = image.squeeze(0)  # (C, H, W)
            clip = self.frame_stacker.push(frame)  # (C, T, H, W)
            clip = clip.unsqueeze(0).to(self.device)  # (1, C, T, H, W)
            features = self.encoder(clip)
        else:
            features = self.encoder(image)
        return self.adapter(features)

    def act(self, obs: dict, explore: bool = True) -> np.ndarray:
        """Select action from observation, conditioning RSSM on previous action."""
        with torch.no_grad():
            image = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(self.device) / 255.0
            embed = self.encode_observation(image)

            if self._current_state is None:
                self._current_state = self.rssm.initial_state(1, self.device)
                self._prev_action = torch.zeros(1, 2, device=self.device)

            # Use ACTUAL previous action, not zeros (Blocker 4 fix)
            self._current_state, _ = self.rssm.observe_step(
                self._current_state, self._prev_action, embed
            )
            action = self.actor(self._current_state.combined)

            if explore:
                action = action + torch.randn_like(action) * 0.3
                action = torch.clamp(action, -1.0, 1.0)

            # Store this action for next step
            self._prev_action = action.clone()

        return action.cpu().numpy().squeeze()

    def reset(self):
        """Reset agent state for new episode."""
        self._current_state = None
        self._prev_action = None
        if self.frame_stacker is not None:
            self.frame_stacker.reset()

    def train_step(self, replay_buffer: list) -> dict:
        """
        Single training step from replay buffer (Blocker 1 fix).

        Performs:
        1. World model training (encoder + RSSM + decoder + reward/continue)
        2. Actor-critic training via imagination rollouts

        Returns:
            Dict of loss values for logging.
        """
        batch_size = self.config["training"]["batch_size"]
        batch_length = self.config["training"]["batch_length"]

        if len(replay_buffer) < batch_size * batch_length:
            return {}

        # Sample batch from replay buffer
        batch = self._sample_batch(replay_buffer, batch_size, batch_length)
        images = batch["images"].to(self.device)       # (B, T, C, H, W)
        actions = batch["actions"].to(self.device)      # (B, T, 2)
        rewards = batch["rewards"].to(self.device)      # (B, T)
        dones = batch["dones"].to(self.device)           # (B, T)

        B, T = images.shape[:2]

        # --- World Model Training ---
        # Encode all frames through the arm's encoder
        # For CNN: each frame independently
        # For temporal encoders: would ideally use clips, but for training
        # from replay we encode per-frame and let the RSSM handle temporality
        images_flat = images.reshape(B * T, *images.shape[2:])  # (B*T, C, H, W)

        if self._needs_temporal:
            # For temporal encoders during training, we add a trivial temporal dim
            # The RSSM provides the real temporal modeling
            images_5d = images_flat.unsqueeze(2)  # (B*T, C, 1, H, W)
            features_flat = self.encoder(images_5d)
        else:
            features_flat = self.encoder(images_flat)

        embeds_flat = self.adapter(features_flat)            # (B*T, D)
        embeds = embeds_flat.reshape(B, T, -1)               # (B, T, D)

        # Run RSSM over sequence
        states, infos = self.rssm.observe_sequence(actions, embeds)

        # Stack states for loss computation
        state_combined = torch.stack([s.combined for s in states], dim=1)  # (B, T, state_dim)
        state_flat = state_combined.reshape(B * T, -1)

        # Reconstruction loss
        recon = self.decoder(state_flat)
        recon_loss = F.mse_loss(recon, images_flat)

        # Reward prediction loss
        reward_pred = self.reward_pred(state_flat).squeeze(-1)
        reward_loss = F.mse_loss(reward_pred, rewards.reshape(-1))

        # Continue prediction loss
        continue_pred = self.continue_pred(state_flat).squeeze(-1)
        continue_target = (1.0 - dones).reshape(-1)
        continue_loss = F.binary_cross_entropy(continue_pred, continue_target)

        # KL loss
        kl_loss = sum(
            RSSM.kl_loss(info["prior_logits"], info["posterior_logits"])
            for info in infos
        ) / T

        # Total world model loss
        world_model_loss = recon_loss + reward_loss + continue_loss + kl_loss

        self.world_model_opt.zero_grad()
        world_model_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.encoder.parameters() if p.requires_grad]
            + list(self.rssm.parameters()),
            self.grad_clip,
        )
        self.world_model_opt.step()

        # --- Actor-Critic Training (Imagination) ---
        # Detach starting state to prevent backprop through world model
        with torch.no_grad():
            start_state = states[-1].detach()

        imagined_states = self.rssm.imagine_sequence(
            start_state, self.actor, self.imagination_horizon
        )

        imag_combined = torch.stack([s.combined for s in imagined_states], dim=1)
        imag_flat = imag_combined.reshape(-1, imag_combined.shape[-1])

        # Predicted rewards and values (detach from world model graph)
        with torch.no_grad():
            imag_rewards = self.reward_pred(imag_flat).reshape(B, self.imagination_horizon)
            imag_continues = self.continue_pred(imag_flat).reshape(B, self.imagination_horizon)

        imag_values = self.critic(imag_flat.detach()).reshape(B, self.imagination_horizon)

        # Compute lambda returns (GAE)
        returns = self._compute_lambda_returns(
            imag_rewards, imag_values.detach(), imag_continues
        )

        # Actor loss (maximize returns) — detach returns from critic
        # Re-evaluate values for the actor graph
        actor_imag_values = self.critic(imag_combined[:, :-1].detach().reshape(-1, imag_combined.shape[-1]))
        actor_loss = -(returns.detach().reshape(-1) - actor_imag_values.squeeze()).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        # Critic loss (predict returns) — separate backward pass
        with torch.no_grad():
            target_returns = returns.detach()

        critic_pred = self.critic(imag_combined[:, :-1].detach().reshape(-1, imag_combined.shape[-1]))
        critic_loss = F.mse_loss(critic_pred.squeeze(), target_returns.reshape(-1))

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        return {
            "world_model/total": world_model_loss.item(),
            "world_model/recon": recon_loss.item(),
            "world_model/reward": reward_loss.item(),
            "world_model/continue": continue_loss.item(),
            "world_model/kl": kl_loss.item(),
            "actor/loss": actor_loss.item(),
            "critic/loss": critic_loss.item(),
        }

    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        continues: torch.Tensor,
    ) -> torch.Tensor:
        """Compute lambda-returns (GAE) for actor training."""
        T = rewards.shape[1]
        returns = torch.zeros_like(rewards[:, :-1])

        last_value = values[:, -1]
        last_return = last_value

        for t in reversed(range(T - 1)):
            bootstrap = self.lambda_gae * last_return + (1 - self.lambda_gae) * values[:, t + 1]
            last_return = rewards[:, t] + self.discount * continues[:, t] * bootstrap
            returns[:, t] = last_return

        return returns

    def _sample_batch(self, replay_buffer: list, batch_size: int, batch_length: int) -> dict:
        """Sample a batch of sequences from the replay buffer."""
        images, actions, rewards, dones = [], [], [], []

        for _ in range(batch_size):
            max_start = max(0, len(replay_buffer) - batch_length)
            start = np.random.randint(0, max_start + 1)
            seq = replay_buffer[start : start + batch_length]

            if len(seq) < batch_length:
                continue

            imgs = [s["obs"]["image"].astype(np.float32) / 255.0 for s in seq]
            acts = [s["action"] for s in seq]
            rews = [s["reward"] for s in seq]
            dns = [float(s["done"]) for s in seq]

            images.append(np.stack(imgs))
            actions.append(np.stack(acts))
            rewards.append(np.array(rews))
            dones.append(np.array(dns))

        return {
            "images": torch.from_numpy(np.stack(images)).float(),
            "actions": torch.from_numpy(np.stack(actions)).float(),
            "rewards": torch.from_numpy(np.stack(rewards)).float(),
            "dones": torch.from_numpy(np.stack(dones)).float(),
        }

    def state_dict(self) -> dict:
        """Get full agent state for checkpointing."""
        return {
            "encoder": self.encoder.state_dict(),
            "adapter": self.adapter.state_dict(),
            "rssm": self.rssm.state_dict(),
            "decoder": self.decoder.state_dict(),
            "reward_pred": self.reward_pred.state_dict(),
            "continue_pred": self.continue_pred.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "world_model_opt": self.world_model_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load full agent state from checkpoint."""
        self.encoder.load_state_dict(state["encoder"])
        self.adapter.load_state_dict(state["adapter"])
        self.rssm.load_state_dict(state["rssm"])
        self.decoder.load_state_dict(state["decoder"])
        self.reward_pred.load_state_dict(state["reward_pred"])
        self.continue_pred.load_state_dict(state["continue_pred"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.world_model_opt.load_state_dict(state["world_model_opt"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])


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

    # Training loop — now actually trains (Blocker 1 fix)
    replay_buffer = []
    metrics_tracker = MetricsTracker(name=f"phase3_{arm}")
    total_steps = 0
    episode = 0
    best_eval_success = 0.0
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

            # TRAINING — the critical missing piece (Blocker 1 fix)
            train_info = agent.train_step(replay_buffer)

            # Logging
            if total_steps % train_cfg["log_every"] == 0:
                if train_info:
                    for key, value in train_info.items():
                        logger.log_scalar(key, value, total_steps)
                logger.log_scalar("train/reward", reward, total_steps)
                logger.log_vram(total_steps)

            obs = next_obs
            total_steps += 1
            metrics_tracker.step(step_info)

        episode_metrics = metrics_tracker.end_episode(
            success=step_info.get("route_completed", False),
            route_completion=step_info.get("route_completion", 0.0),
        )
        episode += 1
        logger.log_scalar("episode/reward", episode_metrics.total_reward, total_steps)

        # Evaluation + Checkpointing (Blocker 1 fix)
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
                eval_tracker.end_episode(
                    success=step_info.get("route_completed", False),
                    route_completion=step_info.get("route_completion", 0.0),
                )

            eval_summary = eval_tracker.summary()
            for k, v in eval_summary.items():
                logger.log_scalar(f"eval/{k}", v, total_steps)

            # Checkpoint (Blocker 1 fix)
            ckpt_state = {
                "agent_state_dict": agent.state_dict(),
                "arm": arm,
                "seed": seed,
                "total_steps": total_steps,
                "config": config,
            }
            eval_success = eval_summary.get("success_rate", 0.0)
            ckpt_manager.save(ckpt_state, total_steps, eval_success)

            if eval_success > best_eval_success:
                best_eval_success = eval_success

    # Final metrics
    wall_clock = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    final_metrics = metrics_tracker.summary()
    final_metrics["wall_clock_seconds"] = wall_clock
    final_metrics["peak_vram_mb"] = peak_vram
    final_metrics["best_eval_success_rate"] = best_eval_success

    env.close()
    logger.close()

    print(f"\nPhase 3 arm '{arm}' complete. Best eval success: {best_eval_success:.2%}")
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
