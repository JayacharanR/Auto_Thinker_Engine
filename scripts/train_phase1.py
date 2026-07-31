"""
Phase 1 Training Script: DreamerV3 Baseline with CarDreamer.

Trains a closed-loop RL agent on a single CarDreamer task using
DreamerV3's RSSM world model with the default CNN encoder.

This script implements the full training loop:
1. Environment interaction (CARLA)
2. RSSM world model learning
3. Actor-critic training via imagination
4. Evaluation with success/collision metrics

Usage:
    python scripts/train_phase1.py --config configs/phase1_dreamer_baseline.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dreamer.encoder_adapter import CNNEncoder, EncoderAdapter
from src.dreamer.rssm_wrapper import RSSM, RSSMState
from src.envs.carla_wrapper import CarlaEnvWrapper
from src.envs.reward import RewardFunction
from src.eval.metrics import MetricsTracker
from src.utils.checkpoint import CheckpointManager, build_checkpoint_state
from src.utils.logging_utils import ExperimentLogger, make_run_name
from src.utils.seeding import seed_everything


class Actor(nn.Module):
    """Actor network: state → action distribution."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        # Output mean and log_std for each action dimension
        layers.append(nn.Linear(hidden_dim, action_dim * 2))
        self.net = nn.Sequential(*layers)
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) combined RSSM state.

        Returns:
            (B, action_dim) sampled actions (tanh-squashed).
        """
        out = self.net(state)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()

        # Reparameterized sample
        noise = torch.randn_like(mean)
        action = torch.tanh(mean + std * noise)
        return action


class Critic(nn.Module):
    """Critic network: state → value estimate."""

    def __init__(self, state_dim: int, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ImageDecoder(nn.Module):
    """Decoder: RSSM state → reconstructed image (for world model loss)."""

    def __init__(
        self,
        state_dim: int,
        channels: int = 3,
        depth: int = 48,
        output_size: int = 128,
    ):
        super().__init__()
        self.output_size = output_size

        # Project state to spatial features
        self.fc = nn.Linear(state_dim, depth * 8 * 4 * 4)
        self.reshape_depth = depth * 8

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(depth * 8, depth * 4, 5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 2, depth, 6, stride=2, padding=2, output_padding=0),
            nn.SiLU(),
            nn.ConvTranspose2d(depth, channels, 6, stride=2, padding=2, output_padding=0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.fc(state)
        x = x.reshape(-1, self.reshape_depth, 4, 4)
        x = self.deconv(x)
        # Resize to exact target size
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return x


class RewardPredictor(nn.Module):
    """Predicts reward from RSSM state (world model component)."""

    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ContinuePredictor(nn.Module):
    """Predicts episode continuation probability (discount model)."""

    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(state))


class DreamerV3Agent:
    """
    Full DreamerV3 agent combining world model + actor-critic.

    Components:
    - Encoder: CNN (default) or swapped encoder (Phase 3)
    - RSSM: Recurrent state space model
    - Decoder: Image reconstruction
    - Reward predictor: Reward from state
    - Continue predictor: Episode termination
    - Actor: Policy network
    - Critic: Value network
    """

    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device

        dreamer_cfg = config["dreamer"]
        rssm_cfg = dreamer_cfg["rssm"]
        training_cfg = config["training"]
        env_cfg = config["environment"]

        img_size = env_cfg["observation"]["camera"]["width"]
        action_dim = 2  # [steer, throttle_brake]

        # Encoder
        enc_cfg = dreamer_cfg["encoder"]
        self.encoder = CNNEncoder(
            depth=enc_cfg["cnn"]["depth"],
            kernels=enc_cfg["cnn"]["kernels"],
            stride=enc_cfg["cnn"]["stride"],
            activation=enc_cfg["cnn"]["activation"],
            input_size=img_size,
        ).to(device)

        # Adapter
        adapter_dim = 1024
        self.adapter = EncoderAdapter(
            input_dim=self.encoder.output_dim,
            target_dim=adapter_dim,
        ).to(device)

        # RSSM
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

        # Decoder
        self.decoder = ImageDecoder(
            state_dim=state_dim,
            channels=3,
            depth=dreamer_cfg["decoder"]["depth"],
            output_size=img_size,
        ).to(device)

        # Reward & continue predictors
        self.reward_pred = RewardPredictor(state_dim).to(device)
        self.continue_pred = ContinuePredictor(state_dim).to(device)

        # Actor-critic
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

        # Optimizers
        lr = training_cfg["learning_rate"]
        self.world_model_opt = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.adapter.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_pred.parameters())
            + list(self.continue_pred.parameters()),
            lr=lr,
            eps=training_cfg["adam_eps"],
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=training_cfg["adam_eps"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=training_cfg["adam_eps"])

        # Discount
        self.discount = ac_cfg["discount"]
        self.lambda_gae = ac_cfg["lambda_gae"]
        self.grad_clip = training_cfg["grad_clip"]
        self.imagination_horizon = training_cfg["imagination_horizon"]

        # State tracking
        self._current_state: Optional[RSSMState] = None

    def act(self, obs: dict, explore: bool = True) -> np.ndarray:
        """Select action from observation."""
        with torch.no_grad():
            image = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(self.device) / 255.0
            embed = self.adapter(self.encoder(image))

            if self._current_state is None:
                self._current_state = self.rssm.initial_state(1, self.device)

            # Use zero action for first observation step
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
        """Reset agent state for new episode."""
        self._current_state = None

    def train_step(self, replay_buffer: list) -> dict:
        """
        Single training step from replay buffer.

        Args:
            replay_buffer: List of (obs, action, reward, done) transitions.

        Returns:
            Dict of loss values for logging.
        """
        if len(replay_buffer) < self.config["training"]["batch_size"] * self.config["training"]["batch_length"]:
            return {}

        # Sample batch from replay buffer
        batch = self._sample_batch(replay_buffer)
        images = batch["images"].to(self.device)       # (B, T, C, H, W)
        actions = batch["actions"].to(self.device)      # (B, T, 2)
        rewards = batch["rewards"].to(self.device)      # (B, T)
        dones = batch["dones"].to(self.device)           # (B, T)

        B, T = images.shape[:2]

        # --- World Model Training ---
        # Encode all frames
        images_flat = images.reshape(B * T, *images.shape[2:])  # (B*T, C, H, W)
        embeds_flat = self.adapter(self.encoder(images_flat))    # (B*T, D)
        embeds = embeds_flat.reshape(B, T, -1)                   # (B, T, D)

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
            list(self.encoder.parameters()) + list(self.rssm.parameters()),
            self.grad_clip,
        )
        self.world_model_opt.step()

        # --- Actor-Critic Training (Imagination) ---
        with torch.no_grad():
            # Start imagination from last observed state
            start_state = states[-1].detach()

        imagined_states = self.rssm.imagine_sequence(
            start_state, self.actor, self.imagination_horizon
        )

        imag_combined = torch.stack([s.combined for s in imagined_states], dim=1)
        imag_flat = imag_combined.reshape(-1, imag_combined.shape[-1])

        # Predicted rewards and values
        imag_rewards = self.reward_pred(imag_flat).reshape(B, self.imagination_horizon)
        imag_values = self.critic(imag_flat).reshape(B, self.imagination_horizon)
        imag_continues = self.continue_pred(imag_flat).reshape(B, self.imagination_horizon)

        # Compute lambda returns (GAE)
        returns = self._compute_lambda_returns(
            imag_rewards, imag_values, imag_continues
        )

        # Actor loss (maximize returns)
        actor_loss = -returns.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        # Critic loss (predict returns)
        with torch.no_grad():
            target_returns = returns.detach()
        critic_pred = self.critic(imag_combined[:, :-1].reshape(-1, imag_combined.shape[-1]))
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

    def _sample_batch(self, replay_buffer: list) -> dict:
        """Sample a batch of sequences from the replay buffer."""
        batch_size = self.config["training"]["batch_size"]
        batch_length = self.config["training"]["batch_length"]

        images, actions, rewards, dones = [], [], [], []

        for _ in range(batch_size):
            # Random starting point
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


def train(config: dict):
    """Main Phase 1 training loop."""
    # Setup
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    seed = exp_cfg["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(seed)

    run_name = exp_cfg.get("run_name") or make_run_name(phase=1, arm="cnn", seed=seed)
    logger = ExperimentLogger(
        log_dir=exp_cfg["log_dir"],
        run_name=run_name,
        use_wandb=exp_cfg.get("use_wandb", False),
        config=config,
    )

    ckpt_manager = CheckpointManager(
        checkpoint_dir=exp_cfg["checkpoint_dir"],
        max_keep=train_cfg["max_keep_checkpoints"],
        metric_name="eval/success_rate",
        metric_mode="max",
    )

    # Create environment
    env = CarlaEnvWrapper(
        config=config["environment"],
        reward_fn=RewardFunction(config["reward"]),
    )
    env.connect()

    # Create agent
    agent = DreamerV3Agent(config, device=device)

    # Training loop
    replay_buffer = []
    metrics_tracker = MetricsTracker(name="phase1_train")
    total_steps = 0
    episode = 0
    best_success_rate = 0.0

    print(f"\nStarting Phase 1 training: {run_name}")
    print(f"Total steps: {train_cfg['total_steps']}")
    print(f"Device: {device}")

    while total_steps < train_cfg["total_steps"]:
        obs, info = env.reset()
        agent.reset()
        metrics_tracker.start_episode()
        episode_reward = 0.0
        episode_steps = 0

        done = False
        while not done and total_steps < train_cfg["total_steps"]:
            # Act
            action = agent.act(obs, explore=True)

            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            # Store transition
            replay_buffer.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "done": done,
            })

            # Keep replay buffer bounded
            max_buffer_size = 100_000
            if len(replay_buffer) > max_buffer_size:
                replay_buffer = replay_buffer[-max_buffer_size:]

            # Train
            train_info = agent.train_step(replay_buffer)

            # Log
            if total_steps % train_cfg["log_every"] == 0 and train_info:
                for key, value in train_info.items():
                    logger.log_scalar(key, value, total_steps)
                logger.log_scalar("train/reward", reward, total_steps)
                logger.log_vram(total_steps)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            metrics_tracker.step(step_info)

        # Episode end
        episode_metrics = metrics_tracker.end_episode(
            success=not terminated and truncated,  # Completed without collision
            route_completion=step_info.get("route_completion", 0.0),
        )
        episode += 1

        logger.log_scalar("episode/reward", episode_reward, total_steps)
        logger.log_scalar("episode/steps", episode_steps, total_steps)

        # Evaluation
        if total_steps % train_cfg["eval_every"] == 0:
            eval_metrics = evaluate(agent, env, config, device)
            for key, value in eval_metrics.items():
                logger.log_scalar(f"eval/{key}", value, total_steps)

            # Checkpoint
            ckpt_state = {"agent_state_dict": agent.state_dict(), "config": config}
            ckpt_manager.save(ckpt_state, total_steps, eval_metrics.get("success_rate", 0.0))

            if eval_metrics.get("success_rate", 0.0) > best_success_rate:
                best_success_rate = eval_metrics["success_rate"]

    # Cleanup
    env.close()
    logger.close()

    print(f"\nPhase 1 training complete. Best success rate: {best_success_rate:.2%}")


def evaluate(agent, env, config, device) -> dict:
    """Run evaluation episodes."""
    num_episodes = config["environment"]["num_eval_episodes"]
    tracker = MetricsTracker(name="eval")

    for _ in range(num_episodes):
        obs, info = env.reset()
        agent.reset()
        tracker.start_episode()
        done = False

        while not done:
            action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            tracker.step(step_info)

        tracker.end_episode(
            success=not terminated and truncated,
            route_completion=step_info.get("route_completion", 0.0),
        )

    return tracker.summary()


def main():
    parser = argparse.ArgumentParser(description="Phase 1: DreamerV3 Baseline Training")
    parser.add_argument(
        "--config",
        default="configs/phase1_dreamer_baseline.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
