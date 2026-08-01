"""
Synthetic integration test: runs a REAL DreamerV3 training step.

This is the highest-value test in the project. It proves the full pipeline
works end-to-end: env → encoder → RSSM → actor-critic → gradients → optimizer.

No CARLA. No GPU. CPU only. Uses FakeDrivingEnv.
"""

import os
import sys
import pathlib
import shutil
import tempfile

import numpy as np
import pytest
import torch

# Add project and dreamerv3-torch to path
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "dreamerv3_torch"))

from tests.fake_driving_env import FakeDrivingEnv


def _make_dreamer_config(logdir: str, image_size=(64, 64)):
    """Build a minimal dreamerv3-torch config for CPU testing."""
    import argparse
    from ruamel.yaml import YAML

    # Load dreamerv3-torch defaults
    configs_path = PROJECT_ROOT / "third_party" / "dreamerv3_torch" / "configs.yaml"
    ryaml = YAML(typ="safe", pure=True)
    all_configs = ryaml.load(configs_path.read_text())
    config = all_configs["defaults"].copy()

    # Override for fast CPU testing
    config.update({
        "logdir": logdir,
        "seed": 42,
        "steps": 200,               # very short
        "device": "cpu",
        "size": list(image_size),
        "action_repeat": 1,
        "time_limit": 50,           # short episodes
        "prefill": 100,             # minimal prefill
        "eval_every": 100,
        "eval_episode_num": 1,
        "log_every": 50,
        "reset_every": 0,
        "compile": False,
        "precision": 32,
        "video_pred_log": False,
        "parallel": False,
        "batch_size": 2,            # tiny batch
        "batch_length": 8,          # short sequences
        "train_ratio": 32,          # train less often
        "pretrain": 1,              # just 1 pretrain step
        "envs": 1,
        "reward_EMA": True,
        "deterministic_run": True,
        "expl_behavior": "greedy",
        "expl_until": 0,
        "dataset_size": 10000,
    })

    return argparse.Namespace(**config)


def _build_cnn_encoder_hook(device="cpu"):
    """Build a CNN encoder hook for testing."""
    from src.dreamer.cardreamer_encoder_hook import DreamerV3EncoderHook

    config = {
        "adapter": {"target_dim": 256, "use_layer_norm": True, "activation": "silu"},
        "encoders": {
            "cnn": {
                "depth": 32,
                "kernels": [4, 4, 4, 4],
                "stride": 2,
                "activation": "silu",
                "output_dim": 512,
                "input_resolution": 64,
            },
        },
    }
    return DreamerV3EncoderHook(arm="cnn", config=config, device=device)


def _build_jepa_encoder_hook(device="cpu"):
    """Build a JEPA encoder hook for testing."""
    from src.dreamer.cardreamer_encoder_hook import DreamerV3EncoderHook

    config = {
        "adapter": {"target_dim": 256, "use_layer_norm": True, "activation": "silu"},
        "encoders": {
            "custom_jepa": {
                "input_resolution": 224,
                "patch_size": 16,
                "embed_dim": 384,
                "depth": 2,       # small for fast test
                "num_heads": 6,
                "output_dim": 384,
                "tubelet_size": 2,
            },
        },
    }
    return DreamerV3EncoderHook(
        arm="custom_jepa", config=config, device=device,
        num_temporal_frames=4,
    )


def _setup_agent(logdir, hook, image_size=(64, 64)):
    """
    Shared setup: build env, config, prefill replay, construct Dreamer agent.

    This is the correct construction order:
    1. Build env (with UUID wrapper for episode tracking)
    2. Build encoder hook
    3. Prefill replay buffer with random actions
    4. Construct Dreamer with custom_encoder (BEFORE RSSM is built)

    Returns: (agent, env, config, train_dataset, train_eps)
    """
    import tools
    from parallel import Damy
    import envs.wrappers as wrappers
    from dreamer import Dreamer, make_dataset

    env = FakeDrivingEnv(image_size=image_size, seed=42)
    # SelectAction extracts the 'action' key from the dict that simulate() passes
    env = wrappers.SelectAction(env, key="action")
    # UUID wrapper adds .id for episode tracking in replay
    env = wrappers.UUID(env)

    config = _make_dreamer_config(logdir, image_size)
    config.num_actions = env.action_space.shape[0]

    logger = tools.Logger(pathlib.Path(logdir), 0)

    # Wrap for dreamerv3-torch's simulate()
    train_envs = [Damy(env)]
    train_eps = tools.load_episodes(
        pathlib.Path(logdir) / "train_eps", limit=config.dataset_size
    )

    # Prefill with random actions
    from torch import distributions as torchd
    random_actor = torchd.independent.Independent(
        torchd.uniform.Uniform(
            torch.tensor(env.action_space.low).unsqueeze(0),
            torch.tensor(env.action_space.high).unsqueeze(0),
        ),
        1,
    )

    def random_agent(o, d, s):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {"action": action, "logprob": logprob}, None

    tools.simulate(
        random_agent, train_envs, train_eps,
        pathlib.Path(logdir) / "train_eps", logger,
        limit=config.dataset_size, steps=config.prefill,
    )

    train_dataset = make_dataset(train_eps, config)

    # KEY: inject encoder at construction time (not after)
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config, logger, train_dataset,
        custom_encoder=hook,
    ).to("cpu")

    return agent, env, config, train_dataset, train_eps


class TestFakeDrivingEnv:
    """Verify the synthetic env works correctly."""

    def test_reset_returns_valid_obs(self):
        env = FakeDrivingEnv(image_size=(64, 64))
        obs = env.reset()
        assert "image" in obs
        assert obs["image"].shape == (64, 64, 3)
        assert obs["image"].dtype == np.uint8

    def test_step_returns_4_tuple(self):
        env = FakeDrivingEnv(image_size=(64, 64))
        obs = env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 4  # obs, reward, done, info
        obs, reward, done, info = result
        assert "image" in obs
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_100_random_steps(self):
        """Run 100 random steps — acceptance test from the plan."""
        env = FakeDrivingEnv(image_size=(64, 64), max_steps=200, seed=42)
        obs = env.reset()
        assert "image" in obs
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            assert obs["image"].shape == (64, 64, 3)
            if done:
                obs = env.reset()


class TestCNNOneStepTraining:
    """
    Milestone A: CNN arm, one training step, gradient audit.

    This is the EXIT GATE for "does the code actually run."
    """

    @pytest.fixture
    def logdir(self):
        d = tempfile.mkdtemp(prefix="dreamer_test_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_cnn_constructs_with_custom_encoder(self, logdir):
        """Agent constructs with custom encoder injected BEFORE RSSM."""
        hook = _build_cnn_encoder_hook("cpu")
        agent, env, config, _, _ = _setup_agent(logdir, hook)

        # Verify encoder is our hook
        assert agent._wm.encoder is hook
        assert agent._wm.embed_size == 256  # adapter target_dim

        # Verify RSSM was sized correctly
        print(f"embed_size: {agent._wm.embed_size}")
        print(f"encoder type: {type(agent._wm.encoder).__name__}")

    def test_cnn_one_training_step(self, logdir):
        """Execute one real training step and verify gradients."""
        hook = _build_cnn_encoder_hook("cpu")
        agent, env, config, train_dataset, _ = _setup_agent(logdir, hook)

        # Enable gradients (NOT requires_grad_(False) on whole agent)
        for param in agent.parameters():
            param.requires_grad = True

        # Get a training batch
        data = next(train_dataset)

        # Execute ONE training step
        agent._train(data)

        # === GRADIENT AUDIT ===
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in hook.encoder.parameters()
        )
        print(f"CNN encoder has gradients: {encoder_has_grad}")

        adapter_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in hook.adapter.parameters()
        )
        print(f"Adapter has gradients: {adapter_has_grad}")

        rssm_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in agent._wm.dynamics.parameters()
        )
        print(f"RSSM has gradients: {rssm_has_grad}")

        # At minimum, RSSM and adapter must have gradients
        assert rssm_has_grad, "RSSM received no gradients — training is broken"
        assert adapter_has_grad, "Adapter received no gradients — encoder swap is broken"

    def test_checkpoint_save_load(self, logdir):
        """Save and load checkpoint without crash."""
        import tools

        hook = _build_cnn_encoder_hook("cpu")
        agent, env, config, _, _ = _setup_agent(logdir, hook)

        # Save checkpoint
        ckpt_path = pathlib.Path(logdir) / "test.pt"
        items = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items, ckpt_path)

        # Load checkpoint
        loaded = torch.load(ckpt_path, weights_only=False)
        agent.load_state_dict(loaded["agent_state_dict"])
        print("Checkpoint save/load: OK")
