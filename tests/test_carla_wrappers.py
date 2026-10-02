"""
CarDreamer -> dreamerv3-torch wrappers, without CARLA.

A fake task mimics CarDreamer's legacy Gym API: dict observations with
``camera`` and ``birdeye_wpt``, and terminal conditions in ``info`` as
one-element bool arrays.
"""

import json
import pathlib
import sys
import tempfile

import gym
import numpy as np
import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "dreamerv3_torch"))

from src.dreamer.carla_wrappers import DreamerObservation, EpisodeMetricsRecorder  # noqa: E402

CONDITIONS = ("is_collision", "out_of_lane", "destination_reached", "time_exceeded")


class FakeCarDreamerTask(gym.Env):
    """Episodes of ``length`` steps that end with the condition ``ending``."""

    def __init__(self, length=5, ending="is_collision"):
        self.length, self.ending = length, ending
        image = gym.spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({
            "camera": image,
            "birdeye_wpt": image,
            "collision": gym.spaces.Box(0, np.inf, (1,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(3)
        self._t = 0

    def _obs(self):
        return {
            "camera": np.full((128, 128, 3), 10, np.uint8),
            "birdeye_wpt": np.full((128, 128, 3), 200, np.uint8),
            "collision": np.zeros(1, np.float32),
        }

    def reset(self):
        self._t = 0
        return self._obs()

    def step(self, action):
        self._t += 1
        done = self._t >= self.length
        info = {k: np.array([done and k == self.ending]) for k in CONDITIONS}
        info.update(wpt_dis=0.5, speed_norm=5.0, ttc=10.0)
        return self._obs(), 1.0, done, info


def _run_episode(env, steps):
    env.reset()
    for _ in range(steps):
        out = env.step(0)
    return out


def test_bev_observation_contract():
    env = DreamerObservation(FakeCarDreamerTask(), obs="bev", size=(64, 64))
    assert set(env.observation_space.spaces) == {"image", "is_first", "is_last", "is_terminal"}
    obs = env.reset()
    assert obs["image"].shape == (64, 64, 3) and obs["image"].dtype == np.uint8
    assert obs["image"].min() == 200  # taken from birdeye_wpt, not the camera
    assert obs["is_first"] == 1 and obs["is_last"] == 0


def test_missing_source_is_rejected():
    task = FakeCarDreamerTask()
    del task.observation_space.spaces["birdeye_wpt"]
    with pytest.raises(ValueError, match="birdeye_wpt"):
        DreamerObservation(task, obs="bev")


@pytest.mark.parametrize("ending, terminal", [
    ("is_collision", True),
    ("out_of_lane", True),
    ("destination_reached", True),
    ("time_exceeded", False),  # truncation: the value after it is not zero
])
def test_terminal_versus_truncation(ending, terminal):
    env = DreamerObservation(FakeCarDreamerTask(length=3, ending=ending))
    obs, _, done, info = _run_episode(env, 3)
    assert done and obs["is_last"] == 1
    assert obs["is_terminal"] == int(terminal)
    assert float(info["discount"]) == (0.0 if terminal else 1.0)


def test_metrics_recorder_writes_one_record_per_episode():
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "episodes.jsonl"
        env = EpisodeMetricsRecorder(DreamerObservation(FakeCarDreamerTask(length=4)), path)
        env.step_fn = lambda: 123
        _run_episode(env, 4)
        env.mode = "eval"
        env.env.env.ending = "destination_reached"
        _run_episode(env, 4)
        records = [json.loads(line) for line in path.read_text().splitlines()]

    assert [r["mode"] for r in records] == ["train", "eval"]
    train, evaluation = records
    assert train["termination"] == "is_collision" and train["collision"] and not train["success"]
    assert train["return"] == 4.0 and train["length"] == 4 and train["agent_step"] == 123
    assert evaluation["termination"] == "destination_reached" and evaluation["success"]
    summary = env.reset_eval().summary()
    assert summary["num_episodes"] == 1 and summary["success_rate"] == 1.0


def _wrapped_task(directory):
    import envs.wrappers as wrappers

    env = DreamerObservation(FakeCarDreamerTask(length=20, ending="time_exceeded"))
    env = EpisodeMetricsRecorder(env, pathlib.Path(directory) / "episodes.jsonl")
    env = wrappers.OneHotAction(env)
    env = wrappers.SelectAction(env, key="action")
    return wrappers.UUID(env)


def test_stack_with_dreamer_simulate_and_one_update():
    """The trainer's wrapper stack works with dreamerv3-torch's replay and agent."""
    import tools
    from dreamer import make_dataset
    from parallel import Damy

    from scripts.train_cardreamer import build_agent
    from tests.test_synthetic_training import _make_dreamer_config

    with tempfile.TemporaryDirectory() as d:
        env = _wrapped_task(d)
        config = _make_dreamer_config(d)
        logger = tools.Logger(pathlib.Path(d), 0)
        cache = tools.load_episodes(pathlib.Path(d) / "train_eps", limit=config.dataset_size)
        acts = env.action_space
        actor = tools.OneHotDist(torch.zeros(acts.shape[0]).repeat(1, 1))

        def random_agent(o, done, s):
            a = actor.sample()
            return {"action": a, "logprob": actor.log_prob(a)}, None

        tools.simulate(random_agent, [Damy(env)], cache, pathlib.Path(d) / "train_eps",
                       logger, limit=config.dataset_size, steps=100)
        episode = next(iter(cache.values()))
        # Time-limit endings are stored as truncations (discount 1).
        assert episode["discount"][-1] == 1.0 and episode["is_terminal"][-1] == 0

        agent = build_agent(env.observation_space, acts, config, logger,
                            make_dataset(cache, config), "cnn", {}, "cpu")
        assert config.actor["dist"] == "onehot"
        assert type(agent._wm.encoder).__name__ == "MultiEncoder"
        agent.requires_grad_(False)
        agent._train(next(agent._dataset))
        assert float(np.mean(agent._metrics["model_grad_norm"])) > 0


# --- Stage 5: camera + route contract, frozen-encoder features -------------

from src.dreamer.carla_wrappers import ROUTE_WAYPOINTS, route_features  # noqa: E402

ROUTE_DIM = 2 * ROUTE_WAYPOINTS + 3


def _fake_ego_state(env):
    # Ego at the origin heading +x at 5 m/s; route ahead then bending right (+y).
    return 0.0, 0.0, 0.0, 5.0, [(10.0, 0.0, 0.0), (20.0, 5.0, 30.0)]


def test_route_features_geometry():
    f = route_features(0.0, 0.0, 0.0, 5.0, [(10.0, 0.0, 0.0), (20.0, 5.0, 30.0)], stride=1)
    assert f.shape == (ROUTE_DIM,) and f.dtype == np.float32
    pairs = f[: 2 * ROUTE_WAYPOINTS].reshape(-1, 2) * 20.0
    np.testing.assert_allclose(pairs[0], [10.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(pairs[1], [20.0, -5.0], atol=1e-5)  # +y is to the right in CARLA
    np.testing.assert_allclose(pairs[-1], pairs[1])  # padded with the last waypoint
    np.testing.assert_allclose(f[-3:], [0.5, 0.0, 1.0], atol=1e-6)

    # Rotating ego and route together leaves the ego-frame features unchanged.
    yaw = np.deg2rad(90.0)
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    pts = [(*(rot @ [10.0, 0.0]), 90.0), (*(rot @ [20.0, 5.0]), 120.0)]
    np.testing.assert_allclose(route_features(0.0, 0.0, 90.0, 5.0, pts, stride=1), f, atol=1e-5)
    assert np.isfinite(route_features(1.0, 2.0, 0.0, 0.0, [])).all()

    # Default stride: every 4th of CarDreamer's 0.8 m waypoints (~30 m lookahead).
    line = [(0.8 * i, 0.0, 0.0) for i in range(60)]
    forward = route_features(0.0, 0.0, 0.0, 0.0, line)[: 2 * ROUTE_WAYPOINTS : 2] * 20.0
    np.testing.assert_allclose(forward, 3.2 * np.arange(ROUTE_WAYPOINTS), atol=1e-5)


def test_camera_route_contract():
    env = DreamerObservation(FakeCarDreamerTask(), obs="camera_route", ego_state_fn=_fake_ego_state)
    assert env.observation_space["route"].shape == (ROUTE_DIM,)
    obs = env.reset()
    assert obs["image"].min() == 10  # front camera, not the BEV
    assert obs["route"].shape == (ROUTE_DIM,)


def _phase2_checkpoint(tmp_path):
    """A tiny Phase 2 checkpoint as scripts/train_phase2_jepa.py writes it."""
    import yaml

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import train_phase2_jepa as phase2

    config = yaml.safe_load((PROJECT_ROOT / "configs" / "phase2_jepa_laptop.yaml").read_text())
    config["data"]["tubelet"].update(num_frames=4, spatial_size=32)
    config["model"]["context_encoder"].update(patch_size=8, embed_dim=48, depth=1, num_heads=2)
    encoder, _, _ = phase2.build_models(config, "cpu")
    path = pathlib.Path(tmp_path) / "best.pt"
    torch.save({"context_encoder_state_dict": encoder.state_dict(), "config": config,
                "regularizer": "ema"}, path)
    return path, encoder


def _phase3_config(checkpoint):
    return {
        "adapter": {"target_dim": 64, "use_layer_norm": True, "activation": "silu"},
        "encoders": {"custom_jepa": {"checkpoint": str(checkpoint), "freeze": True}},
    }


def test_feature_extractor_matches_checkpoint(tmp_path):
    from src.dreamer.cardreamer_encoder_hook import build_feature_extractor

    path, encoder = _phase2_checkpoint(tmp_path)
    extractor = build_feature_extractor("custom_jepa", _phase3_config(path), "cpu")
    assert extractor.dim == 48 and extractor.resolution == 32
    for p_loaded, p_saved in zip(extractor.encoder.parameters(), encoder.parameters()):
        torch.testing.assert_close(p_loaded, p_saved)
    # Channel-reversed view with negative strides, as CarDreamer's camera returns.
    frame = np.random.RandomState(0).randint(0, 255, (128, 128, 3), np.uint8)[:, :, ::-1]
    first = extractor(frame)
    assert first.shape == (48,) and np.isfinite(first).all()
    extractor.reset()
    np.testing.assert_allclose(extractor(frame), first, rtol=1e-4, atol=1e-5)


def test_feature_extractor_rejects_mismatched_checkpoint(tmp_path):
    from src.dreamer.cardreamer_encoder_hook import build_feature_extractor

    path, _ = _phase2_checkpoint(tmp_path)
    ckpt = torch.load(path, weights_only=False)
    ckpt["context_encoder_state_dict"].pop("norm.weight")
    torch.save(ckpt, path)
    with pytest.raises(RuntimeError, match="norm.weight"):
        build_feature_extractor("custom_jepa", _phase3_config(path), "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="dreamerv3-torch's MLP allocates a constant on cuda")
def test_frozen_arm_with_route_trains_on_stored_features(tmp_path):
    """custom_jepa + camera_route: features stored in replay, route MLP shared with cnn."""
    import envs.wrappers as wrappers
    import tools
    from dreamer import make_dataset
    from parallel import Damy

    from scripts.train_cardreamer import build_agent
    from src.dreamer.cardreamer_encoder_hook import (
        PrecomputedFeatureEncoder,
        build_feature_extractor,
    )
    from tests.test_synthetic_training import _make_dreamer_config

    path, _ = _phase2_checkpoint(tmp_path)
    phase3 = _phase3_config(path)
    extractor = build_feature_extractor("custom_jepa", phase3, "cpu")
    env = DreamerObservation(FakeCarDreamerTask(length=20, ending="time_exceeded"),
                             obs="camera_route", feature_extractor=extractor,
                             ego_state_fn=_fake_ego_state)
    env = wrappers.UUID(wrappers.SelectAction(wrappers.OneHotAction(env), key="action"))

    config = _make_dreamer_config(str(tmp_path))
    logger = tools.Logger(pathlib.Path(tmp_path), 0)
    cache = tools.load_episodes(pathlib.Path(tmp_path) / "eps", limit=config.dataset_size)
    actor = tools.OneHotDist(torch.zeros(env.action_space.shape[0]).repeat(1, 1))

    def random_agent(o, done, s):
        a = actor.sample()
        return {"action": a, "logprob": actor.log_prob(a)}, None

    tools.simulate(random_agent, [Damy(env)], cache, pathlib.Path(tmp_path) / "eps",
                   logger, limit=config.dataset_size, steps=100)
    episode = next(iter(cache.values()))
    assert np.asarray(episode["feat"]).shape[1:] == (48,)

    agent = build_agent(env.observation_space, env.action_space, config, logger,
                        make_dataset(cache, config), "custom_jepa", phase3, "cpu")
    hook = agent._wm.encoder
    assert isinstance(hook, PrecomputedFeatureEncoder)
    assert hook.vector_encoder.mlp_shapes == {"route": (ROUTE_DIM,)}
    assert not hook.vector_encoder.cnn_shapes
    assert "route" in agent._wm.heads["decoder"].mlp_shapes

    cnn_config = _make_dreamer_config(str(tmp_path))
    cnn_space = gym.spaces.Dict(
        {k: v for k, v in env.observation_space.spaces.items() if k != "feat"}
    )
    cnn = build_agent(cnn_space, env.action_space, cnn_config, logger,
                      make_dataset(cache, cnn_config), "cnn", phase3, "cpu")
    assert set(cnn._wm.encoder.cnn_shapes) == {"image"}
    # Same route branch architecture in both arms.
    assert str(cnn._wm.encoder._mlp) == str(hook.vector_encoder._mlp)

    agent.requires_grad_(False)
    before = [p.clone() for p in hook.adapter.parameters()]
    agent._train(next(agent._dataset))
    assert any((p - q).abs().max() > 0 for p, q in zip(hook.adapter.parameters(), before))
