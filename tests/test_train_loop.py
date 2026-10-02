"""
End-to-end check of train_arm (the Stage 3/5 runner) without CARLA: a fake
CarDreamer task, a tiny Dreamer on CPU, checkpoint cadence, eval, metrics
and resume.
"""

import json
import re
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "dreamerv3_torch"))

import scripts.train_cardreamer as trainer  # noqa: E402
from src.dreamer.carla_wrappers import DreamerObservation, EpisodeMetricsRecorder  # noqa: E402
from tests.test_carla_wrappers import FakeCarDreamerTask  # noqa: E402

TINY = (
    "prefill=60", "batch_size=2", "batch_length=8", "train_ratio=2", "pretrain=1",
    "eval_every=100", "eval_episode_num=1", "precision=32", "log_every=50",
    "dataset_size=10000", "units=64", "dyn_hidden=64", "dyn_deter=64",
    "dyn_stoch=8", "dyn_discrete=8", "encoder={cnn_depth: 8}", "decoder={cnn_depth: 8}",
    "imag_horizon=4",
)


def fake_make_carla_env(task_name, obs="bev", action="discrete", image_size=(64, 64),
                        metrics_path=None, feature_extractor=None, env_overrides=()):
    import envs.wrappers as wrappers

    env = DreamerObservation(FakeCarDreamerTask(length=40, ending="time_exceeded"),
                             obs=obs, size=image_size)
    env = recorder = EpisodeMetricsRecorder(env, metrics_path)
    env = wrappers.UUID(wrappers.SelectAction(wrappers.OneHotAction(env), key="action"))
    return env, recorder


def test_train_arm_checkpoints_evaluates_and_resumes(tmp_path, monkeypatch, capsys):
    torch.set_num_threads(2)  # keep CPU free for anything else running
    monkeypatch.setattr(trainer, "make_carla_env", fake_make_carla_env)
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    run = dict(arm="cnn", task="fake", seed=0, phase3_config={}, overrides=TINY,
               logdir=str(tmp_path / "run"), checkpoint_every=50)

    metrics = trainer.train_arm(steps=260, **run)
    out = capsys.readouterr().out
    # Prefill to 60, then checkpoints every 50 steps inside each 100-step eval chunk.
    assert re.findall(r"\[train\] env step (\d+)/260", out) == ["110", "160", "210", "260"]
    assert len(re.findall(r"\[train\] eval @", out)) == 2
    assert metrics["env_steps"] == 260 and metrics["updates"] > 0
    assert torch.load(tmp_path / "run" / "latest.pt", weights_only=False)["step"] == 260
    eval_lines = (tmp_path / "run" / "eval.jsonl").read_text().splitlines()
    evals = [json.loads(line) for line in eval_lines]
    assert [e["agent_step"] for e in evals] == [160, 260]
    assert (tmp_path / "run" / "episodes.jsonl").read_text().count('"mode": "train"') >= 5

    with pytest.raises(FileExistsError):
        trainer.train_arm(steps=300, **run)

    metrics = trainer.train_arm(steps=300, resume=True, **run)
    out = capsys.readouterr().out
    assert "Prefilling" not in out and re.findall(r"env step (\d+)/300", out) == ["300"]
    assert metrics["env_steps"] == 300


def test_task_overrides_are_validated():
    sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "CarDreamer"))
    from car_dreamer import load_task_configs
    from car_dreamer.toolkit import Flags

    argv = trainer.task_argv("carla_right_turn_simple", "discrete", ("reward.scales.time=0.1",))
    config, _ = Flags(load_task_configs("carla_right_turn_simple")).parse_known(argv)
    assert config.env.reward.scales.time == 0.1 and config.env.action.discrete is True
    with pytest.raises(ValueError, match="Unknown CarDreamer options"):
        trainer.task_argv("carla_right_turn_simple", "discrete", ("reward.scales.tme=0.1",))
