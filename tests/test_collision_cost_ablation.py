"""Check real reward code and equal-budget, canonical-evaluation pilot contracts."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_collision_cost_ablation import Ablation, parse_args, plan
from scripts.run_traffic_curriculum import Queue

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/CarDreamer"))


@pytest.fixture
def reward_task(monkeypatch):
    module = importlib.import_module("car_dreamer.carla_wpt_env")

    class RewardTask(module.CarlaWptEnv):
        def get_ego_planner(self):
            return None

    task = object.__new__(RewardTask)
    task._config = SimpleNamespace(reward=SimpleNamespace(desired_speed=4, scales={
        "waypoint": 2.0, "speed": 0.5, "collision": 30.0, "collision_fixed": 0.0,
        "out_of_lane": 3.0, "destination_reached": 20.0, "time": 0.1,
    }))
    task.num_completed = 1
    task.waypoints = [(1, 0, 0)] * 4
    task.planner_stats = {}
    task._world = SimpleNamespace(carla_world=None, carla_map=None)
    task.ego = object()
    task.colliding = False
    task.velocity = (0, 0)
    task.is_collision = lambda: task.colliding
    monkeypatch.setattr(module, "get_vehicle_pos", lambda ego: (0, 0))
    monkeypatch.setattr(module, "get_vehicle_velocity", lambda ego: task.velocity)
    monkeypatch.setattr(module.TTCCalculator, "get_ttc", lambda *args: 0)
    monkeypatch.setattr(module.CarlaBaseEnv, "reset", lambda self: "reset observation")
    task.reset()
    return task


@pytest.mark.parametrize("speed", [0.0, 0.01, 4.0])
def test_fixed_cost_changes_only_first_collision_and_resets(reward_task, speed):
    task = reward_task
    task.velocity = (speed, 0)
    baseline, baseline_info = task.reward()
    task._config.reward.scales["collision_fixed"] = 300.0
    unchanged, info = task.reward()
    assert unchanged == baseline and info["r_collision_fixed"] == 0.0
    task.colliding = True
    task._config.reward.scales["collision_fixed"] = 0.0
    original, original_info = task.reward()
    assert original_info["r_collision"] == pytest.approx(-30 * speed)
    task.reset()
    task._config.reward.scales["collision_fixed"] = 300.0
    changed, changed_info = task.reward()
    assert changed == pytest.approx(original - 300)
    assert changed_info["r_collision"] == pytest.approx(-300 - 30 * speed)
    # Latched sensor / delayed terminal step must not charge another fixed cost.
    repeated, repeated_info = task.reward()
    assert repeated == original and repeated_info["r_collision_fixed"] == 0
    assert task.reset() == "reset observation"
    assert task.reward()[0] == pytest.approx(original - 300)
    for key in ("r_waypoints", "r_speed", "r_out_of_lane", "r_destination", "r_time"):
        assert changed_info[key] == original_info[key]


def test_legacy_reward_config_without_fixed_key_is_unchanged(reward_task):
    task = reward_task
    task.colliding, task.velocity = True, (4, 0)
    del task._config.reward.scales["collision_fixed"]
    _, info = task.reward()
    assert info["r_collision"] == -120 and info["r_collision_fixed"] == 0


@pytest.mark.parametrize("task", ["simple", "medium", "hard"])
def test_benchmark_default_and_evaluation_override_win(task):
    from car_dreamer import load_task_configs
    from car_dreamer.toolkit import Flags

    from scripts.train_cardreamer import task_argv

    name = f"carla_right_turn_{task}"
    config = load_task_configs(name)
    assert config.env.reward.scales.collision_fixed == 0.0
    argv = task_argv(name, env_overrides=(
        "reward.scales.time=0.1", "reward.scales.collision_fixed=300.0",
        "reward.scales.collision_fixed=0.0",
    ))
    resolved, unknown = Flags(config).parse_known(argv)
    assert not unknown and resolved.env.reward.scales.collision_fixed == 0.0
    assert resolved.env.reward.scales.time == 0.1


def make_queue(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    args = parse_args(["--out", str(tmp_path / "queue"), "--run-dir", str(tmp_path / "runs"),
                       "--init-from", str(source)])
    queue = Ablation(args, steps=plan(args))
    queue.source = source
    return queue


def test_pilots_use_same_fresh_source_budget_and_final_checkpoint(tmp_path, monkeypatch):
    queue = make_queue(tmp_path)
    launches = []
    monkeypatch.setattr(queue, "environment", lambda name: {})

    def launch(name, env, capture_dir=None):
        launches.append(env)
        variant = name.rsplit("_seed", 1)[0]
        out = queue.args.run_dir / variant
        out.mkdir(parents=True)
        (out / "metrics.json").write_text(json.dumps({"env_steps": 30000}))
        assert capture_dir is None
        return True

    monkeypatch.setattr(queue, "run", launch)
    try:
        for variant in ("control", "fixed300"):
            assert queue.train_pilot(variant) == queue.args.run_dir / variant / "latest.pt"
        assert all(e["STEPS"] == "30000" and e["OBS"] == "bev" for e in launches)
        for env in launches:
            assert f"--init-from {queue.source}" in env["TRAIN_ARGS"]
            assert "--resume" not in env["TRAIN_ARGS"]
            assert "--early-stop-evals 0" in env["TRAIN_ARGS"]
        assert "collision_fixed=0.0" in launches[0]["TRAIN_ARGS"]
        assert "collision_fixed=300.0" in launches[1]["TRAIN_ARGS"]
    finally:
        queue.lock.close()


def test_standalone_evaluation_restores_canonical_reward(tmp_path, monkeypatch):
    queue = make_queue(tmp_path)
    calls = []
    monkeypatch.setattr(Queue, "run", lambda self, name, env, capture_dir=None: calls.append(env))
    try:
        queue.run("fixed300_medium_seed2029", {"RUN_MODE": "evaluate", "EVAL_ARGS": "--no-shield"})
        assert calls[0]["EVAL_ARGS"].endswith("--env-set reward.scales.collision_fixed=0.0")
    finally:
        queue.lock.close()


def test_completed_pilot_requires_matching_reward_and_source(tmp_path):
    queue = make_queue(tmp_path)
    out = queue.args.run_dir / "fixed300"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps({"env_steps": 30000,
        "env_overrides": ["reward.scales.time=0.1", "reward.scales.collision_fixed=0.0"],
        "init_from": str(queue.source)}))
    try:
        with pytest.raises(ValueError, match="does not match protocol"):
            queue.train_pilot("fixed300")
    finally:
        queue.lock.close()


@pytest.mark.parametrize("saved_cost", [0.0, 300.0])
def test_resume_requires_same_reward_and_keeps_replay_state(tmp_path, monkeypatch, saved_cost):
    import torch

    queue = make_queue(tmp_path)
    out = queue.args.run_dir / "fixed300"
    out.mkdir(parents=True)
    torch.save({"step": 12000, "run_spec": {
        "arm": "cnn", "obs": "bev", "action": "discrete", "seed": 42,
        "init_from": str(queue.source),
        "env_overrides": ["reward.scales.time=0.1", f"reward.scales.collision_fixed={saved_cost}"],
    }}, out / "latest.pt")
    replay = out / "train_eps"
    replay.mkdir()
    (replay / "saved.npz").write_bytes(b"existing replay")
    launches = []
    monkeypatch.setattr(queue, "environment", lambda name: {})

    def launch(name, env, capture_dir=None):
        launches.append(env)
        (out / "metrics.json").write_text(json.dumps({"env_steps": 30000}))
        return True

    monkeypatch.setattr(queue, "run", launch)
    try:
        if saved_cost == 0:
            with pytest.raises(ValueError, match="Resume checkpoint does not match"):
                queue.train_pilot("fixed300")
            assert launches == []
        else:
            assert queue.train_pilot("fixed300") == out / "latest.pt"
            assert "--resume" in launches[0]["TRAIN_ARGS"]
            assert "--init-from" not in launches[0]["TRAIN_ARGS"]
        assert (replay / "saved.npz").read_bytes() == b"existing replay"
    finally:
        queue.lock.close()


def test_collision_trace_preserves_observations_and_episode_metrics(tmp_path):
    from src.dreamer.carla_wrappers import DreamerObservation, EpisodeMetricsRecorder
    from tests.test_carla_wrappers import FakeCarDreamerTask

    class CollisionTask(FakeCarDreamerTask):
        def step(self, action):
            obs, reward, done, info = super().step(action)
            info.update(collision_detected=True, r_collision=-300.0,
                        r_collision_speed=0.0, r_collision_fixed=-300.0)
            return obs, reward - 300, done, info

    env = EpisodeMetricsRecorder(
        DreamerObservation(CollisionTask(length=1)), tmp_path / "episodes.jsonl")
    env.reset()
    obs, reward, done, _ = env.step(0)
    assert reward == -299 and done
    assert "r_collision" not in obs and "collision_detected" not in obs
    episode = json.loads((tmp_path / "episodes.jsonl").read_text())
    event = json.loads((tmp_path / "collision_events.jsonl").read_text())
    assert episode["collision"] and not episode["success"]
    assert event["r_collision_fixed"] == -300 and event["total_reward"] == -299


def test_runtime_recovery_preserves_reward_budget_and_original_hashes():
    from scripts.run_collision_cost_ablation import runtime_recovery_matches

    saved = {"steps_per_pilot": 30000, "variants": {"control": 0, "fixed300": 300},
             "code_sha256": {"scripts/run_traffic_curriculum.py": "old",
                             "scripts/train_cardreamer.py": "training"}}
    current = {**saved, "code_sha256": {**saved["code_sha256"],
                                        "scripts/run_traffic_curriculum.py": "fixed"}}
    recovery = {"original_protocol_sha256": "original", "from_code_sha256": saved["code_sha256"],
                "to_code_sha256": current["code_sha256"]}
    assert runtime_recovery_matches(saved, current, recovery, "original")
    assert not runtime_recovery_matches(saved, current, None, "original")
    assert not runtime_recovery_matches(saved, current, recovery, "different original")
    assert not runtime_recovery_matches(saved, {**current, "steps_per_pilot": 40000},
                                        recovery, "original")
    changed_training = {**current, "code_sha256": {**current["code_sha256"],
                                                 "scripts/train_cardreamer.py": "changed"}}
    assert not runtime_recovery_matches(saved, changed_training,
        {**recovery, "to_code_sha256": changed_training["code_sha256"]}, "original")


def test_failed_invocation_records_error_and_new_skips(tmp_path, monkeypatch):
    queue = make_queue(tmp_path)
    (queue.out / "status.txt").write_text(
        "2026-10-05 21:03:15 === collision-cost ablation started ===\n"
        "2026-10-05 21:08:15 SKIP fixed300_seed42 (old invocation)\n")

    def fail():
        raise ValueError("preparation failed")

    monkeypatch.setattr(queue, "prepare", fail)
    try:
        with pytest.raises(ValueError, match="preparation failed"):
            queue.execute()
        results = json.loads((queue.out / "results.json").read_text())
        assert not results["completed"] and "preparation failed" in results["error"]
        current_status = (queue.out / "status.txt").read_text().rsplit(
            "=== collision-cost ablation started ===", 1)[-1]
        assert "SKIP fixed300_seed42" in current_status
    finally:
        queue.lock.close()
