"""Continuation must preserve experience, score joint outcomes and retain skill."""

import json

from scripts.run_overnight_bev_improvement import (
    Overnight,
    clone_training_state,
    improvement,
    make_steps,
    parse_args,
    rank,
)


def measurements(counts=(21, 25, 25), simple=10, collisions=None, lanes=0):
    result = {}
    for seed, safe in zip((2028, 2029, 2030), counts):
        result[f"medium_seed{seed}"] = {
            "collision_free_arrivals": safe,
            "summary": {"collision_rate": (30 - safe) / 30 if collisions is None else collisions,
                        "out_of_lane_rate": lanes / 30},
        }
    result["simple_seed2029"] = {
        "collision_free_arrivals": simple, "summary": {"collision_rate": 0}}
    return result


def test_selection_rejects_overlap_reward_gain_retention_loss_and_stream_regression():
    baseline = measurements()
    better = measurements((22, 26, 26))
    assert improvement(better, baseline)
    assert not improvement(measurements((22, 25, 25)), baseline)  # one extra episode
    assert not improvement(measurements((22, 26, 26), simple=8), baseline)
    assert not improvement(measurements((18, 30, 30)), baseline)  # weaker worst stream
    assert not improvement(measurements((22, 26, 26), collisions=0.4), baseline)
    assert not improvement(measurements((22, 26, 26), lanes=1), baseline)
    overlap = measurements()
    for m in overlap.values():
        m["summary"].update(success_rate=1, mean_reward=9999)
    assert rank(overlap) == rank(baseline)
    assert not improvement(overlap, baseline)


def test_cloned_replay_and_optimizer_checkpoint_are_independent_of_source(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "branch"
    (source / "train_eps").mkdir(parents=True)
    (source / "train_eps" / "episode.npz").write_bytes(b"saved experience")
    (source / "latest.pt").write_bytes(b"weights and optimizer state")
    (source / "metrics.json").write_text('{"env_steps": 30000}')
    clone_training_state(source, destination)
    assert (destination / "latest.pt").read_bytes() == b"weights and optimizer state"
    assert (destination / "train_eps" / "episode.npz").read_bytes() == b"saved experience"
    assert not (destination / "metrics.json").exists()
    (destination / "latest.pt").write_bytes(b"new state")
    (destination / "train_eps" / "episode.npz").write_bytes(b"new replay")
    assert (source / "latest.pt").read_bytes() == b"weights and optimizer state"
    assert (source / "train_eps" / "episode.npz").read_bytes() == b"saved experience"


def test_training_extends_checkpoint_even_if_previous_budget_metrics_exist(tmp_path, monkeypatch):
    import scripts.run_overnight_bev_improvement as module

    args = parse_args(["--out", str(tmp_path / "queue"), "--run-dir", str(tmp_path / "run")])
    args.run_dir.mkdir()
    (args.run_dir / "metrics.json").write_text('{"env_steps": 60000}')
    queue = Overnight(args, steps=make_steps(args))
    state = {"step": 60000}
    spec = {"seed": 42, "task": "carla_right_turn_medium",
            "env_overrides": ["reward.scales.time=0.1", "reward.scales.collision_fixed=0.0"]}
    monkeypatch.setattr(module, "checkpoint_step", lambda path: (state["step"], spec))
    monkeypatch.setattr(queue, "environment", lambda name: {})
    calls = []

    def run(name, env):
        calls.append(env)
        state["step"] = 90000
        return True

    monkeypatch.setattr(queue, "run", run)
    try:
        queue.train_block(1, 90000, spec)
        assert len(calls) == 1 and calls[0]["STEPS"] == "90000"
        assert "--resume" in calls[0]["TRAIN_ARGS"]
        assert "--init-from" not in calls[0]["TRAIN_ARGS"]
        assert "--early-stop-evals 0" in calls[0]["TRAIN_ARGS"]
    finally:
        queue.lock.close()


def test_continuation_bar_counts_only_additional_steps(tmp_path, monkeypatch):
    import scripts.watch_queue as watcher

    run = tmp_path / "run"
    run.mkdir()
    (run / "episodes.jsonl").write_text('\n' + json.dumps({"agent_step": 75000}) + '\n')
    (tmp_path / "status.txt").write_text(
        "2026-10-06 00:00:00  === overnight BEV improvement started ===\n"
        "2026-10-06 00:00:01  START continue_block1\n")
    (tmp_path / "steps.json").write_text(json.dumps([
        {"name": "continue_block1", "hours": 1.5, "logdir": str(run),
         "start": 60000, "target": 90000}]))
    monkeypatch.setattr(watcher, "OUT", tmp_path)
    lines = watcher.render({})
    assert "50.0%" in lines[0] and "75,000/90,000" in lines[0]
    assert "50.0%" in lines[1]


def test_full_controller_keeps_baseline_after_regression_and_checks_unseen_streams(
        tmp_path, monkeypatch):
    import scripts.run_overnight_bev_improvement as module

    source = tmp_path / "source"
    (source / "train_eps").mkdir(parents=True)
    (source / "train_eps" / "episode.npz").write_bytes(b"replay")
    (source / "latest.pt").write_bytes(b"baseline weights and optimizers")
    args = parse_args(["--out", str(tmp_path / "queue"),
                       "--run-dir", str(tmp_path / "branch"), "--blocks", "1"])
    queue = Overnight(args, steps=make_steps(args))
    spec = {"task": "carla_right_turn_medium", "seed": 42, "env_overrides": []}
    monkeypatch.setattr(module, "checkpoint_step", lambda path: (30000, spec))
    monkeypatch.setattr(queue, "prepare", lambda: None)
    monkeypatch.setattr(queue, "journal", lambda *args: None)
    monkeypatch.setattr(queue, "check_health", lambda: False)
    monkeypatch.setattr(queue, "finish_pilot", lambda: {})
    monkeypatch.setattr(queue, "choose_source", lambda _: {"winner": "reference", "choices": {
        "reference": {"source_dir": str(source), "checkpoint": str(source / "latest.pt"),
                      "evaluations": measurements()}}})

    def train(block, target, spec):
        (args.run_dir / "latest.pt").write_bytes(b"regressed weights")

    monkeypatch.setattr(queue, "train_block", train)
    requests = []

    def measure(name, checkpoint, task, seed, episodes):
        requests.append((name, seed, episodes))
        safe = 10 if task == "simple" else 20
        return {"collision_free_arrivals": safe,
                "summary": {"num_episodes": episodes,
                            "collision_rate": (episodes - safe) / episodes,
                            "out_of_lane_rate": 0}}

    monkeypatch.setattr(queue, "measure", measure)
    try:
        queue.execute()
        result = json.loads((args.out / "results.json").read_text())
        assert result["completed"] and not result["blocks"][0]["accepted"]
        assert (args.out / "best.pt").read_bytes() == (source / "latest.pt").read_bytes()
        assert (args.run_dir / "latest.pt").read_bytes() == b"regressed weights"
        assert result["best_step"] == 30000
        assert {seed for name, seed, _ in requests if name.startswith("holdout")} == {
            3031, 4031, 5031}
        assert (args.out / "summary.md").exists()
    finally:
        queue.lock.close()
