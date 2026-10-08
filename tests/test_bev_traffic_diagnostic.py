"""A completed lower-budget diagnostic must still resume when its target grows."""

import json
from types import SimpleNamespace

from scripts.run_bev_traffic_diagnostic import Diagnostic, plan


def make_queue(tmp_path, target):
    run = tmp_path / "run"
    run.mkdir()
    (run / "latest.pt").write_bytes(b"latest checkpoint")
    (run / "best.pt").write_bytes(b"selected checkpoint")
    (run / "metrics.json").write_text(json.dumps({"env_steps": 30000,
                                                "early_stopped_at": None}))
    args = SimpleNamespace(out=tmp_path / "queue", run_dir=run,
                           init_from=tmp_path / "original.pt", steps=target,
                           max_hours=3, eval_seed=2026)
    return Diagnostic(args, steps=plan(args))


def test_higher_target_resumes_completed_run_with_same_contract(tmp_path, monkeypatch):
    queue = make_queue(tmp_path, 60000)
    launches = []
    monkeypatch.setattr(queue, "environment", lambda *_: {})

    def launch(name, env, capture_dir=None):
        launches.append(env)
        return True

    monkeypatch.setattr(queue, "run", launch)
    try:
        assert queue.train_bev() == queue.args.run_dir / "best.pt"
        assert len(launches) == 1
        env = launches[0]
        assert env["STEPS"] == "60000"
        assert env["OBS"] == "bev" and env["ACTION"] == "discrete"
        assert "--resume" in env["TRAIN_ARGS"]
        assert "--init-from" not in env["TRAIN_ARGS"]
        assert "reward.scales.time=0.1" in env["TRAIN_ARGS"]
    finally:
        queue.lock.close()


def test_same_completed_target_does_not_launch_duplicate_training(tmp_path, monkeypatch):
    queue = make_queue(tmp_path, 30000)

    def launch(*args, **kwargs):
        raise AssertionError("Completed target should not train again")

    monkeypatch.setattr(queue, "run", launch)
    try:
        assert queue.train_bev() == queue.args.run_dir / "best.pt"
    finally:
        queue.lock.close()
