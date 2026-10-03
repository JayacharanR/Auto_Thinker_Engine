"""Phase 3 comparison bookkeeping, with training stubbed out."""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.train_cardreamer as trainer  # noqa: E402


def _fake_run(logdir: Path, seed: int, successes=(0.2, 0.9, 1.0)):
    logdir.mkdir(parents=True, exist_ok=True)
    with open(logdir / "eval.jsonl", "w") as f:
        for i, rate in enumerate(successes, 1):
            f.write(json.dumps({"agent_step": i * 10000, "success_rate": rate}) + "\n")
    final = {"success_rate": successes[-1], "collision_rate": 0.1 * seed % 1,
             "out_of_lane_rate": 0.0, "mean_reward": 100.0 + seed}
    (logdir / "metrics.json").write_text(json.dumps({
        "final_eval": final, "wall_clock_sec": 7200, "peak_vram_mb": 4096,
        "train_steps_per_sec": 15.0,
    }))


def test_summarize_run(tmp_path):
    _fake_run(tmp_path, seed=1)
    summary = trainer.summarize_run(tmp_path, success_threshold=0.8)
    assert summary["steps_to_threshold"] == 20000
    assert summary["success_rate"] == 1.0 and summary["eval_return"] == 101.0
    assert summary["wall_clock_hours"] == 2.0 and summary["peak_vram_gb"] == 4.0
    _fake_run(tmp_path, seed=1, successes=(0.1, 0.5))
    assert np.isnan(trainer.summarize_run(tmp_path)["steps_to_threshold"])


def test_comparison_trains_missing_runs_and_aggregates(tmp_path, monkeypatch):
    calls = []

    def fake_train_arm(arm, task, seed, phase3_config, steps, logdir, resume, **kwargs):
        calls.append((arm, seed, resume))
        _fake_run(Path(logdir), seed)

    monkeypatch.setattr(trainer, "train_arm", fake_train_arm)
    _fake_run(tmp_path / "cnn_seed1", seed=1)  # already finished: reused
    (tmp_path / "cnn_seed2").mkdir()
    (tmp_path / "cnn_seed2" / "latest.pt").write_bytes(b"")  # interrupted: resumed

    config = {"experiment": {"seeds": [1, 2]}}
    trainer.run_comparison("task", config, steps=100, arms=("cnn", "vjepa2"),
                           output_dir=str(tmp_path), obs="bev")

    assert calls == [("cnn", 2, True), ("vjepa2", 1, False), ("vjepa2", 2, False)]
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert [(r["arm"], r["seed"]) for r in result["runs"]] == [
        ("cnn", 1), ("cnn", 2), ("vjepa2", 1), ("vjepa2", 2)
    ]
    table = (tmp_path / "comparison.md").read_text()
    assert "| steps_to_threshold |" in table and "cnn" in table and "vjepa2" in table


def test_aggregate_only_skips_unfinished_runs(tmp_path, monkeypatch):
    def must_not_train(**kwargs):
        raise AssertionError("aggregate-only must not train")

    monkeypatch.setattr(trainer, "train_arm", must_not_train)
    _fake_run(tmp_path / "cnn_seed42", seed=42)
    trainer.run_comparison("task", {}, steps=100, arms=("cnn", "vjepa2"), seeds=[42],
                           output_dir=str(tmp_path), train_missing=False, obs="camera_route")
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert [(r["arm"], r["seed"]) for r in result["runs"]] == [("cnn", 42)]
    assert result["seeds"] == [42]


def test_aggregate_only_with_partial_runs_never_trains(tmp_path, monkeypatch):
    def must_not_train(**kwargs):
        raise AssertionError("aggregate-only must not train")

    monkeypatch.setattr(trainer, "train_arm", must_not_train)
    _fake_run(tmp_path / "cnn_seed42", seed=42)
    _fake_run(tmp_path / "vjepa2_seed42", seed=42, successes=(0.0, 1.0))
    (tmp_path / "vjepa2_seed42" / "metrics.json").unlink()  # stopped before finishing
    trainer.run_comparison("task", {}, steps=100, arms=("cnn", "vjepa2"), seeds=[42],
                           output_dir=str(tmp_path), train_missing=False,
                           include_partial=True, obs="camera_route")
    runs = json.loads((tmp_path / "comparison.json").read_text())["runs"]
    vjepa2 = next(r for r in runs if r["arm"] == "vjepa2")
    assert vjepa2["partial"] == 1.0 and vjepa2["steps_to_threshold"] == 20000
    assert vjepa2["success_by_12.5k"] == 0.0 and vjepa2["success_by_22.5k"] == 1.0
