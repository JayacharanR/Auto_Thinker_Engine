"""Safety and selection checks for the unattended curriculum controller."""

import json
import subprocess
import sys
import time

import pytest

from scripts.run_traffic_curriculum import Queue, capture_best, parse_args, promotion_gate


def test_promotion_requires_safe_traffic_and_retained_turning():
    traffic = {"success_rate": 0.8, "collision_rate": 0.2}
    retention = {"success_rate": 0.9, "collision_rate": 0.0}
    assert promotion_gate(traffic, retention)
    assert not promotion_gate({**traffic, "collision_rate": 0.3}, retention)
    assert not promotion_gate({**traffic, "success_rate": 0.7}, retention)
    assert not promotion_gate(traffic, {**retention, "success_rate": 0.8})


def test_best_checkpoint_survives_later_regression_and_incomplete_json(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    evals = tmp_path / "eval.jsonl"
    first = {"agent_step": 5000, "success_rate": 0.7, "collision_rate": 0.3,
             "mean_reward": 100.0}
    checkpoint.write_bytes(b"first weights")
    evals.write_text(json.dumps(first) + "\n")
    capture_best(tmp_path)
    checkpoint.write_bytes(b"regressed weights")
    regression = {**first, "agent_step": 7500, "success_rate": 0.1}
    with evals.open("a") as f:
        f.write(json.dumps(regression) + '\n{"agent_step":')
    capture_best(tmp_path)
    assert (tmp_path / "best.pt").read_bytes() == b"first weights"
    assert json.loads((tmp_path / "best_eval.json").read_text()) == first


def test_duplicate_refusal_preserves_stop_and_progress_metadata(tmp_path):
    args = parse_args(["--out", str(tmp_path), "--run-dir", str(tmp_path / "runs")])
    first = Queue(args)
    first.stop.write_text("manual stop\n")
    (tmp_path / "steps.json").write_text("original progress metadata")
    with pytest.raises(SystemExit, match="already running"):
        Queue(args)
    assert first.stop.read_text() == "manual stop\n"
    assert (tmp_path / "steps.json").read_text() == "original progress metadata"
    first.lock.close()


def test_watcher_shows_started_queue_while_waiting_for_gpu(tmp_path, monkeypatch):
    import scripts.watch_queue as watcher

    (tmp_path / "status.txt").write_text(
        "2026-10-05 01:40:30  === overnight traffic curriculum started ===\n")
    (tmp_path / "steps.json").write_text(json.dumps([
        {"name": "baseline_custom_jepa_medium", "hours": 0.1}]))
    monkeypatch.setattr(watcher, "OUT", tmp_path)
    lines = watcher.render({})
    assert "waiting" in lines[0]
    assert "0.0%" in lines[1]
    assert "steps done 0/1" in lines[1]


def test_watcher_eta_includes_training_before_first_rate(tmp_path, monkeypatch):
    import scripts.watch_queue as watcher

    (tmp_path / "status.txt").write_text(
        "2026-10-05 21:03:15  === collision-cost ablation started ===\n"
        "2026-10-05 21:03:15  START control_seed42\n")
    (tmp_path / "steps.json").write_text(json.dumps([
        {"name": "control_seed42", "hours": 1.4,
         "logdir": str(tmp_path / "control"), "target": 30000},
        {"name": "fixed300_seed42", "hours": 1.4},
    ]))
    monkeypatch.setattr(watcher, "OUT", tmp_path)
    assert "ETA ~2h48m" in watcher.render({})[1]


def test_watcher_rate_excludes_random_prefill_step_jump():
    from scripts.watch_queue import rate_from_records

    records = [{"time": 0, "agent_step": 0}, {"time": 20, "agent_step": 0},
               {"time": 30, "agent_step": 2500}, {"time": 80, "agent_step": 2800}]
    assert rate_from_records(records) == 6.0
    assert rate_from_records(records[:3]) is None


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("sensors", 10),
                                   FileNotFoundError("sensors")])
@pytest.mark.parametrize("overheated", [False, True])
def test_missing_cpu_telemetry_keeps_training_and_critical_guards(tmp_path, monkeypatch,
                                                                error, overheated):
    queue = Queue(parse_args(["--out", str(tmp_path / "queue")]))

    def run(command, **kwargs):
        if command[0] == "sensors":
            raise error
        assert command[0] == "nvidia-smi"
        return subprocess.CompletedProcess(command, 0,
                                           "95,7800,80,70\n" if overheated else "55,7800,80,70\n")

    monkeypatch.setattr(subprocess, "run", run)
    if overheated:
        queue.hot_since = time.monotonic() - 91
    try:
        assert queue.check_health() is overheated
        assert "cpu=?" in (queue.out / "health.log").read_text()
        assert type(error).__name__ in (queue.out / "monitor_warnings.log").read_text()
        if overheated:
            assert "GPU >= 92 C" in queue.stop.read_text()
    finally:
        queue.lock.close()


def test_controller_exception_records_failed_stage_and_requests_recovery(tmp_path, monkeypatch):
    queue = Queue(parse_args(["--out", str(tmp_path / "queue")]))
    calls = []

    def health():
        calls.append(True)
        if len(calls) > 1:
            raise RuntimeError("monitor failed")
        return False

    class Child:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(queue, "check_health", health)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(queue, "stop_owned_client", lambda *a: None)
    try:
        with pytest.raises(RuntimeError, match="monitor failed"):
            queue.run("fixed300_seed42", {})
        assert queue.stop.exists()
        assert "FAIL fixed300_seed42" in (queue.out / "status.txt").read_text()
    finally:
        queue.lock.close()


def test_stop_signals_only_owned_main_client_once(tmp_path):
    directory = tmp_path / "scripts"
    directory.mkdir()
    script = directory / "train_cardreamer.py"
    script.write_text('''import os, signal, sys, time
from pathlib import Path
out = Path(sys.argv[1])
worker = os.fork()
if worker == 0:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True: time.sleep(0.02)
def stop(*_):
    out.write_text("checkpoint saved")
    os.kill(worker, signal.SIGTERM)
    os.waitpid(worker, 0)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
out.with_suffix(".ready").write_text(str(worker))
while True: time.sleep(0.02)
''')
    saved = tmp_path / "saved"
    launcher = subprocess.Popen(["bash", "-c", '"$1" "$2" "$3" & wait',
                                 "launcher", sys.executable, str(script), str(saved)])
    queue = Queue(parse_args(["--out", str(tmp_path / "queue")]))
    try:
        deadline = time.monotonic() + 5
        while not saved.with_suffix(".ready").exists():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        signaled = set()
        queue.stop_owned_client(launcher.pid, signaled)
        queue.stop_owned_client(launcher.pid, signaled)
        launcher.wait(timeout=5)
        assert len(signaled) == 1  # main trainer only; worker is cleaned up by main
        assert saved.read_text() == "checkpoint saved"
    finally:
        if launcher.poll() is None:
            queue.stop_owned_client(launcher.pid, set())
            launcher.wait(timeout=5)
        queue.lock.close()
