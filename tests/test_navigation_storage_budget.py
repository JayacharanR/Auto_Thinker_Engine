import json
import shutil
import time

import pytest

from src.navigation import storage_budget as storage


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(
        total=40 * 10**9, used=10 * 10**9, free=30 * 10**9))
    budget = tmp_path / "outputs/guard"
    budget.mkdir(parents=True)
    state = dict(status="running", updated_at=time.time(), effective_floor_bytes=10**10)
    (budget / "progress.json").write_text(json.dumps(state))
    (budget / "watched_roots.json").write_text(json.dumps(dict(roots=[])))
    root = tmp_path / "outputs/candidate"
    root.mkdir()
    return root, budget, state


def test_register_owned_only_and_reject_stale_guardian(tmp_path, monkeypatch):
    root, budget, state = setup(tmp_path, monkeypatch)
    storage.register_output(root, budget)
    assert json.loads((budget / "watched_roots.json").read_text())["roots"] == [str(root)]
    with pytest.raises(ValueError, match="Only owned"):
        storage.register_output(tmp_path / "unrelated", budget)
    state["updated_at"] -= 30
    (budget / "progress.json").write_text(json.dumps(state))
    with pytest.raises(InterruptedError, match="live approved"):
        storage.require_budget(budget)


def test_loss_of_guardian_stops_owned_queue_and_preserves_artifacts(tmp_path, monkeypatch):
    root, budget, state = setup(tmp_path, monkeypatch)
    data = root / "retained.bin"
    data.write_bytes(b"must survive the resource stop")
    with storage.BudgetWatch(root, budget, poll_seconds=.01):
        state["status"] = "budget_reached"
        (budget / "progress.json").write_text(json.dumps(state))
        deadline = time.monotonic() + 2
        while not (root / "STOP").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (root / "STOP").exists()
    # STOP is deliberately published first. Exiting joins the monitor before
    # testing the later atomic audit receipt, rather than racing its write.
    assert (root / "storage_stop.json").exists()
    assert data.read_bytes() == b"must survive the resource stop"
